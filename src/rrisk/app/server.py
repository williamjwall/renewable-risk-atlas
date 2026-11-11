"""Local web app: `rrisk app` -> http://127.0.0.1:8765

Serves the front-end and a small JSON API:
  GET  /api/bootstrap    everything the map/tables need (cached until the warehouse changes)
  GET  /api/status       pipeline steps with state, last run and errors
  POST /api/run          start a pipeline job  {"steps": [...] | null, "force": false}
  GET  /api/job          progress/log of the current or last job
  GET  /api/schema       warehouse tables and columns
  GET  /api/footprint/{site_id}  a US site's real outline (GeoJSON geometry)
  POST /api/query        read-only SQL  {"sql": "...", "limit": 1000}
  POST /api/query.csv    same, full result as CSV
Binds to localhost only.
"""

from __future__ import annotations

import datetime as dt
import decimal
import io
import json
import threading
import time
from pathlib import Path

import duckdb
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .. import config, pipeline
from . import payload

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="rrisk", docs_url="/api/docs")
app.add_middleware(GZipMiddleware, minimum_size=2000)

_cache: dict = {"mtime": None, "body": None}
_cache_lock = threading.Lock()
_job_lock = threading.Lock()
_job: dict = {"running": False, "steps": [], "log": [], "ok": None, "started": None, "finished": None}


def _ro():
    return duckdb.connect(str(config.WAREHOUSE), read_only=True, config={"enable_external_access": False})


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/api/bootstrap")
def bootstrap():
    if not config.WAREHOUSE.exists():
        return JSONResponse({"error": "No data yet. Open the Data tab and run Update."}, status_code=404)
    with _cache_lock:
        m = config.WAREHOUSE.stat().st_mtime
        if _cache["mtime"] != m:
            _cache["body"] = json.dumps(payload.build(), separators=(",", ":"), allow_nan=False)
            _cache["mtime"] = m
        return Response(_cache["body"], media_type="application/json")


@app.get("/api/footprint/{site_id:path}")
def footprint(site_id: str):
    if not config.WAREHOUSE.exists():
        return JSONResponse({"error": "No data yet."}, status_code=404)
    con = _ro()
    try:
        row = con.execute("select kind, area_km2, radius_km, geojson from footprints where site_id = ?",
                          [site_id]).fetchone()
    except duckdb.CatalogException:
        row = None
    finally:
        con.close()
    if row is None:
        return JSONResponse({"error": "No outline for this site."}, status_code=404)
    return {"site_id": site_id, "kind": row[0], "area_km2": row[1], "radius_km": row[2],
            "geometry": json.loads(row[3])}


@app.get("/api/status")
def status():
    return {
        "steps": pipeline.status(),
        "job": {k: v for k, v in _job.items() if k != "log"},
        "settings": config.SETTINGS,
        "warehouse": config.WAREHOUSE.exists(),
    }


class RunReq(BaseModel):
    steps: list[str] | None = None
    force: bool = False


def _run_job(steps, force):
    def log(line):
        _job["log"].append(line)
        del _job["log"][:-600]

    try:
        ok = pipeline.update(steps, force=force, log=log)
    except BaseException as e:  # noqa: BLE001 — SystemExit from bad step names too
        log(f"✗ {e}")
        ok = False
    _job.update(running=False, ok=ok, finished=time.strftime("%H:%M:%S"))


@app.post("/api/run")
def run(req: RunReq):
    with _job_lock:
        if _job["running"]:
            return JSONResponse({"error": "A job is already running."}, status_code=409)
        _job.update(running=True, steps=req.steps or ["update"], log=[], ok=None,
                    started=time.strftime("%H:%M:%S"), finished=None)
        threading.Thread(target=_run_job, args=(req.steps, req.force), daemon=True).start()
    return {"started": True}


@app.get("/api/job")
def job(since: int = 0):
    return {**{k: v for k, v in _job.items() if k != "log"}, "log": _job["log"][since:], "n": len(_job["log"])}


@app.get("/api/schema")
def schema():
    if not config.WAREHOUSE.exists():
        return {"tables": []}
    con = _ro()
    try:
        rows = con.sql(
            "select table_name, column_name, data_type from information_schema.columns "
            "order by table_name, ordinal_position"
        ).fetchall()
    finally:
        con.close()
    tables: dict[str, list] = {}
    for t, c, ty in rows:
        tables.setdefault(t, []).append([c, ty])
    from .. import docs

    order = list(docs.TABLES)
    names = sorted(tables, key=lambda t: (order.index(t) if t in order else 99, t))
    return {"tables": [{"name": t, "doc": docs.TABLES.get(t, ""),
                        "columns": [[c, ty, docs.COLUMNS.get(c, "") if t in ("site_risk", "sites", "site_scores") else ""]
                                    for c, ty in tables[t]]} for t in names]}


class QueryReq(BaseModel):
    sql: str
    limit: int = 1000


def _jsonable(v):
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
        return v
    return str(v)


@app.post("/api/query")
def query(req: QueryReq):
    t0 = time.time()
    try:
        con = _ro()
        try:
            cur = con.execute(req.sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(req.limit + 1)
        finally:
            con.close()
    except duckdb.Error as e:
        return JSONResponse({"error": str(e).split("\n")[0][:500]}, status_code=400)
    return {
        "columns": cols,
        "rows": [[_jsonable(v) for v in r] for r in rows[: req.limit]],
        "truncated": len(rows) > req.limit,
        "ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/query.csv")
def query_csv(req: QueryReq):
    try:
        con = _ro()
        try:
            df = con.execute(req.sql).df()
        finally:
            con.close()
    except duckdb.Error as e:
        return PlainTextResponse(str(e).split("\n")[0], status_code=400)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="rrisk_query.csv"'})


def serve(host: str | None = None, port: int | None = None, open_browser: bool = True) -> None:
    import uvicorn

    host = host or config.SETTINGS["app"]["host"]
    port = int(port or config.SETTINGS["app"]["port"])
    url = f"http://{host}:{port}"
    if open_browser:
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"rrisk app on {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
