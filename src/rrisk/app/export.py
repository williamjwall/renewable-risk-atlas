"""Static export of the app (read-only: no SQL box or pipeline).

Writes the page as one HTML file plus a ``footprints/`` folder beside it holding
the US site outlines, sharded into small JSON files the page fetches on demand
(the live app serves them from /api/footprint instead). Host both together; opened
straight from disk (file://) the page still works, just without outlines.

It also writes ``share/``: per-site link-preview facts (name, place, score and a
satellite frame), sharded the same way, plus ``og.jpg`` beside the page as the
default preview image. ``middleware.js`` at the repo root reads these to give a
shared ``?site=`` link a preview with that site's name and a satellite photo.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import duckdb

from .. import config
from . import payload

STATIC = Path(__file__).with_name("static")
SHARDS = 256  # keep in sync with fpShard() in static/index.html


def shard(site_id: str) -> int:
    """32-bit FNV-1a over UTF-16 code units, matching the page's JS."""
    h = 0x811C9DC5
    b = site_id.encode("utf-16-le")
    for k in range(0, len(b), 2):
        h = ((h ^ int.from_bytes(b[k : k + 2], "little")) * 0x01000193) & 0xFFFFFFFF
    return h % SHARDS


def _footprints(folder: Path) -> dict[str, dict]:
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    try:
        rows = con.execute("select site_id, geojson from footprints").fetchall()
    except duckdb.CatalogException:
        rows = []
    finally:
        con.close()
    buckets: dict[int, dict] = {}
    for sid, gj in rows:
        buckets.setdefault(shard(sid), {})[sid] = json.loads(gj)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for b, sites in buckets.items():
        (folder / f"{b}.json").write_text(json.dumps(sites, separators=(",", ":")))
    return {sid: json.loads(gj) for sid, gj in rows}


OG_W, OG_H, R_EARTH = 1200, 630, 6378137.0


def _merc(lon: float, lat: float) -> tuple[float, float]:
    return math.radians(lon) * R_EARTH, R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _points(coords) -> list:
    if coords and isinstance(coords[0], (int, float)):
        return [coords]
    return [p for c in coords for p in _points(c)]


def og_frame(lat: float, lon: float, tech: str, mw: float | None, geom: dict | None) -> list[float]:
    """Web-Mercator bbox at the preview's 1200×630 aspect, framed like satFrame() in the page."""
    pts = _points(geom.get("coordinates") or []) if geom else []
    if pts:
        ms = [_merc(p[0], p[1]) for p in pts]
        x0, x1 = min(m[0] for m in ms), max(m[0] for m in ms)
        y0, y1 = min(m[1] for m in ms), max(m[1] for m in ms)
    else:
        # no outline: size the frame from capacity (wind farms spread far wider)
        cx, cy = _merc(lon, lat)
        km = max(0.5, min(15.0, math.sqrt((mw or 1) * (0.3 if "wind" in tech else 0.03))))
        m = km * 1000 / math.cos(math.radians(lat))
        x0, y0, x1, y1 = cx - m, cy - m, cx + m, cy + m
    w, h = (x1 - x0) * 1.3 or 1000.0, (y1 - y0) * 1.3 or 1000.0
    if w / h < OG_W / OG_H:
        w = h * OG_W / OG_H
    else:
        h = w * OG_H / OG_W
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [round(v, 1) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]


def _share(folder: Path, geoms: dict[str, dict]) -> int:
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    try:
        rows = con.execute(
            """select site_id, name, city, state, country_name, primary_tech, total_mw, score, tier, lat, lon
               from site_risk where lat is not null and lon is not null"""
        ).fetchall()
    finally:
        con.close()
    buckets: dict[int, dict] = {}
    for sid, name, city, st, cn, tech, mw, score, tier, lat, lon in rows:
        buckets.setdefault(shard(sid), {})[sid] = {
            "name": name or sid,
            "place": ", ".join(x for x in (city, st, cn) if x),
            "tech": tech,
            "mw": None if mw is None or math.isnan(mw) else round(mw, 1),
            "score": None if score is None or math.isnan(score) else round(score),
            "tier": tier,
            "bb": og_frame(lat, lon, tech or "", mw, geoms.get(sid)),
        }
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for b, sites in buckets.items():
        (folder / f"{b}.json").write_text(json.dumps(sites, separators=(",", ":"), ensure_ascii=False))
    return len(rows)


def build(out: str | Path | None = None) -> Path:
    out = Path(out or config.OUTPUT / "rrisk_atlas.html")
    data = json.dumps(payload.build(), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (STATIC / "index.html").read_text().replace(
        "<!--__EMBED__-->", f'<script type="application/json" id="embedded">{data}</script>'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    geoms = _footprints(out.parent / "footprints")
    n_share = _share(out.parent / "share", geoms)
    shutil.copyfile(STATIC / "og.jpg", out.parent / "og.jpg")
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) + {len(geoms)} outlines in "
          f"{out.parent / 'footprints'}/ + {n_share} link previews in {out.parent / 'share'}/")
    return out
