"""Search the GDELT DOC 2.0 API for candidate loss-event coverage.

GDELT indexes worldwide online news (machine-translated from ~65 languages)
and returns at most 250 articles per request, so we tile the query over time
windows. GDELT asks for <= 1 request per 5 s but throttles harder when busy;
we space requests 30 s apart, back off on its throttle message, and cache every
window so an interrupted harvest resumes.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from ... import config

API = "https://api.gdeltproject.org/api/v2/doc/doc"
RAW = config.RAW / "gdelt"

# Asset term x damage term. GDELT needs OR-groups in parentheses and quoted
# phrases; keep each query short (long queries are rejected).
ASSET_QUERIES = {
    "wind": '("wind turbine" OR "wind farm" OR "wind project")',
    "solar": '("solar farm" OR "solar project" OR "solar plant" OR "solar panels")',
}
DAMAGE_QUERIES = {
    "wind": "(collapse OR collapsed OR fire OR blade OR toppled OR struck OR damaged)",
    "solar": "(hail OR tornado OR fire OR damaged OR destroyed OR storm)",
}


class GdeltThrottled(RuntimeError):
    pass


def _get(params: dict, min_gap: float | None = None, tries: int = 8) -> dict | None:
    """GET with spacing and patient backoff. Returns None if GDELT keeps throttling.

    GDELT's stated limit is one request per 5 s, but in practice it throttles
    much harder at busy times, so we default to a 30 s gap (news.gdelt_gap_seconds in rrisk.toml).
    """
    gap = float(config.SETTINGS["news"]["gdelt_gap_seconds"]) if min_gap is None else min_gap
    wait = _get.last + gap - time.monotonic()  # type: ignore[attr-defined]
    if wait > 0:
        time.sleep(wait)
    delay = 60.0
    for _ in range(tries):
        _get.last = time.monotonic()  # type: ignore[attr-defined]
        try:
            r = httpx.get(API, params=params, timeout=60, headers={"User-Agent": config.USER_AGENT})
        except httpx.HTTPError as e:
            print(f"  [gdelt] network error {e}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            continue
        body = r.text.strip()
        if r.status_code == 429 or body.startswith("Please limit"):
            time.sleep(delay)
            delay = min(delay * 1.5, 300)
            continue
        if not body:
            return {"articles": []}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # GDELT returns plain-text errors (e.g. query too short/long).
            raise ValueError(f"GDELT error: {body[:200]}")
    return None


_get.last = 0.0  # type: ignore[attr-defined]


def _windows(start: date, end: date, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def search(
    start: date,
    end: date,
    techs=config.TECHS,
    window_days: int = 30,
    extra: str = "",
) -> None:
    """Tile the asset x damage queries over [start, end), newest window first,
    caching each response. Re-running resumes where it stopped."""
    RAW.mkdir(parents=True, exist_ok=True)
    wins = list(_windows(start, end, window_days))[::-1]
    for a, b in wins:
        for tech in techs:
            q = f"{ASSET_QUERIES[tech]} {DAMAGE_QUERIES[tech]} {extra}".strip()
            key = hashlib.sha1(f"{q}|{a}|{b}".encode()).hexdigest()[:16]
            cache = RAW / f"{tech}_{a:%Y%m%d}_{b:%Y%m%d}_{key}.json"
            if cache.exists():
                continue
            params = {
                "query": q,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 250,
                # relevance ranking keeps the best 250 when a window is capped
                "sort": "hybridrel",
                "startdatetime": f"{a:%Y%m%d}000000",
                "enddatetime": f"{b:%Y%m%d}000000",
            }
            try:
                data = _get(params)
            except ValueError as e:
                print(f"  [gdelt] {tech} {a}: {e}", flush=True)
                continue
            if data is None:
                print(f"  [gdelt] {tech} {a}..{b}: still throttled, skipped (re-run to resume)", flush=True)
                continue
            cache.write_text(json.dumps(data))
            n = len(data.get("articles", []) or [])
            cap = "  (hit the 250 cap)" if n >= 250 else ""
            print(f"  [gdelt] {tech} {a}..{b}: {n} articles{cap}", flush=True)


def collect() -> pd.DataFrame:
    """All cached GDELT responses -> one de-duplicated article table."""
    rows = []
    for f in sorted(RAW.glob("*.json")):
        tech = f.name.split("_", 1)[0]
        for art in json.loads(f.read_text()).get("articles", []) or []:
            rows.append({"query_tech": tech, **art})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    return df.drop_duplicates("url").reset_index(drop=True)


def run(start: str, end: str | None = None, window_days: int = 30) -> pd.DataFrame:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
    search(s, e, window_days=window_days)
    df = collect()
    out = config.PROCESSED / "news_articles.parquet"
    df.to_parquet(out, index=False)
    print(f"[gdelt] {len(df):,} unique candidate articles in {out.name}")
    return df
