"""Static export of the app (read-only: no SQL box or pipeline).

Writes the page as one HTML file plus a ``footprints/`` folder beside it holding
the US site outlines, sharded into small JSON files the page fetches on demand
(the live app serves them from /api/footprint instead). Host both together; opened
straight from disk (file://) the page still works, just without outlines.
"""

from __future__ import annotations

import json
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


def _footprints(folder: Path) -> int:
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
    return len(rows)


def build(out: str | Path | None = None) -> Path:
    out = Path(out or config.OUTPUT / "rrisk_atlas.html")
    data = json.dumps(payload.build(), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (STATIC / "index.html").read_text().replace(
        "<!--__EMBED__-->", f'<script type="application/json" id="embedded">{data}</script>'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    n = _footprints(out.parent / "footprints")
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) + {n} outlines in {out.parent / 'footprints'}/")
    return out
