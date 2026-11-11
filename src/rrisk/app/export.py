"""Static, single-file export of the app (read-only: no SQL box or pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from . import payload

STATIC = Path(__file__).with_name("static")


def build(out: str | Path | None = None) -> Path:
    out = Path(out or config.OUTPUT / "rrisk_atlas.html")
    data = json.dumps(payload.build(), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (STATIC / "index.html").read_text().replace(
        "<!--__EMBED__-->", f'<script type="application/json" id="embedded">{data}</script>'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
