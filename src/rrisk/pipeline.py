"""Pipeline steps, their inputs/outputs, and an incremental runner.

A step is *stale* when an output is missing or its inputs changed since its last
successful run (fingerprint of input paths, sizes and mtimes).
``update()`` runs the stale steps in order (downloads only when files are missing).
Run history is kept in data/pipeline_state.json for the CLI and the app.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Callable

from . import config

R, P, O = config.RAW, config.PROCESSED, config.OUTPUT
C = Path(__file__).parent  # code: a step re-runs when its own source changes


@dataclass
class Step:
    name: str
    title: str
    run: Callable[[], object]
    inputs: list[str] = field(default_factory=list)   # glob patterns
    outputs: list[str] = field(default_factory=list)  # file paths
    manual: bool = False
    only_if_missing: bool = False  # downloads: never "stale", only missing
    help: str = ""

    def input_files(self) -> list[Path]:
        return [Path(p) for pat in self.inputs for p in glob(str(pat))]

    def missing(self) -> list[str]:
        return [o for o in self.outputs if not Path(o).exists()]

    def signature(self) -> str:
        """Fingerprint of the inputs (path, size, mtime)."""
        h = hashlib.sha1()
        for p in sorted(self.input_files()):
            st = p.stat()
            h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()

    def stale(self, state: dict | None = None) -> bool:
        if self.missing():
            return True
        if self.only_if_missing or not self.inputs:
            return False
        rec = (state if state is not None else load_state()).get(self.name, {})
        if rec.get("inputs_sig"):
            return rec["inputs_sig"] != self.signature()
        # No record yet: fall back to timestamps.
        ins = self.input_files()
        if not ins:
            return False
        return max(p.stat().st_mtime for p in ins) > min(Path(o).stat().st_mtime for o in self.outputs)


# --- step bodies (imported lazily so `rrisk status` stays fast) -------------------

def _download(group):
    def f():
        from .sources import download
        download.run([group])
    return f


def _sites():
    from .sources import sites
    sites.build()


def _hazard_us():
    from .hazard import us_convective
    us_convective.build()


def _hazard_tc():
    from .hazard import tropical_cyclone
    tropical_cyclone.build()


def _hazard_global():
    from .hazard import reanalysis
    reanalysis.build()


def _score():
    from .models import score
    score.build()


def _validate():
    from . import validate
    validate.run()


def _export():
    from .app import export
    export.build()


STEPS: list[Step] = [
    Step("download_global", "Download global data", _download("global"),
         outputs=[R / "gppd/gppd_v1_3.zip", R / "ibtracs/ibtracs.since1980.csv"], only_if_missing=True,
         help="WRI Global Power Plant Database and IBTrACS cyclone tracks (~150 MB)."),
    Step("download_maps", "Download map outlines", _download("maps"),
         outputs=[R / "us_atlas/states-10m.json", R / "world_atlas/countries-50m.json"], only_if_missing=True),
    Step("download_us", "Download US data", _download("us"),
         outputs=[R / "uswtdb/uswtdbCSV.zip", R / "uspvdb/uspvdbGeoJSON.zip", R / "storm_events/details_2025.csv.gz",
                  R / "radar_hail/hail-2025.csv.gz"], only_if_missing=True,
         help="EIA-860, US Wind Turbine and Solar PV Databases, NOAA Storm Events, NEXRAD hail (~3 GB)."),
    Step("sites", "Build site table", _sites,
         inputs=[R / "uswtdb/*.zip", R / "uspvdb/*.zip", R / "eia860/*.zip", R / "gppd/*.zip", R / "gem/*.xlsx",
                 C / "sources/*sites.py", C / "sources/footprints.py"],
         outputs=[P / "sites.parquet", P / "footprints.parquet"],
         help="US (EIA-860 + USWTDB + USPVDB outlines) and global (GPPD / GEM) sites."),
    Step("hazard_us", "US hail, wind & tornado hazard", _hazard_us,
         inputs=[P / "sites.parquet", R / "storm_events/*.gz", R / "radar_hail/*.gz", C / "hazard/us_convective.py"],
         outputs=[P / "hazard_us.parquet", P / "hazard_us_grid.npz"], help="About 4 minutes."),
    Step("hazard_tc", "Global cyclone hazard", _hazard_tc,
         inputs=[P / "sites.parquet", R / "ibtracs/*.csv", C / "hazard/tropical_cyclone.py"],
         outputs=[P / "hazard_tc.parquet", P / "hazard_tc_grid.npz"]),
    Step("hazard_global", "Global wind, rain & heat hazard", _hazard_global,
         inputs=[P / "sites.parquet", C / "hazard/reanalysis.py"],
         outputs=[P / "hazard_global.parquet"],
         help="NASA POWER reanalysis; ~20 min the first time (cached after)."),
    Step("score", "Scores & warehouse", _score,
         inputs=[P / "sites.parquet", P / "footprints.parquet", P / "hazard_us.parquet", P / "hazard_tc.parquet", P / "hazard_global.parquet",
                 C / "models/score.py"],
         outputs=[P / "site_scores.parquet", config.WAREHOUSE]),
    Step("validate", "Validation back-tests", _validate,
         inputs=[P / "storm_reports.parquet", P / "radar_hail.parquet", config.WAREHOUSE, C / "validate.py"],
         outputs=[O / "validation.json"], help="Re-checks the hazard model against years it never saw (~30 s)."),
    Step("export", "Static HTML export", _export,
         inputs=[config.WAREHOUSE, O / "validation.json", C / "app/static/index.html", C / "app/payload.py"], outputs=[O / "rrisk_atlas.html"],
         help="A single shareable HTML file of the app (read-only)."),
]
BY_NAME = {s.name: s for s in STEPS}


# --- state & running ----------------------------------------------------------

def load_state() -> dict:
    try:
        return json.loads(config.STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    tmp = config.STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.STATE_FILE)


class _Tee(io.TextIOBase):
    def __init__(self, sink: Callable[[str], None], echo):
        self.sink, self.echo, self.buf = sink, echo, ""

    def write(self, s):
        self.echo.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.sink(line)
        return len(s)

    def flush(self):
        self.echo.flush()


def run_step(step: Step, log: Callable[[str], None] = lambda s: None) -> bool:
    state = load_state()
    lines: list[str] = []

    def sink(line):
        lines.append(line)
        log(line)

    t0 = time.time()
    ok, err = True, None
    with contextlib.redirect_stdout(_Tee(sink, sys.__stdout__)):
        print(f"▶ {step.title}")
        try:
            step.run()
        except Exception as e:  # noqa: BLE001 — report any failure to the user
            ok, err = False, f"{type(e).__name__}: {e}"
            print(f"✗ {step.name} failed: {err}")
            for ln in traceback.format_exc().splitlines()[-6:]:
                print("   " + ln)
        else:
            print(f"✓ {step.title} ({time.time() - t0:.0f}s)")
    state = load_state()
    prev = state.get(step.name, {})
    state[step.name] = {
        "inputs_sig": step.signature() if ok else prev.get("inputs_sig"),
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "ok": ok,
        "error": err,
        "summary": [ln for ln in lines if not ln.startswith(("▶", "✓"))][-4:],
    }
    _save_state(state)
    return ok


def plan(names: list[str] | None = None, force: bool = False) -> list[Step]:
    if names:
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown step(s) {unknown}; steps: {', '.join(BY_NAME)}")
        return [BY_NAME[n] for n in names]
    return [s for s in STEPS if not s.manual and (force or s.stale())]


def update(names: list[str] | None = None, force: bool = False, log=lambda s: None) -> bool:
    """Run named steps, or every stale automatic step (re-checking as we go,
    since an upstream step can make downstream ones stale)."""
    if names:
        for s in plan(names, force):
            if not run_step(s, log):
                return False
        return True
    ran = False
    for s in STEPS:
        if s.manual or not (force or s.stale()):
            continue
        ran = True
        if not run_step(s, log):
            return False
    if not ran:
        log("Everything is up to date.")
        print("Everything is up to date.")
    return True


def status() -> list[dict]:
    state = load_state()
    rows = []
    for s in STEPS:
        st = state.get(s.name, {})
        missing = s.missing()
        rows.append({
            "name": s.name, "title": s.title, "manual": s.manual, "help": s.help,
            "state": "missing" if missing else ("stale" if s.stale() else "ok"),
            "missing": [str(Path(m).relative_to(config.ROOT)) if str(m).startswith(str(config.ROOT)) else str(m)
                        for m in missing],
            "last_run": st.get("last_run"), "seconds": st.get("seconds"),
            "ok": st.get("ok"), "error": st.get("error"), "summary": st.get("summary", []),
        })
    return rows
