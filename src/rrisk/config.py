"""Paths, user settings and modelling constants.

User-adjustable settings live in ``rrisk.toml`` at the project root (all keys
optional; see ``rrisk.example.toml``). Modelling constants below are part of the
method and documented in the README.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get("RRISK_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUT = DATA / "output"
WAREHOUSE = DATA / "warehouse.duckdb"
STATE_FILE = DATA / "pipeline_state.json"

# Technologies covered. Battery storage was removed: the model could not predict
# battery fires, and its weather score was misleading on its own.
TECHS = ("wind", "solar")

for _p in (RAW, PROCESSED, OUTPUT):
    _p.mkdir(parents=True, exist_ok=True)


# --- user settings (rrisk.toml) ---------------------------------------------
_DEFAULTS = {
    "app": {"host": "127.0.0.1", "port": 8765},
}


def _load_settings() -> dict:
    s = {k: dict(v) for k, v in _DEFAULTS.items()}
    path = ROOT / "rrisk.toml"
    if path.exists():
        with path.open("rb") as fh:
            for section, vals in tomllib.load(fh).items():
                s.setdefault(section, {}).update(vals)
    return s


SETTINGS = _load_settings()


# --- US convective hazard -----------------------------------------------------
# Storm Events is consistent for all event types from 1996 on. Use complete years only.
REPORT_YEARS = range(1996, 2026)
# NEXRAD hail-signature archive (SWDI nx3hail). Pre-2010 coverage/algorithm changed a lot.
RADAR_YEARS = range(2010, 2026)

# SPC convention: "probability of an event within 25 miles of a point".
NEIGHBOURHOOD_KM = 40.2
# Radar storm-cell tracks are dense enough to look at the site itself.
RADAR_SITE_KM = 10.0
# Extra radius allowed for a site's physical spread (wind farms can span 80 km).
FOOTPRINT_CAP_KM = 5.0
# Gaussian smoothing of the report climatology (SPC uses ~120 km on its 80 km grid).
SMOOTH_SIGMA_KM = 60.0
GRID_DEG = 0.25

HAIL_1IN = 1.0
HAIL_2IN = 2.0
WIND_KT_EXTREME = 65.0  # "significant severe" thunderstorm wind (75 mph)

# Radar MEHS (max expected hail size) over-estimates size. Thresholds below were
# calibrated so that radar hail-day frequency within 40 km matches spotter-report
# frequency at ~1,600 spatially thinned sites (median radar/report ratio 1.1 for
# >=1", 1.4 for >=2"; log-rate correlation 0.88 / 0.85). See README.
RADAR_MEHS_FOR_1IN = 2.25
RADAR_MEHS_FOR_2IN = 3.5

# --- Global tropical-cyclone hazard (IBTrACS) -------------------------------------
TC_YEARS = range(1980, 2026)
TC_GRID_DEG = 0.5
# Default wind radii (km) when the track lacks observed radii. Typical global
# medians: gale (34 kt) ~150 km, hurricane force (64 kt) ~45 km.
TC_R34_DEFAULT_KM = 150.0
TC_R64_DEFAULT_KM = 45.0
# Major-hurricane winds (>= 96 kt) are confined to the inner core: half of R64.
TC_R96_FRACTION = 0.5

EARTH_KM = 6371.0088

USER_AGENT = "rrisk/0.2 (renewable site risk research)"
