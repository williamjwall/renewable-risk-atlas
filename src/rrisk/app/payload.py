"""The data bundle the front-end needs (used by the local app and the static export)."""

from __future__ import annotations

import base64
import json
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from .. import config, docs

TIER_INDEX = {"Low": 0, "Moderate": 1, "High": 2, "Very high": 3}

US_LAYERS = {
    "rhail2": ("Hail ≥ 2″ near site (radar)", "days/yr"),
    "rhail1": ("Hail ≥ 1″ near site (radar)", "days/yr"),
    "hail1": ("Hail ≥ 1″ within 25 mi (reports)", "days/yr"),
    "hail2": ("Hail ≥ 2″ within 25 mi (reports)", "days/yr"),
    "wind65": ("Gusts ≥ 75 mph within 25 mi", "days/yr"),
    "tor1": ("EF1+ tornado within 25 mi", "days/yr"),
}
TC_LAYERS = {
    "tc64": ("Hurricane/typhoon-force winds", "storms/yr"),
    "tc34": ("Tropical-storm-force winds", "storms/yr"),
    "tc96": ("Major hurricane winds (Cat 3+)", "storms/yr"),
}


def _nice(v: float) -> float:
    if v <= 0:
        return 0.0
    mag = 10 ** np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if v <= m * mag:
            return float(round(m * mag, 6))
    return float(10 * mag)


def _encode(a: np.ndarray, label: str, unit: str) -> dict | None:
    vals = a[np.isfinite(a) & (a > 0)]
    if not len(vals):
        return None
    mx = float(np.nanmax(a))
    q = np.full(a.shape, 255, dtype=np.uint8)
    ok = np.isfinite(a)
    q[ok] = np.clip(np.round(a[ok] / mx * 254), 0, 254).astype(np.uint8)
    br = sorted({_nice(float(np.quantile(vals, p))) for p in (0.3, 0.5, 0.7, 0.85, 0.95, 0.99)})
    return {"data": base64.b64encode(q.ravel().tobytes()).decode(), "max": mx, "breaks": br,
            "label": label, "unit": unit}


def grids() -> dict:
    out = {}
    p = config.PROCESSED / "hazard_us_grid.npz"
    if p.exists():
        g = np.load(p)
        layers = {k: _encode(np.where(g["mask"], g[k], np.nan).astype(float), *v) for k, v in US_LAYERS.items()}
        out["us"] = {"lat0": float(g["lat"][0]), "lon0": float(g["lon"][0]), "nlat": int(len(g["lat"])),
                     "nlon": int(len(g["lon"])), "step": config.GRID_DEG,
                     "layers": {k: v for k, v in layers.items() if v}}
    p = config.PROCESSED / "hazard_tc_grid.npz"
    if p.exists():
        g = np.load(p)
        layers = {k: _encode(np.where(g[k] > 0, g[k], np.nan).astype(float), *v) for k, v in TC_LAYERS.items()}
        out["tc"] = {"lat0": float(g["lat"][0]), "lon0": float(g["lon"][0]), "nlat": int(len(g["lat"])),
                     "nlon": int(len(g["lon"])), "step": config.TC_GRID_DEG,
                     "layers": {k: v for k, v in layers.items() if v}}
    return out


SITE_SQL = """
select site_id id, name, operator op, country cc, country_name cn, state st, county, city, lat, lon,
       primary_tech tech, techs, total_mw mw, first_year fy, retired ret, coverage cov, source src,
       turbines nturb, turbine_maker tmaker, turbine_model tmodel, solar_tracking trk,
       score, tier, score_basis basis, score_global sg, tier_global tg,
       hazard_index hi, hazard_index_us hiu, hazard_index_global hig, drivers,
       rhail2_rate rh2r, rhail2_p rh2p, rhail1_rate rh1r, rhail1_p rh1p,
       hail2_rate h2r, hail2_p h2p, hail1_rate h1r, hail1_p h1p,
       wind65_rate w65r, wind65_p w65p, tor1_rate t1r, tor1_p t1p,
       tc34_rate tc34r, tc34_p tc34p, tc64_rate tc64r, tc64_p tc64p, tc96_rate tc96r, tc96_p tc96p,
       xwind_rate xwr, xwind_p xwp, xrain_rate xrr, xrain_p xrp, heat_rate htr, heat_p htp,
       wind_rl50_ms wrl, rain_rl50_mm rrl, footprint_kind fpk, footprint_area_km2 fpa
from site_risk where lat is not null and lon is not null
"""


def _columnar(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = [bool(v) if pd.notna(v) else False for v in s]
        elif pd.api.types.is_float_dtype(s):
            dig = 4 if c in ("lat", "lon") else 3
            out[c] = [None if pd.isna(v) else round(float(v), dig) for v in s]
        elif pd.api.types.is_integer_dtype(s):
            out[c] = [None if pd.isna(v) else int(v) for v in s]
        else:
            out[c] = [None if pd.isna(v) else str(v) for v in s]
    return out


def sites(con) -> dict:
    df = con.sql(SITE_SQL).df()
    df["tier"] = df["tier"].map(TIER_INDEX)
    df["tg"] = df["tg"].map(TIER_INDEX)
    for c in ("fy", "nturb"):
        df[c] = df[c].astype("Int64")
    return _columnar(df)


def atlas() -> dict:
    w = json.loads((config.RAW / "world_atlas" / "countries-50m.json").read_text())
    us = json.loads((config.RAW / "us_atlas" / "states-10m.json").read_text())
    return {"world": w, "us": us}


def _validation() -> dict | None:
    p = config.OUTPUT / "validation.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text())
    return {"hazard_backtest": v.get("hazard_backtest")}


def build() -> dict:
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    try:
        return {
            "meta": {
                "generated": date.today().isoformat(),
                "caveat": "Scores rank how exposed a site is to the modelled weather perils, relative to other "
                          "sites of the same technology. They are not loss estimates: there are no insured values, "
                          "damage curves or claims behind them.",
                "score": json.loads((config.OUTPUT / "score_meta.json").read_text()),
                "years": {"reports": f"{config.REPORT_YEARS[0]}–{config.REPORT_YEARS[-1]}",
                          "radar": f"{config.RADAR_YEARS[0]}–{config.RADAR_YEARS[-1]}",
                          "tc": f"{config.TC_YEARS[0]}–{config.TC_YEARS[-1]}", "reanalysis": "1981–2025"},
            },
            "docs": {"tables": docs.TABLES, "columns": docs.COLUMNS},
            "sites": sites(con),
            "grids": grids(),
            "atlas": atlas(),
            "validation": _validation(),
        }
    finally:
        con.close()
