"""Weather-hazard scores per site, and the DuckDB warehouse.

Two scores, because hazard data are richer in the US than elsewhere:

* ``score_us`` (US sites): hail, thunderstorm gusts, tornadoes, tropical
  cyclones.
* ``score_global`` (every site, worldwide, US included): tropical cyclones
  (IBTrACS) and damaging wind, extreme rain and extreme heat (NASA POWER /
  MERRA-2). No hail outside the US.

``score`` is score_us where available, else score_global; ``score_basis`` says
which. Each is the percentile of a *hazard index* within the site's technology:

    hazard index = sum over perils of  weight[tech][peril] x  min(rate / mean rate, REL_CAP)

so 1.0 is an average site and 2.0 is twice as exposed. Weights say which perils
damage which technology and are judgement. Scores rank exposure; they are not
loss costs.
"""

from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pandas as pd

from .. import config

PERILS = {
    "rhail1": 'Hail ≥ 1" near site (radar)',
    "hail2": 'Hail ≥ 2" within 25 mi',
    "rhail2": 'Hail ≥ 2" near site (radar)',
    "wind65": "Thunderstorm gusts ≥ 75 mph",
    "tor1": "EF1+ tornadoes",
    "tc34": "Tropical-storm winds",
    "tc64": "Hurricane/typhoon winds",
    "tc96": "Major hurricane winds",
    "xwind": "Damaging wind days",
    "xrain": "Extreme rain days",
    "heat": "Extreme heat days",
}
# Validation (rrisk validate) showed radar >=2" hail within 10 km barely repeats
# between 2010-17 and 2018-25 (rank corr 0.30) while report-based >=2" hail
# within 25 mi does (0.87), so most of the 2" weight sits on the latter.
WEIGHTS_US = {
    "solar": {"rhail1": 0.20, "hail2": 0.25, "rhail2": 0.15, "wind65": 0.10, "tor1": 0.15, "tc64": 0.15},
    "wind": {"rhail1": 0.05, "hail2": 0.07, "rhail2": 0.03, "wind65": 0.30, "tor1": 0.35, "tc64": 0.20},
}
WEIGHTS_GLOBAL = {
    "solar": {"tc34": 0.05, "tc64": 0.20, "tc96": 0.10, "xwind": 0.25, "xrain": 0.25, "heat": 0.15},
    "wind": {"tc34": 0.05, "tc64": 0.25, "tc96": 0.15, "xwind": 0.45, "xrain": 0.05, "heat": 0.05},
}
REL_CAP = 10.0
TIERS = [(90, "Very high"), (70, "High"), (40, "Moderate"), (0, "Low")]


def _tier(s) -> str | None:
    return next(name for cut, name in TIERS if s >= cut) if pd.notna(s) else None


def _means(d: pd.DataFrame, weights: dict, mask: pd.Series) -> dict:
    keys = {k for w in weights.values() for k in w}
    return {k: float(d.loc[mask, f"{k}_rate"].mean()) for k in keys}


def _index(d, tech_col, weights, means, mask) -> tuple[pd.Series, dict]:
    """Hazard index per site, plus each peril's weighted contribution."""
    idx = pd.Series(np.nan, index=d.index)
    contrib: dict[str, pd.Series] = {}
    for tech, w in weights.items():
        m = mask & (tech_col == tech)
        tot = pd.Series(0.0, index=d.index)
        for k, wk in w.items():
            rel = np.minimum(d[f"{k}_rate"] / means[k], REL_CAP) if means[k] > 0 else d[f"{k}_rate"] * 0
            c = (wk * rel).where(m)
            contrib[k] = contrib.get(k, pd.Series(np.nan, index=d.index)).fillna(c)
            tot = tot + (wk * rel).fillna(0)
        has_any = d[[f"{k}_rate" for k in w]].notna().any(axis=1)
        idx = idx.fillna(tot.where(m & has_any))
    return idx, contrib


def _pct(idx: pd.Series, groups: pd.Series) -> pd.Series:
    return np.log(idx.clip(lower=0.02)).groupby(groups).rank(pct=True).mul(100).round(1)


def build() -> pd.DataFrame:
    P = config.PROCESSED
    sites = pd.read_parquet(P / "sites.parquet")
    d = (sites.merge(pd.read_parquet(P / "hazard_us.parquet"), on="site_id", how="left")
         .merge(pd.read_parquet(P / "hazard_tc.parquet"), on="site_id", how="left")
         .merge(pd.read_parquet(P / "hazard_global.parquet"), on="site_id", how="left"))
    us = d["country"] == "USA"
    prim = d["primary_tech"]
    everyone = pd.Series(True, index=d.index)
    means_us = _means(d, WEIGHTS_US, us)
    means_gl = _means(d, WEIGHTS_GLOBAL, everyone)
    idx_us, c_us = _index(d, prim, WEIGHTS_US, means_us, us)
    idx_gl, c_gl = _index(d, prim, WEIGHTS_GLOBAL, means_gl, everyone)

    out = d[["site_id"]].copy()
    out["hazard_index_us"] = idx_us
    out["hazard_index_global"] = idx_gl
    out["score_us"] = _pct(idx_us, prim.where(idx_us.notna()))
    out["score_global"] = _pct(idx_gl, prim.where(idx_gl.notna()))
    out["score"] = out["score_us"].fillna(out["score_global"])
    out["score_basis"] = np.where(out["score_us"].notna(), "us_full",
                                  np.where(out["score_global"].notna(), "global", None))
    out["hazard_index"] = out["hazard_index_us"].fillna(out["hazard_index_global"])
    out["tier"] = out["score"].map(_tier)
    out["tier_global"] = out["score_global"].map(_tier)

    drivers = []
    for i in d.index:
        c = c_us if us[i] and pd.notna(idx_us[i]) else c_gl
        parts = sorted(((v[i], k) for k, v in c.items() if pd.notna(v[i])), reverse=True)
        tot = sum(v for v, _ in parts) or 1
        txt = [f"{PERILS[k]} ({v / tot:.0%} of index)" for v, k in parts[:2] if v / tot >= 0.15]
        drivers.append("; ".join(txt) if txt else "no single dominant peril")
    out["drivers"] = drivers
    out.to_parquet(P / "site_scores.parquet", index=False)
    (config.OUTPUT / "score_meta.json").write_text(json.dumps({
        "perils": PERILS, "weights_us": WEIGHTS_US, "weights_global": WEIGHTS_GLOBAL,
        "means_us": means_us, "means_global": means_gl, "rel_cap": REL_CAP,
        "tiers": [[c, n] for c, n in TIERS],
    }, indent=2))
    warehouse()
    print(f"[score] {out['score'].notna().sum():,} of {len(out):,} sites scored "
          f"({out['score_us'].notna().sum():,} US full, {out['score_global'].notna().sum():,} global)")
    return out


SITE_RISK_VIEW = """
create or replace view site_risk as
select
  s.site_id, s.plant_name as name, s.utility_name as operator, s.country, s.country_name,
  s.state, s.county, s.city, s.lat, s.lon, s.techs, s.primary_tech, s.total_mw,
  s.wind_mw, s.solar_mw, s.solar_tracking, s.first_year, s.retired, s.source, s.coverage,
  s.wt_n_turbines as turbines, s.wt_manu as turbine_maker, s.wt_model as turbine_model,
  s.wt_hub_m as hub_height_m, s.wt_rotor_m as rotor_m,
  s.fp_kind as footprint_kind, s.fp_area_km2 as footprint_area_km2, s.footprint_km as footprint_radius_km,
  sc.score, sc.tier, sc.score_basis, sc.score_us, sc.score_global, sc.tier_global,
  sc.hazard_index, sc.hazard_index_us, sc.hazard_index_global, sc.drivers,
  h.hail1_rate, h.hail1_p, h.hail2_rate, h.hail2_p, h.wind65_rate, h.wind65_p, h.tor1_rate, h.tor1_p,
  h.rhail1_rate, h.rhail1_p, h.rhail2_rate, h.rhail2_p,
  t.tc34_rate, t.tc34_p, t.tc64_rate, t.tc64_p, t.tc96_rate, t.tc96_p,
  g.xwind_rate, g.xwind_p, g.xrain_rate, g.xrain_p, g.heat_rate, g.heat_p,
  g.wind_rl50_ms, g.rain_rl50_mm, g.tmax_rl50_c
from sites s
left join site_scores sc using (site_id)
left join hazard_us h using (site_id)
left join hazard_tc t using (site_id)
left join hazard_global g using (site_id)
"""

TABLES = ["sites", "footprints", "turbines", "generators", "hazard_us", "hazard_tc", "hazard_global", "site_scores",
          "storm_reports"]


def warehouse() -> None:
    """Rebuild the DuckDB file beside the live one, then swap it in atomically,
    so the app can keep reading while the pipeline runs."""
    tmp = config.WAREHOUSE.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    for t in TABLES:
        p = config.PROCESSED / f"{t}.parquet"
        if p.exists():
            con.execute(f"create table {t} as select * from read_parquet('{p}')")
    con.execute(SITE_RISK_VIEW)
    con.close()
    os.replace(tmp, config.WAREHOUSE)
