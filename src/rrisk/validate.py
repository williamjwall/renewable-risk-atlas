"""Validation: does the model hold up against data it never saw?

1. Hazard back-test (US). Fit hail/gust/tornado frequencies on 1996-2010
   reports, predict 2011-2025 at every US site:
   - rank skill: Spearman correlation of predicted vs observed rates
   - lift: observed frequency in the top vs bottom predicted decile
   - calibration: predicted P(>=1 event-year) vs observed share of test years
   Done for the smoothed model and for raw site counts (does smoothing help?).
2. Radar back-test: radar hail 2010-2017 -> 2018-2025.
3. Known losses: where documented loss sites land in the score distribution.
Writes data/output/validation.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .hazard import us_convective as hz

SPLIT_YEAR = 2011


def _year_key(ev: pd.DataFrame) -> pd.DataFrame:
    """Re-key events by calendar year so event_days counts distinct *years*."""
    d = ev.copy()
    d["day"] = (np.datetime64("1950-01-01") + d["day"].to_numpy().astype("timedelta64[D]")).astype("datetime64[Y]").astype(int)
    return d


def _smoothed_rate(ev, lat, lon, n_years, mask, glat, glon):
    from scipy.interpolate import RegularGridInterpolator

    G_lat, G_lon = np.meshgrid(glat, glon, indexing="ij")
    raw = hz.event_days(G_lat.ravel(), G_lon.ravel(), ev, config.NEIGHBOURHOOD_KM).reshape(G_lat.shape) / n_years
    sm = np.nan_to_num(hz._smooth(raw, mask))
    return RegularGridInterpolator((glat, glon), sm, bounds_error=False, fill_value=0)(np.c_[lat, lon])


def _skill(pred_rate, obs_rate, obs_year_frac, n_test_years):
    ok = np.isfinite(pred_rate)
    pr, ob, fr = pred_rate[ok], obs_rate[ok], obs_year_frac[ok]
    rho = stats.spearmanr(pr, ob).statistic
    dec = pd.qcut(pd.Series(pr).rank(method="first"), 10, labels=False)
    by = pd.DataFrame({"dec": dec, "pred": pr, "obs": ob, "pred_p": 1 - np.exp(-pr), "obs_frac": fr}).groupby("dec").mean()
    top, bot = by["obs"].iloc[-1], by["obs"].iloc[0]
    return {
        "spearman": round(float(rho), 3),
        "top_decile_obs_rate": round(float(top), 3),
        "bottom_decile_obs_rate": round(float(bot), 3),
        "lift_top_vs_bottom": round(float(top / bot), 1) if bot > 0 else None,
        "calibration_by_decile": [
            {"decile": int(k) + 1, "predicted_p": round(float(r.pred_p), 3), "observed_share_of_years": round(float(r.obs_frac), 3)}
            for k, r in by.iterrows()
        ],
        "mean_abs_calibration_error": round(float((by["pred_p"] - by["obs_frac"]).abs().mean()), 3),
    }


def hazard_backtest() -> dict:
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet")
    sites = sites[sites["country"] == "USA"].reset_index(drop=True)
    lat, lon = sites["lat"].to_numpy(), sites["lon"].to_numpy()
    reports = hz.load_reports()
    ev = hz.hazard_events(reports)
    split = (np.datetime64(f"{SPLIT_YEAR}-01-01") - np.datetime64("1950-01-01")).astype(int)
    n_tr, n_te = SPLIT_YEAR - config.REPORT_YEARS[0], config.REPORT_YEARS[-1] + 1 - SPLIT_YEAR
    glat, glon = hz.GRID_LAT, hz.GRID_LON
    G_lat, G_lon = np.meshgrid(glat, glon, indexing="ij")
    mask = (hz.event_days(G_lat.ravel(), G_lon.ravel(), ev["coverage"][ev["coverage"].day < split],
                          config.NEIGHBOURHOOD_KM) > 0).reshape(G_lat.shape)
    out = {"train": f"{config.REPORT_YEARS[0]}-{SPLIT_YEAR - 1}", "test": f"{SPLIT_YEAR}-{config.REPORT_YEARS[-1]}",
           "n_sites": int(len(sites)), "perils": {}}
    for h in ("hail1", "hail2", "wind65", "tor1"):
        tr, te = ev[h][ev[h].day < split], ev[h][ev[h].day >= split]
        raw_tr = hz.event_days(lat, lon, tr, config.NEIGHBOURHOOD_KM) / n_tr
        sm_tr = _smoothed_rate(tr, lat, lon, n_tr, mask, glat, glon)
        obs = hz.event_days(lat, lon, te, config.NEIGHBOURHOOD_KM) / n_te
        frac = hz.event_days(lat, lon, _year_key(te), config.NEIGHBOURHOOD_KM) / n_te
        out["perils"][h] = {"smoothed_model": _skill(sm_tr, obs, frac, n_te), "raw_counts": _skill(raw_tr, obs, frac, n_te)}
        s = out["perils"][h]["smoothed_model"]
        print(f"[validate] {h}: spearman {s['spearman']}, lift {s['lift_top_vs_bottom']}x, "
              f"calibration error {s['mean_abs_calibration_error']} (raw counts spearman "
              f"{out['perils'][h]['raw_counts']['spearman']})")
    return out


def radar_backtest() -> dict:
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet")
    sites = sites[sites["country"] == "USA"].reset_index(drop=True)
    lat, lon = sites["lat"].to_numpy(), sites["lon"].to_numpy()
    radar = hz.load_radar()
    r = config.RADAR_SITE_KM
    out = {}
    for h, thr in (("rhail1", config.RADAR_MEHS_FOR_1IN), ("rhail2", config.RADAR_MEHS_FOR_2IN)):
        pts = radar[radar.maxsize >= thr]
        tr, te = pts[pts.year <= 2017], pts[pts.year >= 2018]
        pr = hz.event_days(lat, lon, tr, r) / 8
        ob = hz.event_days(lat, lon, te, r) / 8
        fr = hz.event_days(lat, lon, te.assign(day=te["year"]), r) / 8
        out[h] = _skill(pr, ob, fr, 8)
        print(f"[validate] {h} radar 2010-17 -> 2018-25: spearman {out[h]['spearman']}, lift {out[h]['lift_top_vs_bottom']}x")
    return out


# Publicly reported losses (site, what happened). Matched by name in the site table.
KNOWN_LOSSES = [
    ("Fighting Jays", "solar", "hail, March 2019 (Fort Bend Co., TX)"),
    ("Midway Solar", "solar", "hail, 2019 (TX)"),
    ("Casa Mesa", "wind", "GE turbine collapse, Feb 2019 (NM)"),
]


def known_losses() -> list[dict]:
    import duckdb

    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    rows = []
    for name, tech, what in KNOWN_LOSSES:
        r = con.execute(
            """select name, state, primary_tech, score, tier, hazard_index, null, rhail2_rate, rhail1_rate,
                      (select avg(rhail2_rate) from site_risk where primary_tech = ? and country = 'USA') avg_rhail2
               from site_risk where name ilike ? and techs like ? order by total_mw desc limit 1""",
            [tech, f"%{name}%", f"%{tech}%"],
        ).fetchone()
        if r:
            rows.append({"site": r[0], "state": r[1], "event": what, "score": r[3], "tier": r[4],
                         "hazard_index": round(r[5], 2) if r[5] else None,
                         "hail2_rel": round(r[7] / r[9], 1) if r[7] is not None and r[9] else None})
        else:
            rows.append({"site": name, "event": what, "score": None, "note": "not found in site table"})
    for r in rows:
        print(f"[validate] {r['site']}: {r.get('event')} -> score {r.get('score')} ({r.get('tier')}), "
              f"hazard index x{r.get('hazard_index')}, 2in hail x{r.get('hail2_rel')}")
    return rows


def run() -> dict:
    res = {"hazard_backtest": hazard_backtest(), "radar_backtest": radar_backtest(), "known_losses": known_losses()}
    (config.OUTPUT / "validation.json").write_text(json.dumps(res, indent=2, default=str))
    return res
