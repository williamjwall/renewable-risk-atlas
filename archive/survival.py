"""Exploratory failure-risk model: how incident hazard varies with equipment,
age, size and weather exposure.

Design
* Unit of analysis: a site's wind / solar asset (from EIA-860 and
  USWTDB). Time scale: years since that asset first operated.
* Observation window: loss data are only collected systematically from
  OBS_START, so assets older than that enter the risk set late
  (left truncation via lifelines ``entry_col``). Exit at the data cut-off or
  retirement. Event = first matched incident of that asset type.
* Cox proportional hazards with a ridge penalty (few events -> keep the
  coefficients from exploding). With tens of events, read hazard ratios as
  "direction and rough size", not as calibrated estimates.
* Also reports crude incident rates by commissioning cohort.

Outputs data/processed/failure_sites.parquet and data/output/failure_model.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .. import config

OBS_START = {"wind": 2017.0, "solar": 2017.0}
DATA_END = 2026.75  # ~ end of Sept 2026, the latest EIA vintage used
MIN_EVENTS = 5
PENALIZER = 0.1
# Credibility weight for the model's relative hazard: n / (n + K) where n is
# the number of events behind the fit. Pulls predictions to 1.0 when thin.
CRED_K = 20


def _hazard() -> pd.DataFrame:
    us = pd.read_parquet(config.PROCESSED / "hazard_us.parquet")
    tc = pd.read_parquet(config.PROCESSED / "hazard_tc.parquet", columns=["site_id", "tc64_rate"])
    return us.merge(tc, on="site_id", how="left")


def _std(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    sd = x.std()
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0


def design(tech: str, sites: pd.DataFrame, events: pd.DataFrame, haz: pd.DataFrame):
    col = f"{tech}_mw"
    has = sites[col].fillna(0) > 0
    if tech == "wind":
        has |= sites["wt_n_turbines"].fillna(0) > 0
    d = sites[has].merge(haz, on="site_id", how="left").copy()
    start = d[f"{tech}_first_date"].fillna(d[f"{tech}_first_year"] + 0.5)
    d = d[start.notna()].copy()
    start = start[d.index]
    retire = d.get(f"{tech}_retire_year")
    end = pd.Series(DATA_END, index=d.index)
    if retire is not None:
        r = pd.to_numeric(retire, errors="coerce")
        retired = d.get(f"{tech}_retired", pd.Series(False, index=d.index)).fillna(False).astype(bool)
        end = np.where(retired & r.notna(), np.minimum(r + 0.5, DATA_END), DATA_END)
    ev = events[(events.asset_type == tech) & events.site_id.notna() & events.event_date.notna()].copy()
    ev["t"] = ev.event_date.dt.year + (ev.event_date.dt.dayofyear - 1) / 365.25
    ev = ev[ev.t >= OBS_START[tech]]
    first = ev.groupby("site_id")["t"].min()
    n_all = ev.groupby("site_id").size()

    d["start"] = start
    d["t_event"] = d["site_id"].map(first)
    d["n_incidents"] = d["site_id"].map(n_all).fillna(0).astype(int)
    d["event"] = d["t_event"].notna().astype(int)
    stop = np.where(d["event"] == 1, d["t_event"], end)
    entry_cal = np.maximum(d["start"], OBS_START[tech])
    d["entry"] = (entry_cal - d["start"]).clip(lower=0)
    d["duration"] = (stop - d["start"]).astype(float)
    d["exposure_yrs"] = (np.asarray(end, dtype=float) - entry_cal).clip(lower=0)
    # Drop assets that never overlap the window, or whose event pre-dates entry.
    d = d[(d["duration"] > d["entry"]) & (d["duration"] > 0)].copy()
    d["duration"] = np.maximum(d["duration"], d["entry"] + 1 / 365)

    X = pd.DataFrame(index=d.index)
    X["cohort_yr"] = d["start"] - 2015
    if tech == "wind":
        X["log_mw"] = np.log1p(d["wind_mw"].fillna(d["wt_mw"]).fillna(0))
        X["turbine_mw"] = d["wt_turbine_mw"].fillna(d["wt_turbine_mw"].median())
        X["rotor_m"] = d["wt_rotor_m"].fillna(d["wt_rotor_m"].median())
        manu = d["wt_manu"].fillna("Unknown")
        for m in ("GE Wind", "Vestas", "Siemens Gamesa Renewable Energy", "Siemens"):
            X[f"manu_{m.split()[0].lower()}"] = (manu == m).astype(float)
        X["log_hail2"] = np.log1p(d["rhail2_rate"].fillna(0))
        X["log_wind65"] = np.log1p(d["wind65_rate"].fillna(0))
        X["log_tor1"] = np.log1p(d["tor1_rate"].fillna(0))
        X["log_tc64"] = np.log1p(d["tc64_rate"].fillna(0))
    else:  # solar
        X["log_mw"] = np.log1p(d["solar_mw"].fillna(0))
        X["tracking"] = d["solar_tracking"].fillna(False).astype(float)
        X["log_hail1"] = np.log1p(d["rhail1_rate"].fillna(0))
        X["log_hail2"] = np.log1p(d["rhail2_rate"].fillna(0))
        X["log_wind65"] = np.log1p(d["wind65_rate"].fillna(0))
    X = X.loc[:, X.std() > 0]
    return d, X


def fit(tech: str, sites, events, haz) -> tuple[pd.DataFrame, dict]:
    from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: F401  (KM kept for notebooks)

    d, X = design(tech, sites, events, haz)
    n_ev = int(d["event"].sum())
    info = {
        "tech": tech,
        "n_assets": int(len(d)),
        "n_events_first": n_ev,
        "n_incidents_total": int(d["n_incidents"].sum()),
        "exposure_asset_years": float(d["exposure_yrs"].sum()),
        "observation_window": f"{OBS_START[tech]:.0f}-{DATA_END:.2f}",
    }
    info["crude_rate_per_asset_year"] = (
        info["n_incidents_total"] / info["exposure_asset_years"] if info["exposure_asset_years"] else None
    )
    cohort = pd.cut(d["start"], [0, 2015, 2019, 2022, 2100], labels=["<=2015", "2016-19", "2020-22", "2023+"])
    by = d.groupby(cohort, observed=True).agg(assets=("site_id", "size"), incidents=("n_incidents", "sum"),
                                              exposure=("exposure_yrs", "sum"))
    by["rate_per_1000_asset_yrs"] = 1000 * by["incidents"] / by["exposure"]
    info["by_cohort"] = by.reset_index().rename(columns={"start": "cohort"}).astype({"cohort": str}).to_dict("records")

    out = d[["site_id"]].copy()
    out[f"fail_{tech}_rel_hazard"] = 1.0
    out[f"fail_{tech}_n_incidents"] = d["n_incidents"].values
    out[f"fail_{tech}_exposure_yrs"] = d["exposure_yrs"].values

    if n_ev < MIN_EVENTS:
        info["model"] = "none"
        info["note"] = (
            f"Only {n_ev} matched {tech} incidents in the window; Cox model not fitted. "
            "Relative hazard set to 1.0 (site score relies on hazard + history)."
        )
        return out, info

    Z = X.apply(_std)
    df = pd.concat([Z, d[["duration", "event", "entry"]]], axis=1)
    cph = CoxPHFitter(penalizer=PENALIZER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="duration", event_col="event", entry_col="entry")
    s = cph.summary
    info["model"] = "cox_ph_ridge"
    info["penalizer"] = PENALIZER
    info["concordance"] = float(cph.concordance_index_)
    info["covariates_standardised"] = True
    info["coefficients"] = [
        {
            "covariate": k,
            "hazard_ratio_per_sd": float(np.exp(s.loc[k, "coef"])),
            "hr_lo95": float(np.exp(s.loc[k, "coef lower 95%"])),
            "hr_hi95": float(np.exp(s.loc[k, "coef upper 95%"])),
            "p": float(s.loc[k, "p"]),
            "sd_of_raw": float(X[k].std()),
        }
        for k in s.index
    ]
    # Proportional-hazards check (Schoenfeld); with few events it's weak.
    try:
        from lifelines.statistics import proportional_hazard_test

        ph = proportional_hazard_test(cph, df, time_transform="rank")
        info["ph_test_min_p"] = float(ph.summary["p"].min())
    except Exception as e:  # noqa: BLE001
        info["ph_test_min_p"] = None
        info["ph_test_error"] = str(e)[:200]
    rel = cph.predict_partial_hazard(Z).to_numpy().ravel()
    rel = rel / np.exp(np.log(rel).mean())  # geometric mean 1
    w = n_ev / (n_ev + CRED_K)
    info["credibility_weight"] = w
    out[f"fail_{tech}_rel_hazard"] = np.exp(w * np.log(rel))
    return out, info


def build() -> pd.DataFrame:
    # Exposure needs reliable commissioning dates, which only EIA-860 gives us,
    # so the failure model is fitted on US sites.
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet")
    sites = sites[sites["country"] == "USA"].reset_index(drop=True)
    events = pd.read_parquet(config.PROCESSED / "loss_events.parquet")
    haz = _hazard()
    report = {"techs": {}, "caveat": (
        "Exploratory. Loss data are thin and come from public sources with uneven "
        "coverage (media attention), so incident counts are "
        "lower bounds and the fitted effects are not calibrated loss frequencies."
    )}
    out = sites[["site_id"]].copy()
    for tech in config.TECHS:
        o, info = fit(tech, sites, events, haz)
        out = out.merge(o, on="site_id", how="left")
        report["techs"][tech] = info
        print(f"[survival] {tech}: {info['n_assets']} assets, {info['n_events_first']} first events, "
              f"model={info['model']}" + (f", C={info['concordance']:.2f}" if "concordance" in info else ""))
    out.to_parquet(config.PROCESSED / "failure_sites.parquet", index=False)
    (config.OUTPUT / "failure_model.json").write_text(json.dumps(report, indent=2, default=str))
    return out
