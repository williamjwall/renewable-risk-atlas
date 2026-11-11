"""Wind and solar sites outside the US.

Default source: WRI Global Power Plant Database v1.3 (CC BY 4.0; data to ~2021,
utility-scale). Optional, fresher: Global Energy Monitor's Global Wind Power
Tracker / Global Solar Power Tracker spreadsheets (free after a short form at
globalenergymonitor.org). Drop the .xlsx files in ``data/raw/gem/``; for each
technology they cover, GEM replaces GPPD.
"""

from __future__ import annotations

import re
import zipfile

import numpy as np
import pandas as pd

from .. import config

GPPD_TECH = {"Wind": "wind", "Solar": "solar"}
# Typical land use (km² per MW) to turn capacity into a footprint radius.
KM2_PER_MW = {"solar": 0.025, "wind": 0.10}


def _site_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Common columns from (site_id, plant_name, utility_name, country, country_name,
    lat, lon, tech, mw, first_year, source)."""
    out = df.copy()
    for t in config.TECHS:
        on = out["tech"] == t
        out[f"{t}_mw"] = np.where(on, out["mw"], np.nan)
        out[f"{t}_first_year"] = np.where(on, out["first_year"], np.nan)
        out[f"{t}_first_date"] = np.where(on, out["first_year"] + 0.5, np.nan)
    out["primary_tech"] = out["tech"]
    out["techs"] = out["tech"]
    out["total_mw"] = out["mw"]
    out["retired"] = False
    out["coverage"] = "global"
    area = out["mw"].fillna(0) * out["tech"].map(KM2_PER_MW).fillna(0.025)
    out["footprint_km"] = np.maximum(np.sqrt(area / np.pi), 0.2)
    out["fp_kind"] = "estimated from MW"
    return out.drop(columns=["tech", "mw"])


def load_gppd() -> pd.DataFrame:
    path = config.RAW / "gppd" / "gppd_v1_3.zip"
    with zipfile.ZipFile(path) as zf, zf.open("global_power_plant_database.csv") as fh:
        g = pd.read_csv(fh, low_memory=False)
    g = g[g["primary_fuel"].isin(GPPD_TECH) & (g["country"] != "USA")]
    df = pd.DataFrame(
        {
            "site_id": "gppd:" + g["gppd_idnr"].astype(str),
            "plant_name": g["name"],
            "utility_name": g["owner"],
            "country": g["country"],
            "country_name": g["country_long"],
            "lat": g["latitude"],
            "lon": g["longitude"],
            "tech": g["primary_fuel"].map(GPPD_TECH),
            "mw": g["capacity_mw"],
            "first_year": np.floor(pd.to_numeric(g["commissioning_year"], errors="coerce")),
            "source": "gppd",
        }
    )
    return df


def _pick(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_gem() -> pd.DataFrame:
    """Global Energy Monitor tracker spreadsheets, if present. Column names vary a
    little between releases, so they are looked up by candidate names."""
    from .countries import iso3

    files = sorted((config.RAW / "gem").glob("*.xlsx")) if (config.RAW / "gem").exists() else []
    frames = []
    for f in files:
        tech = "wind" if "wind" in f.name.lower() else "solar" if "solar" in f.name.lower() else None
        if tech is None:
            print(f"[sites] GEM: can't tell technology of {f.name}; name it *wind* or *solar*")
            continue
        for sheet, d in pd.read_excel(f, sheet_name=None).items():
            c = d.columns
            col = {
                "country": _pick(c, "Country/Area", "Country"),
                "name": _pick(c, "Project Name"),
                "phase": _pick(c, "Phase Name"),
                "mw": _pick(c, "Capacity (MW)", "Capacity (MWac)", "Capacity Rating"),
                "status": _pick(c, "Status"),
                "start": _pick(c, "Start year", "Start Year"),
                "lat": _pick(c, "Latitude"),
                "lon": _pick(c, "Longitude"),
                "owner": _pick(c, "Owner", "Owner(s)", "Operator"),
            }
            if not all(col[k] for k in ("country", "name", "mw", "lat", "lon")):
                continue  # notes / about sheets
            d = d.rename(columns={v: k for k, v in col.items() if v})
            if "status" in d:
                d = d[d["status"].astype(str).str.lower().isin(["operating", "retired", "mothballed"])]
            d["iso"] = d["country"].map(iso3)
            d = d[d["iso"].notna() & (d["iso"] != "USA")]
            d["mw"] = pd.to_numeric(d["mw"], errors="coerce")
            g = d.groupby(["iso", "name"], as_index=False).agg(
                country_name=("country", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
                mw=("mw", "sum"), first_year=("start", "min") if "start" in d else ("mw", lambda s: np.nan),
                owner=("owner", "first") if "owner" in d else ("name", lambda s: None),
            )
            slug = g["name"].astype(str).map(lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-"))
            frames.append(pd.DataFrame({
                "site_id": f"gem:{tech}:" + g["iso"] + ":" + slug,
                "plant_name": g["name"], "utility_name": g["owner"], "country": g["iso"],
                "country_name": g["country_name"], "lat": pd.to_numeric(g["lat"], errors="coerce"),
                "lon": pd.to_numeric(g["lon"], errors="coerce"), "tech": tech, "mw": g["mw"],
                "first_year": pd.to_numeric(g["first_year"], errors="coerce"), "source": "gem",
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_global() -> pd.DataFrame:
    gppd = load_gppd()
    gem = load_gem()
    if len(gem):
        covered = set(gem["tech"])
        gppd = gppd[~gppd["tech"].isin(covered)]
        print(f"[sites] GEM trackers found for {sorted(covered)}; GPPD used for the rest")
    df = pd.concat([gppd, gem], ignore_index=True).dropna(subset=["lat", "lon"])
    df = df.drop_duplicates("site_id")
    out = _site_frame(df)
    print(f"[sites] global (ex-US): {len(out):,} sites in {out['country'].nunique()} countries "
          f"({out['primary_tech'].value_counts().to_dict()})")
    return out
