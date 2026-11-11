"""Build the US site table from EIA-860 (all wind and solar plants)
and the US Wind Turbine Database (turbine-level detail rolled up to projects).

Writes turbines.parquet and generators.parquet; returns the US site rows that
``sources.sites`` merges with the global ones.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import footprints

TECH_MAP = {
    "Onshore Wind Turbine": "wind",
    "Offshore Wind Turbine": "wind",
    "Solar Photovoltaic": "solar",
    "Solar Thermal without Energy Storage": "solar",
    "Solar Thermal with Energy Storage": "solar",
}


def _eia_zip() -> Path:
    zips = sorted((config.RAW / "eia860").glob("eia860*.zip"))
    if not zips:
        raise FileNotFoundError("run `rrisk download eia860` first")
    return zips[-1]


def _read_eia_sheet(zf: zipfile.ZipFile, prefix: str, sheet: str | int = 0) -> pd.DataFrame:
    name = next(n for n in zf.namelist() if n.startswith(prefix))
    with zf.open(name) as fh:
        df = pd.read_excel(fh, sheet_name=sheet, header=1)
    # Footnote rows at the bottom have no plant code.
    return df[pd.to_numeric(df["Plant Code"], errors="coerce").notna()].copy()


def load_turbines() -> pd.DataFrame:
    zpath = config.RAW / "uswtdb" / "uswtdbCSV.zip"
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            t = pd.read_csv(fh, low_memory=False)
    t = t.rename(columns={"xlong": "lon", "ylat": "lat"})
    t["eia_id"] = pd.to_numeric(t["eia_id"], errors="coerce").astype("Int64")
    for c in ("t_cap", "t_hh", "t_rd", "t_ttlh", "p_year", "t_retro_yr"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    # USWTDB uses -9999 for missing in older releases.
    t = t.replace(-9999, np.nan)
    t["t_manu"] = t["t_manu"].fillna("Unknown").str.strip()
    t["t_model"] = t["t_model"].fillna("Unknown").str.strip()
    return t


def load_generators() -> pd.DataFrame:
    with zipfile.ZipFile(_eia_zip()) as zf:
        frames = []
        for sheet in ("Operable", "Retired and Canceled"):
            g = _read_eia_sheet(zf, "3_1_Generator", sheet)
            g["sheet"] = sheet
            frames.append(g)
        gen = pd.concat(frames, ignore_index=True)
        solar = pd.concat(
            [_read_eia_sheet(zf, "3_3_Solar", s) for s in ("Operable", "Retired and Canceled")],
            ignore_index=True,
        )

    gen["tech"] = gen["Technology"].map(TECH_MAP)
    gen = gen[gen["tech"].notna()].copy()
    # The retired sheet also lists cancelled projects that never operated.
    gen = gen[~gen["Status"].astype(str).str.upper().str.startswith("CN")]
    gen = gen[~((gen["sheet"] == "Retired and Canceled") & gen["Operating Year"].isna())]

    keep = {
        "Plant Code": "plant_code",
        "Generator ID": "gen_id",
        "Plant Name": "plant_name",
        "Utility Name": "utility_name",
        "State": "state",
        "County": "county",
        "Technology": "technology",
        "tech": "tech",
        "Nameplate Capacity (MW)": "mw",
        "Operating Year": "op_year",
        "Operating Month": "op_month",
        "Retirement Year": "retire_year",
        "Retirement Month": "retire_month",
        "Status": "status",
        "sheet": "sheet",
    }
    gen = gen[[c for c in keep if c in gen.columns]].rename(columns=keep)

    so_cols = {
        "Plant Code": "plant_code",
        "Generator ID": "gen_id",
        "Single-Axis Tracking?": "single_axis",
        "Fixed Tilt?": "fixed_tilt",
        "DC Net Capacity (MW)": "dc_mw",
    }
    solar = solar[list(so_cols)].rename(columns=so_cols)

    for df in (gen, solar):
        df["plant_code"] = df["plant_code"].astype(int)
        df["gen_id"] = df["gen_id"].astype(str)
    gen = gen.merge(solar, on=["plant_code", "gen_id"], how="left")
    for c in ("mw", "dc_mw", "op_year", "op_month", "retire_year", "retire_month"):
        if c in gen:
            gen[c] = pd.to_numeric(gen[c], errors="coerce")
    return gen


def load_plants() -> pd.DataFrame:
    with zipfile.ZipFile(_eia_zip()) as zf:
        p = _read_eia_sheet(zf, "2___Plant")
    cols = {
        "Plant Code": "plant_code",
        "Plant Name": "plant_name",
        "Utility Name": "utility_name",
        "City": "city",
        "State": "state",
        "County": "county",
        "Zip": "zip",
        "Latitude": "lat",
        "Longitude": "lon",
        "NERC Region": "nerc",
        "Balancing Authority Code": "ba",
    }
    p = p[list(cols)].rename(columns=cols)
    p["plant_code"] = p["plant_code"].astype(int)
    p["lat"] = pd.to_numeric(p["lat"], errors="coerce")
    p["lon"] = pd.to_numeric(p["lon"], errors="coerce")
    return p


def clean_objects(df: pd.DataFrame) -> pd.DataFrame:
    """EIA sheets mix ints, blanks and strings in one column; make object cols strings."""
    for c in df.columns[df.dtypes == object]:
        vals = df[c].dropna()
        if len(vals) and vals.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            df[c] = df[c].astype("boolean")
            continue
        df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip() or None)
    return df


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else None


def assign_turbine_sites(turbines: pd.DataFrame) -> pd.DataFrame:
    """Give each turbine the site_id of its EIA plant (or USWTDB project if no EIA id)."""
    t = turbines.copy()
    # A few turbines lack the EIA id their project-mates carry; borrow it.
    named = t["p_name"].notna() & ~t["p_name"].str.startswith("unknown", na=False)
    fill = (
        t[named & t["eia_id"].notna()]
        .groupby(["t_state", "p_name"])["eia_id"]
        .agg(lambda s: s.mode().iat[0])
    )
    miss = named & t["eia_id"].isna()
    t.loc[miss, "eia_id"] = [
        fill.get((st, pn), pd.NA) for st, pn in zip(t.loc[miss, "t_state"], t.loc[miss, "p_name"])
    ]
    t["site_id"] = np.where(
        t["eia_id"].notna(),
        "eia:" + t["eia_id"].astype("string"),
        "wtdb:" + t["t_state"].fillna("") + ":" + t["p_name"].fillna("unknown"),
    )
    return t


def wind_projects(t: pd.DataFrame) -> pd.DataFrame:
    """Roll turbines (with site_id) up to one row per site."""
    g = t.groupby("site_id")
    out = pd.DataFrame(
        {
            "wt_n_turbines": g.size(),
            "wt_lat": g["lat"].mean(),
            "wt_lon": g["lon"].mean(),
            "wt_project": g["p_name"].agg(_mode),
            "wt_state": g["t_state"].agg(_mode),
            "wt_county": g["t_county"].agg(_mode),
            "wt_year": g["p_year"].min(),
            "wt_manu": g["t_manu"].agg(_mode),
            "wt_model": g["t_model"].agg(_mode),
            "wt_turbine_mw": g["t_cap"].mean() / 1000.0,
            "wt_hub_m": g["t_hh"].mean(),
            "wt_rotor_m": g["t_rd"].mean(),
            "wt_tip_m": g["t_ttlh"].mean(),
            "wt_mw": g["t_cap"].sum(min_count=1) / 1000.0,
            "wt_p_cap": g["p_cap"].max(),
            "wt_offshore": g["t_offshore"].max(),
            "wt_retrofit": g["t_retrofit"].max(),
            # Spread of the turbine field; used as a footprint radius.
            "wt_extent_km": g.apply(
                lambda d: float(
                    np.hypot(
                        (d["lat"].max() - d["lat"].min()) * 111.0,
                        (d["lon"].max() - d["lon"].min()) * 111.0 * np.cos(np.radians(d["lat"].mean())),
                    )
                    / 2
                ),
                include_groups=False,
            ),
        }
    ).reset_index()
    return out


def build_us() -> pd.DataFrame:
    """US sites from EIA-860 + USWTDB (also writes turbines/generators tables)."""
    turbines = assign_turbine_sites(load_turbines())
    gens = load_generators()
    plants = load_plants()

    turbines.to_parquet(config.PROCESSED / "turbines.parquet", index=False)
    clean_objects(gens).to_parquet(config.PROCESSED / "generators.parquet", index=False)

    # --- per-plant technology roll-up from EIA generators -------------------
    def tech_agg(tech: str) -> pd.DataFrame:
        g = gens[gens["tech"] == tech].groupby("plant_code")
        d = pd.DataFrame(
            {
                f"{tech}_mw": g["mw"].sum(),
                f"{tech}_first_year": g["op_year"].min(),
                # decimal year of first operation (month known in EIA-860)
                f"{tech}_first_date": (g["op_year"].min() * 0 + (
                    gens[gens["tech"] == tech]
                    .assign(_d=lambda x: x["op_year"] + (x["op_month"].fillna(6) - 1) / 12)
                    .groupby("plant_code")["_d"].min()
                )),
                f"{tech}_last_year": g["op_year"].max(),
                f"{tech}_n_gens": g.size(),
                # A plant is retired for a tech only if every generator of it is.
                f"{tech}_retired": g["sheet"].agg(lambda s: bool((s == "Retired and Canceled").all())),
                f"{tech}_retire_year": g["retire_year"].max(),
            }
        )
        if tech == "solar":
            d["solar_tracking"] = g["single_axis"].agg(lambda s: bool((s == "Y").any()))
            d["solar_dc_mw"] = g["dc_mw"].sum(min_count=1)
        return d

    techs = [tech_agg(t) for t in config.TECHS]
    eia = plants.set_index("plant_code").join(techs, how="left")
    eia = eia[eia[[f"{t}_mw" for t in config.TECHS]].notna().any(axis=1)]
    eia = eia.reset_index()
    eia["site_id"] = "eia:" + eia["plant_code"].astype(str)
    eia["source"] = "eia860"

    # --- merge turbine detail ------------------------------------------------
    wp = wind_projects(turbines)
    sites = eia.merge(wp, on="site_id", how="outer", indicator=True)
    only_wtdb = sites["_merge"] == "right_only"
    sites.loc[only_wtdb, "source"] = "uswtdb"
    sites.loc[only_wtdb, "plant_name"] = sites.loc[only_wtdb, "wt_project"]
    sites.loc[only_wtdb, "state"] = sites.loc[only_wtdb, "wt_state"]
    sites.loc[only_wtdb, "county"] = sites.loc[only_wtdb, "wt_county"].str.replace(
        r"\s+County$", "", regex=True
    )
    sites.loc[only_wtdb, "wind_mw"] = sites.loc[only_wtdb, "wt_mw"].fillna(
        pd.to_numeric(sites.loc[only_wtdb, "wt_p_cap"], errors="coerce")
    )
    sites.loc[only_wtdb, "wind_first_year"] = sites.loc[only_wtdb, "wt_year"]
    sites.loc[only_wtdb, "wind_first_date"] = sites.loc[only_wtdb, "wt_year"] + 0.5
    sites = sites.drop(columns="_merge")
    # Turbine centroid is a better wind-farm location than the EIA plant point.
    has_wt = sites["wt_lat"].notna()
    sites.loc[has_wt, "lat"] = sites.loc[has_wt, "wt_lat"]
    sites.loc[has_wt, "lon"] = sites.loc[has_wt, "wt_lon"]
    # Real outlines: solar arrays (USPVDB) and turbine fields (USWTDB).
    fp = footprints.build(sites, turbines)
    # Array centroid is a better solar-farm location than the EIA plant point.
    # Wind farms keep their turbine centroid; mostly-solar hybrids use the arrays.
    pv = fp.dropna(subset=["pv_lat", "pv_lon"]).set_index("site_id")
    mostly_solar = sites["solar_mw"].fillna(0) > sites["wind_mw"].fillna(0)
    move = sites["site_id"].isin(pv.index) & (~has_wt | mostly_solar)
    sites.loc[move, "lat"] = sites.loc[move, "site_id"].map(pv["pv_lat"]).to_numpy()
    sites.loc[move, "lon"] = sites.loc[move, "site_id"].map(pv["pv_lon"]).to_numpy()
    sites = sites.dropna(subset=["lat", "lon"])
    sites = sites.merge(footprints.finish(fp, sites), on="site_id", how="left")

    mw =sites[[f"{t}_mw" for t in config.TECHS]].fillna(0)
    sites["primary_tech"] = mw.idxmax(axis=1).str.replace("_mw", "", regex=False)
    sites["total_mw"] = mw.sum(axis=1)
    present = mw.gt(0)
    present["wind_mw"] |= sites["wt_n_turbines"].fillna(0).gt(0)
    sites["techs"] = present.apply(
        lambda r: "+".join(t for t, on in zip(config.TECHS, r) if on), axis=1
    )
    sites["first_year"] = sites[[f"{t}_first_year" for t in config.TECHS]].min(axis=1)
    retired_cols = [c for c in (f"{t}_retired" for t in config.TECHS) if c in sites]
    sites["retired"] = sites[retired_cols].apply(
        lambda r: bool(r.dropna().astype(bool).all()) if r.notna().any() else False, axis=1
    )
    # Footprint radius for "at site" hazard: the real outline where we have one,
    # else MW → km² at typical land use (solar ≈ 0.025 km²/MW) or the turbine spread.
    solar_r = np.sqrt(sites["solar_mw"].fillna(0) * 0.025 / np.pi)
    estimate = np.maximum(sites["wt_extent_km"].fillna(0).to_numpy(), solar_r.to_numpy())
    sites["footprint_km"] = np.maximum(sites["fp_radius_km"].fillna(pd.Series(estimate, index=sites.index)), 0.2)
    sites["fp_kind"] = sites["fp_kind"].fillna("estimated from MW")
    sites["plant_code"] = sites["plant_code"].astype("Int64")
    sites = sites.sort_values("site_id").reset_index(drop=True)
    sites["country"] = "USA"
    sites["country_name"] = "United States"
    sites["coverage"] = "us_full"
    print(
        f"[sites] US: {len(sites):,} sites "
        f"({(sites['wind_mw']>0).sum():,} wind, {(sites['solar_mw']>0).sum():,} solar); {len(turbines):,} turbines; "
        f"{len(gens):,} generators"
    )
    return sites
