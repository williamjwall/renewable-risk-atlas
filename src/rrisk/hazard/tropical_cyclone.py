"""Global tropical-cyclone wind hazard from IBTrACS (1980-2025).

For every site: storms per year that brought
  tc34  gale-force winds (>= 34 kt) — within the storm's 34-kt radius
  tc64  hurricane/typhoon-force winds (>= 64 kt) — within the 64-kt radius
  tc96  major-hurricane winds (>= 96 kt, Cat 3+) — within half the 64-kt radius

Method
* Intensity: US agencies' 1-minute sustained wind where available (covers all
  basins via JTWC), else the WMO agency's wind converted from 10-minute to
  1-minute with the usual 0.88 factor.
* Radii: mean of the four quadrant radii reported by US agencies; global
  default radii where missing (config.TC_R*_DEFAULT_KM).
* Only tropical/subtropical stages count (extratropical and disturbance
  stages are dropped).
* Tracks are interpolated to <= 15 km spacing so fast storms don't skip sites.
* Frequency counts distinct storms; p = 1 - exp(-rate).
Computed at sites and on a 0.5° global grid for the map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .. import config

NM_KM = 1.852
TEN_TO_ONE_MIN = 0.88
ONE_MIN_AGENCIES = {"hurdat_atl", "hurdat_epa", "cphc", "atcf"}
LABELS = {
    "tc34": "Tropical storm (gale-force winds)",
    "tc64": "Hurricane/typhoon-force winds",
    "tc96": "Major hurricane winds (Cat 3+)",
}


def load_tracks() -> pd.DataFrame:
    out = config.PROCESSED / "tc_tracks_v2.parquet"
    src = config.RAW / "ibtracs" / "ibtracs.since1980.csv"
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return pd.read_parquet(out)
    quad = [f"USA_R{r}_{q}" for r in (34, 64) for q in ("NE", "SE", "SW", "NW")]
    cols = ["SID", "SEASON", "ISO_TIME", "NATURE", "LAT", "LON", "TRACK_TYPE", "WMO_WIND", "WMO_AGENCY", "USA_WIND", *quad]
    t = pd.read_csv(src, skiprows=[1], usecols=cols, keep_default_na=False, na_values=["", " "], low_memory=False)
    t = t[t["TRACK_TYPE"] == "main"]
    # Drop extratropical (ET) and disturbance (DS) stages: after transition the
    # storm is an ordinary mid-latitude low, which this tracker would otherwise
    # carry into e.g. the UK and Norway.
    t = t[~t["NATURE"].isin(["ET", "DS"])]
    for c in ["LAT", "LON", "WMO_WIND", "USA_WIND", *quad]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    wmo = np.where(t["WMO_AGENCY"].isin(ONE_MIN_AGENCIES), t["WMO_WIND"], t["WMO_WIND"] / TEN_TO_ONE_MIN)
    t["wind"] = t["USA_WIND"].fillna(pd.Series(wmo, index=t.index))

    def radius(r, default):
        q = t[[f"USA_R{r}_{x}" for x in ("NE", "SE", "SW", "NW")]].where(lambda v: v > 0)
        km = q.mean(axis=1) * NM_KM
        return km.fillna(default)

    t["r34"] = radius(34, config.TC_R34_DEFAULT_KM)
    t["r64"] = radius(64, config.TC_R64_DEFAULT_KM)
    t["time"] = pd.to_datetime(t["ISO_TIME"])
    t["year"] = t["SEASON"].astype(int)
    t = t[t["year"].isin(config.TC_YEARS) & t["wind"].notna()]
    t = t[["SID", "year", "time", "LAT", "LON", "wind", "r34", "r64"]].rename(columns={"LAT": "lat", "LON": "lon"})
    t["lon"] = ((t["lon"] + 180) % 360) - 180
    t.to_parquet(out, index=False)
    return t


def densify(t: pd.DataFrame, step_km: float = 15.0) -> pd.DataFrame:
    """Linear interpolation between consecutive fixes of the same storm."""
    t = t.sort_values(["SID", "time"]).reset_index(drop=True)
    same = t["SID"].eq(t["SID"].shift(-1)).to_numpy()
    lat, lon = t["lat"].to_numpy(), t["lon"].to_numpy()
    dlon = (np.roll(lon, -1) - lon + 180) % 360 - 180  # dateline-safe
    dist = np.hypot((np.roll(lat, -1) - lat) * 111, dlon * 111 * np.cos(np.radians(lat)))
    n = np.where(same, np.clip(np.ceil(dist / step_km), 1, 60), 1).astype(int)
    idx = np.repeat(np.arange(len(t)), n)
    frac = np.concatenate([np.arange(k) / k for k in n])
    nxt = np.minimum(idx + 1, len(t) - 1)
    has_next = same[idx]
    frac = np.where(has_next, frac, 0.0)

    def lerp(col):
        a = t[col].to_numpy(float)
        return a[idx] + frac * (a[nxt] - a[idx])

    out = pd.DataFrame({
        "storm": pd.factorize(t["SID"])[0][idx],
        "year": t["year"].to_numpy()[idx],
        "lat": lerp("lat"),
        "lon": ((lon[idx] + frac * dlon[idx] + 180) % 360) - 180,
        "wind": lerp("wind"), "r34": lerp("r34"), "r64": lerp("r64"),
    })
    return out


def storm_counts(t_lat, t_lon, ev: pd.DataFrame, r_km: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Distinct storms passing within each event point's own radius of each target."""
    n = len(t_lat)
    if ev.empty or n == 0:
        return np.zeros(n, dtype=int)
    tree = BallTree(np.radians(np.c_[t_lat, t_lon]), metric="haversine")
    storms = ev["storm"].to_numpy(np.int64)
    span = int(storms.max()) + 1
    pts = np.radians(ev[["lat", "lon"]].to_numpy())
    rr = np.asarray(r_km, float) / config.EARTH_KM
    keys = []
    for s in range(0, len(pts), chunk):
        ind = tree.query_radius(pts[s : s + chunk], r=rr[s : s + chunk])
        lens = np.fromiter((len(i) for i in ind), dtype=np.int64, count=len(ind))
        if lens.sum():
            keys.append(np.unique(np.concatenate(ind) * span + np.repeat(storms[s : s + chunk], lens)))
    if not keys:
        return np.zeros(n, dtype=int)
    uniq = np.unique(np.concatenate(keys))
    return np.bincount(uniq // span, minlength=n)


def _events(d: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    e34 = d[d["wind"] >= 34]
    e64 = d[d["wind"] >= 64]
    e96 = d[d["wind"] >= 96]
    return {
        "tc34": (e34, e34["r34"].to_numpy()),
        "tc64": (e64, e64["r64"].to_numpy()),
        "tc96": (e96, e96["r64"].to_numpy() * config.TC_R96_FRACTION),
    }


def build() -> pd.DataFrame:
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet", columns=["site_id", "lat", "lon"])
    tracks = load_tracks()
    d = densify(tracks)
    ny = len(config.TC_YEARS)
    ev = _events(d)
    out = sites[["site_id"]].copy()
    glat = np.arange(-60, 60 + 1e-9, config.TC_GRID_DEG)
    glon = np.arange(-180, 180, config.TC_GRID_DEG)
    G_lat, G_lon = np.meshgrid(glat, glon, indexing="ij")
    grids = {}
    for k, (e, r) in ev.items():
        n = storm_counts(sites["lat"].to_numpy(), sites["lon"].to_numpy(), e, r)
        lam = n / ny
        out[f"{k}_n_storms"] = n
        out[f"{k}_rate"] = lam
        out[f"{k}_p"] = 1 - np.exp(-lam)
        g = storm_counts(G_lat.ravel(), G_lon.ravel(), e, r).reshape(G_lat.shape) / ny
        grids[k] = g
        print(f"[tc] {k}: {(n > 0).sum():,} sites ever exposed; max {lam.max():.2f} storms/yr")
    out["tc_years"] = f"{config.TC_YEARS[0]}-{config.TC_YEARS[-1]}"
    out.to_parquet(config.PROCESSED / "hazard_tc.parquet", index=False)
    np.savez_compressed(config.PROCESSED / "hazard_tc_grid.npz", lat=glat, lon=glon, **grids)
    return out
