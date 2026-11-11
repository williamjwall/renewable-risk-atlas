"""Per-site convective hazard: annual frequency/probability of damaging hail,
extreme thunderstorm wind and tornadoes.

Method (a deliberately simple version of a cat-model hazard module):

1. Event sets
   * NOAA Storm Events point reports 1996-2025: hail >= 1", hail >= 2",
     thunderstorm gusts >= 65 kt, tornadoes EF1+ (paths sampled every ~2 km).
   * NEXRAD hail-signature cells (NCEI SWDI nx3hail) 2010-2025 with the
     radar's maximum expected hail size (MEHS) >= 1" / >= 2" and POSH >= 50 %.
2. Hazard frequency = number of distinct *event days* per year, because many
   reports from one storm are not independent losses.
3. Reports: SPC convention — event within 40 km (25 mi) of the point —
   computed on a 0.25° grid, then Gaussian-smoothed (normalised convolution,
   sigma 60 km) to damp the population/reporting bias of spotter reports, and
   sampled at each site.
4. Radar: days with a radar hail cell within 10 km + the site's footprint
   radius. Less population-biased and closer to "the site got hit". MEHS
   over-estimates size, so the size thresholds are calibrated against the
   reports (config.RADAR_MEHS_FOR_1IN / _2IN).
5. Annual probability of >= 1 event: p = 1 - exp(-lambda)  (Poisson).

Outputs data/processed/hazard_us.parquet and hazard_us_grid.npz (US sites only).
"""

from __future__ import annotations

import glob

import duckdb
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import BallTree

from .. import config

DAY0 = np.datetime64("1950-01-01", "D")
HAZARDS = ("hail1", "hail2", "wind65", "tor1")
RADAR_HAZARDS = ("rhail1", "rhail2")
LABELS = {
    "hail1": 'Hail >= 1" within 25 mi (reports)',
    "hail2": 'Hail >= 2" within 25 mi (reports)',
    "wind65": "Thunderstorm gust >= 65 kt (75 mph) within 25 mi (reports)",
    "tor1": "EF1+ tornado path within 25 mi (reports)",
    "rhail1": 'Radar-indicated hail >= 1" at/near site',
    "rhail2": 'Radar-indicated hail >= 2" at/near site',
}

# CONUS grid; non-CONUS sites fall back to unsmoothed site-level counts.
GRID_LAT = np.arange(24.0, 50.0 + 1e-9, config.GRID_DEG)
GRID_LON = np.arange(-125.0, -66.5 + 1e-9, config.GRID_DEG)


# ----------------------------------------------------------------------------
# Event sets
# ----------------------------------------------------------------------------

def load_reports(rebuild: bool = False) -> pd.DataFrame:
    """Storm Events point reports for the hazard classes + a coverage set."""
    out = config.PROCESSED / "storm_reports.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    files = sorted(glob.glob(str(config.RAW / "storm_events" / "details_*.csv.gz")))
    files = [f for f in files if int(f[-11:-7]) in config.REPORT_YEARS]
    con = duckdb.connect()
    df = con.sql(
        f"""
        select
            try_cast(EVENT_ID as bigint)                     as event_id,
            try_cast(EPISODE_ID as bigint)                   as episode_id,
            EVENT_TYPE                                       as event_type,
            make_date(try_cast(BEGIN_YEARMONTH as int) // 100,
                      try_cast(BEGIN_YEARMONTH as int) % 100,
                      try_cast(BEGIN_DAY as int))            as date,
            try_cast(MAGNITUDE as double)                    as magnitude,
            MAGNITUDE_TYPE                                   as magnitude_type,
            TOR_F_SCALE                                      as tor_scale,
            try_cast(BEGIN_LAT as double)                    as lat,
            try_cast(BEGIN_LON as double)                    as lon,
            try_cast(END_LAT as double)                      as end_lat,
            try_cast(END_LON as double)                      as end_lon,
            STATE                                            as state,
            DAMAGE_PROPERTY                                  as damage_property
        from read_csv({files!r}, union_by_name=true, all_varchar=true,
                      strict_mode=false, ignore_errors=true)
        where EVENT_TYPE in ('Hail', 'Thunderstorm Wind', 'Tornado')
          and try_cast(BEGIN_LAT as double) is not null
        """
    ).df()
    df = df[df["lat"].between(17, 72) & df["lon"].between(-180, -60)]
    df.to_parquet(out, index=False)
    print(f"[hazard] {len(df):,} hail/wind/tornado reports cached")
    return df


def _tor_rank(scale: pd.Series) -> pd.Series:
    return pd.to_numeric(scale.fillna("").str.extract(r"F(\d)")[0], errors="coerce")


def hazard_events(reports: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Points (lat, lon, day) for each hazard class."""
    r = reports.copy()
    r["day"] = (r["date"].values.astype("datetime64[D]") - DAY0).astype(int)
    out = {
        "hail1": r[(r.event_type == "Hail") & (r.magnitude >= config.HAIL_1IN)],
        "hail2": r[(r.event_type == "Hail") & (r.magnitude >= config.HAIL_2IN)],
        "wind65": r[(r.event_type == "Thunderstorm Wind") & (r.magnitude >= config.WIND_KT_EXTREME)],
    }
    tor = r[(r.event_type == "Tornado") & (_tor_rank(r.tor_scale) >= 1)].copy()
    tor["end_lat"] = tor["end_lat"].fillna(tor["lat"])
    tor["end_lon"] = tor["end_lon"].fillna(tor["lon"])
    # Sample tornado paths every ~2 km so long tracks count along their length.
    dist = np.hypot(
        (tor.end_lat - tor.lat) * 111, (tor.end_lon - tor.lon) * 111 * np.cos(np.radians(tor.lat))
    )
    n = np.clip(np.ceil(dist / 2.0).astype(int), 1, 200).to_numpy()
    idx = np.repeat(np.arange(len(tor)), n + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
    t = tor.iloc[idx]
    out["tor1"] = pd.DataFrame(
        {
            "lat": t.lat.to_numpy() + frac * (t.end_lat.to_numpy() - t.lat.to_numpy()),
            "lon": t.lon.to_numpy() + frac * (t.end_lon.to_numpy() - t.lon.to_numpy()),
            "day": t.day.to_numpy(),
        }
    )
    out["coverage"] = r  # any severe report: defines where the network "sees"
    return {k: v[["lat", "lon", "day"]].reset_index(drop=True) for k, v in out.items()}


def load_radar(rebuild: bool = False) -> pd.DataFrame:
    """Radar hail cells with MEHS >= 1", thinned to one point per ~2 km cell per day."""
    out = config.PROCESSED / "radar_hail.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    frames = []
    con = duckdb.connect()
    for y in config.RADAR_YEARS:
        f = config.RAW / "radar_hail" / f"hail-{y}.csv.gz"
        if not f.exists():
            print(f"[hazard] missing radar year {y}; skipping")
            continue
        d = con.sql(
            f"""
            select
              -- convective day: 12Z-12Z
              datediff('day', date '1950-01-01',
                       cast(strptime(cast(ZTIME as varchar), '%Y%m%d%H%M%S') - interval 12 hour as date)) as day,
              round(LAT / 0.02) * 0.02 as lat,
              round(LON / 0.02) * 0.02 as lon,
              max(MAXSIZE) as maxsize
            from read_csv('{f}', comment='#', header=false,
                          names=['ZTIME','LON','LAT','WSR_ID','CELL_ID','RANGE','AZIMUTH','SEVPROB','PROB','MAXSIZE'],
                          types={{'ZTIME':'BIGINT','LON':'DOUBLE','LAT':'DOUBLE','SEVPROB':'DOUBLE','MAXSIZE':'DOUBLE'}},
                          ignore_errors=true)
            where MAXSIZE >= {config.HAIL_1IN} and SEVPROB >= 50
            group by all
            """
        ).df()
        d["year"] = y
        frames.append(d)
        print(f"[hazard] radar {y}: {len(d):,} thinned hail cells")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    return df


# ----------------------------------------------------------------------------
# Neighbourhood counting
# ----------------------------------------------------------------------------

def _tree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    return BallTree(np.radians(np.c_[lat, lon]), metric="haversine")


def event_days(
    t_lat: np.ndarray,
    t_lon: np.ndarray,
    events: pd.DataFrame,
    radius_km: float | np.ndarray,
    chunk: int = 100_000,
) -> np.ndarray:
    """Distinct event days with >= 1 event within radius of each target.

    ``radius_km`` may be per-target (site footprint); we query with the max
    radius and filter by exact great-circle distance.
    """
    n = len(t_lat)
    if events.empty or n == 0:
        return np.zeros(n, dtype=int)
    radius = np.broadcast_to(np.asarray(radius_km, dtype=float), (n,))
    rmax = float(radius.max())
    tree = _tree(t_lat, t_lon)
    days = events["day"].to_numpy(np.int64)
    span = int(days.max()) + 1
    keys = []
    ev = np.radians(events[["lat", "lon"]].to_numpy())
    for s in range(0, len(ev), chunk):
        ind, dist = tree.query_radius(ev[s : s + chunk], r=rmax / config.EARTH_KM, return_distance=True)
        lens = np.fromiter((len(i) for i in ind), dtype=np.int64, count=len(ind))
        if lens.sum() == 0:
            continue
        tgt = np.concatenate(ind)
        dkm = np.concatenate(dist) * config.EARTH_KM
        day = np.repeat(days[s : s + chunk], lens)
        ok = dkm <= radius[tgt]
        keys.append(np.unique(tgt[ok] * span + day[ok]))
    if not keys:
        return np.zeros(n, dtype=int)
    uniq = np.unique(np.concatenate(keys))
    return np.bincount(uniq // span, minlength=n)


def _poisson_ci(k: np.ndarray, years: float, level: float = 0.90):
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0) / years
    hi = stats.chi2.ppf(1 - a, 2 * (k + 1)) / 2 / years
    return lo, hi


def _smooth(rate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalised Gaussian convolution so coasts/borders aren't diluted by zeros."""
    sig_lat = config.SMOOTH_SIGMA_KM / (111.0 * config.GRID_DEG)
    sig_lon = config.SMOOTH_SIGMA_KM / (111.0 * np.cos(np.radians(37.0)) * config.GRID_DEG)
    num = ndimage.gaussian_filter(rate * mask, (sig_lat, sig_lon), mode="constant")
    den = ndimage.gaussian_filter(mask.astype(float), (sig_lat, sig_lon), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.05, num / den, np.nan)
    return out


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

def build(rebuild_events: bool = False) -> pd.DataFrame:
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet")
    sites = sites[sites["country"] == "USA"].reset_index(drop=True)
    reports = load_reports(rebuild_events)
    ev = hazard_events(reports)
    n_rep_years = len(config.REPORT_YEARS)

    glat, glon = np.meshgrid(GRID_LAT, GRID_LON, indexing="ij")
    g_lat, g_lon = glat.ravel(), glon.ravel()

    coverage = event_days(g_lat, g_lon, ev["coverage"], config.NEIGHBOURHOOD_KM).reshape(glat.shape)
    mask = coverage > 0

    grids: dict[str, np.ndarray] = {}
    out = sites[["site_id", "lat", "lon", "footprint_km"]].copy()
    in_grid = (
        out.lat.between(GRID_LAT[0], GRID_LAT[-1]) & out.lon.between(GRID_LON[0], GRID_LON[-1])
    ).to_numpy()

    for h in HAZARDS:
        raw_grid = event_days(g_lat, g_lon, ev[h], config.NEIGHBOURHOOD_KM).reshape(glat.shape)
        rate_grid = raw_grid / n_rep_years
        sm = _smooth(rate_grid, mask)
        grids[h] = sm
        interp = RegularGridInterpolator((GRID_LAT, GRID_LON), np.nan_to_num(sm), bounds_error=False)
        k_site = event_days(out.lat.to_numpy(), out.lon.to_numpy(), ev[h], config.NEIGHBOURHOOD_KM)
        raw = k_site / n_rep_years
        lam = np.where(in_grid, interp(np.c_[out.lat, out.lon]), raw)
        lam = np.where(np.isnan(lam), raw, lam)
        lo, hi = _poisson_ci(k_site, n_rep_years)
        out[f"{h}_n_days"] = k_site
        out[f"{h}_rate_raw"] = raw
        out[f"{h}_rate_lo90"] = lo
        out[f"{h}_rate_hi90"] = hi
        out[f"{h}_rate"] = lam
        out[f"{h}_p"] = 1 - np.exp(-lam)
        print(f"[hazard] {h}: median {np.median(lam):.3f} days/yr, max {lam.max():.2f}")

    # --- radar ---------------------------------------------------------------
    radar = load_radar(rebuild_events)
    n_rad_years = radar["year"].nunique() if len(radar) else 1
    # Same near-site radius for every site so rates are comparable across
    # technologies; large wind farms only get a little extra for their spread.
    site_r = config.RADAR_SITE_KM + np.minimum(out["footprint_km"].to_numpy(), config.FOOTPRINT_CAP_KM)
    for h, thr in (("rhail1", config.RADAR_MEHS_FOR_1IN), ("rhail2", config.RADAR_MEHS_FOR_2IN)):
        pts = radar[radar.maxsize >= thr]
        k = event_days(out.lat.to_numpy(), out.lon.to_numpy(), pts, site_r)
        lam = k / n_rad_years
        lo, hi = _poisson_ci(k, n_rad_years)
        out[f"{h}_n_days"] = k
        out[f"{h}_rate"] = lam
        out[f"{h}_rate_lo90"] = lo
        out[f"{h}_rate_hi90"] = hi
        out[f"{h}_p"] = 1 - np.exp(-lam)
        print(f"[hazard] {h}: median {np.median(lam):.3f} days/yr, max {lam.max():.2f}")
        # coarse radar grid for the map layer
        rg = event_days(g_lat, g_lon, pts, config.RADAR_SITE_KM).reshape(glat.shape) / n_rad_years
        grids[h] = np.where(mask, rg, np.nan)

    out["hazard_years_reports"] = f"{config.REPORT_YEARS[0]}-{config.REPORT_YEARS[-1]}"
    out["hazard_years_radar"] = f"{config.RADAR_YEARS[0]}-{config.RADAR_YEARS[-1]}"
    out = out.drop(columns=["lat", "lon", "footprint_km"])
    out.to_parquet(config.PROCESSED / "hazard_us.parquet", index=False)
    np.savez_compressed(
        config.PROCESSED / "hazard_us_grid.npz", lat=GRID_LAT, lon=GRID_LON, mask=mask, **grids
    )
    return out
