"""Global wind, rain and heat hazard from NASA POWER (MERRA-2 reanalysis, 1981-2025).

Free, no key. One request per MERRA-2 grid cell (0.5° x 0.625°) that holds a
site; responses are cached under data/raw/power/ so only new cells are fetched.

Per site (from its cell's daily series):
  xwind  days/yr with daily max 10 m wind >= 15 m/s (near-gale, sustained)
  xrain  days/yr with >= 50 mm rain (flash-flood / erosion proxy)
  heat   days/yr with max temperature >= 35 °C (PV thermal stress and derating)
plus 50-year return levels of the annual maxima (Gumbel, method of moments)
for wind (m/s) and 1-day rain (mm). p = 1 - exp(-rate).

Reanalysis winds are hourly means on a ~50 km grid, so they understate local
gusts (and miss hail entirely); they are used for *relative* exposure, which is
what the global score needs.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from .. import config

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
CACHE = config.RAW / "power"
DLAT, DLON = 0.5, 0.625
START, END = "19810101", "20251231"
THRESH = {"xwind": ("WS10M_MAX", 15.0), "xrain": ("PRECTOTCORR", 50.0), "heat": ("T2M_MAX", 35.0)}
LABELS = {
    "xwind": "Damaging wind days (≥ 15 m/s, reanalysis)",
    "xrain": "Extreme rain days (≥ 50 mm)",
    "heat": "Extreme heat days (≥ 35 °C)",
}


def _cell(lat, lon):
    return int(np.round(lat / DLAT)), int(np.round(lon / DLON))


def fetch_cell(ci: int, cj: int, retries: int = 4) -> pd.DataFrame | None:
    path = CACHE / f"{ci}_{cj}.csv"
    if not path.exists():
        params = {"parameters": "WS10M_MAX,PRECTOTCORR,T2M_MAX", "community": "RE",
                  "latitude": round(ci * DLAT, 3), "longitude": round(cj * DLON, 3),
                  "start": START, "end": END, "format": "CSV"}
        for k in range(retries):
            try:
                r = httpx.get(API, params=params, timeout=120, headers={"User-Agent": config.USER_AGENT})
                if r.status_code == 200 and "-END HEADER-" in r.text:
                    path.write_text(r.text)
                    break
                if r.status_code in (422, 400):  # outside model domain etc.
                    return None
            except httpx.HTTPError:
                pass
            time.sleep(5 * (k + 1))
        else:
            return None
    text = path.read_text()
    body = text.split("-END HEADER-", 1)[1]
    df = pd.read_csv(io.StringIO(body))
    return df.replace(-999, np.nan)


def _gumbel_rl(annual_max: pd.Series, T: float = 50.0) -> float:
    x = annual_max.dropna()
    if len(x) < 10:
        return np.nan
    beta = x.std() * np.sqrt(6) / np.pi
    mu = x.mean() - 0.5772 * beta
    return float(mu - beta * np.log(-np.log(1 - 1 / T)))


def cell_stats(df: pd.DataFrame) -> dict:
    years = df["YEAR"].nunique()
    out = {}
    for k, (col, thr) in THRESH.items():
        rate = float((df[col] >= thr).sum() / years)
        out[f"{k}_rate"], out[f"{k}_p"] = rate, 1 - np.exp(-rate)
    am = df.groupby("YEAR")[["WS10M_MAX", "PRECTOTCORR", "T2M_MAX"]].max()
    out["wind_rl50_ms"] = _gumbel_rl(am["WS10M_MAX"])
    out["rain_rl50_mm"] = _gumbel_rl(am["PRECTOTCORR"])
    out["tmax_rl50_c"] = _gumbel_rl(am["T2M_MAX"])
    return out


def build(workers: int = 4) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet", columns=["site_id", "lat", "lon"])
    cells = sorted({_cell(a, b) for a, b in zip(sites["lat"], sites["lon"])})
    todo = [c for c in cells if not (CACHE / f"{c[0]}_{c[1]}.csv").exists()]
    print(f"[reanalysis] {len(cells):,} grid cells, {len(todo):,} to download from NASA POWER")
    stats = {}

    def work(c):
        df = fetch_cell(*c)
        return c, (cell_stats(df) if df is not None and len(df) else None)

    with ThreadPoolExecutor(workers) as ex:
        for n, (c, s) in enumerate(ex.map(work, cells), 1):
            if s:
                stats[c] = s
            if n % 250 == 0:
                print(f"[reanalysis] {n:,}/{len(cells):,} cells", flush=True)
    rows = []
    for sid, a, b in zip(sites["site_id"], sites["lat"], sites["lon"]):
        rows.append({"site_id": sid, **(stats.get(_cell(a, b)) or {})})
    out = pd.DataFrame(rows)
    out["reanalysis_years"] = f"{START[:4]}-{END[:4]}"
    out.to_parquet(config.PROCESSED / "hazard_global.parquet", index=False)
    missing = out["xwind_rate"].isna().sum() if "xwind_rate" in out else len(out)
    print(f"[reanalysis] {len(out) - missing:,} sites with data, {missing:,} without")
    return out
