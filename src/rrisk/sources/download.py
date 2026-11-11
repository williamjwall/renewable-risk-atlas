"""Fetch the public source datasets into data/raw/<source>/.

Every file lands in its own source directory and is only ever *read* as data
(parsed by pandas/duckdb); nothing downloaded is executed or imported.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .. import config

USWTDB_CSV = "https://energy.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip"
EIA860_INDEX = "https://www.eia.gov/electricity/data/eia860/"
STORM_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
RADAR_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/"
US_ATLAS = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json"
WORLD_ATLAS = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
USPVDB_ZIP = "https://energy.usgs.gov/uspvdb/assets/data/uspvdbGeoJSON.zip"
GPPD_ZIP = "https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip"
IBTRACS_CSV = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, read=300.0),
    )


def fetch(url: str, dest: Path, *, force: bool = False, retries: int = 3) -> Path:
    """Stream ``url`` to ``dest`` atomically; skip if it already exists."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _client() as c, c.stream("GET", url) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            print(f"  ok  {dest.relative_to(config.ROOT)} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except (httpx.HTTPError, OSError) as e:
            print(f"  retry {attempt}/{retries} {url}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"failed to download {url}")


def _listing(url: str) -> str:
    with _client() as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text


def uswtdb() -> Path:
    return fetch(USWTDB_CSV, config.RAW / "uswtdb" / "uswtdbCSV.zip")


def eia860() -> Path:
    """Latest final-release EIA-860 annual zip (falls back to early release)."""
    html = _listing(EIA860_INDEX)
    final = sorted(set(re.findall(r'href="(xls/eia860(\d{4})\.zip)"', html)), key=lambda t: t[1])
    early = sorted(set(re.findall(r'href="(xls/eia860(\d{4})ER\.zip)"', html)), key=lambda t: t[1])
    rel, year = (final or early)[-1]
    return fetch(EIA860_INDEX + rel, config.RAW / "eia860" / Path(rel).name)


def storm_events(years=config.REPORT_YEARS) -> list[Path]:
    html = _listing(STORM_INDEX)
    names = set(re.findall(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz", html))
    latest: dict[int, str] = {}
    for y, c in names:
        y = int(y)
        if y in years and c > latest.get(y, ""):
            latest[y] = c
    jobs = [
        (
            STORM_INDEX + f"StormEvents_details-ftp_v1.0_d{y}_c{c}.csv.gz",
            config.RAW / "storm_events" / f"details_{y}.csv.gz",
        )
        for y, c in sorted(latest.items())
    ]
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(lambda j: fetch(*j), jobs))


def radar_hail(years=config.RADAR_YEARS) -> list[Path]:
    jobs = [
        (RADAR_BASE + f"hail-{y}.csv.gz", config.RAW / "radar_hail" / f"hail-{y}.csv.gz")
        for y in years
    ]
    with ThreadPoolExecutor(3) as ex:
        return list(ex.map(lambda j: fetch(*j), jobs))


def us_atlas() -> Path:
    return fetch(US_ATLAS, config.RAW / "us_atlas" / "states-10m.json")


def world_atlas() -> Path:
    return fetch(WORLD_ATLAS, config.RAW / "world_atlas" / "countries-50m.json")


def uspvdb() -> Path:
    return fetch(USPVDB_ZIP, config.RAW / "uspvdb" / "uspvdbGeoJSON.zip")


def gppd() -> Path:
    return fetch(GPPD_ZIP, config.RAW / "gppd" / "gppd_v1_3.zip")


def ibtracs(force: bool = False) -> Path:
    return fetch(IBTRACS_CSV, config.RAW / "ibtracs" / "ibtracs.since1980.csv", force=force)


# name -> (function, group). "us" sources are large; "global" are small.
SOURCES = {
    "uswtdb": (uswtdb, "us"),
    "uspvdb": (uspvdb, "us"),
    "eia860": (eia860, "us"),
    "storms": (storm_events, "us"),
    "radar": (radar_hail, "us"),
    "atlas": (us_atlas, "maps"),
    "world_atlas": (world_atlas, "maps"),
    "gppd": (gppd, "global"),
    "ibtracs": (ibtracs, "global"),
}


def run(which: list[str] | None = None) -> None:
    """Download sources by name or group (us, global, maps); default all."""
    names = []
    for w in which or list(SOURCES):
        names += [n for n, (_, g) in SOURCES.items() if g == w] or [w]
    for name in dict.fromkeys(names):
        if name not in SOURCES:
            raise SystemExit(f"unknown source {name!r}; choose from {', '.join(SOURCES)} or a group")
        print(f"[download] {name}")
        SOURCES[name][0]()
