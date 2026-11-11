"""Country name -> ISO-3166 alpha-3, for matching loss events to sites worldwide."""

from __future__ import annotations

import functools
import re
import zipfile

import pandas as pd

from .. import config

ALIASES = {
    "us": "USA", "usa": "USA", "u.s.": "USA", "u.s.a.": "USA", "united states": "USA",
    "united states of america": "USA", "america": "USA",
    "uk": "GBR", "u.k.": "GBR", "britain": "GBR", "great britain": "GBR", "england": "GBR",
    "scotland": "GBR", "wales": "GBR", "northern ireland": "GBR", "united kingdom": "GBR",
    "south korea": "KOR", "korea": "KOR", "republic of korea": "KOR", "korea, republic of": "KOR",
    "north korea": "PRK", "russia": "RUS", "russian federation": "RUS", "czechia": "CZE",
    "czech republic": "CZE", "taiwan": "TWN", "vietnam": "VNM", "viet nam": "VNM", "iran": "IRN",
    "turkey": "TUR", "türkiye": "TUR", "turkiye": "TUR", "uae": "ARE", "united arab emirates": "ARE",
    "ivory coast": "CIV", "côte d'ivoire": "CIV", "holland": "NLD", "netherlands": "NLD",
    "the netherlands": "NLD", "china": "CHN", "people's republic of china": "CHN", "prc": "CHN",
    "hong kong": "HKG", "macau": "MAC", "laos": "LAO", "bolivia": "BOL", "venezuela": "VEN",
    "tanzania": "TZA", "syria": "SYR", "moldova": "MDA", "palestine": "PSE",
    "dominican republic": "DOM", "philippines": "PHL", "australia": "AUS", "japan": "JPN",
    "germany": "DEU", "france": "FRA", "spain": "ESP", "italy": "ITA", "india": "IND",
    "brazil": "BRA", "canada": "CAN", "mexico": "MEX", "chile": "CHL", "belgium": "BEL",
    "switzerland": "CHE", "sweden": "SWE", "norway": "NOR", "denmark": "DNK", "finland": "FIN",
    "poland": "POL", "ireland": "IRL", "portugal": "PRT", "greece": "GRC", "austria": "AUT",
    "south africa": "ZAF", "egypt": "EGY", "morocco": "MAR", "saudi arabia": "SAU",
    "new zealand": "NZL", "argentina": "ARG", "thailand": "THA", "indonesia": "IDN",
    "malaysia": "MYS", "singapore": "SGP", "pakistan": "PAK", "israel": "ISR",
}


@functools.cache
def _gppd_names() -> dict[str, str]:
    path = config.RAW / "gppd" / "gppd_v1_3.zip"
    if not path.exists():
        return {}
    with zipfile.ZipFile(path) as zf, zf.open("global_power_plant_database.csv") as fh:
        d = pd.read_csv(fh, usecols=["country", "country_long"]).drop_duplicates()
    return {n.lower(): c for c, n in zip(d["country"], d["country_long"])}


def iso3(name) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    k = re.sub(r"\s+", " ", name.strip().lower())
    if len(k) == 3 and k.upper() in set(_gppd_names().values()) | set(ALIASES.values()):
        return k.upper()
    return ALIASES.get(k) or _gppd_names().get(k)


@functools.cache
def names() -> dict[str, str]:
    """ISO3 -> display name."""
    out = {v: k.title() for k, v in _gppd_names().items()}
    out.update({"USA": "United States", "GBR": "United Kingdom", "KOR": "South Korea"})
    return out
