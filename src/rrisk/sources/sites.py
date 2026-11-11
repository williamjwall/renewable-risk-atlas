"""One site table for the world: US detail (EIA-860 + USWTDB) + global sites."""

from __future__ import annotations

import pandas as pd

from .. import config
from .global_sites import build_global
from .us_sites import build_us, clean_objects


def _digest(path) -> tuple:
    df = pd.read_parquet(path)
    return int(pd.util.hash_pandas_object(df, index=False).sum()), tuple(df.columns)


def build() -> pd.DataFrame:
    us = build_us()
    gl = build_global()
    sites = pd.concat([us, gl], ignore_index=True)
    sites = clean_objects(sites)
    out = config.PROCESSED / "sites.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    sites.to_parquet(tmp, index=False)
    if out.exists() and _digest(tmp) == _digest(out):
        # Unchanged: keep the old file (and its timestamp) so downstream steps
        # like the 4-minute hazard model aren't re-run for nothing.
        tmp.unlink()
        print(f"[sites] unchanged ({len(sites):,} sites)")
    else:
        tmp.replace(out)
        print(f"[sites] total {len(sites):,} sites in {sites['country'].nunique()} countries")
    return sites
