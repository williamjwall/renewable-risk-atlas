"""News-to-database: GDELT search -> article text -> Claude extraction."""

from __future__ import annotations


def run_extract(backend: str = "claude-cli", limit: int | None = None, model: str | None = None,
                since: str | None = None):
    """Screen, fetch and extract every not-yet-processed candidate article
    (up to ``limit`` new ones) and append the results to news_events_raw.parquet."""
    import pandas as pd

    from ... import config
    from . import extract, fetch

    src = config.PROCESSED / "news_articles.parquet"
    if not src.exists():
        raise FileNotFoundError("no candidate articles yet: run the news_search step first")
    arts = pd.read_parquet(src)
    cand = fetch.prefilter(arts)
    if since:
        cand = cand[cand["seendate"] >= pd.Timestamp(since)]
    if model:
        extract.MODEL = model
    done = {r["url"] for r in cand.to_dict("records") if extract._cache_get(r)}
    todo = cand[~cand["url"].isin(done)]
    print(f"[news] {len(cand):,} of {len(arts):,} articles pass the title screen; "
          f"{len(done):,} already extracted, {len(todo):,} new")
    if limit:
        todo = todo.head(limit)
    batch = pd.concat([cand[cand["url"].isin(done)], fetch.fetch(todo) if len(todo) else todo])
    ev = extract.extract(batch, backend=backend)
    for c in ev.columns[ev.dtypes == object]:
        ev[c] = ev[c].astype("string")
    out = config.PROCESSED / "news_events_raw.parquet"
    ev.to_parquet(out, index=False)
    print(f"[news] {len(ev)} extracted event mentions in total")
    return ev
