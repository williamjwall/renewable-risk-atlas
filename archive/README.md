# Archived: loss database and failure model

Removed from the app on request (scores are now weather hazard only). Kept here
so the work isn't lost; nothing in `src/rrisk` imports it.

- `losses/events.py` — merge in-house + news incidents, match to sites, de-duplicate
- `losses/news/` — GDELT search, article fetch, Claude extraction (structured output)
- `survival.py` — Cox failure model per technology
- `incidents/` — template for an in-house incident list
- `tests/` — the tests that covered them

To revive: move the folders back under `src/rrisk/`, re-add the `events`,
`survival`, `news_search` and `news_extract` steps in `pipeline.py`, and the
history/failure factors in `models/score.py`. Dependencies they need:
`rapidfuzz`, `trafilatura`, `anthropic`, `lifelines`.
