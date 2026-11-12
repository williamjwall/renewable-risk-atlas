# Renewable Site Risk Atlas (`rrisk`)

Weather-hazard exposure and a risk score for **20,858 wind and solar sites in
103 countries**, built from free public data, with a local web app to explore,
query and update it.

## Use it

```bash
cd ~/code/renewable-risk-atlas
uv run rrisk app          # opens http://127.0.0.1:8765 in your browser
```

| Tab | For |
|---|---|
| **Map** | Every site coloured by tier, hazard layers, region presets, and a site card showing which perils make up its score |
| **Sites** | The same filters as a sortable table; download as CSV |
| **Guide** | How the score works (formula, weights, a worked example), every peril and how far it's been validated, data sources, searchable terms, FAQ |
| **Query** | A no-SQL query builder, SQL editor with examples, column descriptions, saved queries and history; results open sites on the map |
| **Data** | Pipeline status, **Update** (rebuilds only what changed), per-step runs with a live log |

The theme button in the top bar switches Light (default) / Dark / System (remembered per browser).
Views can be bookmarked: `http://127.0.0.1:8765/?layer=tc64&country=PHL#map`
(`layer`, `country` as ISO-3, `tech`, `basis=global`).

### Terminal equivalents

```bash
uv run rrisk update            # rebuild only what changed (same as the Update button)
uv run rrisk status            # what's built, stale, or failed
uv run rrisk run hazard_global score       # run specific steps
uv run rrisk site "Moss Landing"           # risk card in the terminal
uv run rrisk query "select name, country_name, score from site_risk order by score desc limit 10"
uv run rrisk validate          # back-test the hazard model
uv run rrisk export            # one shareable HTML file -> data/output/rrisk_atlas.html
```

### Fresh machine

```bash
uv sync
uv run rrisk download          # ~3.2 GB (US radar hail is most of it)
uv run rrisk update            # ~30 min the first time (NASA POWER downloads), minutes after
uv run rrisk app
```

## What the score is

For each peril, a site's yearly rate is divided by the average rate, capped at
10×, weighted by how much that peril damages the technology, and summed:

```
hazard index = Σ weight(tech, peril) × min(site rate ÷ average rate, 10)
score        = percentile of the hazard index among sites of the same technology
```

* **US model** (US sites): hail ≥1″ near site (radar), hail ≥2″ within 25 mi
  (reports) and near site (radar), thunderstorm gusts ≥75 mph, EF1+ tornadoes,
  hurricane winds.
* **Global model** (every site, US included, so countries compare): tropical-storm,
  hurricane and major-hurricane winds; damaging wind, extreme rain and extreme
  heat days from reanalysis. No hail or tornado data exists for free outside the US.

Tiers: Low < 40 ≤ Moderate < 70 ≤ High < 90 ≤ Very high. The score ranks
exposure; it is not a loss estimate. Weights are judgement and live in
`src/rrisk/models/score.py`; the Guide tab shows them and a worked example.

## Project layout

```
src/rrisk/
  config.py            paths, settings (rrisk.toml), modelling constants
  pipeline.py          steps, their inputs/outputs, incremental runner
  cli.py               `rrisk ...`
  docs.py              table and column descriptions (Query tab, Guide)
  validate.py          back-tests
  sources/             raw data -> site table (EIA-860, USWTDB, USPVDB outlines, GPPD/GEM)
  hazard/
    us_convective.py     US hail, thunderstorm gusts, tornadoes
    tropical_cyclone.py  global hurricane/typhoon wind (IBTrACS)
    reanalysis.py        global damaging wind, extreme rain, extreme heat (NASA POWER)
  models/score.py      hazard index, scores, DuckDB warehouse
  app/                 local web app (FastAPI), data bundle, static export, front-end
tests/                 unit tests on synthetic data (`uv run pytest`)
archive/               removed features kept for reference (loss database, news
                       extraction, failure model) — not imported
data/                  raw downloads, processed tables, warehouse.duckdb (git-ignored)
```

### Pipeline steps

`rrisk update` runs every step whose inputs (data files or its own source code)
changed since its last successful run.

| Step | Builds | Notes |
|---|---|---|
| download_global / download_maps / download_us | raw data | only if missing |
| sites | `sites`, `turbines`, `generators` | skipped downstream if unchanged |
| hazard_us | `hazard_us`, `storm_reports` | ~40 s with cached events, ~4 min from raw |
| hazard_tc | `hazard_tc` | ~15 s |
| hazard_global | `hazard_global` | ~20–50 min the first time (downloads), about a minute after |
| score | `site_scores`, `data/warehouse.duckdb` | |
| validate | `data/output/validation.json` | ~35 s |
| export | `data/output/rrisk_atlas.html` | |

The view **`site_risk`** joins everything: one row per site. Every column is
described in the Query tab and in `src/rrisk/docs.py`.

## Data

| Source | Coverage | Used for |
|---|---|---|
| [EIA-860](https://www.eia.gov/electricity/data/eia860/) | US, 2025 | site list, MW, dates, solar tracking |
| [US Wind Turbine Database](https://energy.usgs.gov/uswtdb/) | US, V9.1 Sept 2026 | turbine maker, model, size, location; wind-farm outlines |
| [US Large-Scale Solar PV Database](https://energy.usgs.gov/uspvdb/) | US, V4.0 Apr 2026 | solar-array outlines: location, area, footprint |
| [WRI Global Power Plant Database](https://datasets.wri.org/dataset/globalpowerplantdatabase) | 102 countries, to ~2021 | global sites |
| [Global Energy Monitor trackers](https://globalenergymonitor.org/) (optional) | worldwide, current | fresher global sites: drop the .xlsx files in `data/raw/gem/` |
| [NOAA Storm Events](https://www.ncei.noaa.gov/stormevents/) | US, 1996–2025 | hail, gust, tornado reports |
| [NEXRAD hail signatures](https://www.ncei.noaa.gov/products/severe-weather-data-inventory) | US, 2010–2025 | radar hail near each site |
| [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | worldwide, 1980–2025 | tropical-cyclone tracks, intensity, wind radii |
| [NASA POWER](https://power.larc.nasa.gov/) (MERRA-2) | worldwide, 1981–2025 | daily max wind, rain, max temperature |

## Validation

`rrisk validate` fits the US hazard on 1996–2010 and checks it against
2011–2025 at every US site: rank correlation 0.77–0.89, the top tenth of sites
saw 38–530× more events than the bottom tenth, calibration within 2–4
percentage points. Radar ≥2″ hail within 10 km is unstable (0.29), so it carries
little weight. The global perils have not been back-tested yet.

## Removed on purpose

* **Battery storage**: battery fires couldn't be predicted from this data.
* **Loss events and the failure model**: public loss records were too thin
  to estimate anything (the code is kept in `archive/`).

## Known limitations

* Outside the US, hail, tornadoes and lightning aren't modelled; flood and
  wildfire aren't modelled anywhere. Extreme rain is a proxy, not a flood model.
* Reanalysis wind is a ~50 km smoothed field; it ranks places but understates gusts.
* GPPD stops around 2021; add GEM trackers for current global sites.
* Exposure only: no insured values, damage curves or claims.
