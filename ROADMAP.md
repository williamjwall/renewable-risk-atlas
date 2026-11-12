# Roadmap

What the atlas is for: a free, sourced, first look at how much weather a wind
or solar site is exposed to, and why, for anyone screening sites or portfolios
(developers, lenders, insurers, analysts). Not a loss model or a price.

## Shipped

- Real site outlines on the hosted site (static `site/footprints/` shards).
- Link to any site: `?site=<id>` opens its card; **Share link** on the card.

## Next, in order

1. **Score your own sites.** Paste or upload coordinates (planned or private
   assets) and get the same peril breakdown, score and CSV. Biggest jump in
   who can use it. Needs the hazard grids shipped to the browser, or lookup
   from the nearest scored sites as a first cut.
2. **Notable past events on the card.** Largest nearby hail, closest
   hurricane track, worst gust, with dates. Turns rates into something people
   remember; the data is already in the warehouse.
3. **Wildfire and flood perils.** USFS Wildfire Hazard Potential and FEMA
   flood zones in the US; JRC flood maps and burned area globally.
4. **Compare sites side by side** (2-4, peril bars aligned).
5. **Printable one-page site report** (print stylesheet: map, outline, score,
   perils, sources).
6. **Climate outlook**: change in heat, extreme rain and wind days by ~2050
   from NASA NEX-GDDP.

## How to ship

`uv run rrisk export --out site/index.html`, commit `site/` with the source,
push `main`; Vercel serves `site/`.
