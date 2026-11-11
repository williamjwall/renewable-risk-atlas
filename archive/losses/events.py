"""Loss-event database: merge in-house and news-extracted incidents,
match each to a site in the site table, and collapse duplicates.

Unified event columns
  event_uid, source, source_ref, asset_type, site_name, operator_or_owner,
  country, state (postal), county, city, lat, lon, event_date, date_precision,
  cause, equipment_manufacturer, equipment_model, capacity_mw, capacity_mwh,
  damage_summary, confidence

Outputs
  loss_mentions.parquet  every mention, with its site match
  loss_events.parquet    de-duplicated incidents (one row per real-world event)
"""

from __future__ import annotations

import hashlib
import re
import warnings

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .. import config
from ..sources.countries import iso3

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Guam": "GU",
}
POSTAL = set(STATES.values())
US_NAMES = {"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america", "america"}
CAUSE_FAMILY = {
    "fire": "fire", "thermal_runaway": "fire", "explosion": "fire", "electrical_fault": "fire",
    "transformer": "fire", "wildfire": "fire", "lightning": "weather", "hail": "weather",
    "wind": "weather", "tornado": "weather", "hurricane": "weather", "flood": "weather",
    "winter_ice": "weather", "earthquake": "weather", "blade_failure": "mechanical",
    "tower_collapse": "mechanical", "gearbox_drivetrain": "mechanical",
    "construction": "other", "vandalism_theft": "other", "other": "other", "unknown": "unknown",
}
GENERIC = re.compile(
    r"\b(solar|wind|energy|storage|battery|batteries|bess|farm|park|project|plant|facility|"
    r"center|centre|station|system|power|llc|inc|the|of|and|i|ii|iii|iv|phase|hybrid|generating)\b",
    re.I,
)


def state_code(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s.upper() in POSTAL:
        return s.upper()
    return STATES.get(s.title()) or STATES.get(s)


def is_us(country) -> bool:
    return isinstance(country, str) and country.strip().lower() in US_NAMES


def _uid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

OWN_COLUMNS = {
    "site_name", "operator_or_owner", "asset_type", "country", "state", "county", "city", "lat", "lon",
    "event_date", "cause", "equipment_manufacturer", "equipment_model", "capacity_mw",
    "capacity_mwh", "damage_summary", "eia_plant_code", "site_id", "loss_usd",
}


def load_own() -> pd.DataFrame:
    """The in-house incident list (see README for the expected columns)."""
    path = config.OWN_INCIDENTS
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    unknown = set(d.columns) - OWN_COLUMNS - {"id", "notes", "source_ref"}
    if unknown:
        print(f"[events] own_incidents.csv: ignoring columns {sorted(unknown)}")
    d["source"] = "own"
    d["source_ref"] = d.get("id", pd.Series(range(len(d)))).astype(str)
    d["event_date"] = pd.to_datetime(d.get("event_date"), errors="coerce")
    d["date_precision"] = "day"
    d["confidence"] = "high"
    d["state"] = d.get("state", pd.Series(dtype=object)).map(state_code)
    d["country"] = d.get("country", pd.Series(["United States"] * len(d))).fillna("United States")
    if "eia_plant_code" in d and "site_id" not in d:
        d["site_id"] = "eia:" + d["eia_plant_code"].astype("Int64").astype(str)
    d["event_uid"] = ["own:" + _uid(r) for r in d["source_ref"]]
    return d


def load_news() -> pd.DataFrame:
    path = config.PROCESSED / "news_events_raw.parquet"
    if not path.exists():
        return pd.DataFrame()
    n = pd.read_parquet(path)
    if n.empty:
        return n
    out = n.rename(columns={"state_or_province": "state_raw", "city_or_locality": "city"}).copy()
    out["state"] = [state_code(s) if is_us(c) else None for s, c in zip(out["state_raw"], out["country"])]
    out["country"] = np.where(out["country"].map(is_us), "United States", out["country"])
    # Partial dates: "2023-05" -> first of month; precision tells the dedup step.
    out["event_date"] = pd.to_datetime(out["event_date"].astype(str).str.slice(0, 10), errors="coerce", format="mixed")
    out["source_ref"] = out["source_url"]
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["event_uid"] = ["news:" + _uid(u, i) for u, i in zip(out["source_url"], out["event_idx"])]
    return out


# ----------------------------------------------------------------------------
# Site matching
# ----------------------------------------------------------------------------

def _core(name) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", GENERIC.sub(" ", name.lower())).strip()


def _hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * config.EARTH_KM * np.arcsin(np.sqrt(a))


TECH_COL = {"wind": "wind_mw", "solar": "solar_mw"}


def match_sites(events: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Score candidate sites in the event's state and keep the best if clear.

    Evidence: distance (if coordinates), name / operator similarity, county,
    city, capacity agreement, and technology. The score is a transparent
    additive heuristic; >= 45 with an 8-point margin over the runner-up = matched.
    """
    sites = sites.copy()
    sites["_core"] = sites["plant_name"].map(_core)
    sites["_wcore"] = sites["wt_project"].map(_core)
    sites["_county"] = sites["county"].fillna("").str.lower().str.replace(r"\s+county$", "", regex=True)
    sites["_city"] = sites["city"].fillna("").str.lower()
    by_state = {s: g for s, g in sites[sites["country"] == "USA"].groupby("state")}
    by_country = {c: g for c, g in sites[sites["country"] != "USA"].groupby("country")}
    known = set(sites["site_id"])

    res = []
    for ev in events.itertuples():
        best = {"site_id": getattr(ev, "site_id", None), "match_score": np.nan, "match_method": None,
                "match_runner_up": np.nan}
        if isinstance(best["site_id"], str) and best["site_id"] in known:
            best.update(match_score=100.0, match_method="given")
            res.append(best)
            continue
        best["site_id"] = None
        if ev.country == "USA" and ev.state in by_state:
            cand = by_state[ev.state]
        elif ev.country != "USA" and ev.country in by_country:
            # Outside the US there are no states/counties in the site data, so
            # matches rest on distance, name, operator and capacity.
            cand = by_country[ev.country]
        else:
            res.append(best)
            continue
        tcol = TECH_COL.get(ev.asset_type)
        if tcol:
            has = cand[tcol].fillna(0) > 0
            if tcol == "wind_mw":
                has |= cand["wt_n_turbines"].fillna(0) > 0
            cand = cand[has]
        if cand.empty:
            res.append(best)
            continue
        score = np.zeros(len(cand))
        method = [[] for _ in range(len(cand))]

        def add(mask, pts, label):
            nonlocal score
            score = score + np.where(mask, pts, 0)
            for i in np.flatnonzero(mask):
                method[i].append(label)

        if pd.notna(ev.lat) and pd.notna(ev.lon):
            d = _hav_km(ev.lat, ev.lon, cand["lat"].to_numpy(), cand["lon"].to_numpy())
            add(d <= 2, 45, "dist<2km")
            add((d > 2) & (d <= 8), 30, "dist<8km")
            add((d > 8) & (d <= 25), 12, "dist<25km")
        core = _core(ev.site_name)
        if core:
            sim = np.array([
                max(fuzz.token_set_ratio(core, a) if a else 0, fuzz.token_set_ratio(core, b) if b else 0)
                for a, b in zip(cand["_core"], cand["_wcore"])
            ])
            add(sim >= 90, 50, "name")
            add((sim >= 75) & (sim < 90), 25, "name~")
        if isinstance(ev.operator_or_owner, str) and ev.operator_or_owner.strip():
            op = ev.operator_or_owner.lower()
            osim = np.array([fuzz.partial_ratio(op, u.lower()) if isinstance(u, str) else 0
                             for u in cand["utility_name"]])
            add(osim >= 85, 10, "operator")
        if isinstance(ev.county, str) and ev.county.strip():
            c = re.sub(r"\s+county$", "", ev.county.strip().lower())
            add(cand["_county"].to_numpy() == c, 15, "county")
        if isinstance(ev.city, str) and ev.city.strip():
            add(cand["_city"].to_numpy() == ev.city.strip().lower(), 20, "city")
        cap = ev.capacity_mw if pd.notna(ev.capacity_mw) else np.nan
        cap_ratio = np.full(len(cand), 9.0)
        if pd.notna(cap) and cap > 0 and tcol:
            smw = cand[tcol].fillna(cand["total_mw"]).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                cap_ratio = np.nan_to_num(np.abs(np.log(np.where(smw > 0, smw, np.nan) / cap)), nan=9.0)
            add(cap_ratio <= np.log(1.25), 20, "capacity")
        if pd.notna(ev.event_date):
            ev_t = ev.event_date.year + (ev.event_date.dayofyear - 1) / 365.25
            dcol = f"{ev.asset_type}_first_date" if f"{ev.asset_type}_first_date" in cand else "first_year"
            start = cand[dcol].fillna(cand["first_year"]).to_numpy(dtype=float)
            # Sites that started operating after the event can't have had it:
            # rule out a start > 1 yr later; penalise a smaller gap (EIA dates
            # can lag commissioning by a few months).
            s0 = np.nan_to_num(start, nan=0)
            score = np.where(s0 > ev_t + 0.1, score - 40, score)
            score = np.where(s0 > ev_t + 1.0, -999.0, score)
            # A reported system age implies a start date that should line up.
            age = getattr(ev, "system_age_yr", np.nan)
            if pd.notna(age):
                add(np.abs((ev_t - age) - np.nan_to_num(start, nan=-99)) <= 1.0, 10, "age")

        order = np.argsort(-score)
        top = score[order[0]]
        tied = order[score[order] >= top - 1e-9]
        if len(tied) > 1:
            # Several plants of one complex (e.g. phases 1/2 at the same
            # address): if they're all within 2 km, take the closest capacity.
            tl, tn = cand["lat"].to_numpy()[tied], cand["lon"].to_numpy()[tied]
            spread = _hav_km(tl[:, None], tn[:, None], tl[None, :], tn[None, :]).max()
            if spread <= 2.0:
                pick = tied[np.argmin(cap_ratio[tied])]
                order = np.r_[pick, [o for o in order if o != pick]]
                for t in tied:
                    if t != pick:
                        score[t] -= 10  # demote siblings so the margin rule passes
                method[pick].append("complex-tiebreak")
        top = score[order[0]]
        second = score[order[1]] if len(order) > 1 else 0.0
        if top >= 45 and top - second >= 8:
            best.update(
                site_id=cand["site_id"].iat[order[0]],
                match_score=float(top),
                match_method="+".join(method[order[0]]),
                match_runner_up=float(second),
            )
        else:
            best.update(match_score=float(top), match_runner_up=float(second),
                        match_method="unmatched:" + "+".join(method[order[0]]))
        res.append(best)
    m = pd.DataFrame(res, index=events.index)
    return events.drop(columns=[c for c in m.columns if c in events.columns]).join(m)


# ----------------------------------------------------------------------------
# De-duplication
# ----------------------------------------------------------------------------

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _date_close(a, pa, b, pb, days=10) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    if "year" in (pa, pb):
        return a.year == b.year
    if "month" in (pa, pb):
        return (a.year, a.month) == (b.year, b.month) or abs((a - b).days) <= 31
    return abs((a - b).days) <= days


def dedupe(m: pd.DataFrame) -> pd.DataFrame:
    """Union mentions that describe the same incident.

    Same incident = (same matched site, or same place/name when unmatched) and
    compatible cause family and dates within 10 days (looser for month/year
    precision).
    """
    m = m.reset_index(drop=True)
    if m.empty:
        return m.assign(n_mentions=0, sources="", source_refs="", mention_uids="", event_id="", cause_family="")
    dsu = _DSU(len(m))
    fam = m["cause"].map(CAUSE_FAMILY).fillna("unknown")
    place = (
        m["country"].fillna("").str.lower() + "|" + m["state"].fillna("") + "|" + m["city"].fillna("").str.lower()
    )
    core = m["site_name"].map(_core)
    rows = list(m.itertuples())
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            a, b = rows[i], rows[j]
            if not _date_close(a.event_date, a.date_precision, b.event_date, b.date_precision):
                continue
            f1, f2 = fam[i], fam[j]
            if f1 != f2 and "unknown" not in (f1, f2):
                continue
            if a.asset_type != b.asset_type and "unknown" not in (a.asset_type, b.asset_type):
                continue
            same = False
            if isinstance(a.site_id, str) and isinstance(b.site_id, str):
                same = a.site_id == b.site_id
            elif core[i] and core[j] and fuzz.token_set_ratio(core[i], core[j]) >= 85:
                same = m["country"][i] == m["country"][j]
            elif place[i] == place[j] and m["city"].notna()[i]:
                same = True
            if same:
                dsu.union(i, j)
    m["incident_id"] = [dsu.find(i) for i in range(len(m))]

    prio = {"own": 0, "news": 1}
    conf = {"high": 0, "medium": 1, "low": 2}
    m["_p"] = m["source"].map(prio).fillna(3) * 10 + m["confidence"].map(conf).fillna(3)
    agg = []
    for _, g in m.sort_values("_p").groupby("incident_id", sort=False):
        top = g.iloc[0].to_dict()
        for col in ("site_id", "site_name", "operator_or_owner", "county", "city", "lat", "lon",
                    "equipment_manufacturer", "equipment_model", "capacity_mw", "capacity_mwh"):
            if pd.isna(top.get(col)) or top.get(col) in ("", None):
                vals = g[col].dropna()
                vals = vals[vals.astype(str).str.len() > 0]
                if len(vals):
                    top[col] = vals.iloc[0]
        if pd.isna(top.get("match_score")) and isinstance(top.get("site_id"), str):
            top["match_score"] = g["match_score"].max()
        top["n_mentions"] = len(g)
        top["sources"] = ",".join(sorted(set(g["source"])))
        top["source_refs"] = " | ".join(g["source_ref"].dropna().astype(str).unique()[:8])
        top["mention_uids"] = ",".join(g["event_uid"])
        agg.append(top)
    ev = pd.DataFrame(agg).drop(columns=["_p", "incident_id"])
    ev["event_id"] = ["ev:" + _uid(u) for u in ev["mention_uids"]]
    ev["cause_family"] = ev["cause"].map(CAUSE_FAMILY).fillna("unknown")
    return ev


KEEP = [
    "event_uid", "source", "source_ref", "asset_type", "site_name", "operator_or_owner", "country",
    "state", "county", "city", "lat", "lon", "event_date", "date_precision", "cause",
    "equipment_manufacturer", "equipment_model", "capacity_mw", "capacity_mwh", "damage_summary",
    "confidence", "site_id", "system_age_yr", "loss_usd", "evidence_quote", "source_title",
]


def build() -> pd.DataFrame:
    sites = pd.read_parquet(config.PROCESSED / "sites.parquet")
    parts = [p for p in (load_own(), load_news()) if len(p)]
    if not parts:
        parts = [pd.DataFrame(columns=KEEP)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # all-NA columns from sparse sources
        mentions = pd.concat([p.reindex(columns=KEEP) for p in parts], ignore_index=True)
    # Drop news mentions the model itself marked as low confidence with no date.
    weak = (mentions.source == "news") & (mentions.confidence == "low") & mentions.event_date.isna()
    # Wind and solar only (battery storage was removed from the model).
    keep_asset = mentions["asset_type"].isin([*config.TECHS, "unknown"]) | mentions["asset_type"].isna()
    mentions = mentions[~weak & keep_asset].copy()
    mentions["country_name"] = mentions["country"]
    mentions["country"] = mentions["country"].map(iso3)
    mentions = match_sites(mentions, sites)
    for c in ("source_ref", "site_name", "operator_or_owner", "county", "city", "equipment_manufacturer",
              "equipment_model", "damage_summary", "evidence_quote", "source_title"):
        mentions[c] = mentions[c].astype("string")
    mentions.to_parquet(config.PROCESSED / "loss_mentions.parquet", index=False)
    events = dedupe(mentions)
    events.to_parquet(config.PROCESSED / "loss_events.parquet", index=False)
    us = events[events.country == "USA"]
    print(
        f"[events] {len(mentions)} mentions -> {len(events)} incidents "
        f"({len(us)} US, {us.site_id.notna().sum()} matched; "
        f"{len(events) - len(us)} elsewhere, {events[events.country != 'USA'].site_id.notna().sum()} matched); "
        f"by source: {mentions.source.value_counts().to_dict()}"
    )
    return events
