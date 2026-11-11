"""Unit tests on synthetic inputs (no downloads needed)."""

import numpy as np
import pandas as pd
import pytest

from rrisk.hazard import us_convective as hazard
from rrisk.losses import events
from rrisk.losses.news import extract, fetch


# --- hazard -----------------------------------------------------------------

def test_event_days_counts_distinct_days_within_radius():
    # One target at (35, -100). Three reports on day 1 within 10 km, one on day 2
    # within 10 km, one on day 3 ~60 km away.
    ev = pd.DataFrame(
        {
            "lat": [35.0, 35.01, 35.02, 35.05, 35.55],
            "lon": [-100.0, -100.01, -100.02, -100.0, -100.0],
            "day": [1, 1, 1, 2, 3],
        }
    )
    k = hazard.event_days(np.array([35.0]), np.array([-100.0]), ev, 40.2)
    assert k.tolist() == [2]
    k_wide = hazard.event_days(np.array([35.0]), np.array([-100.0]), ev, 80.0)
    assert k_wide.tolist() == [3]


def test_event_days_per_target_radius():
    ev = pd.DataFrame({"lat": [35.1], "lon": [-100.0], "day": [7]})  # ~11 km north
    k = hazard.event_days(np.array([35.0, 35.0]), np.array([-100.0, -100.0]), ev, np.array([5.0, 15.0]))
    assert k.tolist() == [0, 1]


def test_poisson_ci_brackets_rate():
    lo, hi = hazard._poisson_ci(np.array([0, 10]), 10.0)
    assert lo[0] == 0 and hi[0] > 0
    assert lo[1] < 1.0 < hi[1]


def test_smooth_keeps_constant_field_constant_inside_mask():
    rate = np.full((40, 40), 2.0)
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    sm = hazard._smooth(rate, mask)
    # normalised convolution: no dilution near the mask edge
    assert np.allclose(sm[5:35, 5:35], 2.0)


# --- events -----------------------------------------------------------------

def test_state_code():
    assert events.state_code("Texas") == "TX"
    assert events.state_code("tx") == "TX"
    assert events.state_code("Ontario") is None
    assert events.state_code(None) is None


def _sites():
    return pd.DataFrame(
        {
            "site_id": ["eia:1", "eia:2", "eia:3"],
            "country": ["USA", "USA", "USA"],
            "plant_name": ["Sunny Acres Solar", "Big Wind Farm", "Bayside Solar"],
            "wt_project": [None, "Big Wind", None],
            "utility_name": ["Acme Solar LLC", "Windco", "Grid Solar Co"],
            "state": ["TX", "TX", "CA"],
            "county": ["Lubbock", "Nolan", "Monterey"],
            "city": ["Lubbock", "Sweetwater", "Moss Landing"],
            "lat": [33.58, 32.47, 36.80],
            "lon": [-101.85, -100.40, -121.78],
            "wind_mw": [np.nan, 200.0, np.nan],
            "solar_mw": [50.0, np.nan, 100.0],
            "total_mw": [50.0, 200.0, 100.0],
            "wt_n_turbines": [np.nan, 100, np.nan],
            "first_year": [2018, 2010, 2020],
            "solar_first_date": [2018.5, np.nan, 2020.9],
            "wind_first_date": [np.nan, 2010.5, np.nan],
        }
    )


def _event(**kw):
    base = dict(
        event_uid="x", source="news", source_ref="u", asset_type="solar", site_name=None,
        operator_or_owner=None, country="USA", state="TX", county=None, city=None,
        lat=np.nan, lon=np.nan, event_date=pd.Timestamp("2023-05-03"), date_precision="day",
        cause="hail", equipment_manufacturer=None, equipment_model=None, capacity_mw=np.nan,
        capacity_mwh=np.nan, damage_summary="", confidence="high", site_id=None, system_age_yr=np.nan,
    )
    base.update(kw)
    return base


def test_match_by_name_and_county():
    ev = pd.DataFrame([_event(site_name="Sunny Acres solar farm", county="Lubbock County", capacity_mw=50)])
    m = events.match_sites(ev, _sites())
    assert m.loc[0, "site_id"] == "eia:1"


def test_no_match_when_site_started_after_event():
    ev = pd.DataFrame([_event(asset_type="solar", state="CA", city="Moss Landing", lat=36.80, lon=-121.78,
                              event_date=pd.Timestamp("2019-01-01"), cause="fire", capacity_mw=100)])
    m = events.match_sites(ev, _sites())
    assert m.loc[0, "site_id"] is None


def test_no_match_outside_us():
    ev = pd.DataFrame([_event(country="AUS", state=None, site_name="Sunny Acres")])
    m = events.match_sites(ev, _sites())
    assert m.loc[0, "site_id"] is None


def test_dedupe_merges_own_and_news_for_same_site_and_week():
    rows = [
        _event(event_uid="own:1", source="own", state="CA", cause="hail",
               event_date=pd.Timestamp("2022-09-20"), site_id="eia:3", confidence="high"),
        _event(event_uid="news:1", source="news", state="CA", cause="hail",
               event_date=pd.Timestamp("2022-09-21"), site_id="eia:3", confidence="medium"),
        _event(event_uid="news:2", source="news", state="CA", cause="hail",
               event_date=pd.Timestamp("2023-09-21"), site_id="eia:3"),
    ]
    m = pd.DataFrame(rows)
    m["match_score"] = 80.0
    out = events.dedupe(m)
    assert len(out) == 2
    merged = out[out.n_mentions == 2].iloc[0]
    assert merged["source"] == "own" and merged["sources"] == "news,own"


def test_dedupe_keeps_different_causes_apart():
    rows = [
        _event(event_uid="news:1", source="news", state="TX", city="Lubbock", cause="hail",
               event_date=pd.Timestamp("2023-06-26")),
        _event(event_uid="news:2", source="news", state="TX", city="Lubbock", cause="tower_collapse",
               event_date=pd.Timestamp("2023-06-26")),
    ]
    out = events.dedupe(pd.DataFrame(rows).assign(match_score=np.nan))
    assert len(out) == 2


# --- news -------------------------------------------------------------------

def test_schema_requires_every_property():
    props = extract.EVENT_SCHEMA["properties"]
    assert set(extract.EVENT_SCHEMA["required"]) == set(props)
    assert extract.EVENT_SCHEMA["additionalProperties"] is False


def test_prefilter_title_screen():
    arts = pd.DataFrame({"title": [
        "Fire breaks out at solar farm near Austin",
        "Wind turbine collapses in Iowa field",
        "Solar market expected to grow 12% by 2030",
        "Hail storm damages Texas solar farm",
    ]})
    kept = fetch.prefilter(arts)["title"].tolist()
    assert "Solar market expected to grow 12% by 2030" not in kept
    assert len(kept) == 3


@pytest.mark.parametrize("v,expected", [("Low", 0), ("Very high", 3)])
def test_tier_index(v, expected):
    from rrisk.app import payload

    assert payload.TIER_INDEX[v] == expected
