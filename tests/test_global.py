"""Global pieces: countries, cyclone hazard, matching outside the US, pipeline staleness."""

import numpy as np
import pandas as pd
import pytest

from rrisk.hazard import tropical_cyclone as tc
from rrisk.sources.countries import iso3


def test_iso3_aliases():
    assert iso3("United States") == "USA"
    assert iso3("UK") == "GBR"
    assert iso3("South Korea") == "KOR"
    assert iso3("Narnia") is None


def test_densify_spacing_and_dateline():
    t = pd.DataFrame({
        "SID": ["A", "A"], "year": [2020, 2020], "time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "lat": [10.0, 10.0], "lon": [179.5, -179.5], "wind": [70.0, 90.0], "r34": [150.0, 150.0], "r64": [40.0, 40.0],
    })
    d = tc.densify(t, step_km=15)
    assert len(d) > 5                       # ~110 km across the dateline, not ~39,000 km the long way
    assert d["lon"].abs().min() > 179.0
    assert d["wind"].between(70, 90).all()


def test_storm_counts_distinct_storms_and_radius():
    ev = pd.DataFrame({"storm": [0, 0, 1], "lat": [20.0, 20.05, 20.3], "lon": [120.0, 120.0, 120.0]})
    r = np.array([10.0, 10.0, 50.0])  # storm 1 is ~33 km away but has a 50 km radius
    n = tc.storm_counts(np.array([20.0]), np.array([120.0]), ev, r)
    assert n.tolist() == [2]
    n_small = tc.storm_counts(np.array([20.0]), np.array([120.0]), ev, np.array([10.0, 10.0, 20.0]))
    assert n_small.tolist() == [1]


def test_pipeline_staleness_tracks_input_changes(tmp_path, monkeypatch):
    from rrisk import config, pipeline

    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    src, out = tmp_path / "in.txt", tmp_path / "out.txt"
    src.write_text("a")
    step = pipeline.Step("demo", "Demo", lambda: out.write_text(src.read_text()), inputs=[src], outputs=[out])
    assert step.stale()                      # output missing
    assert pipeline.run_step(step)
    assert not step.stale()                  # fingerprint recorded
    src.write_text("bb")
    assert step.stale()                      # input changed


def test_reanalysis_cell_stats():
    from rrisk.hazard import reanalysis

    days = pd.date_range("1981-01-01", "2020-12-31")
    df = pd.DataFrame({"YEAR": days.year, "WS10M_MAX": 5.0, "PRECTOTCORR": 1.0, "T2M_MAX": 20.0})
    df.loc[df.index[::365], "WS10M_MAX"] = 20.0      # ~1 damaging-wind day a year
    df.loc[df.index[::730], "PRECTOTCORR"] = 80.0    # ~1 heavy-rain day every 2 years
    s = reanalysis.cell_stats(df)
    assert s["xwind_rate"] == pytest.approx(1.0, abs=0.05)
    assert s["xrain_rate"] == pytest.approx(0.5, abs=0.05)
    assert s["heat_rate"] == 0
    assert s["xwind_p"] == pytest.approx(1 - np.exp(-s["xwind_rate"]))


def test_hazard_index_is_relative_to_average():
    from rrisk.models import score

    d = pd.DataFrame({"a_rate": [1.0, 2.0, 3.0], "b_rate": [0.0, 0.0, 6.0]})
    w = {"solar": {"a": 0.5, "b": 0.5}}
    means = score._means(d, w, pd.Series(True, index=d.index))
    idx, contrib = score._index(d, pd.Series(["solar"] * 3), w, means, pd.Series(True, index=d.index))
    assert idx.round(3).tolist() == [0.25, 0.5, 2.25]          # 0.5*a/2 + 0.5*b/2
    assert (contrib["a"] + contrib["b"]).round(3).tolist() == idx.round(3).tolist()
