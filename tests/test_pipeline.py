"""Unit tests on synthetic inputs (no downloads needed)."""

import numpy as np
import pandas as pd
import pytest

from rrisk.hazard import us_convective as hazard


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


@pytest.mark.parametrize("v,expected", [("Low", 0), ("Very high", 3)])
def test_tier_index(v, expected):
    from rrisk.app import payload

    assert payload.TIER_INDEX[v] == expected
