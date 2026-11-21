"""Tests for the experiment metrics."""

import numpy as np
import pandas as pd
import pytest

from src.analyze import (bootstrap_interval, hamming, metrics_by_n, site_frequencies,
                         wilson_interval)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)


def test_hamming_counts_sites_in_exactly_one_plan():
    assert hamming("0-1", [0, 1]) == 0
    assert hamming("0-2", [0, 1]) == 2
    assert hamming("0-1-2", [0, 1]) == 1


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = np.arange(50, dtype=float)
    a = bootstrap_interval(values, np.random.default_rng(1))
    assert a == bootstrap_interval(values, np.random.default_rng(1))
    assert a[0] < values.mean() < a[1]


@pytest.fixture
def runs():
    # four good runs at N = 10 (plans 0-1, 0-1, 0-2, 0-1-2) and one failed run
    return pd.DataFrame({
        "N": [10] * 5,
        "status": [0, 0, 0, 0, -1],
        "plan": ["0-1", "0-1", "0-2", "0-1-2", ""],
        "promised": [100.0, 110.0, 120.0, 130.0, np.nan],
        "realized": [105.0, 112.0, 118.0, 140.0, np.nan],
        "gap": [5.0, 2.0, -2.0, 10.0, np.nan],
        "excess": [0.0, 0.0, 4.0, 8.0, np.nan],
        "rel_excess": [0.0, 0.0, 4.0, 8.0, np.nan],
    })


def test_metrics_match_hand_calculation(runs):
    m = metrics_by_n(runs, [0, 1]).iloc[0]
    assert m["runs"] == 5 and m["runs_status0"] == 4
    assert m["completion_rate"] == pytest.approx(0.8)
    assert m["p_reference"] == pytest.approx(0.5)                 # two of four chose {0,1}
    assert m["distinct_plans"] == 3
    assert m["mean_hamming"] == pytest.approx((0 + 0 + 2 + 1) / 4)
    assert m["promised_mean"] == pytest.approx(115.0)
    assert m["gap_mean"] == pytest.approx(3.75)
    assert m["gap_se"] == pytest.approx(np.std([5, 2, -2, 10], ddof=1) / 2)
    assert m["gap_negative_share"] == pytest.approx(0.25)
    assert m["rel_excess_median"] == pytest.approx(2.0)
    assert m["rel_excess_max"] == pytest.approx(8.0)
    assert m["promised_sd_times_sqrt_n"] == pytest.approx(np.std([100, 110, 120, 130], ddof=1) * np.sqrt(10))


def test_site_frequencies(runs):
    f = site_frequencies(runs)
    assert f.loc[10, 0] == pytest.approx(1.0)                     # every good run opens site 0
    assert f.loc[10, 1] == pytest.approx(0.75)
    assert f.loc[10, 2] == pytest.approx(0.5)
    assert f.loc[10, 3] == pytest.approx(0.0)