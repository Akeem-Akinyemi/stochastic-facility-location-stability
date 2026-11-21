"""Tests for the exploratory gap-difference interval."""

import numpy as np
import pandas as pd
import pytest

from src.exploratory_gap import gap_difference


@pytest.fixture
def evaluated(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"N": 100, "status": 0, "gap": g} for g in rng.normal(10, 20, 40)]
    rows += [{"N": 200, "status": 0, "gap": g} for g in rng.normal(30, 20, 40)]
    rows += [{"N": 200, "status": -1, "gap": 1e5}]               # failed run with an extreme finite gap
    path = tmp_path / "evaluated.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, rows


def test_difference_and_interval(evaluated):
    path, rows = evaluated
    r = gap_difference(path, 100, 200)
    a = [x["gap"] for x in rows if x["N"] == 100]
    b = [x["gap"] for x in rows if x["N"] == 200 and x["status"] == 0]
    assert r["difference"] == pytest.approx(np.mean(b) - np.mean(a))
    assert r["lo"] < r["difference"] < r["hi"]
    assert (r["n1"], r["n2"]) == (40, 40)
    assert r == gap_difference(path, 100, 200)                   # fixed seed: reproducible


def test_nonfinite_gap_in_a_successful_run_is_rejected(tmp_path):
    rows = [{"N": 100, "status": 0, "gap": float(g)} for g in range(10)]
    rows += [{"N": 200, "status": 0, "gap": float(g)} for g in range(9)]
    rows += [{"N": 200, "status": 0, "gap": np.nan}]
    path = tmp_path / "evaluated.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        gap_difference(path, 100, 200)


def test_too_few_runs_are_rejected(evaluated):
    path, _ = evaluated
    with pytest.raises(ValueError):
        gap_difference(path, 100, 300)                           # no runs at N = 300