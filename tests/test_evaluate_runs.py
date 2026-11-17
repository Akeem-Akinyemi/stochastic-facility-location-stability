"""Tests for evaluating the plans chosen in an experiment CSV."""

import csv
import json

import numpy as np
import pytest

from src.city import make_city
from src.demand import make_demand_model
from src.evaluate import PlanEvaluator
from src.evaluate_runs import evaluate_runs
from src.experiment import FIELDS


@pytest.fixture
def files(tmp_path):
    runs = tmp_path / "runs.csv"
    rows = [
        {"city_seed": 1, "N": 10, "rep": 0, "plan": "0-4-8-10", "promised": 5000.0, "mip_gap": 0,
         "solve_seconds": 1, "time_limit": 900, "status": 0, "message": ""},
        {"city_seed": 1, "N": 10, "rep": 1, "plan": "0-3-8-10", "promised": 5200.0, "mip_gap": 0,
         "solve_seconds": 1, "time_limit": 900, "status": 0, "message": ""},
        {"city_seed": 1, "N": 10, "rep": 2, "plan": "", "promised": float("nan"), "mip_gap": float("nan"),
         "solve_seconds": 1, "time_limit": 900, "status": -1, "message": "no solution"},
    ]
    with open(runs, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps({"city_seed": 1, "reference_plan": [0, 4, 8, 10],
                                 "mean_demand_plan": [0, 3, 8, 10], "greedy_plan": [0, 4, 9, 10]}))
    return runs, plans, tmp_path


def test_realized_costs_gaps_and_excess(files):
    runs, plans, tmp = files
    summary = evaluate_runs(1, runs, plans, tmp / "out.csv", tmp / "summary.json")
    with open(tmp / "out.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    city = make_city(seed=1)
    ev = PlanEvaluator(city, make_demand_model(city).sample(1000, np.random.default_rng(101)))
    y_ref = np.zeros(12, dtype=int)
    y_ref[[0, 4, 8, 10]] = 1
    y_other = np.zeros(12, dtype=int)
    y_other[[0, 3, 8, 10]] = 1

    assert float(rows[0]["realized"]) == pytest.approx(ev.expected_cost(y_ref))
    assert float(rows[0]["excess"]) == pytest.approx(0.0, abs=1e-9)          # it is the reference plan
    assert float(rows[1]["realized"]) == pytest.approx(ev.expected_cost(y_other))
    assert float(rows[1]["gap"]) == pytest.approx(ev.expected_cost(y_other) - 5200.0)
    assert float(rows[1]["rel_excess"]) == pytest.approx(
        100 * (ev.expected_cost(y_other) - ev.expected_cost(y_ref)) / ev.expected_cost(y_ref))
    assert rows[2]["realized"] == ""                                          # run without a plan
    assert summary["check_max_abs_difference"] < 1e-6
    assert summary["distinct_plans_evaluated"] == 2


def test_final_evaluation_set_requires_a_clean_tree(files, monkeypatch):
    import src.evaluate_runs as module
    runs, plans, tmp = files
    for dirty in (True, None):
        monkeypatch.setattr(module, "_environment", lambda d=dirty: {"uncommitted_changes": d})
        with pytest.raises(RuntimeError):
            evaluate_runs(1, runs, plans, tmp / "out.csv", tmp / "summary.json", final=True)
        assert not (tmp / "out.csv").exists()


def test_plans_file_must_match_the_city(files):
    runs, plans, tmp = files
    with pytest.raises(ValueError):
        evaluate_runs(2, runs, plans, tmp / "out.csv", tmp / "summary.json")