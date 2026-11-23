
"""Tests for paired plan comparisons."""

import csv
import json

import numpy as np
import pytest

from src.city import make_city
from src.demand import make_demand_model
from src.evaluate import PlanEvaluator
from src.paired_comparisons import (
    compare_city,
    most_common_alternative,
    paired_statistics,
)


def test_constant_difference():
    a = np.array([10.0, 20.0, 30.0, 40.0])

    result = paired_statistics(a, a + 5.0)

    assert result["mean_difference"] == pytest.approx(5.0)
    assert result["paired_se"] == pytest.approx(0.0)
    assert result["ci95_lower"] == pytest.approx(5.0)
    assert result["ci95_upper"] == pytest.approx(5.0)


def test_paired_standard_error():
    a = np.array([10.0, 20.0, 30.0, 40.0])
    b = np.array([11.0, 23.0, 32.0, 44.0])

    result = paired_statistics(a, b)

    expected_se = np.std(b - a, ddof=1) / np.sqrt(len(a))

    assert result["mean_difference"] == pytest.approx(2.5)
    assert result["paired_se"] == pytest.approx(expected_se)
    assert result["ci95_lower"] < 2.5 < result["ci95_upper"]


def test_identical_plans_have_zero_difference():
    a = np.array([10.0, 15.0, 25.0])

    result = paired_statistics(a, a)

    assert result["mean_difference"] == 0.0
    assert result["paired_se"] == 0.0
    assert result["ci95_lower"] == 0.0
    assert result["ci95_upper"] == 0.0


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        paired_statistics([1.0], [2.0])

    with pytest.raises(ValueError):
        paired_statistics(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_nonfinite_costs_are_rejected():
    with pytest.raises(ValueError):
        paired_statistics(
            [1.0, np.nan],
            [2.0, 3.0],
        )

    with pytest.raises(ValueError):
        paired_statistics(
            [1.0, 2.0],
            [3.0, np.inf],
        )


def test_most_common_alternative_ignores_reference_and_breaks_ties():
    plans = [
        "0-1",
        "0-1",
        "0-1",
        "0-2",
        "0-2",
        "0-3",
        "0-3",
        "",
    ]

    assert most_common_alternative(
        plans,
        [0, 1],
    ) == ("0-2", 2)

    assert most_common_alternative(
        ["0-1", "1-0"],
        [0, 1],
    ) == (None, 0)


@pytest.fixture
def files(tmp_path):
    runs = tmp_path / "runs.csv"

    with open(runs, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["N", "rep", "plan", "status"],
        )

        writer.writeheader()

        selected_plans = [
            "0-4-8-10",
            "0-3-8-10",
            "0-3-8-10",
            "0-4-9-10",
        ]

        for rep, plan in enumerate(selected_plans):
            writer.writerow({
                "N": 10,
                "rep": rep,
                "plan": plan,
                "status": 0,
            })

        writer.writerow({
            "N": 10,
            "rep": 4,
            "plan": "",
            "status": -1,
        })

    plans = tmp_path / "plans.json"

    plans.write_text(json.dumps({
        "city_seed": 1,
        "reference_plan": [0, 4, 8, 10],
        "mean_demand_plan": [0, 3, 8, 10],
        "greedy_plan": [0, 4, 9, 10],
    }))

    return runs, plans, tmp_path


def test_comparison_includes_opening_cost_difference(files):
    runs, plans, tmp = files

    rows = compare_city(
        1,
        runs,
        plans,
        tmp / "out.csv",
    )

    city = make_city(seed=1)

    from src.evaluate_runs import EVAL_SETS
    from src.experiment import CV, CORR_LENGTH

    seed, n_eval = EVAL_SETS["dev"]

    evaluation = make_demand_model(
        city,
        cv=CV,
        corr_length=CORR_LENGTH,
    ).sample(
        n_eval,
        np.random.default_rng(seed),
    )

    ev = PlanEvaluator(city, evaluation)

    def plan(sites):
        y = np.zeros(len(city.open_cost), dtype=int)
        y[sites] = 1
        return y

    first = rows[0]

    assert first["alternative_plan"] == "0-3-8-10"
    assert first["alternative_selection_count"] == 2

    reference = plan([0, 4, 8, 10])
    alternative = plan([0, 3, 8, 10])

    expected = (
        ev.expected_cost(alternative)
        - ev.expected_cost(reference)
    )

    assert first["mean_difference"] == pytest.approx(expected)

    second_stage_difference = (
        ev.scenario_costs(alternative).mean()
        - ev.scenario_costs(reference).mean()
    )

    opening_difference = (
        float(np.dot(city.open_cost, alternative))
        - float(np.dot(city.open_cost, reference))
    )

    assert opening_difference != 0

    assert first["mean_difference"] == pytest.approx(
        second_stage_difference + opening_difference
    )

    assert [r["comparison"] for r in rows] == [
        "most frequent non-reference plan",
        "mean-demand plan",
        "greedy rule",
    ]

    assert (tmp / "out.csv").exists()


def test_final_evaluation_requires_clean_tree(files, monkeypatch):
    import src.paired_comparisons as module

    runs, plans, tmp = files

    for dirty in (True, None):
        monkeypatch.setattr(
            module,
            "_environment",
            lambda d=dirty: {"uncommitted_changes": d},
        )

        with pytest.raises(RuntimeError):
            compare_city(
                1,
                runs,
                plans,
                tmp / "out.csv",
                final=True,
            )

        assert not (tmp / "out.csv").exists()


def test_plans_file_must_match_city(files):
    runs, plans, tmp = files

    with pytest.raises(ValueError):
        compare_city(
            2,
            runs,
            plans,
            tmp / "out.csv",
        )
