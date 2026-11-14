"""Tests for the screening-and-refinement reference plan."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.reference import best_plan_by_enumeration
from tests.test_toy import D, KEY, K, c, f, pi

toy = SimpleNamespace(open_cost=f, capacity=K, service_cost=c, penalty=pi)


def test_enumeration_on_toy_example():
    # With top=8 every plan is a finalist, so the result must equal the answer key.
    out = best_plan_by_enumeration(toy, D, n_screen=2, top=8)
    assert tuple(out["y"]) == (1, 1, 0)
    assert out["cost"] == pytest.approx(411.67, abs=0.01)
    assert out["screen_margin"] is None and out["screen_min_z"] is None
    assert out["runner_up_gap"] == pytest.approx(440.0 - 411.67, abs=0.01)  # plan {1,3}


def test_screening_margin_matches_answer_key():
    # Screening on all three scenarios reproduces the answer key. Keeping the top 4,
    # the best excluded plan is {2} with V = 506.67, so the margin is 506.67 - 411.67.
    out = best_plan_by_enumeration(toy, D, n_screen=3, top=4)
    assert out["screen_margin"] == pytest.approx(506.67 - 411.67, abs=0.01)


def test_screening_min_z_matches_answer_key():
    out = best_plan_by_enumeration(toy, D, n_screen=3, top=4)
    totals = {p: f @ np.array(p) + np.array(KEY[p][:3]) for p in KEY}
    winner = totals[(1, 1, 0)]
    excluded = sorted(KEY, key=lambda p: totals[p].mean())[4:]
    z = []
    for p in excluded:
        d = totals[p] - winner
        z.append(d.mean() / (d.std(ddof=1) / np.sqrt(3)))
    assert out["screen_min_z"] == pytest.approx(min(z))


@pytest.mark.parametrize("kwargs", [{"n_screen": 0}, {"n_screen": 4}, {"top": 0}, {"top": 9}])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        best_plan_by_enumeration(toy, D, **{"n_screen": 2, "top": 4, **kwargs})


def test_invalid_scenarios_are_rejected():
    with pytest.raises(ValueError):
        best_plan_by_enumeration(toy, D[:, :3])            # wrong number of zones
    with pytest.raises(ValueError):
        best_plan_by_enumeration(toy, -D)                  # negative demand
    with pytest.raises(ValueError):
        best_plan_by_enumeration(toy, np.where(D > 15, np.nan, D))
        