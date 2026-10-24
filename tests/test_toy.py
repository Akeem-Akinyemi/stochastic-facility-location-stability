"""Regression test: the sample from docs/model_notes.md (our toy example)."""

import itertools

import numpy as np
import pytest

from src.model import second_stage_cost, solve_saa

f = np.array([100., 120., 140.])
K = np.array([50., 60., 40.])
c = np.array([[2, 3, 6, 7], [6, 2, 3, 4], [7, 5, 2, 1]], float)
pi = 12.0
D = np.array([[10, 15, 10, 5], [20, 20, 20, 10], [30, 25, 30, 20]], float)

# Answer key: plan -> (Q(A,1), Q(A,2), Q(A,3), V(A))
KEY = {
    (0, 0, 0): (480, 840, 1260, 860.00),
    (0, 0, 1): (170, 460, 840, 630.00),
    (0, 1, 0): (140, 320, 700, 506.67),
    (0, 1, 1): (115, 210, 350, 485.00),
    (1, 0, 0): (160, 400, 780, 546.67),
    (1, 0, 1): (90, 150, 360, 440.00),
    (1, 1, 0): (100, 180, 295, 411.67),
    (1, 1, 1): (75, 130, 200, 495.00),
}


@pytest.mark.parametrize("plan", list(KEY))
def test_second_stage_costs(plan):
    y = np.array(plan)
    q = [second_stage_cost(y, K, c, pi, d) for d in D]
    assert np.allclose(q, KEY[plan][:3])
    assert f @ y + np.mean(q) == pytest.approx(KEY[plan][3], abs=0.01)


def test_saa_picks_best_plan():
    res = solve_saa(f, K, c, pi, D)
    assert tuple(res["y"]) == (1, 1, 0)
    assert res["objective"] == pytest.approx(411.67, abs=0.01)


def test_saa_matches_brute_force():
    best = min(itertools.product([0, 1], repeat=3),
               key=lambda p: f @ np.array(p) + np.mean([second_stage_cost(p, K, c, pi, d) for d in D]))
    assert tuple(solve_saa(f, K, c, pi, D)["y"]) == best