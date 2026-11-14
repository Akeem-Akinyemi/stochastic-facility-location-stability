"""Tests for the plan evaluator, using the toy answer key and the full instance."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.city import make_city
from src.demand import make_demand_model
from src.evaluate import PlanEvaluator
from tests.test_toy import D, KEY, K, c, f, pi

toy = SimpleNamespace(open_cost=f, capacity=K, service_cost=c, penalty=pi)


@pytest.mark.parametrize("use_highs", [False, True])
@pytest.mark.parametrize("plan", list(KEY))
def test_matches_toy_answer_key(plan, use_highs):
    if use_highs:
        pytest.importorskip("highspy")
    ev = PlanEvaluator(toy, D, use_highs=use_highs)
    y = np.array(plan)
    assert np.allclose(ev.scenario_costs(y), KEY[plan][:3])
    assert ev.expected_cost(y) == pytest.approx(KEY[plan][3], abs=0.01)


def test_results_are_cached():
    ev = PlanEvaluator(toy, D, use_highs=False)
    y = np.array([1, 1, 0])
    assert ev.scenario_costs(y) is ev.scenario_costs(y)


def test_backends_agree_on_full_instance():
    pytest.importorskip("highspy")
    city = make_city(seed=1)
    scen = make_demand_model(city).sample(25, np.random.default_rng(3))
    for sites in ([0, 4, 8, 10], [2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
        y = np.zeros(12, dtype=int)
        y[sites] = 1
        a = PlanEvaluator(city, scen, use_highs=False).scenario_costs(y)
        b = PlanEvaluator(city, scen, use_highs=True).scenario_costs(y)
        assert np.allclose(a, b, atol=1e-6)