"""Checks on the full-size instance (12 sites, 40 zones) with a small sample."""

import numpy as np
import pytest

from src.city import make_city
from src.demand import make_demand_model
from src.model import build_saa, solve_saa
from src.verify import verify_solution


@pytest.fixture(scope="module")
def solved():
    city = make_city(seed=1)
    D = make_demand_model(city).sample(8, np.random.default_rng(7))
    args = (city.open_cost, city.capacity, city.service_cost, city.penalty, D)
    return args, solve_saa(*args)


def test_solver_reports_optimal(solved):
    _, res = solved
    assert res["status"] == 0


def test_solution_is_feasible_and_consistent(solved):
    args, res = solved
    r = verify_solution(*args, res)
    assert r["y_integrality"] < 1e-6
    assert r["min_variable"] > -1e-6
    assert r["balance_residual"] < 1e-6
    assert r["capacity_violation"] < 1e-6
    assert abs(r["objective_from_variables"]) < 1e-5
    assert abs(r["objective_independent"]) < 1e-5


@pytest.mark.parametrize("N", [10, 50, 200])
def test_model_size_matches_formula(N):
    city = make_city(seed=1)
    D = make_demand_model(city).sample(N, np.random.default_rng(1))
    cost, A, *_ = build_saa(city.open_cost, city.capacity, city.service_cost, city.penalty, D)
    I, J = 12, 40
    assert A.shape == (N * (J + I), I + N * I * J + N * J)
    assert cost.size == I + N * I * J + N * J