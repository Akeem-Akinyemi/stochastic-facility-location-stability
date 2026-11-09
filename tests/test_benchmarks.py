"""Tests for the benchmark plans."""

from types import SimpleNamespace

from src.benchmarks import greedy_plan, mean_demand_plan
from src.city import make_city
from tests.test_toy import D, K, c, f, pi

toy = SimpleNamespace(open_cost=f, capacity=K, service_cost=c, penalty=pi,
                      mean_demand=D.mean(axis=0))


def test_mean_demand_plan_on_toy_example():
    # Matches the hand-checked result: the mean-demand plan opens sites 1 and 3.
    assert tuple(mean_demand_plan(toy)) == (1, 0, 1)


def test_greedy_plan_on_toy_example():
    # Sites 1 and 2 both cost 2 per unit of capacity; the stable order picks site 1 first.
    assert tuple(greedy_plan(toy)) == (1, 1, 0)


def test_greedy_plan_covers_mean_demand():
    city = make_city(seed=1)
    y = greedy_plan(city)
    assert city.capacity[y == 1].sum() >= city.mean_demand.sum()
    