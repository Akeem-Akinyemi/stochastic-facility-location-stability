"""
benchmarks.py: plans we compare the scenario-based plans against.

mean_demand_plan : the optimal plan if demand is assumed equal to its mean in every zone.
greedy_plan      : a rule of thumb that ignores both distances and uncertainty.
"""

import numpy as np

from src.model import solve_saa


def mean_demand_plan(city):
    """Optimal plan for the single demand vector equal to the mean demand."""
    res = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty,
                    city.mean_demand[None, :])
    return res["y"]


def greedy_plan(city):
    """
    Open sites in increasing order of opening cost per unit of capacity until the
    open capacity covers the mean total demand. Ties are broken by site index.
    """
    order = np.argsort(city.open_cost / city.capacity, kind="stable")
    y = np.zeros(len(order), dtype=int)
    capacity = 0.0
    for i in order:
        if capacity >= city.mean_demand.sum():
            break
        y[i] = 1
        capacity += city.capacity[i]
    return y


if __name__ == "__main__":
    from src.city import make_city
    from src.demand import make_demand_model
    from src.evaluate import PlanEvaluator

    city = make_city(seed=1)
    dev = make_demand_model(city).sample(1000, np.random.default_rng(101))
    ev = PlanEvaluator(city, dev)

    saa = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty,
                    make_demand_model(city).sample(100, np.random.default_rng(5100)))["y"]
    plans = {"mean-demand plan": mean_demand_plan(city), "greedy rule": greedy_plan(city),
             "scenario plan (N=100)": saa}
    base = ev.expected_cost(saa)
    for name, y in plans.items():
        cost = ev.expected_cost(y)
        print(f"{name:22s} sites {np.flatnonzero(y).tolist()!s:18s} cost {cost:8.1f}   "
              f"vs scenario plan {cost - base:+7.1f} ({100 * (cost - base) / base:+.2f}%)")