"""
brute_force_check.py: check the solver against full enumeration on the real instance.

With 12 sites there are 2**12 = 4096 possible plans. For a small sample we
evaluate every plan with the separate second-stage linear programs and compare
the best one with the plan chosen by the mixed-integer model.
"""

import itertools
import time

import numpy as np

from src.city import make_city
from src.demand import make_demand_model
from src.model import second_stage_cost, solve_saa


def enumerate_plans(city, D):
    """Return (cost, plan) for all plans, sorted from cheapest to most expensive."""
    results = []
    for bits in itertools.product([0, 1], repeat=len(city.open_cost)):
        y = np.array(bits)
        q = [second_stage_cost(y, city.capacity, city.service_cost, city.penalty, d) for d in D]
        results.append((city.open_cost @ y + np.mean(q), tuple(np.flatnonzero(y).tolist())))
    return sorted(results)


if __name__ == "__main__":
    city = make_city(seed=1)
    D = make_demand_model(city).sample(5, np.random.default_rng(7))

    t0 = time.time()
    ranked = enumerate_plans(city, D)
    print(f"Enumerated {len(ranked)} plans in {time.time() - t0:.0f}s")
    print("Three cheapest plans:")
    for cost, plan in ranked[:3]:
        print(f"  {cost:.4f}  {plan}")

    res = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty, D)
    print(f"Model:  {res['objective']:.4f}  {tuple(np.flatnonzero(res['y']).tolist())}")
    assert abs(res["objective"] - ranked[0][0]) < 1e-6, "model and enumeration disagree"
    print("Model and enumeration agree.")