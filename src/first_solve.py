"""
first_solve.py: solve the SAA model once on the pilot city and read the output.

For each scenario count N we solve on N training scenarios, then evaluate the
chosen plan on a larger development set (seed 101) the model has never seen.
"""

import time

import numpy as np

from src.city import make_city
from src.demand import make_demand_model
from src.model import second_stage_cost, solve_saa


def evaluate(y, city, scenarios):
    """Average total cost of plan y on the given scenarios (opening cost + mean second stage)."""
    q = np.array([second_stage_cost(y, city.capacity, city.service_cost, city.penalty, d)
                  for d in scenarios])
    return city.open_cost @ y + q.mean(), q


if __name__ == "__main__":
    city = make_city(seed=1)                      # pilot city: used for calibration only
    model = make_demand_model(city)
    dev = model.sample(1000, np.random.default_rng(101))    # development evaluation set

    print(f"Total mean demand {city.mean_demand.sum():.1f}, total capacity {city.capacity.sum():.0f}")
    for N in (10, 50, 100):
        train = model.sample(N, np.random.default_rng(5000 + N))
        t0 = time.time()
        res = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty, train)
        secs = time.time() - t0
        y = res["y"]
        realized, q = evaluate(y, city, dev)
        # Any open site can serve any zone, and serving is always cheaper than the
        # penalty, so unmet demand is exactly demand beyond the total open capacity.
        cap_open = city.capacity[y == 1].sum()
        unmet_share = np.mean(np.maximum(dev.sum(axis=1) - cap_open, 0)) / dev.sum(axis=1).mean()
        print(f"\nN={N}  solved in {secs:.1f}s  status: {res['message']}")
        print(f"  open sites: {np.flatnonzero(y).tolist()}  (count {y.sum()}, opening cost {city.open_cost @ y:.0f})")
        print(f"  promised cost {res['objective']:.1f}   realized cost on dev set {realized:.1f}   gap {realized - res['objective']:.1f}")
        print(f"  unmet share of demand: {unmet_share:.3f}")
        