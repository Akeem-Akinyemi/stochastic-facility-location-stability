"""pilot_checks.py: how stable is the plan on the pilot city? (Stage K sanity checks)"""

from collections import Counter

import numpy as np

from src.city import make_city
from src.demand import make_demand_model
from src.model import second_stage_cost, solve_saa


def plan_frequencies(city, model, N, reps, seed0):
    """Solve on `reps` independent samples of size N and count how often each plan is chosen."""
    counts = Counter()
    for r in range(reps):
        train = model.sample(N, np.random.default_rng(seed0 + r))
        y = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty, train)["y"]
        counts[tuple(np.flatnonzero(y).tolist())] += 1
    return counts


def neighbour_costs(city, y, scenarios):
    """Cost on common scenarios of plan y and every plan that adds, drops or swaps one site."""
    def cost(p):
        q = [second_stage_cost(p, city.capacity, city.service_cost, city.penalty, d) for d in scenarios]
        return city.open_cost @ p + np.mean(q)

    plans = {tuple(np.flatnonzero(y)): y}
    for i in range(len(y)):                      # add or drop one site
        p = y.copy(); p[i] = 1 - p[i]
        plans[tuple(np.flatnonzero(p))] = p
    for i in np.flatnonzero(y):                  # swap one open site for one closed site
        for k in np.flatnonzero(1 - y):
            p = y.copy(); p[i] = 0; p[k] = 1
            plans[tuple(np.flatnonzero(p))] = p
    return sorted((cost(p), plan) for plan, p in plans.items())


if __name__ == "__main__":
    city = make_city(seed=1)
    model = make_demand_model(city)

    print("Plans chosen across independent samples:")
    for N, reps in ((10, 20), (30, 8)):
        print(f"  N={N}: {dict(plan_frequencies(city, model, N, reps, seed0=9000 + 100 * N))}")

    base = np.zeros(12, dtype=int)
    base[[0, 4, 8, 10]] = 1
    dev = model.sample(300, np.random.default_rng(101))
    print("\nBest plans among one-site neighbours (cost on 300 development scenarios):")
    for cost, plan in neighbour_costs(city, base, dev)[:6]:
        print(f"  {cost:8.1f}  {plan}")