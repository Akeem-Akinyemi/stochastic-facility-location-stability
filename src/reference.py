"""
reference.py: a large-sample reference plan, found by screening and refinement.

With 12 sites there are only 4096 plans. All of them are screenedon the first
n_screen scenarios of a reference sample, then evaluate the most promising ones
on the whole reference sample (which contains the screening scenarios) and keep
the cheapest. This gives the best plan we found; it is not proven optimal for the
true demand distribution. A plan excluded by screening is never re-evaluated, and
the reference sample carries sampling error. The diagnostics returned below are
descriptive: they are computed on the same data that selected the plans, so they
are not formal confidence statements. Enumerating all plans only works for small
instances (2**12 = 4096 plans here, but over a million for 20 sites).
"""

import itertools

import numpy as np

from src.evaluate import PlanEvaluator


def best_plan_by_enumeration(city, scenarios, n_screen=100, top=50):
    """
    Screen every plan on the first n_screen scenarios, then evaluate the `top`
    cheapest on all scenarios and keep the cheapest of those.

    Returns a dict with
      y             : the selected plan
      cost          : its cost on the full scenario set
      screen_rank   : the rank it had in the screening pass
      screen_margin : screened cost of the best excluded plan minus the screened cost
                      of the selected plan (None if every plan was a finalist)
      screen_min_z  : over all excluded plans, the smallest ratio of their mean screened
                      cost difference to the selected plan to its paired standard error
                      (None if every plan was a finalist or n_screen < 2). Large values
                      mean every excluded plan was clearly behind in the screening pass.
      runner_up_gap, runner_up_se : mean and paired standard error of the cost
                      difference between the runner-up and the selected plan
                      on the full set (None if there is only one finalist)
      finalists     : the five best finalists as (cost, sites)
    """
    scenarios = np.asarray(scenarios, dtype=float)
    n_sites = len(city.open_cost)
    if scenarios.ndim != 2 or scenarios.shape[1] != city.service_cost.shape[1]:
        raise ValueError("scenarios must have one column per demand zone")
    if not np.all(np.isfinite(scenarios)) or np.any(scenarios < 0):
        raise ValueError("scenarios must be finite and non-negative")
    if not 1 <= n_screen <= len(scenarios):
        raise ValueError("n_screen must be between 1 and the number of scenarios")
    if not 1 <= top <= 2 ** n_sites:
        raise ValueError("top must be between 1 and the number of possible plans")

    def to_plan(key):
        y = np.zeros(n_sites, dtype=int)
        y[list(key)] = 1
        return y

    plans = [np.array(bits) for bits in itertools.product([0, 1], repeat=n_sites)]
    screen = PlanEvaluator(city, scenarios[:n_screen])
    screened = sorted((screen.expected_cost(y), tuple(np.flatnonzero(y).tolist())) for y in plans)

    full = PlanEvaluator(city, scenarios)
    refined = sorted((full.expected_cost(to_plan(key)), key) for _, key in screened[:top])
    best_cost, best_key = refined[0]

    def screen_total(key):
        return city.open_cost[list(key)].sum() + screen.scenario_costs(to_plan(key))

    screened_cost = {key: cost for cost, key in screened}
    margin = min_z = None
    if top < len(screened):
        margin = screened[top][0] - screened_cost[best_key]
        if n_screen >= 2:
            diffs = np.array([screen_total(key) - screen_total(best_key)
                              for _, key in screened[top:]])
            se = diffs.std(axis=1, ddof=1) / np.sqrt(n_screen)
            z = np.divide(diffs.mean(axis=1), se, out=np.full(len(diffs), np.inf), where=se > 0)
            min_z = float(z.min())

    gap = se_gap = None
    if len(refined) > 1:
        def full_total(key):
            return city.open_cost[list(key)].sum() + full.scenario_costs(to_plan(key))
        diff = full_total(refined[1][1]) - full_total(best_key)
        gap, se_gap = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))

    return {"y": to_plan(best_key), "cost": best_cost,
            "screen_rank": [key for _, key in screened].index(best_key) + 1,
            "screen_margin": margin, "screen_min_z": min_z,
            "runner_up_gap": gap, "runner_up_se": se_gap,
            "finalists": refined[:5]}


if __name__ == "__main__":
    import time

    from src.city import make_city
    from src.demand import make_demand_model

    city = make_city(seed=1)
    reference = make_demand_model(city).sample(5000, np.random.default_rng(303))
    t0 = time.time()
    out = best_plan_by_enumeration(city, reference)
    print(f"Done in {time.time() - t0:.0f}s")
    print(f"Reference plan: {np.flatnonzero(out['y']).tolist()}  cost {out['cost']:.1f}  "
          f"(screening rank {out['screen_rank']} of 4096)")
    print(f"Screening margin of the best excluded plan: {out['screen_margin']:.1f}")
    print(f"Smallest z-score among all excluded plans: {out['screen_min_z']:.2f}")
    print(f"Runner-up is worse by {out['runner_up_gap']:.1f} (paired SE {out['runner_up_se']:.1f})")
    print("Best five finalists on the full reference sample:")
    for cost, key in out["finalists"]:
        print(f"  {cost:8.1f}  {list(key)}")