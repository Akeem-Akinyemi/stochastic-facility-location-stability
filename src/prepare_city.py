"""
prepare_city.py: reference, mean-demand and greedy plans for one city.

Uses only the reference sample (seed 303). The final evaluation set is not touched.
The defaults are the protocol values; smaller values are only for quick tests.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.benchmarks import greedy_plan, mean_demand_plan
from src.city import make_city
from src.demand import make_demand_model
from src.experiment import CORR_LENGTH, CV
from src.reference import best_plan_by_enumeration

REFERENCE_SEED = 303


def sites(y):
    return [int(i) for i in np.flatnonzero(y)]


def prepare_city(city_seed, out_json, n_reference=5000, n_screen=100, top=50):
    """Compute the three plans for a city and save them, with diagnostics, as JSON."""
    city = make_city(seed=city_seed)
    model = make_demand_model(city, cv=CV, corr_length=CORR_LENGTH)
    reference_sample = model.sample(n_reference, np.random.default_rng(REFERENCE_SEED))

    t0 = time.time()
    found = best_plan_by_enumeration(city, reference_sample, n_screen=n_screen, top=top)
    result = {
        "city_seed": city_seed,
        "settings": {"reference_seed": REFERENCE_SEED, "n_reference": n_reference,
                     "n_screen": n_screen, "top": top},
        "reference_plan": sites(found["y"]),
        "reference_cost_on_reference_sample": float(found["cost"]),
        "screen_rank": int(found["screen_rank"]),
        "screen_margin": None if found["screen_margin"] is None else float(found["screen_margin"]),
        "screen_min_z": None if found["screen_min_z"] is None else float(found["screen_min_z"]),
        "runner_up_gap": None if found["runner_up_gap"] is None else float(found["runner_up_gap"]),
        "runner_up_se": None if found["runner_up_se"] is None else float(found["runner_up_se"]),
        "finalists": [[float(c), list(k)] for c, k in found["finalists"]],
        "mean_demand_plan": sites(mean_demand_plan(city)),
        "greedy_plan": sites(greedy_plan(city)),
        "seconds": round(time.time() - t0, 1),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", type=int, required=True)
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--n-reference", type=int, default=5000)
    parser.add_argument("--n-screen", type=int, default=100)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()
    res = prepare_city(args.city, args.out, args.n_reference, args.n_screen, args.top)
    print(f"Reference plan {res['reference_plan']} (screening rank {res['screen_rank']}), "
          f"mean-demand plan {res['mean_demand_plan']}, greedy plan {res['greedy_plan']}")
    print(f"Screening margin {res['screen_margin']}, smallest z {res['screen_min_z']}, "
          f"runner-up gap {res['runner_up_gap']} (paired SE {res['runner_up_se']}), {res['seconds']}s")