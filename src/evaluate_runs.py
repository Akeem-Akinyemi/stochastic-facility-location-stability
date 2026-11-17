"""
evaluate_runs.py: realized costs of the plans chosen in an experiment CSV.

Every chosen plan is evaluated on one common evaluation set. The development set
(seed 101, 1,000 scenarios) is the default; the final set (seed 202, 10,000
scenarios) is used only with final=True, which also requires a clean git tree.

Added per run: number of open sites, realized cost, gap (realized minus promised),
excess over the reference plan (realized cost minus the reference plan's realized
cost on the same scenarios), and relative excess in percent. Relative excess can be
negative because the reference plan is not a proven optimum.
Before finishing, five plans are re-evaluated with the original recourse LP
(second_stage_cost) on the first 1,000 scenarios as an independent check.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.city import make_city
from src.demand import make_demand_model
from src.evaluate import PlanEvaluator
from src.experiment import CORR_LENGTH, CV, _environment
from src.model import second_stage_cost

EVAL_SETS = {"dev": (101, 1000), "final": (202, 10000)}
CHECK_SCENARIOS = 1000
CHECK_PLANS = 5
EXTRA_FIELDS = ["n_open", "realized", "gap", "excess", "rel_excess"]


def _to_plan(sites, n_sites):
    y = np.zeros(n_sites, dtype=int)
    y[[int(s) for s in sites]] = 1
    return y


def evaluate_runs(city_seed, runs_csv, plans_json, out_csv, summary_json, final=False):
    """Evaluate every chosen plan; write the extended CSV and a JSON summary. Returns the summary."""
    if final and _environment()["uncommitted_changes"] is not False:
        raise RuntimeError("tracked files differ from the last commit (or git is unavailable); "
                           "commit your changes before using the final evaluation set")
    seed, n_eval = EVAL_SETS["final" if final else "dev"]

    city = make_city(seed=city_seed)
    n_sites = len(city.open_cost)
    plans = json.loads(Path(plans_json).read_text())
    if plans["city_seed"] != city_seed:
        raise ValueError("plans file belongs to a different city")
    evaluation = make_demand_model(city, cv=CV, corr_length=CORR_LENGTH).sample(
        n_eval, np.random.default_rng(seed))
    ev = PlanEvaluator(city, evaluation)

    reference = _to_plan(plans["reference_plan"], n_sites)
    reference_cost = ev.expected_cost(reference)

    with open(runs_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)

    out_rows = []
    for r in rows:
        if r["plan"]:
            y = _to_plan(r["plan"].split("-"), n_sites)
            realized = ev.expected_cost(y)
            excess = realized - reference_cost
            r.update(n_open=int(y.sum()), realized=realized, gap=realized - float(r["promised"]),
                     excess=excess, rel_excess=100 * excess / reference_cost)
        else:                                     # the solver returned no plan
            r.update(n_open="", realized="", gap="", excess="", rel_excess="")
        out_rows.append(r)

    # Independent check of the evaluator against the original recourse LP.
    popular = [p for p, _ in Counter(r["plan"] for r in rows if r["plan"]).most_common()]
    to_check = [plans["reference_plan"]] + [[int(s) for s in p.split("-")] for p in popular]
    unique_checks = []
    for p in to_check:
        if p not in unique_checks:
            unique_checks.append(p)
    max_diff = 0.0
    for p in unique_checks[:CHECK_PLANS]:
        y = _to_plan(p, n_sites)
        direct = np.array([second_stage_cost(y, city.capacity, city.service_cost, city.penalty, d)
                           for d in evaluation[:CHECK_SCENARIOS]])
        max_diff = max(max_diff, float(np.max(np.abs(direct - ev.scenario_costs(y)[:CHECK_SCENARIOS]))))
    if max_diff > 1e-6:
        raise RuntimeError(f"evaluator disagrees with the recourse LP by {max_diff:.2e}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {"city_seed": city_seed, "evaluation_set": "final" if final else "dev",
               "evaluation_seed": seed, "n_evaluation_scenarios": n_eval,
               "reference_plan": plans["reference_plan"], "reference_realized_cost": float(reference_cost),
               "benchmarks": {name: float(ev.expected_cost(_to_plan(plans[key], n_sites)))
                              for name, key in [("mean_demand_plan", "mean_demand_plan"),
                                                ("greedy_plan", "greedy_plan")]},
               "distinct_plans_evaluated": len({r["plan"] for r in rows if r["plan"]}),
               "runs": len(rows), "check_max_abs_difference": max_diff,
               "plans_checked_against_recourse_lp": len(unique_checks[:CHECK_PLANS]),
               "environment": _environment()}
    Path(summary_json).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", type=int, required=True)
    parser.add_argument("--runs", required=True, help="experiment CSV")
    parser.add_argument("--plans", required=True, help="JSON from prepare_city")
    parser.add_argument("--out", required=True, help="extended CSV to write")
    parser.add_argument("--summary", required=True, help="summary JSON to write")
    parser.add_argument("--final", action="store_true",
                        help="use the final evaluation set (seed 202); reported results only")
    args = parser.parse_args()
    s = evaluate_runs(args.city, args.runs, args.plans, args.out, args.summary, args.final)
    print(f"Evaluated {s['runs']} runs ({s['distinct_plans_evaluated']} distinct plans) on the "
          f"{s['evaluation_set']} set ({s['n_evaluation_scenarios']} scenarios).")
    print(f"Reference plan {s['reference_plan']}: realized cost {s['reference_realized_cost']:.1f}")
    for name, cost in s["benchmarks"].items():
        print(f"  {name}: {cost:.1f} ({100 * (cost / s['reference_realized_cost'] - 1):+.2f}% vs reference)")
    print(f"Checked {s['plans_checked_against_recourse_lp']} plans against the recourse LP; "
          f"largest difference {s['check_max_abs_difference']:.1e}")