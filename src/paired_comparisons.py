
"""
Paired cost comparisons of plans on the common evaluation set.

For each city, compare the reference plan with:
1. The most frequently chosen non-reference plan.
2. The mean-demand plan.
3. The greedy rule.

Total scenario cost includes both facility opening costs and
second-stage costs.

The paired standard error is calculated from scenario-by-scenario
cost differences on the same evaluation sample.

The 95% confidence interval uses a normal approximation.

The development evaluation set is used by default. The final
evaluation set requires --final and a clean tracked Git tree.
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
from src.evaluate_runs import EVAL_SETS, _to_plan
from src.experiment import CORR_LENGTH, CV, _environment


def paired_statistics(a, b):
    """
    Calculate the mean paired difference (b - a),
    its standard error, and a normal 95% confidence interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError(
            "a and b must be one-dimensional arrays of equal "
            "length, containing at least two observations"
        )

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Scenario costs must be finite")

    d = b - a

    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))

    return {
        "mean_difference": mean,
        "paired_se": se,
        "ci95_lower": mean - 1.96 * se,
        "ci95_upper": mean + 1.96 * se,
        "n_scenarios": len(d),
    }


def most_common_alternative(plans, reference_sites):
    """
    Return the most frequently chosen non-reference plan.

    Ties are broken lexicographically by plan string.
    """
    reference = {int(s) for s in reference_sites}

    counts = Counter(
        p
        for p in plans
        if p and {int(s) for s in p.split("-")} != reference
    )

    if not counts:
        return None, 0

    best = min(counts, key=lambda p: (-counts[p], p))

    return best, counts[best]


def compare_city(city_seed, runs_csv, plans_json, out_csv, final=False):
    """
    Compare reference and alternative plans on common scenarios.

    Write the results to CSV and return the result rows.
    """
    if final and _environment()["uncommitted_changes"] is not False:
        raise RuntimeError(
            "Tracked files differ from the last commit, "
            "or Git is unavailable. Commit changes before "
            "using the final evaluation set."
        )

    evaluation_name = "final" if final else "dev"
    seed, n_eval = EVAL_SETS[evaluation_name]

    city = make_city(seed=city_seed)
    n_sites = len(city.open_cost)

    plans = json.loads(Path(plans_json).read_text())

    if plans["city_seed"] != city_seed:
        raise ValueError("Plans file belongs to a different city")

    evaluation = make_demand_model(
        city,
        cv=CV,
        corr_length=CORR_LENGTH,
    ).sample(
        n_eval,
        np.random.default_rng(seed),
    )

    ev = PlanEvaluator(city, evaluation)

    def totals(sites):
        """
        Return opening cost plus second-stage cost
        for every evaluation scenario.
        """
        y = _to_plan(sites, n_sites)

        opening_cost = float(np.dot(city.open_cost, y))
        second_stage_costs = ev.scenario_costs(y)

        return opening_cost + second_stage_costs

    with open(runs_csv, newline="") as fh:
        chosen = [
            row["plan"]
            for row in csv.DictReader(fh)
            if int(row["status"]) == 0
        ]

    alternative, count = most_common_alternative(
        chosen,
        plans["reference_plan"],
    )

    reference_sites = [int(s) for s in plans["reference_plan"]]
    reference_totals = totals(reference_sites)

    candidates = []

    if alternative is not None:
        candidates.append((
            "most frequent non-reference plan",
            [int(s) for s in alternative.split("-")],
            count,
        ))

    candidates.append((
        "mean-demand plan",
        plans["mean_demand_plan"],
        "",
    ))

    candidates.append((
        "greedy rule",
        plans["greedy_plan"],
        "",
    ))

    results = []

    for label, sites, frequency in candidates:
        alternative_totals = totals(sites)

        stats = paired_statistics(
            reference_totals,
            alternative_totals,
        )

        reference_cost = float(reference_totals.mean())
        alternative_cost = float(alternative_totals.mean())

        results.append({
            "city_seed": city_seed,
            "evaluation_set": evaluation_name,
            "comparison": label,
            "reference_plan": "-".join(map(str, reference_sites)),
            "alternative_plan": "-".join(map(str, sites)),
            "alternative_selection_count": frequency,
            "reference_cost": reference_cost,
            "alternative_cost": alternative_cost,
            "relative_difference_percent":
                100 * stats["mean_difference"] / reference_cost,
            **stats,
        })

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--city",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--runs",
        required=True,
        help="Experiment CSV",
    )

    parser.add_argument(
        "--plans",
        required=True,
        help="JSON file from prepare_city",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV",
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help="Use the final evaluation set (seed 202)",
    )

    args = parser.parse_args()

    results = compare_city(
        args.city,
        args.runs,
        args.plans,
        args.out,
        args.final,
    )

    for r in results:
        print(
            f"{r['comparison']}: "
            f"plan {r['alternative_plan']}, "
            f"difference {r['mean_difference']:.3f}, "
            f"paired SE {r['paired_se']:.3f}, "
            f"95% CI "
            f"[{r['ci95_lower']:.3f}, "
            f"{r['ci95_upper']:.3f}], "
            f"relative difference "
            f"{r['relative_difference_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
