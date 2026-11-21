
"""
exploratory_gap.py: EXPLORATORY analysis, not part of the frozen protocol.

Prints uncertainty columns from the main metrics table and calculates a
bootstrap percentile interval for the difference in mean optimism gap
(realized minus promised cost) between two scenario counts.

The training samples at different N are independent, so the two groups
are resampled independently. The analysis conditions on the fixed final
evaluation set and does not include its Monte Carlo uncertainty.
"""

import argparse

import numpy as np
import pandas as pd

SEED = 11
N_BOOT = 10_000


def gap_difference(evaluated_csv, n1, n2, seed=SEED, n_boot=N_BOOT):
    """Bootstrap interval for mean gap at n2 minus mean gap at n1."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive.")

    d = pd.read_csv(evaluated_csv)
    d = d[d["status"] == 0]

    a = d.loc[d["N"] == n1, "gap"].to_numpy(dtype=float)
    b = d.loc[d["N"] == n2, "gap"].to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each scenario-count group needs at least two successful runs.")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Gap values must all be finite.")

    rng = np.random.default_rng([seed, n1, n2])

    boot_a = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    boot_b = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)

    diff = boot_b - boot_a
    lo, hi = np.percentile(diff, [2.5, 97.5])

    return {
        "mean_n1": a.mean(),
        "mean_n2": b.mean(),
        "difference": b.mean() - a.mean(),
        "lo": lo,
        "hi": hi,
        "bootstrap_share_nonpositive": float(np.mean(diff <= 0)),
        "n1": len(a),
        "n2": len(b),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated", default="results/main_2024_evaluated.csv")
    parser.add_argument("--metrics", default="results/main_2024_metrics.csv")
    parser.add_argument("--n1", type=int, default=100)
    parser.add_argument("--n2", type=int, default=200)
    args = parser.parse_args()

    pd.set_option("display.width", 250)
    m = pd.read_csv(args.metrics)

    cols = [
        "N", "gap_mean", "gap_se", "gap_mean_lo", "gap_mean_hi",
        "rel_excess_mean", "rel_excess_max",
        "realized_mean", "promised_mean"
    ]
    print(m[cols].round(3).to_string(index=False))

    r = gap_difference(args.evaluated, args.n1, args.n2)

    print(
        f"\nEXPLORATORY: mean gap at N={args.n2} minus "
        f"mean gap at N={args.n1}"
    )
    print(f"  Observations: {r['n1']} and {r['n2']}")
    print(
        f"  {r['mean_n2']:.2f} - {r['mean_n1']:.2f} "
        f"= {r['difference']:.2f}"
    )
    print(
        f"  Bootstrap 95% percentile interval: "
        f"{r['lo']:.2f} to {r['hi']:.2f}"
    )
    print(
        f"  Bootstrap proportion of differences <= 0: "
        f"{r['bootstrap_share_nonpositive']:.4f}"
    )
