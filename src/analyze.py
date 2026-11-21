"""
analyze.py: metrics of the baseline experiment, as defined in docs/protocol.md.

Reads the evaluated CSV (from evaluate_runs) and the plans JSON, uses only runs with
solver status 0 for the statistics, and writes a metrics table, a table of site
selection frequencies, and figures. Bootstrap intervals resample the training
repetitions within each N; they condition on the fixed evaluation set and do not
include the Monte Carlo error of its scenarios. The bootstrap seed is fixed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 7
N_BOOT = 10_000
N_SITES = 12


def wilson_interval(k, n, z=1.96):
    """Wilson 95% interval for a proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_interval(values, rng, stat="mean", n_boot=N_BOOT):
    """Percentile 95% interval of the mean or median, resampling the repetitions."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = values[idx]
    stats = draws.mean(axis=1) if stat == "mean" else np.median(draws, axis=1)
    return tuple(np.percentile(stats, [2.5, 97.5]))


def hamming(plan, reference_sites):
    """Number of sites opened in exactly one of the two plans."""
    sites = {int(s) for s in plan.split("-")} if plan else set()
    return len(sites ^ set(reference_sites))


def metrics_by_n(df, reference_sites, seed=BOOTSTRAP_SEED):
    """One row of metrics per scenario count N, from runs with solver status 0."""
    rows = []
    for N, group in df.groupby("N"):
        ok = group[group["status"] == 0]
        n = len(ok)
        rng = np.random.default_rng([seed, int(N)])
        k_ref = int(sum(set(int(s) for s in p.split("-")) == set(reference_sites) for p in ok["plan"]))
        p_lo, p_hi = wilson_interval(k_ref, n)
        row = {"N": int(N), "runs": len(group), "runs_status0": n,
               "completion_rate": n / len(group),
               "p_reference": k_ref / n, "p_reference_lo": p_lo, "p_reference_hi": p_hi,
               "distinct_plans": ok["plan"].nunique(),
               "mean_hamming": float(np.mean([hamming(p, reference_sites) for p in ok["plan"]]))}
        for name in ("promised", "realized"):
            lo, hi = bootstrap_interval(ok[name], rng)
            row.update({f"{name}_mean": ok[name].mean(), f"{name}_sd": ok[name].std(ddof=1),
                        f"{name}_mean_lo": lo, f"{name}_mean_hi": hi})
        lo, hi = bootstrap_interval(ok["gap"], rng)
        row.update({"gap_mean": ok["gap"].mean(), "gap_se": ok["gap"].std(ddof=1) / np.sqrt(n),
                    "gap_mean_lo": lo, "gap_mean_hi": hi,
                    "gap_negative_share": float((ok["gap"] < 0).mean())})
        for name in ("excess", "rel_excess"):
            col = ok[name]
            row.update({f"{name}_mean": col.mean(), f"{name}_median": col.median(),
                        f"{name}_p95": col.quantile(0.95), f"{name}_max": col.max()})
        lo, hi = bootstrap_interval(ok["rel_excess"], rng, stat="median")
        row.update({"rel_excess_median_lo": lo, "rel_excess_median_hi": hi,
                    "promised_sd_times_sqrt_n": row["promised_sd"] * np.sqrt(N)})
        rows.append(row)
    return pd.DataFrame(rows)


def site_frequencies(df, n_sites=N_SITES):
    """Share of runs (status 0) that opened each site, for every N."""
    ok = df[df["status"] == 0]
    table = {}
    for N, group in ok.groupby("N"):
        counts = np.zeros(n_sites)
        for p in group["plan"]:
            for s in p.split("-"):
                counts[int(s)] += 1
        table[int(N)] = counts / len(group)
    return pd.DataFrame(table, index=range(n_sites)).T.rename_axis("N")


def style_n_axis(ax, N):
    """Logarithmic x axis labelled with the scenario counts actually used."""
    ax.set_xscale("log")
    ax.set_xticks(list(N))
    ax.set_xticklabels([str(n) for n in N])
    ax.minorticks_off()


def make_figures(metrics, freqs, reference_sites, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = metrics["N"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{prefix}_fig_{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(N, metrics["p_reference"], yerr=[metrics["p_reference"] - metrics["p_reference_lo"],
                metrics["p_reference_hi"] - metrics["p_reference"]], marker="o", capsize=3)
    style_n_axis(ax, N); ax.set_ylim(0, 1)
    ax.set_xlabel("Scenarios N"); ax.set_ylabel("Share of runs choosing the reference plan")
    ax.set_title("Decision stability (Wilson 95% intervals)")
    save(fig, "reference_probability")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(N, metrics["gap_mean"], yerr=[metrics["gap_mean"] - metrics["gap_mean_lo"],
                metrics["gap_mean_hi"] - metrics["gap_mean"]], marker="o", capsize=3)
    ax.axhline(0, color="grey", lw=1); style_n_axis(ax, N)
    ax.set_xlabel("Scenarios N"); ax.set_ylabel("Mean gap (realized - promised)")
    ax.set_title("Promised versus realized cost (bootstrap 95% intervals)")
    save(fig, "gap")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(N, metrics["rel_excess_median"], yerr=[metrics["rel_excess_median"] - metrics["rel_excess_median_lo"],
                metrics["rel_excess_median_hi"] - metrics["rel_excess_median"]], marker="o", capsize=3, label="median")
    ax.plot(N, metrics["rel_excess_p95"], marker="s", ls="--", label="95th percentile")
    ax.axhline(2, color="red", lw=1, ls=":", label="2% (hypothesis 3)")
    style_n_axis(ax, N)
    ax.set_xlabel("Scenarios N"); ax.set_ylabel("Relative excess over reference plan (%)")
    ax.set_title("Cost stability"); ax.legend()
    save(fig, "relative_excess")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(freqs.values, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(freqs.shape[1])); ax.set_xticklabels(freqs.columns)
    ax.set_yticks(range(freqs.shape[0])); ax.set_yticklabels(freqs.index)
    ax.set_xlabel("Site"); ax.set_ylabel("Scenarios N")
    ax.set_title(f"Share of runs opening each site (reference: {reference_sites})")
    fig.colorbar(im, ax=ax)
    save(fig, "site_frequencies")


def analyze(evaluated_csv, plans_json, prefix):
    df = pd.read_csv(evaluated_csv)
    df["plan"] = df["plan"].fillna("").astype(str)
    reference = json.loads(Path(plans_json).read_text())["reference_plan"]
    metrics = metrics_by_n(df, reference)
    freqs = site_frequencies(df)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)
    freqs.to_csv(f"{prefix}_site_frequencies.csv")
    make_figures(metrics, freqs, reference, prefix)
    return metrics, freqs, reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--prefix", required=True, help="output path prefix, e.g. results/main_2024")
    args = parser.parse_args()
    metrics, freqs, reference = analyze(args.evaluated, args.plans, args.prefix)
    pd.set_option("display.width", 200)
    show = ["N", "runs_status0", "completion_rate", "p_reference", "distinct_plans", "mean_hamming",
            "gap_mean", "gap_negative_share", "rel_excess_median", "rel_excess_p95", "promised_sd_times_sqrt_n"]
    print(f"Reference plan: {reference}")
    print(metrics[show].round(3).to_string(index=False))
    print("Hypothesis 3 (median relative excess below 2% at every N):",
          bool((metrics["rel_excess_median"] < 2).all()))