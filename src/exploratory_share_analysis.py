"""
exploratory_share_analysis.py: EXPLORATORY analysis, not part of the frozen protocol.

For each city we study the share of runs that choose the reference plan at each
scenario count. Runs are independent (separate training samples), so for a fixed
city and N the number of runs choosing the reference plan is binomial.

1. Newcombe (method 10) intervals for the change in the share between adjacent
   scenario counts. The 95% level is Bonferroni-adjusted within each city for the
   number of adjacent contrasts of that city (the family is defined per city).
   The intervals are approximate, not exact finite-sample simultaneous intervals.
2. A one-sided Cochran-Armitage trend test (alternative: the share increases with N)
   with log N scores. The statistic does not depend on the base of the logarithm.
   The Bonferroni adjustment does not cover the trend tests.

Counts are derived from the raw evaluated runs, after checking that every expected
run is present exactly once and finished with status 0, and are reconciled with the
saved metrics files. No optimization or evaluation scenarios are used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analyze import wilson_interval

def parse_sites(value):
    parts = value if isinstance(value, list) else str(value).split("-")
    sites = [int(s) for s in parts]
    if not sites or len(set(sites)) != len(sites) or any(s < 0 or s >= 12 for s in sites):
        raise ValueError(f"invalid facility plan: {value}")
    return frozenset(sites)

CITIES = (2024, 2025, 2026)
EXPECTED_SIZES = {2024: (10, 20, 50, 100, 200), 2025: (10, 50, 100), 2026: (10, 50, 100)}
RUNS_PER_SIZE = 50
FAMILY_ALPHA = 0.05


def _check_counts(k, n):
    if not (float(n).is_integer() and float(k).is_integer() and n >= 1 and 0 <= k <= n):
        raise ValueError(f"invalid counts: k={k}, n={n}")


def z_for_confidence(confidence):
    """Two-sided normal quantile for a confidence level strictly between 0 and 1."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return float(norm.ppf(1 - (1 - confidence) / 2))


def newcombe_difference(k1, n1, k2, n2, confidence=0.95):
    """Difference p2 - p1 with its Newcombe (method 10) interval at the given confidence."""
    _check_counts(k1, n1)
    _check_counts(k2, n2)
    z = z_for_confidence(confidence)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    d = p2 - p1
    return d, d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def cochran_armitage(counts, sizes, scores):
    """Trend statistic z and the one-sided (increasing) and two-sided p-values."""
    x = np.asarray(counts, dtype=float)
    n = np.asarray(sizes, dtype=float)
    s = np.asarray(scores, dtype=float)
    if not (x.shape == n.shape == s.shape and x.ndim == 1 and len(x) >= 2):
        raise ValueError("counts, sizes and scores must be one-dimensional arrays of equal length (at least 2)")
    for k, m in zip(x, n):
        _check_counts(k, m)
    if not np.isfinite(s).all() or np.ptp(s) == 0:
        raise ValueError("scores must be finite and not all equal")
    total = n.sum()
    p_bar = x.sum() / total
    if p_bar in (0.0, 1.0):
        raise ValueError("the trend test is undefined when every run (or no run) chose the reference plan")
    centered = s - np.average(s, weights=n)
    t = np.sum(centered * (x - n * p_bar))
    variance = p_bar * (1 - p_bar) * np.sum(n * centered ** 2)
    z = t / np.sqrt(variance)
    return float(z), float(norm.sf(z)), float(2 * norm.sf(abs(z)))


def reference_counts(evaluated_csv, plans_json, city):
    """Per scenario count: all runs, runs with status 0, and runs choosing the reference plan."""
    plans = json.loads(Path(plans_json).read_text(encoding="utf-8"))
    if plans["city_seed"] != city:
        raise ValueError(f"{city}: plans file belongs to another city")
    reference = parse_sites(plans["reference_plan"])
    d = pd.read_csv(evaluated_csv, dtype={"plan": str})
    required = ["city_seed", "N", "rep", "status", "plan"]
    if not set(required).issubset(d.columns) or d[required].isna().any().any():
        raise ValueError(f"{city}: missing required columns or values")
    if not (d["city_seed"] == city).all():
        raise ValueError(f"{city}: evaluated file belongs to another city")
    for col in ("N", "rep", "status"):
        v = pd.to_numeric(d[col], errors="coerce")
        if not np.isfinite(v).all() or not (v == np.floor(v)).all():
            raise ValueError(f"{city}: invalid {col}")
        d[col] = v.astype(int)
    if (d["N"] <= 0).any():
        raise ValueError("N must be positive")
    if d.duplicated(["N", "rep"]).any():
        raise ValueError(f"{city}: duplicated (N, rep) identifiers")
    ok = d[(d["status"] == 0) & d["plan"].notna()]
    ok = ok.assign(is_reference=[parse_sites(p) == reference for p in ok["plan"]])
    table = d.groupby("N").agg(n_runs=("rep", "count")).join(
        ok.groupby("N").agg(n=("rep", "count"), k=("is_reference", "sum"))).fillna(0).reset_index()
    table["n"] = table["n"].astype(int)
    table["k"] = table["k"].astype(int)
    return table


def analyze_city(results_dir, city, expected_sizes=None, runs_per_size=RUNS_PER_SIZE):
    """Contrasts between adjacent scenario counts and the trend test for one city."""
    expected_sizes = EXPECTED_SIZES[city] if expected_sizes is None else tuple(expected_sizes)
    folder = Path(results_dir)
    counts = reference_counts(folder / f"main_{city}_evaluated.csv", folder / f"plans_{city}.json", city)

    if tuple(sorted(counts["N"])) != tuple(sorted(expected_sizes)):
        raise ValueError(f"{city}: scenario counts {sorted(counts['N'])} differ from the expected {sorted(expected_sizes)}")
    if not ((counts["n_runs"] == runs_per_size) & (counts["n"] == runs_per_size)).all():
        raise ValueError(f"{city}: every scenario count needs {runs_per_size} runs, all with status 0")

    raw = pd.read_csv(folder / f"main_{city}_evaluated.csv")
    for N, group in raw.groupby("N"):
        if set(group["rep"]) != set(range(runs_per_size)):
            raise ValueError(f"{city}: unexpected repetition identifiers at N={N}")

    # Reconcile with the saved metrics (p_reference was computed by analyze.py).
    metrics = pd.read_csv(folder / f"main_{city}_metrics.csv").set_index("N")
    if not metrics.index.is_unique or set(metrics.index) != set(expected_sizes):
        raise ValueError(f"{city}: invalid metrics scenario counts")
    for col in ("p_reference", "runs_status0", "runs"):
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
        if not np.isfinite(metrics[col]).all():
            raise ValueError(f"{city}: nonfinite metrics {col}")
    if not metrics["p_reference"].between(0, 1).all() or not (metrics["runs"] == runs_per_size).all():
        raise ValueError(f"{city}: invalid saved metrics")
    for row in counts.itertuples():
        expected = metrics.loc[row.N, "p_reference"] * metrics.loc[row.N, "runs_status0"]
        if row.n != metrics.loc[row.N, "runs_status0"] or abs(row.k - expected) > 1e-6:
            raise ValueError(f"{city}: counts at N={row.N} disagree with the saved metrics")

    counts = counts.sort_values("N").reset_index(drop=True)
    family = len(counts) - 1
    confidence = 1 - FAMILY_ALPHA / family
    contrasts = []
    for a, b in zip(counts.itertuples(), counts.iloc[1:].itertuples()):
        d, lo, hi = newcombe_difference(a.k, a.n, b.k, b.n, confidence)
        _, lo95, hi95 = newcombe_difference(a.k, a.n, b.k, b.n, 0.95)
        contrasts.append({"city": city, "N_from": int(a.N), "N_to": int(b.N),
                          "k_from": int(a.k), "n_from": int(a.n), "k_to": int(b.k), "n_to": int(b.n),
                          "difference": d, "family_size": family, "adjusted_confidence": confidence,
                          "ci_lo_adjusted": lo, "ci_hi_adjusted": hi,
                          "excludes_zero_adjusted": bool(lo > 0 or hi < 0),
                          "ci_lo_95": lo95, "ci_hi_95": hi95})
    z, p_one, p_two = cochran_armitage(counts["k"], counts["n"], np.log(counts["N"]))
    trend = {"city": city, "sizes": " ".join(map(str, counts["N"])), "counts": " ".join(map(str, counts["k"])),
             "z": z, "p_one_sided_increasing": p_one, "p_two_sided": p_two}
    return pd.DataFrame(contrasts), trend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out-prefix", default="results/exploratory_share")
    args = parser.parse_args()

    frames, trends = [], []
    for city in CITIES:
        c, t = analyze_city(args.results, city)
        frames.append(c)
        trends.append(t)
    contrasts, trend = pd.concat(frames, ignore_index=True), pd.DataFrame(trends)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(f"{prefix}_contrasts.csv", index=False)
    trend.to_csv(f"{prefix}_trends.csv", index=False)
    Path(f"{prefix}_checks.json").write_text(json.dumps({
        "status": "exploratory, not part of the frozen protocol",
        "cities": list(CITIES), "family_alpha_per_city": FAMILY_ALPHA,
        "family_size_per_city": {int(c): int(g["family_size"].iloc[0]) for c, g in contrasts.groupby("city")},
        "contrasts": len(contrasts), "trend_tests": len(trend),
        "trend_test": "Cochran-Armitage, scores log N, one-sided (increasing share)",
        "counts_checked": "all expected runs present once with status 0; reconciled with saved metrics"}, indent=2) + "\n")

    pd.set_option("display.width", 250)
    print(contrasts[["city", "N_from", "N_to", "difference", "adjusted_confidence", "ci_lo_adjusted",
                     "ci_hi_adjusted", "excludes_zero_adjusted", "ci_lo_95", "ci_hi_95"]].round(3).to_string(index=False))
    print()
    print(trend.to_string(index=False, formatters={"z": "{:.6f}".format, "p_one_sided_increasing": "{:.6e}".format, "p_two_sided": "{:.6e}".format}))


if __name__ == "__main__":
    main()
