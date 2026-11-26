"""Tests for the exploratory share analysis."""

import json

import numpy as np
import pandas as pd
import pytest

from src.exploratory_share_analysis import (analyze_city, cochran_armitage, newcombe_difference,
                                            reference_counts, z_for_confidence)

REFERENCE = (0, 1)


def is_reference(plan, reference=REFERENCE):
    """Plans are equal when they open the same sites, in any order."""
    return {int(s) for s in plan.split("-")} == set(reference)


def test_z_for_confidence():
    assert z_for_confidence(0.95) == pytest.approx(1.95996, abs=1e-4)
    assert z_for_confidence(0.9875) == pytest.approx(2.4977, abs=1e-3)      # Bonferroni, 4 contrasts
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            z_for_confidence(bad)


def test_newcombe_difference_known_values():
    d, lo, hi = newcombe_difference(9, 50, 14, 50)
    assert (d, lo, hi) == pytest.approx((0.10, -0.07, 0.26), abs=0.01)
    d, lo, hi = newcombe_difference(16, 50, 28, 50)
    assert (d, lo, hi) == pytest.approx((0.24, 0.05, 0.41), abs=0.01)
    for bad in [(-1, 50, 5, 50), (51, 50, 5, 50), (5, 0, 5, 50), (2.5, 50, 5, 50)]:
        with pytest.raises(ValueError):
            newcombe_difference(*bad)


def test_trend_statistic_known_values():
    z, p_one, p_two = cochran_armitage([9, 14, 16, 28, 40], [50] * 5, np.log([10, 20, 50, 100, 200]))
    assert z == pytest.approx(6.7809, abs=1e-3)
    assert p_one == pytest.approx(5.972e-12, rel=1e-2)
    assert p_two == pytest.approx(2 * p_one)
    # the statistic does not depend on the base of the logarithm
    assert cochran_armitage([9, 14, 16, 28, 40], [50] * 5, np.log10([10, 20, 50, 100, 200]))[0] == pytest.approx(z)
    # equal shares: no trend
    assert cochran_armitage([20, 20, 20], [50] * 3, np.log([10, 50, 100]))[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        cochran_armitage([50, 50, 50], [50] * 3, np.log([10, 50, 100]))     # every run chose the reference plan
    with pytest.raises(ValueError):
        cochran_armitage([5, 10, 20], [50] * 3, [1.0, 1.0, 1.0])            # constant scores
    with pytest.raises(ValueError):
        cochran_armitage([5, 60, 20], [50] * 3, np.log([10, 50, 100]))      # count larger than the group


def write_city(folder, city, k_by_n, runs=50, reference=REFERENCE, order="forward"):
    """k_by_n: number of runs choosing the reference plan at each N (the others choose '0-2')."""
    rows, metrics = [], []
    for n, k in k_by_n.items():
        ref = "-".join(map(str, reference if order == "forward" else reference[::-1]))
        alt = str(next(s for s in range(12) if s not in reference))
        plans = [ref] * k + [alt] * (runs - k)
        rows += [{"city_seed": city, "N": n, "rep": r, "plan": p, "status": 0} for r, p in enumerate(plans)]
        metrics.append({"N": n, "runs_status0": runs, "runs": runs,
                        "p_reference": sum(is_reference(p, reference) for p in plans) / runs})
    pd.DataFrame(rows).to_csv(folder / f"main_{city}_evaluated.csv", index=False)
    (folder / f"plans_{city}.json").write_text(json.dumps({"city_seed": city, "reference_plan": list(reference)}))
    pd.DataFrame(metrics).to_csv(folder / f"main_{city}_metrics.csv", index=False)


def test_the_reference_plan_is_matched_as_a_set(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33}, order="reversed")     # plans written as '1-0'
    counts = reference_counts(tmp_path / "main_2025_evaluated.csv", tmp_path / "plans_2025.json", 2025)
    assert counts["k"].tolist() == [17, 25, 33]


def test_known_counts_reproduce_the_verified_numbers(tmp_path):
    write_city(tmp_path, 2024, {10: 9, 20: 14, 50: 16, 100: 28, 200: 40})
    contrasts, trend = analyze_city(tmp_path, 2024)
    assert len(contrasts) == 4 and (contrasts["family_size"] == 4).all()
    assert contrasts["adjusted_confidence"].iloc[0] == pytest.approx(0.9875)
    last = contrasts.iloc[-1]
    assert (last["N_from"], last["N_to"]) == (100, 200)
    assert last["difference"] == pytest.approx(0.24)
    assert last["ci_lo_adjusted"] == pytest.approx(0.0069, abs=1e-3)         # narrowly above zero
    assert bool(last["excludes_zero_adjusted"]) is True
    assert bool(contrasts.iloc[2]["excludes_zero_adjusted"]) is False        # 50 -> 100: lower bound -0.0066
    assert (contrasts["n_from"] == 50).all() and (contrasts["n_to"] == 50).all()
    assert trend["z"] == pytest.approx(6.7809, abs=1e-3)


def test_counts_must_agree_with_the_saved_metrics(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / "main_2025_metrics.csv"
    m = pd.read_csv(path)
    m.loc[0, "p_reference"] = 0.9
    m.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        analyze_city(tmp_path, 2025)


def test_plans_file_must_match_the_city(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    with pytest.raises(ValueError, match="another city"):
        reference_counts(tmp_path / "main_2025_evaluated.csv", tmp_path / "plans_2025.json", 2026)


def test_duplicate_run_identifiers_are_rejected(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / "main_2025_evaluated.csv"
    d = pd.read_csv(path)
    d.loc[1, "rep"] = 0                                                       # now (10, 0) appears twice
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        analyze_city(tmp_path, 2025)


def test_missing_scenario_count_is_rejected(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25})                              # N = 100 is missing
    with pytest.raises(ValueError, match="differ from the expected"):
        analyze_city(tmp_path, 2025)


def test_failed_or_missing_runs_are_rejected(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / "main_2025_evaluated.csv"
    d = pd.read_csv(path)
    d.loc[d.index[0], "status"] = -1                                          # one failed run
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="status 0"):
        analyze_city(tmp_path, 2025)
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33}, runs=49)            # one run missing at every N
    with pytest.raises(ValueError, match="needs 50 runs"):
        analyze_city(tmp_path, 2025)

@pytest.mark.parametrize('kind', ['rep', 'city', 'plan', 'nan', 'extra', 'duplicate'])
def test_invalid_saved_data(tmp_path, kind):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / 'main_2025_evaluated.csv'
    if kind in ('rep', 'city', 'plan'):
        d = pd.read_csv(path)
        col, value = {'rep': ('rep', 100), 'city': ('city_seed', 2024), 'plan': ('plan', '0-0')}[kind]
        d.loc[0, col] = value
    else:
        path = tmp_path / 'main_2025_metrics.csv'
        d = pd.read_csv(path)
        if kind == 'nan':
            d.loc[0, 'p_reference'] = np.nan
        else:
            row = d.iloc[[0]].copy()
            if kind == 'extra':
                row['N'] = 200
            d = pd.concat([d, row], ignore_index=True)
    d.to_csv(path, index=False)
    with pytest.raises(ValueError):
        analyze_city(tmp_path, 2025)


def test_nondefault_reference(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33}, reference=(2, 4, 6), order='reversed')
    contrasts, _ = analyze_city(tmp_path, 2025)
    assert contrasts['k_from'].tolist() == [17, 25]

@pytest.mark.parametrize('column,value', [('rep', 100), ('city_seed', 2024), ('plan', '0-0'), ('plan', '12'), ('N', np.nan)])
def test_invalid_run_data(tmp_path, column, value):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / 'main_2025_evaluated.csv'
    d = pd.read_csv(path)
    d.loc[0, column] = value
    d.to_csv(path, index=False)
    with pytest.raises(ValueError):
        analyze_city(tmp_path, 2025)

@pytest.mark.parametrize('value', [np.nan, np.inf, -0.1, 1.1])
def test_invalid_metric(tmp_path, value):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / 'main_2025_metrics.csv'
    d = pd.read_csv(path)
    d.loc[0, 'p_reference'] = value
    d.to_csv(path, index=False)
    with pytest.raises(ValueError):
        analyze_city(tmp_path, 2025)

def test_alternative_reference(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33}, reference=(2,4,6), order='reversed')
    contrasts, _ = analyze_city(tmp_path, 2025)
    assert contrasts.k_from.tolist() == [17,25]

def test_duplicate_metrics(tmp_path):
    write_city(tmp_path, 2025, {10: 17, 50: 25, 100: 33})
    path = tmp_path / 'main_2025_metrics.csv'
    d = pd.read_csv(path)
    pd.concat([d,d.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError, match='metrics scenario counts'):
        analyze_city(tmp_path, 2025)
