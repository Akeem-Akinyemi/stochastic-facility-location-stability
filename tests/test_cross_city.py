"""Focused tests for cross-city consolidation and numerical validation."""
import json

import pandas as pd
import pytest

from src.cross_city import build_tables, load_city, make_figure


def sample_city(folder, city, reference_cost=5000.0):
    rows = []
    for n in ((10, 20, 50, 100, 200) if city == 2024 else (10, 50, 100)):
        rows.append(dict(N=n, runs=50, runs_status0=50, completion_rate=1.0,
                         p_reference=0.5, p_reference_lo=0.3, p_reference_hi=0.7,
                         distinct_plans=2, mean_hamming=1.0, gap_mean=100.0,
                         gap_se=10.0, gap_negative_share=0.3,
                         rel_excess_mean=1.0, rel_excess_median=0.0,
                         rel_excess_p95=2.0, rel_excess_max=3.0,
                         promised_sd_times_sqrt_n=1000.0))
    pd.DataFrame(rows).to_csv(folder / f'main_{city}_metrics.csv', index=False)
    (folder / f'main_{city}_summary.json').write_text(json.dumps({
        'city_seed': city, 'evaluation_set': 'final', 'evaluation_seed': 202,
        'n_evaluation_scenarios': 10000, 'reference_realized_cost': reference_cost,
        'reference_plan': [0, 1],
    }))
    paired = []
    for name, alt in [('most frequent non-reference plan', '1-2'),
                      ('mean-demand plan', '1-2'), ('greedy rule', '3-4')]:
        paired.append(dict(comparison=name, reference_plan='0-1', alternative_plan=alt,
                           alternative_selection_count=10 if name == "most frequent non-reference plan" else None, reference_cost=reference_cost,
                           alternative_cost=reference_cost + 100,
                           mean_difference=100, paired_se=10,
                           ci95_lower=80.4, ci95_upper=119.6,
                           relative_difference_percent=100 * 100 / reference_cost,
                           n_scenarios=10000))
    pd.DataFrame(paired).to_csv(folder / f'main_{city}_paired.csv', index=False)


def test_tables_and_duplicate_plan_labels(tmp_path):
    for city in (2024, 2025, 2026):
        sample_city(tmp_path, city)
    full, common, paired = build_tables(tmp_path)
    assert len(full) == 11
    assert len(common) == 9
    assert set(common.N) == {10, 50, 100}
    assert common.iloc[0].gap_pct_of_reference_cost == pytest.approx(2.0)
    assert paired.groupby('city')['same_alternative_as_another_label'].sum().tolist() == [2, 2, 2]


def test_figure(tmp_path):
    for city in (2024, 2025, 2026):
        sample_city(tmp_path, city)
    _, common, _ = build_tables(tmp_path)
    make_figure(common, tmp_path / 'figure.png')
    assert (tmp_path / 'figure.png').stat().st_size > 0


def test_reject_duplicate_scenario_count(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_metrics.csv'
    df = pd.read_csv(path)
    df.loc[1, 'N'] = 10
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicated'):
        load_city(tmp_path, 2024)


def test_reject_inconsistent_reference_cost(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_paired.csv'
    df = pd.read_csv(path)
    df.loc[0, 'reference_cost'] = 4000
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='reference cost'):
        load_city(tmp_path, 2024)


def test_reject_incorrect_paired_difference(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_paired.csv'
    df = pd.read_csv(path)
    df.loc[0, 'mean_difference'] = 200
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='cost difference'):
        load_city(tmp_path, 2024)


def test_reject_missing_common_n(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_metrics.csv'
    df = pd.read_csv(path)
    df = df[df.N != 50]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing common'):
        load_city(tmp_path, 2024)


def test_negative_excess_is_warning(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_metrics.csv'
    df = pd.read_csv(path)
    df['rel_excess_median'] = -0.2
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning, match='beat the reference plan'):
        load_city(tmp_path, 2024)


def test_reference_plan_mismatch(tmp_path):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_paired.csv'
    df = pd.read_csv(path)
    df['reference_plan'] = '0-2'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='differs from the summary'):
        load_city(tmp_path, 2024)


@pytest.mark.parametrize('row,count', [(1,5), (0,0), (0,1.5), (0,float('inf')), (0,float('nan')), (0,251)])
def test_invalid_selection_counts(tmp_path, row, count):
    sample_city(tmp_path, 2024)
    path = tmp_path / 'main_2024_paired.csv'
    df = pd.read_csv(path)
    df.loc[row, 'alternative_selection_count'] = count
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='selection counts'):
        load_city(tmp_path, 2024)
