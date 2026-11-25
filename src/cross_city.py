"""Consolidate and validate saved stochastic facility-location experiments.

No optimization or demand generation is performed by this module.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = (2024, 2025, 2026)
COMMON_N = (10, 50, 100)
METRIC_COLUMNS = (
    'N', 'runs', 'runs_status0', 'completion_rate', 'p_reference',
    'p_reference_lo', 'p_reference_hi', 'distinct_plans', 'mean_hamming',
    'gap_mean', 'gap_se', 'gap_negative_share', 'rel_excess_mean',
    'rel_excess_median', 'rel_excess_p95', 'rel_excess_max',
    'promised_sd_times_sqrt_n',
)
PAIRED_COLUMNS = (
    'comparison', 'reference_plan', 'alternative_plan',
    'alternative_selection_count', 'reference_cost', 'alternative_cost',
    'mean_difference', 'paired_se', 'ci95_lower', 'ci95_upper',
    'relative_difference_percent', 'n_scenarios',
)


def _require_columns(frame, names, label):
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f'{label}: missing columns {sorted(missing)}')


def _close(actual, expected, label, atol=1e-5, rtol=1e-8):
    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f'{label}: {actual} != {expected}')


def load_city(results_dir, city):
    directory = Path(results_dir)
    metrics = pd.read_csv(directory / f'main_{city}_metrics.csv')
    paired = pd.read_csv(directory / f'main_{city}_paired.csv')
    summary = json.loads((directory / f'main_{city}_summary.json').read_text(encoding='utf-8'))
    _require_columns(metrics, METRIC_COLUMNS, f'{city} metrics')
    _require_columns(paired, PAIRED_COLUMNS, f'{city} paired')
    if summary.get('city_seed') != city:
        raise ValueError(f'{city}: summary city_seed mismatch')
    if summary.get('evaluation_set') != 'final':
        raise ValueError(f'{city}: expected final evaluation set')
    if summary.get('evaluation_seed') != 202 or summary.get('n_evaluation_scenarios') != 10000:
        raise ValueError(f'{city}: unexpected final evaluation specification')
    reference_cost = float(summary['reference_realized_cost'])
    if not np.isfinite(reference_cost) or reference_cost <= 0:
        raise ValueError(f'{city}: invalid reference cost')
    if metrics.empty or metrics['N'].isna().any() or metrics['N'].duplicated().any():
        raise ValueError(f'{city}: missing or duplicated scenario counts')
    if not set(COMMON_N).issubset(set(metrics['N'])):
        raise ValueError(f'{city}: missing common scenario count')
    numeric = metrics.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f'{city}: nonfinite metrics')
    if not ((metrics['runs'] == 50) & (metrics['runs_status0'] == 50) &
            np.isclose(metrics['completion_rate'], 1.0)).all():
        raise ValueError(f'{city}: incomplete or unexpected run counts')
    if not metrics['p_reference'].between(0, 1).all():
        raise ValueError(f'{city}: invalid reference frequency')
    if not ((metrics['p_reference_lo'] <= metrics['p_reference']) &
            (metrics['p_reference'] <= metrics['p_reference_hi'])).all():
        raise ValueError(f'{city}: invalid reference-frequency intervals')
    if (metrics[['rel_excess_mean', 'rel_excess_median', 'rel_excess_p95', 'rel_excess_max']] < -1e-8).any().any():
        warnings.warn(f'{city}: negative relative-excess summary; some plan beat the reference plan on the final set', UserWarning)
    if len(paired) != 3 or paired['comparison'].duplicated().any():
        raise ValueError(f'{city}: expected three distinct comparison labels')
    expected_labels = {'most frequent non-reference plan', 'mean-demand plan', 'greedy rule'}
    if set(paired['comparison']) != expected_labels:
        raise ValueError(f'{city}: unexpected paired comparison labels')
    finite_columns = [c for c in PAIRED_COLUMNS if c not in (
        'comparison', 'reference_plan', 'alternative_plan', 'alternative_selection_count')]
    if not np.isfinite(paired[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError(f'{city}: nonfinite paired values')
    counts = paired.set_index('comparison')['alternative_selection_count']
    count = float(counts['most frequent non-reference plan'])
    if (not np.isfinite(count) or count < 1 or not count.is_integer()
            or count > metrics['runs_status0'].sum()
            or counts[['mean-demand plan', 'greedy rule']].notna().any()):
        raise ValueError(f'{city}: unexpected alternative selection counts')
    expected_reference = '-'.join(str(site) for site in summary['reference_plan'])
    if not (paired['reference_plan'].astype(str) == expected_reference).all():
        raise ValueError(f'{city}: paired reference plan differs from the summary')
    for row in paired.itertuples(index=False):
        _close(float(row.reference_cost), reference_cost, f'{city} paired reference cost')
        _close(float(row.mean_difference), float(row.alternative_cost - row.reference_cost), f'{city} paired cost difference')
        _close(float(row.relative_difference_percent), 100 * row.mean_difference / reference_cost,
               f'{city} paired relative difference', atol=1e-5)
        _close(float(row.ci95_lower), row.mean_difference - 1.96 * row.paired_se,
               f'{city} paired lower CI', atol=1e-5)
        _close(float(row.ci95_upper), row.mean_difference + 1.96 * row.paired_se,
               f'{city} paired upper CI', atol=1e-5)
        if row.n_scenarios != 10000 or row.paired_se < 0:
            raise ValueError(f'{city}: invalid paired sample size or standard error')
    metrics = metrics.copy()
    metrics.insert(0, 'city', city)
    metrics['reference_cost'] = reference_cost
    metrics['gap_pct_of_reference_cost'] = 100 * metrics['gap_mean'] / reference_cost
    paired = paired.copy()
    paired.insert(0, 'city', city)
    # Identical plans should be recognized, not treated as independent comparisons.
    paired['same_alternative_as_another_label'] = paired['alternative_plan'].duplicated(keep=False)
    return metrics, paired


def build_tables(results_dir, cities=CITIES):
    items = [load_city(results_dir, city) for city in cities]
    full = pd.concat([item[0] for item in items], ignore_index=True).sort_values(['city', 'N']).reset_index(drop=True)
    common = full[full['N'].isin(COMMON_N)].sort_values(['N', 'city']).reset_index(drop=True)
    paired = pd.concat([item[1] for item in items], ignore_index=True)
    if len(common) != len(cities) * len(COMMON_N):
        raise ValueError('Common-scenario table has an unexpected number of rows')
    return full, common, paired


def make_figure(common, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = (
        ('p_reference', 'Reference-selection frequency'),
        ('gap_pct_of_reference_cost', 'Mean optimism gap (% of reference cost)'),
        ('rel_excess_p95', '95th-percentile relative excess (%)'),
    )
    for ax, (column, title) in zip(axes, panels):
        for city, group in common.groupby('city'):
            group = group.sort_values('N')
            if column == 'p_reference':
                yerr = np.vstack([group[column] - group['p_reference_lo'],
                                  group['p_reference_hi'] - group[column]])
                ax.errorbar(group['N'], group[column], yerr=yerr, marker='o', capsize=3, label=str(city))
            else:
                ax.plot(group['N'], group[column], marker='o', label=str(city))
        ax.set_title(title)
        ax.set_xlabel('Training scenarios (N)')
        ax.set_xticks(COMMON_N)
        ax.set_xticklabels([str(n) for n in COMMON_N])
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[0].legend(title='City')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--out-prefix', default='results/cross_city')
    args = parser.parse_args()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        full, common, paired = build_tables(args.results)
    notes = [str(w.message) for w in caught]
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(f'{prefix}_full.csv', index=False)
    common.to_csv(f'{prefix}_common.csv', index=False)
    paired.to_csv(f'{prefix}_paired.csv', index=False)
    checks = {
        'cities': list(CITIES),
        'total_settings': len(full),
        'total_runs': int(full['runs'].sum()),
        'common_settings': len(common),
        'paired_rows': len(paired),
        'duplicated_alternative_labels_by_city': {
            str(city): int(group['same_alternative_as_another_label'].sum())
            for city, group in paired.groupby('city')
        },
        'validation': 'passed',
        'warnings': notes,
        'note': 'Reference plans are best among evaluated candidates, not globally certified optima.',
    }
    Path(f'{prefix}_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    make_figure(common, f'{prefix}_fig.png')
    print(json.dumps(checks, indent=2))
    columns = ['city', 'N', 'p_reference', 'distinct_plans', 'mean_hamming',
               'gap_pct_of_reference_cost', 'rel_excess_mean', 'rel_excess_median', 'rel_excess_p95']
    print('\nCOMMON SCENARIO COUNTS\n' + common[columns].round(4).to_string(index=False))
    print('\nPAIRED COMPARISONS\n' + paired[['city', 'comparison', 'mean_difference', 'paired_se',
                                                  'ci95_lower', 'ci95_upper', 'relative_difference_percent',
                                                  'same_alternative_as_another_label']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
