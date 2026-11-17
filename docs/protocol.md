# Experimental protocol: baseline stability experiment

This document fixes the design of the experiment before any main-city run. Changes
after the freeze are listed in the deviations log at the end.

Frozen on: Sat Nov 14 23:01:44 2026 +0100 (git tag `protocol-v1`)

## 1. Frozen parameters

City (`src/city.py`): 12 candidate sites, 40 demand zones, 10 km by 10 km square,
transport cost 2 per km, penalty 40 per unit of unmet demand, capacities uniform in
[80, 160], opening cost (300 + 3 x capacity) x land factor plus noise, as coded.

Demand (`src/demand.py`): lognormal with mean equal to the city's mean demand,
coefficient of variation 0.4, and log-demand correlation exp(-distance / 3 km).

Solver (`src/model.py`): HiGHS through SciPy, relative optimality gap 1e-9. A solver
status of 0 means optimal within the solver's numerical tolerances.

Calibration: the pilot city (seed 1) was solved once with these parameters. The
acceptance rule (3 to 8 open sites, unmet demand below about 5%) was proposed before
that solve. This file records it retrospectively, and the repository cannot
independently verify the timing. Result: 4 sites open, 2.4% of demand unmet. No
parameter has been changed since. Any future change is recorded in the deviations log.

## 2. Cities and data sets

| Role | Seed or construction |
|---|---|
| Pilot city (calibration and timing only) | 1 |
| Main city | 2024 |
| Robustness cities | 2025, 2026 |
| Training samples | `default_rng([city_seed, N, rep, 1])` |
| Development evaluation set (debugging only) | `default_rng(101)`, 1,000 scenarios |
| Reference sample | `default_rng(303)`, 5,000 scenarios |
| Final evaluation set (reported results only) | `default_rng(202)`, 10,000 scenarios |

The development, reference and final evaluation samples use the same fixed seeds for
every city. The cities differ, so the demand samples differ. Training samples use
city-specific seed lists. The independence of these streams rests on the design of
NumPy's seeding from seed lists, not on any observed correlation.

The final evaluation set is never used to select plans, tune settings, or debug.
Pilot-city results are used only for calibration and timing and are not reported as
findings.

## 3. Reference, benchmark and evaluation

- **Reference plan:** all 4,096 plans are screened on the first 100 scenarios of the
  reference sample, and the 50 cheapest are evaluated on all 5,000. This is the best
  plan found, not a proven optimum. A plan excluded by screening is never re-evaluated,
  and the reference sample carries sampling error. We report the screening margin,
  the smallest screening z-score among excluded plans, and the paired standard error
  of the runner-up gap. These diagnostics are descriptive, since they are computed on
  the data that selected the plans.
- **Mean-demand plan:** the optimal plan for the single demand vector equal to the mean.
- **Greedy rule:** open sites by increasing opening cost per unit of capacity until
  capacity covers the mean total demand. The rule is fixed and not tuned.
- **Evaluation:** every plan is evaluated with `PlanEvaluator` on the same final
  evaluation set within a city. The implementation is validated against the toy
  answer key, and its two backends (highspy and the SciPy fallback) agree to 1e-6 on
  the full instance. The backend and package versions are recorded. Before the final
  evaluation, five plans per city are re-evaluated with the original recourse LP
  (`second_stage_cost`) on the first 1,000 final-evaluation scenarios as a check.

Enumeration over all plans only works for small instances.

## 4. Baseline experiment

Training scenarios are drawn directly from the true demand model.

- **Main city 2024:** N in {10, 20, 50, 100, 200}, 50 repetitions per N.
- **Robustness cities 2025 and 2026:** N in {10, 50, 100}, 50 repetitions per N.

### Timing probe (pilot city, before any main-city run)

Solve N = 200 four times (the same four training samples) at each worker count: 1, 2,
and the count planned for the main run. Record the batch wall-clock time printed by
the runner and the peak total memory across all worker processes, measured the same
way for every worker count. Rules fixed in advance:

1. Use the worker count with the shortest batch wall-clock time whose peak total
   memory stays below 70% of the machine's RAM. If two counts are within 10%, take
   the smaller.
2. Project the time for the main city's 50 repetitions at N = 200 as
   50 x (batch wall-clock time / 4). This is an approximate pilot-city estimate: it
   ignores start-up overhead and a partly filled last batch, and other cities may be
   easier or harder. Only the main city has N = 200. If the projection exceeds
   4 hours, N = 200 uses 20 repetitions instead.

If the fallback is used, comparisons involving N = 200 have lower precision: standard
errors are about 58% larger (sqrt(50/20)). Four repetitions give preliminary timing
evidence only. The scientific design (cities, scenario counts, metrics, hypotheses)
is frozen. This computational rule is the only thing that can reduce the N = 200
repetitions, and it is fixed here before any main-city result exists.

Timing decision:
workers = 4, N = 200 repetitions = 50, date = 2026-11-16.

Pilot timing results (N = 200, four repetitions):
1 worker: 184 s
2 workers: 93 s
4 workers: 65 s

Machine: 14 logical processors, 15.40 GB RAM.
Estimated peak memory: approximately 1.6 GB
(combined-process peak not independently verified).

The four-worker configuration had the shortest batch time.
Projected runtime for 50 repetitions: 13.5 minutes.
The four-hour fallback was not triggered.

### Solver settings and failures

- Each solve has a 900 s time limit and a relative gap of 1e-9.
- The solver status, gap and message of every run are recorded.
- Runs that do not finish with status 0 stay in the results file and are reported.
  They may be redone with `--retry` and a larger time limit. Retry only touches the
  requested (N, repetition) combinations, copies the old attempts to an append-only
  log first, and replaces the file atomically. Any such change is recorded in the
  deviations log.
- Main-city runs are launched with `--require-clean` from a committed, tagged state.

## 5. Recorded for every run

In `<name>.csv`: city seed, N, repetition, chosen plan, promised cost (the SAA
objective), MIP gap, solve time, time limit, solver status, solver message.

Next to the CSV: `<name>.config.json` (fixed settings), `<name>.sessions.jsonl` (every
invocation, with its arguments, package versions, git commit and whether tracked
files had uncommitted changes), and `<name>.failed_attempts.csv` if any run was retried.

Afterwards, on the final evaluation set: realized cost, gap (realized minus promised),
excess over the reference plan (realized cost minus the reference plan's realized
cost on the same scenarios), and relative excess
(excess / reference realized cost x 100%). Relative excess can be negative: the
reference plan is not a proven optimum, and a plan can beat it on the final set.

## 6. Metrics, for each N

Decision stability:
- Probability of choosing the reference plan, with a Wilson 95% interval.
- Selection frequency of each site.
- Number of distinct plans chosen; mean Hamming distance to the reference plan.

Cost stability and out-of-sample performance:
- Mean and standard deviation of the promised cost and of the realized cost.
- Mean gap with its standard error, and the share of runs with a negative gap.
- Mean, median, 95th percentile and maximum excess and relative excess over the
  reference plan.

Reporting:
- Number of runs, completion rate (share of runs with solver status 0), and the
  number of observations behind every statistic.
- Bootstrap percentile intervals over the training repetitions. They condition on the
  fixed final evaluation set and do not include the Monte Carlo error of its 10,000
  scenarios. For headline comparisons we also report paired standard errors across
  the final evaluation scenarios.

Main results use runs with status 0. A sensitivity analysis adds runs that stopped at
the time limit with a solution. If the completion rate falls below 90% at any N, that
N is described as incomplete.

Decision stability and cost stability are different questions and are reported
separately. Plans can differ while costing almost the same.

## 7. Hypotheses stated before the run

1. The probability of choosing the reference plan rises with N, and the number of
   distinct plans falls. This presumes the reference plan is close to the true
   optimum. If several plans are nearly tied, the probability need not approach one,
   and we then rely on the cost-based metrics, which carry equal weight.
2. The mean gap is positive for small N (the promise is optimistic) and shrinks toward
   zero as N grows, although individual runs can have negative gaps. Theory: for a
   minimization problem, the expected optimal training value is at most the true
   optimal value, which is at most the expected true cost of the chosen plan, so the
   expected gap is non-negative when the final evaluation set is independent of
   training.
3. Despite variation in the chosen plans, the median relative excess cost over the
   reference plan, estimated on the final evaluation set, stays below 2% at every
   scenario count studied.
4. We expect the spread of the promised cost to shrink with N, roughly like 1/sqrt(N).
   This is a sampling-rate expectation, not a proven property of the optimized
   objective at these sample sizes.

These may turn out wrong, and that is a valid finding.

## 8. Limitations

The reference plan is not proven optimal. Diagnostics computed on the data that
selected plans are descriptive and are not formal confidence statements. Enumeration
over all plans does not scale beyond small instances. Timing and memory estimates come
from the pilot city and one machine. Results come from synthetic cities, not real data.

The baseline draws training scenarios from the same known distribution that defines
the evaluation, so it studies sampling variability only, not robustness to model
misspecification. Scenario-generation methods and stabilization are separate later
experiments.

## 9. Deviations log

## 9. Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-11-16 | `.gitignore` updated after the tag (commit d9ffc1ee) to exclude generated experiment outputs. | No effect on code or design. |
| 2026-11-16 | Peak combined memory was not systematically measured for each worker configuration. An approximate observation of 1.6 GB was reported, but its scope and peak status were not verified. Four workers were selected based on the fastest observed batch time and available system RAM; strict compliance with the 70% memory criterion remained unverified. | Memory was not measured as section 4 prescribes. Monitoring was planned for the main run. |
| 2026-11-17 | The sampled peak combined working-set memory of all Python processes during the 4-worker main run was 1.77 GB (sampled every 2 seconds, idle baseline 0.05 GB), below the 70% limit (10.78 GB of 15.40 GB RAM). | Main-run monitoring provides supporting evidence that memory was not a constraint for 4 workers. The earlier timing probes remain unverified, and brief peaks between samples may have been missed. |
| 2026-11-17 | Four N = 100 solves (reps 26, 27, 29, 30) took approximately 279–291 seconds, compared with typical solve times of 10–25 seconds. All returned status 0. | A temporary system slowdown is suspected but unconfirmed. The unusually long solve times will be retained in the raw data and reported transparently; medians will be used for timing summaries. |
| 2026-11-16 | Main-city experiment completed: 250/250 runs with solver status 0, using 4 workers. Total wall time was 1251.8 seconds. | No failed runs or retries were required. |
| 2026-11-16 | Combined Python-process working-set memory was monitored every 2 seconds during the main run. Highest observed value was 1.77 GB, below the 10.78 GB threshold. | Provides supporting evidence for the four-worker configuration. Short-lived peaks may have been missed; earlier pilot memory measurements remain unverified. |
