# Protocol Deviations and Implementation Changes

This document records decisions, deviations, implementation changes, and computational observations made after the experimental protocol was frozen.

**Original protocol:** `docs/protocol.md` at Git tag `protocol-v1`
**Protocol freeze:** 14 November 2026

The original frozen protocol remains accessible through the Git tag. Entries below document subsequent developments without retrospectively changing the original experimental design.

## 16 November 2026 — Computational timing decision

**Decision:** Four worker processes and 50 replications at \(N=200\) were selected for the main-city experiment.

**Pilot timing results:**

| Workers | Time for four \(N=200\) replications |
|---|---:|
| 1 | 184 seconds |
| 2 | 93 seconds |
| 4 | 65 seconds |

The four-worker configuration had the shortest observed batch time.

**Memory limitation:** Peak combined memory was not systematically verified for every pilot configuration. An approximate observation of 1.6 GB was recorded, but its scope and peak status were uncertain.

**Protocol implication:** The worker configuration was selected using pilot timing results and available system memory. Strict verification of the preregistered 70% memory criterion was incomplete at the time of selection. The four-hour fallback was not triggered.

**Scientific impact:** This decision concerned computational execution and did not change the mathematical model, random seeds, evaluation criteria, or hypotheses.

## 16 November 2026 — Generated-output exclusions

**Change:** `.gitignore` was updated after the protocol tag (commit `d9ffc1ee`) to exclude generated experiment outputs by default.

**Reason:** To avoid accidentally staging temporary or generated files.

**Scientific impact:** No change to the experimental design or optimization implementation. Selected final results can still be committed explicitly.

## 16 November 2026 — Main-city experiment completed

**Observation:** The main-city 2024 experiment completed all 250 optimization runs with solver status 0.

- Worker processes: 4
- Total wall-clock time: 1251.8 seconds
- Failed runs: 0
- Retries required: 0

**Scientific impact:** The complete set of preregistered main-city replications was obtained without excluding failed runs.

## 17 November 2026 — Memory-monitoring evidence

**Observation:** Combined Python-process working-set memory was sampled approximately every two seconds during the four-worker main experiment.

- Highest sampled working-set memory: 1.77 GB
- Reported machine memory: 15.40 GB
- 70% threshold: 10.78 GB
- Approximate idle baseline: 0.05 GB

The highest sampled value was below the monitoring threshold.

**Limitation:** Sampling every two seconds may miss short-lived memory peaks. These observations do not retrospectively establish that the earlier pilot configurations satisfied the memory criterion.

## 17 November 2026 — Unusually long optimization runs

**Observation:** Four \(N=100\) replications (26, 27, 29, and 30) took approximately 279–291 seconds, compared with typical solve times of approximately 10–25 seconds.

All four returned solver status 0.

**Possible explanation:** Temporary system slowdown was suspected but not confirmed.

**Treatment:** The observations were retained in the raw experiment data. Median solution times are reported to reduce sensitivity to unusually long runs.

## 19 November 2026 — Clean-tree validation improvement

**Change:** Updated `_environment()` in `src/experiment.py` to detect untracked files under `src/` and `tests/`, in addition to modifications of tracked files.

The session metadata also records the detected untracked source-file names.

**Reason:** Previously, an experiment could potentially use uncommitted source files that were not represented by the recorded Git commit identifier.

**Main-city experiment:** City 2024 was completed using the earlier clean-tree validation implementation.

**Robustness experiments:** Cities 2025 and 2026 will use the strengthened validation implementation.

**Scientific impact:** The change affects launch validation and environment metadata. It does not modify the mathematical model, demand generation, optimization algorithm, scenario counts, or evaluation methodology.

## Protocol integrity

The changes documented here do not alter the preregistered scientific hypotheses or experimental settings.

The original protocol remains preserved under `protocol-v1`. Any future departures from the experimental design must be documented explicitly, including their reasons and potential effects on interpretation.

## 21 November 2026 — City 2026 experiment interrupted and resumed

**Observation:** The city 2026 experiment (N = 10, 50, 100; 50 repetitions each) was interrupted after 20 runs (session 1, started 13:10, commit a6d8a3be, no end record) and resumed with the same command (session 2, started 13:18, 130 runs, commit 21a44c4f). The final file has 150 rows, one per (N, repetition), with no duplicates and no failures.

**Code identity:** `git diff a6d8a3be 21a44c4f -- src tests` is empty; the intervening commit changed only result files (city 2025).

**Scientific impact:** None. The runner is resumable by design and each run's training sample depends only on (city, N, repetition). All 150 expected runs were completed without duplicate identifiers or solver failures. The source code and tests were unchanged between sessions.