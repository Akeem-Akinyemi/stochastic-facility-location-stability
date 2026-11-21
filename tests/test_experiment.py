"""Tests for the repeated-solve experiment runner."""

import csv
import json

import numpy as np
import pytest

from src.experiment import NO_SOLUTION, run_experiment, solve_one, training_rng


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_training_samples_are_reproducible_and_distinct():
    a = training_rng(2024, 10, 0).standard_normal(5)
    assert np.array_equal(a, training_rng(2024, 10, 0).standard_normal(5))
    for other in [(2025, 10, 0), (2024, 20, 0), (2024, 10, 1)]:
        assert not np.array_equal(a, training_rng(*other).standard_normal(5))


def test_solve_one_is_reproducible_and_records_the_gap():
    r1 = solve_one((1, 4, 0, 60.0))
    r2 = solve_one((1, 4, 0, 60.0))
    assert r1["plan"] == r2["plan"] and r1["promised"] == r2["promised"]
    assert r1["status"] == 0 and r1["mip_gap"] < 1e-6


def test_time_limit_failure_is_recorded_not_raised():
    row = solve_one((1, 15, 0, 1e-4))                 # far too little time
    assert row["status"] in (1, NO_SOLUTION)
    assert row["message"] != ""


def test_run_is_resumable(tmp_path):
    out = tmp_path / "runs.csv"
    assert run_experiment(1, [3, 4], 2, out, workers=1) == 4
    assert run_experiment(1, [3, 4], 2, out, workers=1) == 0      # nothing left to do
    assert run_experiment(1, [3, 4], 3, out, workers=1) == 2      # only the new repetition
    rows = read(out)
    assert len(rows) == 6 and len({(r["N"], r["rep"]) for r in rows}) == 6


def test_failed_runs_are_kept_and_retried_on_request(tmp_path):
    out = tmp_path / "runs.csv"
    run_experiment(1, [15], 1, out, workers=1, time_limit=1e-4)
    assert int(read(out)[0]["status"]) != 0
    assert run_experiment(1, [15], 1, out, workers=1) == 0        # kept, not silently redone
    assert run_experiment(1, [15], 1, out, workers=1, time_limit=120.0, retry=True) == 1
    rows = read(out)
    assert len(rows) == 1 and int(rows[0]["status"]) == 0
    # the failed attempt is preserved in an append-only log, with its time limit
    failed = read(out.with_suffix(".failed_attempts.csv"))
    assert len(failed) == 1 and float(failed[0]["time_limit"]) == 1e-4
    assert int(failed[0]["status"]) != 0


def test_every_invocation_is_logged(tmp_path):
    out = tmp_path / "runs.csv"
    run_experiment(1, [3], 1, out, workers=1)
    run_experiment(1, [3], 2, out, workers=1)
    lines = [json.loads(line) for line in out.with_suffix(".sessions.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["start", "end", "start", "end"]
    assert "numpy" in lines[0]["environment"] and lines[2]["runs_to_do"] == 1


def test_runs_under_other_settings_are_rejected(tmp_path):
    out = tmp_path / "runs.csv"
    run_experiment(1, [3], 1, out, workers=1)
    with pytest.raises(ValueError):
        run_experiment(2, [3], 1, out, workers=1)                 # different city, same file


def test_old_file_format_is_rejected(tmp_path):
    out = tmp_path / "runs.csv"
    out.write_text("city_seed,N,rep,plan,promised,solve_seconds,status\n")
    with pytest.raises(ValueError):
        run_experiment(1, [3], 1, out, workers=1)


def test_retry_only_touches_the_requested_runs(tmp_path):
    out = tmp_path / "runs.csv"
    run_experiment(1, [15, 16], 1, out, workers=1, time_limit=1e-4)    # both fail
    assert run_experiment(1, [15], 1, out, workers=1, time_limit=120.0, retry=True) == 1
    by_n = {int(r["N"]): int(r["status"]) for r in read(out)}
    assert by_n[15] == 0                       # retried and solved
    assert by_n[16] != 0                       # not requested: must stay in the file untouched
    archived = read(out.with_suffix(".failed_attempts.csv"))
    assert [int(r["N"]) for r in archived] == [15]


@pytest.mark.parametrize("dirty, allowed", [(False, True), (True, False), (None, False)])
def test_require_clean_blocks_modified_or_unknown_code(tmp_path, monkeypatch, dirty, allowed):
    import src.experiment as experiment
    monkeypatch.setattr(experiment, "_environment", lambda: {"uncommitted_changes": dirty})
    out = tmp_path / "runs.csv"
    if allowed:
        assert run_experiment(1, [3], 1, out, workers=1, require_clean=True) == 1
    else:
        with pytest.raises(RuntimeError):
            run_experiment(1, [3], 1, out, workers=1, require_clean=True)
        assert not out.exists()


def test_untracked_source_files_count_as_uncommitted(tmp_path, monkeypatch):
    import subprocess

    import src.experiment as experiment

    def git(*args):
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")

    git("add", ".")
    git("commit", "-m", "init")
    monkeypatch.chdir(tmp_path)

    # A clean repository should be detected as clean.
    assert experiment._environment()["uncommitted_changes"] is False

    # An untracked result file should not make the repository dirty.
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "out.csv").write_text("a\n")
    assert experiment._environment()["uncommitted_changes"] is False

    # An untracked source file should make the repository dirty.
    (tmp_path / "src" / "b.py").write_text("y = 2\n")
    env = experiment._environment()

    assert env["uncommitted_changes"] is True
    assert env["untracked_source_files"] == ["src/b.py"]

    # Committing the new source file should restore the clean state.
    git("add", "src/b.py")
    git("commit", "-m", "add b")
    assert experiment._environment()["uncommitted_changes"] is False

    # Modifying a tracked source file should make the repository dirty.
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert experiment._environment()["uncommitted_changes"] is True