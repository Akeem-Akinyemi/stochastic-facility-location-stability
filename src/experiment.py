"""
experiment.py: repeated scenario-based solves on one city, recorded to a CSV file.

For every scenario count N and repetition index we draw a fresh training sample,
solve the SAA model, and record the chosen plan, the cost the sample promises, and
how the solver ended. Solves run in parallel and are appended to the file as they
finish, so an interrupted run can be resumed without repeating finished work.

"""

import argparse
import csv
import datetime
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.city import make_city
from src.demand import make_demand_model
from src.model import MIP_REL_GAP, solve_saa

FIELDS = ["city_seed", "N", "rep", "plan", "promised", "mip_gap",
          "solve_seconds", "time_limit", "status", "message"]
TRAIN_ROLE = 1       # separates training samples from the evaluation and reference samples
CV = 0.4             # demand model settings, frozen in the protocol
CORR_LENGTH = 3.0
NO_SOLUTION = -1     # status recorded when the solver returned no solution at all


def training_rng(city_seed, N, rep):
    """Random generator for the training sample of one run; distinct for every (city, N, rep)."""
    return np.random.default_rng([city_seed, N, rep, TRAIN_ROLE])


def solve_one(task):
    """
    Draw the training sample for one run, solve the SAA model, and return a result row.
    status is the solver's status code (0 = optimal within the solver's tolerances,
    1 = limit reached with a solution) or NO_SOLUTION. Unexpected errors are not
    caught: they stop the run.
    """
    city_seed, N, rep, time_limit = task
    city = make_city(seed=city_seed)
    D = make_demand_model(city, cv=CV, corr_length=CORR_LENGTH).sample(N, training_rng(city_seed, N, rep))
    row = {"city_seed": city_seed, "N": N, "rep": rep, "time_limit": time_limit, "message": ""}
    t0 = time.time()
    try:
        res = solve_saa(city.open_cost, city.capacity, city.service_cost, city.penalty,
                        D, time_limit=time_limit)
        row.update(plan="-".join(str(i) for i in np.flatnonzero(res["y"])),
                   promised=res["objective"], mip_gap=res["mip_gap"], status=res["status"])
        if res["status"] != 0:
            row["message"] = res["message"][:120]
    except RuntimeError as err:                   # the solver returned no solution
        row.update(plan="", promised=float("nan"), mip_gap=float("nan"),
                   status=NO_SOLUTION, message=str(err)[:120])
    row["solve_seconds"] = round(time.time() - t0, 3)
    return row


def _check_config(out, city_seed):
    """Save the fixed settings next to the CSV; refuse to mix runs made under other settings."""
    config = {"city_seed": city_seed, "cv": CV, "corr_length": CORR_LENGTH,
              "mip_rel_gap": MIP_REL_GAP, "fields": FIELDS}
    path = out.with_suffix(".config.json")
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"{path} holds different settings; use a new output file")
    else:
        path.write_text(json.dumps(config, indent=2))


def _read_rows(out):
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{out} has a different format; use a new output file")
        return list(reader)


def _git(*args):
    """Output of a git command, or None if git is unavailable or the command fails."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _environment():
    """
    Package versions and git state, recorded in the session log. uncommitted_changes is
    True if tracked files differ from the last commit, False if they match, and None if
    git could not tell. Untracked files (such as result files) are ignored on purpose.
    """
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"python": platform.python_version(), "numpy": version("numpy"),
            "scipy": version("scipy"), "highspy": version("highspy"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "uncommitted_changes": None if status is None else bool(status)}


def _append_log(path, record):
    """Append one JSON line to an append-only log."""
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_rows_atomically(out, rows):
    """Replace the CSV with the given rows without ever leaving a half-written file."""
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


def run_experiment(city_seed, sizes, reps, out_csv, workers=1, time_limit=900.0,
                   retry=False, require_clean=False):
    """
    Run all missing (N, rep) combinations and append them to out_csv. Returns the number run.
    With retry=True, rows of the requested (N, rep) combinations whose solve was not proven
    optimal are archived and run again; all other rows are left untouched. With
    require_clean=True the run refuses to start unless the tracked code matches the last commit.
    """
    environment = _environment()
    if require_clean and environment["uncommitted_changes"] is not False:
        raise RuntimeError("tracked files differ from the last commit (or git is unavailable); "
                           "commit your changes before starting this run")
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    _check_config(out, city_seed)

    rows = _read_rows(out) if out.exists() else []
    if retry:
        requested = {(city_seed, N, rep) for N in sizes for rep in range(reps)}

        def to_redo(r):
            return (int(r["city_seed"]), int(r["N"]), int(r["rep"])) in requested and int(r["status"]) != 0

        failed = [r for r in rows if to_redo(r)]
        rows = [r for r in rows if not to_redo(r)]
        if failed:
            log = out.with_suffix(".failed_attempts.csv")     # append-only audit trail
            write_header = not log.exists()
            with open(log, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(failed)
            _write_rows_atomically(out, rows)
    done = {(int(r["city_seed"]), int(r["N"]), int(r["rep"])) for r in rows}

    # Largest problems first, so the long solves do not all end up at the back of the queue.
    tasks = sorted(((city_seed, N, rep, time_limit) for N in sizes for rep in range(reps)
                    if (city_seed, N, rep) not in done), key=lambda t: -t[1])
    new_file = not out.exists() or out.stat().st_size == 0
    count = 0
    started = time.time()
    session_log = out.with_suffix(".sessions.jsonl")
    _append_log(session_log, {"event": "start", "time": datetime.datetime.now().isoformat(timespec="seconds"),
                              "sizes": list(sizes), "reps": reps, "workers": workers,
                              "time_limit": time_limit, "retry": retry, "runs_to_do": len(tasks),
                              "environment": environment})
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        def record(row):
            nonlocal count
            writer.writerow(row)
            fh.flush()
            count += 1
            print(f"[{count}/{len(tasks)}] N={row['N']} rep={row['rep']} status={row['status']} "
                  f"plan={row['plan']} {row['solve_seconds']}s", flush=True)

        if workers <= 1:
            for task in tasks:
                record(solve_one(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(solve_one, t) for t in tasks]):
                    record(future.result())
    wall = time.time() - started
    _append_log(session_log, {"event": "end", "runs_done": count, "wall_seconds": round(wall, 1)})
    print(f"Finished {count} runs in {wall:.0f}s with {workers} worker(s)", flush=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", type=int, required=True, help="city seed")
    parser.add_argument("--sizes", type=int, nargs="+", required=True, help="scenario counts N")
    parser.add_argument("--reps", type=int, required=True, help="repetitions per N")
    parser.add_argument("--out", required=True, help="output CSV path (one file per city)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--time-limit", type=float, default=900.0, help="seconds per solve")
    parser.add_argument("--retry", action="store_true",
                        help="redo the requested runs that did not finish with status 0")
    parser.add_argument("--require-clean", action="store_true",
                        help="refuse to run if tracked files differ from the last commit")
    args = parser.parse_args()
    run_experiment(args.city, args.sizes, args.reps, args.out, args.workers,
                   args.time_limit, args.retry, args.require_clean)