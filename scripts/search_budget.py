#!/usr/bin/env python3
"""How far a solve-time search gets for the seconds it is given.

The online search is the product: nothing is known about a task in
advance, so the hint is computed when the task arrives and the only
question is how to spend the seconds that task gets. This sweeps the one
setting that decides that - how many iterations a single search runs -
and reports what each buys.

Measured on 20 tasks before this existed, at one search each:

    1 x 40 iterations,  4 rollouts   8/20 carried a hint, 9.7s mean
    4 x 40 iterations,  4 rollouts  10/20                37.7s
    1 x 160 iterations, 4 rollouts  10/20                16.8s
    1 x 40 iterations, 16 rollouts   9/20                15.7s

Four searches and one search four times as long reached the same
coverage, at twice the cost - a new search rebuilds the tree, more
iterations continue the one already built. Which is why this sweeps
depth. Two of twenty is well inside noise, though, so the point of a
bigger run is to say whether the ordering is real and where depth stops
paying.

Reported per setting: how many tasks carried a hint at all, how far the
search got, how many it solved outright, and what it cost. Coverage is
the figure that matters for a prompt - a task with no hint gets no block.

Usage:
    python scripts/search_budget.py --tasks 0-50 --iterations 40 160 640
    python scripts/search_budget.py --tasks 0-100 --split evaluation \\
        --workers 4 --out budget.json

--workers spreads tasks across processes, which is the axis worth
parallelising: separate tasks share nothing, where repeats of one task
were measured to be the weakest way to spend the same seconds. Each
worker holds an interpreter that reached 2 GB during a search, so 4 is
right for a machine with cores to spare and nothing else loaded.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.search_hints import (SearchSettings, render_block,  # noqa: E402
                             search_task)
from scripts.compare_reward_approaches import load_tasks, parse_span  # noqa: E402


def one_task(payload):
    """One task under one setting, as the numbers a sweep compares."""
    triple, settings = payload
    started = time.perf_counter()
    try:
        found = search_task(triple, settings)
    except Exception as exc:  # a task that fails is not a task that scored 0
        return {"task": triple[0], "why": type(exc).__name__}
    task_id = triple[0]
    shaped = {"names": {str(i): n for i, n in found["actions"].items()},
              "solutions": {task_id: found["solutions"]},
              "partials": {task_id: found["partials"]},
              "effective": {task_id: found["effective"]}}
    text = render_block(triple, shaped, found["actions"], settings.moves,
                        settings.min_gain, settings.episode_len,
                        settings.skip_solved)
    return {"task": task_id,
            "seconds": time.perf_counter() - started,
            "peak": found["peak"],
            "solved": bool(found["solutions"]),
            "moves": len(found["effective"]),
            "carried": bool(text),
            "why": None}


def sweep(tasks, settings, workers):
    payloads = [(triple, settings) for triple in tasks]
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(one_task, payloads))
    return [one_task(payload) for payload in payloads]


def report(label, rows):
    scored = [row for row in rows if row["why"] is None]
    if not scored:
        print(f"{label:>22s}  every task failed")
        return {}
    seconds = [row["seconds"] for row in scored]
    peaks = [row["peak"] for row in scored]
    summary = {
        "tasks": len(scored),
        "failed": len(rows) - len(scored),
        "carried": sum(1 for row in scored if row["carried"]),
        "solved": sum(1 for row in scored if row["solved"]),
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.mean(seconds),
        "max_seconds": max(seconds),
        "mean_peak": statistics.mean(peaks),
        "moved": sum(1 for row in scored if row["peak"] > 0),
    }
    print(f"{label:>22s}  hint on {summary['carried']:3d}/{summary['tasks']:<3d}  "
          f"solved {summary['solved']:2d}  moved {summary['moved']:3d}  "
          f"peak {summary['mean_peak']:.3f}  "
          f"median {summary['median_seconds']:6.1f}s  "
          f"mean {summary['mean_seconds']:6.1f}s  "
          f"max {summary['max_seconds']:6.1f}s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=parse_span, default="20")
    parser.add_argument("--split", default="training")
    parser.add_argument("--dataset", default="ARC")
    parser.add_argument("--iterations", type=int, nargs="+", default=[40, 160, 640])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds one search may take; what it found up to "
                             "the cut is kept")
    parser.add_argument("--min-gain", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1,
                        help="tasks in flight at once, a core each")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks, total = load_tasks(args.dataset, args.tasks, args.split)
    if not tasks:
        raise SystemExit(f"--tasks {args.tasks[0]}-{args.tasks[1]} selects nothing "
                         f"of {total} in {args.dataset} {args.split}")
    print(f"{len(tasks)} tasks of {total} in {args.dataset} {args.split}, "
          f"{args.repeats} search(es) each, {args.workers} worker(s)\n")

    results = {}
    for iterations in args.iterations:
        settings = SearchSettings(iterations=iterations, repeats=args.repeats,
                                  rollouts=args.rollouts, timeout=args.timeout,
                                  min_gain=args.min_gain)
        started = time.perf_counter()
        rows = sweep(tasks, settings, args.workers)
        label = f"{args.repeats} x {iterations} iters"
        summary = report(label, rows)
        summary["wall_seconds"] = time.perf_counter() - started
        results[str(iterations)] = {"summary": summary, "rows": rows}

    print("\n  coverage is what a prompt sees: a task with no hint gets no block.")
    print("  compare the cost columns before reading anything into two tasks "
          "of difference - the spread between identical runs is that wide.")

    if args.out:
        args.out.write_text(json.dumps(
            {"split": args.split, "span": list(args.tasks),
             "repeats": args.repeats, "rollouts": args.rollouts,
             "results": results}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
