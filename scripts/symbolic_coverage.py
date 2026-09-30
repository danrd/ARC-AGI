#!/usr/bin/env python3
"""What the symbolic solvers solve, measured over a dataset split.

A solver's claim on its own says almost nothing - symbolic_module.checked_solve
records that upscale_or_covering claims 350 of 400 training tasks and is right
on 13 - so each task is scored twice per solver:

    claimed        solve() returned a grid
    correct        that grid is the test answer
    kept           checked_solve() also returned one: the rule reproduced
                   every training pair it was shown when that pair was held out
    kept correct   kept, and the grid is the test answer

The number that matters is kept correct, and precision after the check
(kept correct / kept). The per-task record also says which tasks each solver
solved, so the set can be joined with what the searches solve - see
--summary-only and the "solved by" line.

Usage:
    python scripts/symbolic_coverage.py --split evaluation --out sym_eval.jsonl
    python scripts/symbolic_coverage.py --split both --workers 2 --timeout 120
    python scripts/symbolic_coverage.py --summary-only --out sym_eval.jsonl

Written as one JSON line per task as it finishes, and a task already in the
file is not run again, so a run that is cut short - the machine restarts, a
worker is killed - resumes where it stopped.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "datasets" / "ARC"
SOLVERS = ("mixer", "upscale_or_covering", "color_restore")


class TimedOut(Exception):
    pass


def load_task_ids(split):
    """Task ids of one split, in file order."""
    with open(DATA / f"{split}_challenges.json") as handle:
        return list(json.load(handle))


def load_task(split, task_id):
    """One ARCTask, the test pair being the first."""
    from rl.arc_task import ARCSubtask, ARCTask

    with open(DATA / f"{split}_challenges.json") as handle:
        challenge = json.load(handle)[task_id]
    with open(DATA / f"{split}_solutions.json") as handle:
        answer = json.load(handle)[task_id]
    subtasks = [ARCSubtask(f"{task_id}_{index}", np.array(pair["input"]), np.array(pair["output"]))
                for index, pair in enumerate(challenge["train"])]
    return ARCTask(label=task_id, subtasks=subtasks,
                   test_inp=np.array(challenge["test"][0]["input"]),
                   test_out=np.array(answer[0]))


class _Alarm:
    """SIGALRM around a block; a no-op where it cannot ring (not the main
    thread, or no seconds)."""

    def __init__(self, seconds):
        self.seconds = int(seconds)

    def __enter__(self):
        if self.seconds > 0:
            def ring(*_):
                raise TimedOut()

            self.previous = signal.signal(signal.SIGALRM, ring)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *_):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous)
        return False


def default_solvers():
    from symbolic.symbolic_module import SymbolicModule

    module = SymbolicModule()
    return {name: getattr(module, name) for name in SOLVERS}


def score_task(task, solvers, timeout=0):
    """{solver: flags} for one task. A solver that runs past `timeout`
    seconds, or raises, is recorded as such and claims nothing."""
    from symbolic.symbolic_module import checked_solve

    answer = np.asarray(task.test_subtask.train_out)
    scored = {}
    for name, solver in solvers.items():
        record = {"claimed": False, "correct": False, "kept": False, "kept_correct": False,
                  "timed_out": False, "error": None}
        started = time.perf_counter()
        try:
            with _Alarm(timeout):
                result = solver.solve(task)
                record["claimed"] = bool(result.success)
                record["correct"] = bool(result.success) and _same(result.grid, answer)
                if result.success:
                    checked = checked_solve(solver, task)
                    record["kept"] = bool(checked.success)
                    record["kept_correct"] = bool(checked.success) and _same(checked.grid, answer)
        except TimedOut:
            record["timed_out"] = True
        except Exception as error:  # noqa: BLE001 - one solver must not sink the split
            record["error"] = repr(error)
        record["seconds"] = round(time.perf_counter() - started, 2)
        scored[name] = record
    return scored


def _same(grid, answer):
    return grid is not None and np.array_equal(np.asarray(grid), answer)


def score_by_id(split, task_id, timeout):
    """Top-level so a process pool can pickle it."""
    return {"split": split, "task": task_id,
            "solvers": score_task(load_task(split, task_id), default_solvers(), timeout)}


def read_records(path):
    """The records already in `path`, by (split, task); a torn last line is
    ignored, as a run killed mid-write leaves one."""
    records = {}
    if not Path(path).exists():
        return records
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            records[(record["split"], record["task"])] = record
    return records


def summarise(records, solvers=SOLVERS):
    """{split: {solver: counts}} plus, per split, how many tasks any solver
    solved (kept and correct)."""
    out = {}
    for split in sorted({record["split"] for record in records}):
        rows = [record for record in records if record["split"] == split]
        counts = {name: collections.Counter() for name in solvers}
        solved_by_any = 0
        for record in rows:
            any_solved = False
            for name in solvers:
                flags = record["solvers"].get(name)
                if flags is None:
                    continue
                for key in ("claimed", "correct", "kept", "kept_correct", "timed_out"):
                    counts[name][key] += bool(flags[key])
                counts[name]["error"] += flags["error"] is not None
                any_solved |= flags["kept_correct"]
            solved_by_any += any_solved
        out[split] = {"tasks": len(rows), "solvers": counts, "solved_by_any": solved_by_any}
    return out


def render(summary):
    lines = []
    for split, block in summary.items():
        lines.append(f"{split}: {block['tasks']} tasks")
        lines.append(f"  {'solver':<22}{'claimed':>8}{'correct':>8}{'kept':>6}{'kept ok':>8}"
                     f"{'precision':>11}{'timeouts':>9}{'errors':>7}")
        for name, count in block["solvers"].items():
            precision = (f"{100 * count['kept_correct'] / count['kept']:.1f}%"
                         if count["kept"] else "-")
            lines.append(f"  {name:<22}{count['claimed']:>8}{count['correct']:>8}{count['kept']:>6}"
                         f"{count['kept_correct']:>8}{precision:>11}{count['timed_out']:>9}"
                         f"{count['error']:>7}")
        lines.append(f"  solved by any solver, kept and correct: {block['solved_by_any']}")
    return "\n".join(lines)


def solved_ids(records, split=None):
    """{solver: sorted task ids} that were kept and correct."""
    by_solver = collections.defaultdict(list)
    for record in records:
        if split is not None and record["split"] != split:
            continue
        for name, flags in record["solvers"].items():
            if flags["kept_correct"]:
                by_solver[name].append(record["task"])
    return {name: sorted(ids) for name, ids in by_solver.items()}


def run(splits, out, workers, timeout, limit=0):
    """Score every task of `splits` not already in `out`, appending as each
    finishes. A worker killed for memory takes its pending tasks with it, so
    the pool is rebuilt for what is left, up to three times."""
    done = read_records(out)
    todo = [(split, task_id) for split in splits
            for task_id in load_task_ids(split)[:limit or None]
            if (split, task_id) not in done]
    print(f"{len(done)} tasks already in {out}; {len(todo)} to go", flush=True)
    breaks = 0
    with open(out, "a", buffering=1) as handle:
        while todo and breaks <= 3:
            finished = set()
            try:
                with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = {pool.submit(score_by_id, split, task_id, timeout): (split, task_id)
                               for split, task_id in todo}
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(json.dumps(record) + "\n")
                        finished.add(futures[future])
                        print(f"[{len(done) + len(finished)}] {record['split']} {record['task']}",
                              flush=True)
            except BrokenProcessPool:
                breaks += 1
                print(f"a worker was killed ({breaks}); rebuilding the pool", flush=True)
            todo = [item for item in todo if item not in finished]
    return read_records(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="evaluation", choices=("training", "evaluation", "both"))
    parser.add_argument("--out", default="symbolic_coverage.jsonl")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120,
                        help="seconds a solver may take on one task, its check included; "
                             "0 for no limit")
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks of a split")
    parser.add_argument("--summary-only", action="store_true",
                        help="summarise what --out already holds and run nothing")
    args = parser.parse_args()
    splits = ("training", "evaluation") if args.split == "both" else (args.split,)
    records = (read_records(args.out) if args.summary_only
               else run(splits, args.out, args.workers, args.timeout, args.limit))
    wanted = [record for record in records.values() if record["split"] in splits]
    print(render(summarise(wanted)))
    for split in splits:
        solved = solved_ids(wanted, split)
        for name, ids in solved.items():
            print(f"solved by {name} ({split}): {' '.join(ids)}")


if __name__ == "__main__":
    main()
