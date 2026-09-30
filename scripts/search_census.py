#!/usr/bin/env python3
"""What the object search solves, and what it leaves behind where it does not.

For each task whose grids keep their size, the search runs the way hints_for
runs it - the base action types first, then the agent rosters - and the task
is recorded as

    verified   one set of action types reproduces every training pair
    first      the first pair alone was solved (a solution fitted to one grid
               is not a rule, and hints_for says so)
    peak       how far the best search got on the first pair, as the share of
               the cells that had to change which it set right

and, for a task that got somewhere and not all the way, what was left: the
grid the furthest partial path reaches, laid against the answer. The cells
still wrong are described by what they are and how they lie - painted onto
background, recoloured, erased; one region or many, rectangles, lines or
single cells - because that says which action the vocabulary lacks. A task
whose leftover is one rectangle wants a fill; three lines want a stroke; a
recolour of one colour into another wants a part-recolour; and a leftover
that is scattered says the search took a wrong turn early, not that an action
is missing.

    python scripts/search_census.py --split evaluation --out census.jsonl
    python scripts/search_census.py --summary-only --out census.jsonl

One JSON line per task as it finishes; a task already in the file is not run
again, so a run cut short resumes. Tasks whose grids change size are recorded
as skipped: every object action paints inside the grid it is given.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "datasets" / "ARC"
EIGHT = np.ones((3, 3))


# ---------------------------------------------------------------------------
# what is left
# ---------------------------------------------------------------------------

def leftover(grid, target, start):
    """The cells `grid` still has wrong against `target`, described.

    `start` is the task's input: the background is its commonest colour, and
    the share closed is of the cells that differed between it and the answer.
    """
    grid, target, start = np.asarray(grid), np.asarray(target), np.asarray(start)
    background = int(np.bincount(start.ravel()).argmax())
    wrong = grid != target
    to_change = int((start != target).sum())
    closed = 1.0 - wrong.sum() / to_change if to_change else 1.0
    kinds = collections.Counter()
    for current, wanted in zip(grid[wrong].tolist(), target[wrong].tolist()):
        if current == background:
            kinds["paint"] += 1
        elif wanted == background:
            kinds["erase"] += 1
        else:
            kinds["recolour"] += 1
    labels, count = ndimage.label(wrong, structure=EIGHT)
    shapes = collections.Counter()
    for index in range(1, count + 1):
        rows, cols = np.where(labels == index)
        height, width = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
        size = len(rows)
        if size == 1:
            shapes["cell"] += 1
        elif size == height * width:
            shapes["line" if min(height, width) == 1 else "rectangle"] += 1
        else:
            shapes["other"] += 1
    transitions = collections.Counter(zip(grid[wrong].tolist(), target[wrong].tolist()))
    return {"wrong": int(wrong.sum()), "to_change": to_change, "closed": round(float(closed), 3),
            "kinds": dict(kinds), "regions": count, "shapes": dict(shapes),
            "transitions": len(transitions)}


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------

def load_tasks(split):
    """{task id: [(pair id, input, output), ...]} in file order."""
    with open(DATA / f"{split}_challenges.json") as handle:
        challenges = json.load(handle)
    return {task_id: [(f"{task_id}_{index}", np.array(pair["input"]), np.array(pair["output"]))
                      for index, pair in enumerate(task["train"])]
            for task_id, task in challenges.items()}


def best_partial(results):
    """The furthest partial path over every branch: (closed share, trace)."""
    best = None
    for found in results.values():
        for closed, trace in found.get("partials", []):
            if best is None or closed > best[0]:
                best = (closed, trace, found)
    return best


def final_grid(pair, trace, found, episode_len):
    """The grid a partial path reaches, replayed on the pair's own env."""
    from rl.search_hints import make_env

    env = make_env(pair, found["actions"], episode_len)
    env.reset()
    for action in trace:
        _obs, _reward, done, truncated, _info = env.step(np.asarray(action))
        if done or truncated:
            break
    return np.asarray(env.grid)


def census_task(task_id, pairs, settings=None):
    """The record of one task."""
    import rl.search_hints as hints

    settings = settings or hints.SearchSettings()
    if any(np.shape(pair[1]) != np.shape(pair[2]) for pair in pairs):
        return {"task": task_id, "skipped": "the grids change size"}
    first, rest = pairs[0], pairs[1:]
    from dataclasses import replace
    settings = replace(settings, colours=tuple(hints.output_colours(*[p[2] for p in pairs])),
                       directions=tuple(dict.fromkeys((*hints.MAIN_DIRECTIONS, *settings.directions))))
    results = hints.search_branches(first, settings)
    text = hints._first_verified(first, rest, results, settings, 3)
    if text is None and set(results) == {"base"}:
        results = hints.each_branch(first, settings)
        text = hints._first_verified(first, rest, results, settings, 3)
    peak = max((found["peak"] for found in results.values()), default=0.0)
    record = {"task": task_id, "verified": text is not None,
              "first": any(found["solutions"] for found in results.values()),
              "peak": round(float(peak), 3), "branches": sorted(results)}
    if text is None:
        partial = best_partial(results)
        if partial is not None and not record["first"]:
            closed, trace, found = partial
            grid = final_grid(first, trace, found, settings.episode_len)
            record["left"] = leftover(grid, first[2], first[1])
            record["left"]["path_len"] = len(trace)
    return record


def census_by_id(split, task_id, timeout):
    """Top-level so a process pool can pickle it."""
    import rl.search_hints as hints

    pairs = load_tasks(split)[task_id]
    return {"split": split, **census_task(task_id, pairs, hints.SearchSettings(timeout=timeout))}


# ---------------------------------------------------------------------------
# the file, the summary
# ---------------------------------------------------------------------------

def read_records(path):
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


def summarise(records):
    """A few counts over the records: what was solved, and what was left."""
    rows = list(records)
    searched = [r for r in rows if "skipped" not in r]
    out = {"tasks": len(rows), "skipped": len(rows) - len(searched),
           "verified": sum(r["verified"] for r in searched),
           "first_only": sum(r["first"] and not r["verified"] for r in searched)}
    unsolved = [r for r in searched if not r["verified"] and not r["first"]]
    out["unsolved"] = len(unsolved)
    out["peak_bands"] = dict(collections.Counter(
        "0" if r["peak"] <= 0 else "<0.5" if r["peak"] < 0.5 else "0.5-0.8" if r["peak"] < 0.8 else ">=0.8"
        for r in unsolved))
    left = [r["left"] for r in unsolved if "left" in r]
    out["with_leftover"] = len(left)
    out["few_regions"] = sum(item["regions"] <= 3 for item in left)
    out["one_transition"] = sum(item["transitions"] == 1 for item in left)
    out["kinds"] = dict(collections.Counter(
        max(item["kinds"], key=item["kinds"].get) for item in left if item["kinds"]))
    shapes = collections.Counter()
    for item in left:
        if item["shapes"]:
            shapes[max(item["shapes"], key=item["shapes"].get)] += 1
    out["dominant_shape"] = dict(shapes)
    return out


def render(summary):
    lines = [f"{summary['tasks']} tasks, {summary['skipped']} skipped (the grids change size)",
             f"verified by the search: {summary['verified']}; solved on the first pair only: "
             f"{summary['first_only']}; not solved: {summary['unsolved']}",
             f"how far the unsolved got: {summary['peak_bands']}"]
    if summary["with_leftover"]:
        lines += [f"of the {summary['with_leftover']} with a leftover to look at:",
                  f"  at most three regions wrong: {summary['few_regions']}",
                  f"  every wrong cell one colour into one other: {summary['one_transition']}",
                  f"  what the wrong cells mostly are: {summary['kinds']}",
                  f"  the regions are mostly: {summary['dominant_shape']}"]
    return "\n".join(lines)


def run(split, out, workers, timeout, limit=0):
    done = read_records(out)
    todo = [(split, task_id) for task_id in list(load_tasks(split))[:limit or None]
            if (split, task_id) not in done]
    print(f"{len(done)} tasks already in {out}; {len(todo)} to go", flush=True)
    breaks = 0
    with open(out, "a", buffering=1) as handle:
        while todo and breaks <= 3:
            finished = set()
            try:
                with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = {pool.submit(census_by_id, s, t, timeout): (s, t) for s, t in todo}
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(json.dumps(record) + "\n")
                        finished.add(futures[future])
                        print(f"[{len(done) + len(finished)}] {record['task']}", flush=True)
            except BrokenProcessPool:
                breaks += 1
                print(f"a worker was killed ({breaks}); rebuilding the pool", flush=True)
            todo = [item for item in todo if item not in finished]
    return read_records(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="evaluation", choices=("training", "evaluation"))
    parser.add_argument("--out", default="search_census.jsonl")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60, help="seconds one search may take")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    records = read_records(args.out) if args.summary_only else run(
        args.split, args.out, args.workers, args.timeout, args.limit)
    print(render(summarise([r for r in records.values() if r["split"] == args.split])))


if __name__ == "__main__":
    main()
