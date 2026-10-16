#!/usr/bin/env python3
"""How many cell strokes a task's pairs need - painted from the input, or after the object search's best path.

search_census says what the object search leaves wrong on a task it does not solve; this says what it would take to
paint that rest by coordinates (rl.coordinate_search: fills, diagonal lines, triangles, each the cells the answer
wants in one colour). Two numbers per pair:

    strokes   the coordinate solution's length from the input alone - every pair is solvable this way, so what
              matters is how many strokes, and whether the count is the same across the pairs (a rule) or grows
              with the grid (a picture copied cell by cell)
    hybrid    for the first pair, the object search's furthest partial path, and then the strokes that finish it:
              the object steps and the stroke count, which is what an action space holding both kinds would need

    python scripts/stroke_census.py --tasks 0ca9ddb6 ... --out strokes.jsonl
    python scripts/stroke_census.py --tasks ... --no-hybrid --out strokes.jsonl     # the coordinate count alone

One JSON line per task as it finishes; a task already in the file is not run again.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def strokes_for(grid_in, grid_out, colours, timeout=60):
    """(length of the shortest coordinate solution, or None, and the wrong cells) from `grid_in` to `grid_out`."""
    from rl.arc_task import ARCSubtask
    from rl.coordinate_search import CoordinateSearchSettings, search_pair
    from rl.search_hints import coordinate_vocabulary

    grid_in, grid_out = np.asarray(grid_in), np.asarray(grid_out)
    found = search_pair(ARCSubtask("pair", grid_in, grid_out), coordinate_vocabulary(colours),
                        CoordinateSearchSettings(timeout=timeout))
    solution = found["solution"]
    return (None if solution is None else len(solution)), int((grid_in != grid_out).sum())


def object_residual(pair, pairs, timeout=60):
    """(the grid the object search's furthest path reaches on `pair`, the path's length), or (None, 0) when it
    found nothing - the way search_census finds it."""
    from dataclasses import replace

    import rl.search_hints as hints
    from search_census import best_partial, final_grid

    settings = hints.SearchSettings(timeout=timeout)
    settings = replace(settings, colours=tuple(hints.output_colours(*[p[2] for p in pairs])),
                       directions=tuple(dict.fromkeys((*hints.MAIN_DIRECTIONS, *settings.directions))))
    results = hints.search_branches(pair, settings)
    if set(results) == {"base"} and not any(found["solutions"] for found in results.values()):
        results = hints.each_branch(pair, settings)
    for found in results.values():
        if found["solutions"]:
            return np.asarray(pair[2]), len(min(found["solutions"], key=len))
    partial = best_partial(results)
    if partial is None:
        return None, 0
    _closed, trace, found = partial
    return final_grid(pair, trace, found, settings.episode_len), len(trace)


def stroke_task(task_id, pairs, hybrid=True, timeout=60):
    """The record of one task."""
    import rl.search_hints as hints

    if any(np.shape(p[1]) != np.shape(p[2]) for p in pairs):
        return {"task": task_id, "skipped": "the grids change size"}
    colours = hints.output_colours(*[p[2] for p in pairs])
    record = {"task": task_id, "pairs": []}
    for _pid, grid_in, grid_out in pairs:
        length, wrong = strokes_for(grid_in, grid_out, colours, timeout)
        record["pairs"].append({"strokes": length, "wrong": wrong})
    if hybrid:
        residual, steps = object_residual(pairs[0], pairs, timeout)
        if residual is not None:
            length, wrong = strokes_for(residual, pairs[0][2], colours, timeout)
            record["hybrid"] = {"object_steps": steps, "strokes": length, "wrong_after_objects": wrong}
    return record


def stroke_by_id(split, task_id, hybrid, timeout):
    from search_census import load_tasks
    return {"split": split, **stroke_task(task_id, load_tasks(split)[task_id], hybrid, timeout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="training", choices=("training", "evaluation"))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out", default="stroke_census.jsonl")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--no-hybrid", action="store_true")
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["task"] for line in Path(args.out).read_text().splitlines() if line.strip()}
    todo = [t for t in args.tasks if t not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with open(args.out, "a", buffering=1) as handle, \
            ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1,
                                mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(stroke_by_id, args.split, t, not args.no_hybrid, args.timeout): t for t in todo}
        for future in as_completed(futures):
            record = future.result()
            handle.write(json.dumps(record) + "\n")
            print(record["task"], [p["strokes"] for p in record.get("pairs", [])], record.get("hybrid"), flush=True)


if __name__ == "__main__":
    main()
