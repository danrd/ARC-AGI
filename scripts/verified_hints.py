#!/usr/bin/env python3
"""Write verified search hints for a split into the file the prompt reads.

A verified hint (rl.search_hints.verified_hint) is said only when a search
reproduced every training pair of a task with the same kinds of step; any
other task gets no block. The hints that were in the prompt before said
something on every task - a partial attempt, a list of single steps - and
with them and the summary in, a run went from 41 solved to 34.

The output is what subsymbolic.arc_resolvers.search_hints_resolver reads
through `project.search_hints`: task id -> text. A task that was checked
and has nothing verified maps to null, which the resolver reads as no
block, and which is also how a rerun knows the task is done - the file is
rewritten after every task, so a run cut short resumes where it stopped.

A task takes about a minute on one core (a search per agent roster on the
first pair, then one small search per further pair), so a split of 400 is
a couple of hours on four workers.

    python scripts/verified_hints.py --split evaluation --workers 4 \\
        --out data/search_hints_verified.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "datasets" / "ARC"


def load_split(split):
    """task id -> its training pairs as (id, input, output) triples."""
    import numpy as np

    with open(DATA / f"{split}_challenges.json") as handle:
        challenges = json.load(handle)
    return {task_id: [(f"{task_id}_{index}", np.array(pair["input"]), np.array(pair["output"]))
                      for index, pair in enumerate(task["train"])]
            for task_id, task in challenges.items()}


def hint_for(task_id, pairs, timeout):
    """(task id, verified text or None, seconds), in a worker."""
    import rl.search_hints as hints

    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            text = hints.verified_hint(pairs, hints.SearchSettings(timeout=timeout))
        except Exception as error:  # noqa: BLE001 - one task must not sink the split
            return task_id, None, time.perf_counter() - started, repr(error)
    return task_id, text, time.perf_counter() - started, None


def write(path, hints):
    """Rewrite the whole file, atomically: a run killed mid-write must not
    leave a file the resolver cannot parse."""
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(hints, handle, indent=1, sort_keys=True)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="evaluation", choices=("training", "evaluation"))
    parser.add_argument("--out", default="data/search_hints_verified.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60,
                        help="seconds one search may take")
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    args = parser.parse_args(argv)

    tasks = load_split(args.split)
    ids = sorted(tasks)[:args.limit or None]
    hints = {}
    if os.path.exists(args.out):
        with open(args.out) as handle:
            hints = json.load(handle)
    todo = [task_id for task_id in ids if task_id not in hints]
    print(f"{len(ids) - len(todo)} of {len(ids)} already in {args.out}; {len(todo)} to go")

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(hint_for, task_id, tasks[task_id], args.timeout)
                   for task_id in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            task_id, text, seconds, error = future.result()
            if error is not None:
                print(f"[{done}/{len(todo)}] {task_id} failed: {error}", flush=True)
                continue
            hints[task_id] = text
            write(args.out, hints)
            verified = sum(value is not None for value in hints.values())
            print(f"[{done}/{len(todo)}] {task_id} {'verified' if text else '-'} "
                  f"({seconds:.0f}s; {verified} of {len(hints)} verified so far)", flush=True)


if __name__ == "__main__":
    main()
