#!/usr/bin/env python3
"""Write the search hints for a split into the file the prompt reads.

A hint (rl.search_hints.hints_for) is said only when a search reproduced
every training pair of a task with the same kinds of step; any other task
gets no block.

The output is what subsymbolic.arc_resolvers.search_hints_resolver reads
through `project.search_hints`, data/search_hints.json unless a config
says otherwise: task id -> text. A task that was checked
and has nothing verified maps to null, which the resolver reads as no
block, and which is also how a rerun knows the task is done - the file is
rewritten after every task, so a run cut short resumes where it stopped.

A task takes about a minute on one core (a search per agent roster on the
first pair, then one small search per further pair). Two workers by
default: a worker has been measured at 4.8 GB on one task, and four of
them were killed for memory in a 15 GB container. Each worker runs one
task and is replaced, so a large task's memory is returned rather than
carried into the next.

A worker killed anyway breaks the whole pool, and every task still in it
comes back unfinished. They are all tried again in a fresh pool, since
which of them was the one to blame cannot be told; a task unfinished
through BREAKS_ALLOWED breaks is written as null - no block - so one task
that cannot fit does not stop the split.

    python scripts/write_search_hints.py --split evaluation --workers 4
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
from concurrent.futures.process import BrokenProcessPool
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
            text = hints.hints_for(pairs, hints.SearchSettings(timeout=timeout))
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
    parser.add_argument("--out", default="data/search_hints.json")
    parser.add_argument("--workers", type=int, default=2)
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

    run(todo, tasks, hints, args)


#: How many broken pools a task may be caught in before it is given up on.
BREAKS_ALLOWED = 3


def run(todo, tasks, hints, args):
    """Every task in `todo` through a pool of workers, into `hints` and the
    file, surviving a worker that is killed (see the module docstring)."""
    breaks = {task_id: 0 for task_id in todo}
    done = 0

    def record(future, unfinished):
        nonlocal done
        task_id, text, seconds, error = future.result()
        unfinished.discard(task_id)
        done += 1
        if error is not None:
            print(f"[{done}] {task_id} failed: {error}", flush=True)
            return
        hints[task_id] = text
        write(args.out, hints)
        verified = sum(value is not None for value in hints.values())
        print(f"[{done}] {task_id} {'verified' if text else '-'} "
              f"({seconds:.0f}s; {verified} of {len(hints)} verified so far)", flush=True)

    while todo:
        unfinished, futures = set(todo), []
        try:
            with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(hint_for, task_id, tasks[task_id], args.timeout)
                           for task_id in todo]
                for future in as_completed(futures):
                    record(future, unfinished)
        except BrokenProcessPool:
            # Futures finish in no particular order, so the broken one can
            # be reached before others that had already finished; those
            # are results, not casualties.
            for future in futures:
                if future.done() and not future.cancelled() and future.exception() is None \
                        and future.result()[0] in unfinished:
                    record(future, unfinished)
            for task_id in unfinished:
                breaks[task_id] += 1
                if breaks[task_id] >= BREAKS_ALLOWED:
                    print(f"{task_id} was unfinished through {BREAKS_ALLOWED} broken "
                          f"pools; written without a hint", flush=True)
                    hints[task_id] = None
            write(args.out, hints)
            print(f"a worker was killed; {len(unfinished)} unfinished tasks go to a "
                  f"fresh pool", flush=True)
        todo = [task_id for task_id in todo if task_id in unfinished and task_id not in hints]

if __name__ == "__main__":
    main()
