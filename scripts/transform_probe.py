#!/usr/bin/env python3
"""How many ARC tasks a whole-grid transformation followed by a cell rule solves (symbolic.transformed_rules).

scripts/rule_probe.py only reaches the tasks whose output keeps the input's shape. This tries, on every task, each
transformation of symbolic.grid_transforms that gives the inputs their outputs' shapes, with the cell rule search on what
is left, and records which transformation and rule were found and whether the two attempts are right. The identity is one
of the transformations, so a task the plain search solves is found here too; "transformed" counts the solved tasks whose
rule sits behind another transformation, which is what this adds.

    python scripts/transform_probe.py --shard 0 --shards 5 --workers 4 --out transform_probe.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from symbolic import rule_search, transformed_rules  # noqa: E402 - after the path is set

DATA = REPO_ROOT / "data" / "datasets" / "ARC"


def probe(task, challenge, solutions, feature_sets, seconds, relational=True):
    """One task's record: every transformation with a rule, the one used, and whether each test input is right."""
    started = time.time()
    rule_search.RELATIONAL = relational
    record = {"task": task, "transform": None, "rule": None, "first": [], "second": [], "found": []}
    first, second = [], []
    for test, solution in zip(challenge["test"], solutions):
        found = transformed_rules.candidates(challenge["train"], np.asarray(test["input"]), feature_sets, seconds)
        if not record["found"]:
            record["found"] = [{"transform": c.transform, "feature_set": c.feature_set, "cost": round(c.cost, 2)}
                               for c in found]
            if found:
                record.update(transform=found[0].transform, rule=found[0].rules[0].describe(), cost=found[0].cost)
        guesses = transformed_rules.attempts(found) if found else []
        right = [bool(g.shape == np.shape(solution) and np.array_equal(g, solution)) for g in guesses]
        first.append(bool(right[:1] and right[0]))
        second.append(any(right))
    record.update(first=first, second=second, seconds=round(time.time() - started, 1))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="*", default=["training", "evaluation"])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--sets", nargs="*", default=["core", "rich", "rich_ext"])
    parser.add_argument("--no-relational", action="store_true")
    parser.add_argument("--search-seconds", type=float, default=600.0,
                        help="how long the searches of one task may take altogether before the rest are left out")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds a task may take before it counts as unsolved")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "experiments" / "transform_probe.json"))
    args = parser.parse_args()
    report = {"sets": args.sets, "relational": not args.no_relational, "shard": [args.shard, args.shards], "splits": {}}
    for split in args.split:
        challenges = json.loads((DATA / f"{split}_challenges.json").read_text())
        solutions = json.loads((DATA / f"{split}_solutions.json").read_text())
        tasks = [t for t in (args.tasks or challenges) if t in challenges][args.shard::args.shards]
        records = []
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {t: pool.submit(probe, t, challenges[t], solutions[t], args.sets, args.search_seconds,
                                      not args.no_relational) for t in tasks}
            for task, future in futures.items():
                try:
                    records.append(future.result(timeout=args.timeout))
                except FutureTimeout:
                    records.append({"task": task, "rule": None, "first": [], "second": [], "timed_out": True})
        solved = [r["task"] for r in records if r["second"] and all(r["second"])]
        moved = [r["task"] for r in records if r["task"] in solved and r["transform"] != "identity"]
        fitted = [r["task"] for r in records if r["rule"]]
        report["splits"][split] = {"tasks": len(challenges), "searched": len(tasks), "rule_found": len(fitted),
                                   "solved_two": len(solved), "transformed": moved, "solved": solved, "records": records}
        print(f"{split}: {len(tasks)} tasks searched, a rule for {len(fitted)}, solved in two {len(solved)}, "
              f"of them behind a transformation {len(moved)}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
