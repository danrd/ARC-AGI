#!/usr/bin/env python3
"""Why the cell rules (behind a transformation) fail on the tasks they fail on - symbolic.rule_diagnosis over every task.

Each task falls in one class:
  rule_right            a rule is found and its first answer is the test output
  rule_wrong:<kind>     a rule is found and wrong: false_change, missed, wrong_colour, mixed or shape (see rule_diagnosis)
  no_rule               every cell follows from its features (no conflicts) and no rule over them is found: the grammar
  ambiguous             cells with the same features need different colours: the features
  no_transformation     no transformation gives the inputs their outputs' shape: the library of transformations
The conflict is measured on the largest feature set (rich_ext) after the reachable transformation with fewest conflicts.

    python scripts/failure_probe.py --shard 0 --shards 5 --workers 4 --out failures.json
"""
import argparse
import json
import resource
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from symbolic import rule_diagnosis, rule_search, transformed_rules  # noqa: E402 - after the path is set

DATA = REPO_ROOT / "data" / "datasets" / "ARC"
MEMORY_LIMIT = 6 * 2 ** 30
CONFLICT_SET = "rich_ext"


def least_conflict(reach):
    """(share of conflicting cells, transformation) of the reachable transformation whose features separate the cells
    best, or (None, None) when none is reachable."""
    best = (None, None)
    for name, pairs, _ in reach:
        x, _, _ = rule_search.table([p["input"] for p in pairs], CONFLICT_SET)
        targets = np.concatenate([p["output"].ravel() for p in pairs])
        share = rule_diagnosis.conflicts(x, targets)
        if best[0] is None or share < best[0]:
            best = (share, name)
    return best


def diagnose(task, challenge, solution, feature_sets, seconds, relational=True):
    """One task's record: its class, and what it was decided by."""
    started = time.time()
    rule_search.RELATIONAL = relational
    train, test_input = challenge["train"], np.asarray(challenge["test"][0]["input"])
    truth = np.asarray(solution[0])
    record = {"task": task, "class": None}
    reach = transformed_rules.reachable(train, test_input)
    if not reach:
        record["class"] = "no_transformation"
    else:
        share, name = least_conflict(reach)
        record.update(conflict=round(share, 4), conflict_transform=name, reachable=[n for n, _, _ in reach][:8])
        found = transformed_rules.candidates(train, test_input, feature_sets, seconds)
        guesses = transformed_rules.attempts(found, 1) if found else []
        if guesses:
            best = found[0]
            record.update(transform=best.transform, cost=best.cost, rule=best.rules[0].describe())
            if guesses[0].shape == truth.shape and np.array_equal(guesses[0], truth):
                record["class"] = "rule_right"
            else:
                counts = rule_diagnosis.error_kinds(guesses[0], truth, best.test_input)
                record.update(errors=counts)
                record["class"] = "rule_wrong:" + rule_diagnosis.kind_of(counts)
        else:
            record["class"] = "no_rule" if share == 0.0 else "ambiguous"
    record["seconds"] = round(time.time() - started, 1)
    return record


def probe(task, challenge, solution, feature_sets, seconds):
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    try:
        return diagnose(task, challenge, solution, feature_sets, seconds)
    except MemoryError:
        return {"task": task, "class": "out_of_memory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="*", default=["training", "evaluation"])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--sets", nargs="*", default=["core", "rich", "rich_ext"])
    parser.add_argument("--search-seconds", type=float, default=400.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=700.0)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "experiments" / "failure_probe.json"))
    args = parser.parse_args()
    report = {"sets": args.sets, "shard": [args.shard, args.shards], "splits": {}}
    for split in args.split:
        challenges = json.loads((DATA / f"{split}_challenges.json").read_text())
        solutions = json.loads((DATA / f"{split}_solutions.json").read_text())
        tasks = [t for t in (args.tasks or challenges) if t in challenges][args.shard::args.shards]
        records = []
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {t: pool.submit(probe, t, challenges[t], solutions[t], args.sets, args.search_seconds) for t in tasks}
            for task, future in futures.items():
                try:
                    records.append(future.result(timeout=args.timeout))
                except FutureTimeout:
                    records.append({"task": task, "class": "timed_out"})
        counts = Counter(r["class"] for r in records)
        report["splits"][split] = {"tasks": len(tasks), "classes": dict(counts), "records": records}
        print(f"{split}: {len(tasks)} tasks " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
