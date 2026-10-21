#!/usr/bin/env python3
"""How many ARC tasks the shortest cell rule solves (symbolic.rule_search), with ARC's two attempts per test.

Every task whose outputs keep their inputs' shape is searched, on the training pairs alone; a task counts as solved
when each of its test inputs gets its output right in one of two attempts. The report keeps the rule found for each
task, so a solved task can be read and a wrong one seen for what it fitted.

    python scripts/rule_probe.py                         # training and evaluation, every same-shape task
    python scripts/rule_probe.py --split training --tasks 3eda0437 a699fb00
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
from symbolic import rule_search  # noqa: E402 - after the path is set

DATA = REPO_ROOT / "data" / "datasets" / "ARC"


def same_shape(challenge, solutions):
    return (all(np.shape(p["input"]) == np.shape(p["output"]) for p in challenge["train"])
            and all(np.shape(t["input"]) == np.shape(s) for t, s in zip(challenge["test"], solutions)))


def probe(task, challenge, solutions, feature_sets):
    """One task's record: the rule, and for each test input whether the first or the second attempt is right."""
    started = time.time()
    record = {"task": task, "rule": None, "feature_set": None, "first": [], "second": []}
    for feature_set in feature_sets:
        rules = rule_search.search(challenge["train"], feature_set)
        if rules:
            record.update(rule=rules[0].describe(), feature_set=feature_set, cost=rules[0].cost, rules=len(rules))
            for test, solution in zip(challenge["test"], solutions):
                guesses = rule_search.attempts(rules, test["input"])
                right = [bool(np.array_equal(g, np.asarray(solution))) for g in guesses]
                record["first"].append(bool(right[:1] and right[0]))
                record["second"].append(any(right))
            break
    record["seconds"] = round(time.time() - started, 1)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="*", default=["training", "evaluation"])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--sets", nargs="*", default=["core", "rich"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds a task may take before it counts as unsolved")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "experiments" / "rule_probe.json"))
    args = parser.parse_args()
    report = {"sets": args.sets, "splits": {}}
    for split in args.split:
        challenges = json.loads((DATA / f"{split}_challenges.json").read_text())
        solutions = json.loads((DATA / f"{split}_solutions.json").read_text())
        tasks = [t for t in (args.tasks or challenges) if t in challenges and same_shape(challenges[t], solutions[t])]
        records = []
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {t: pool.submit(probe, t, challenges[t], solutions[t], args.sets) for t in tasks}
            for task, future in futures.items():
                try:
                    records.append(future.result(timeout=args.timeout))
                except FutureTimeout:
                    records.append({"task": task, "rule": None, "first": [], "second": [], "timed_out": True})
        solved = [r["task"] for r in records if r["second"] and all(r["second"])]
        first = [r["task"] for r in records if r["first"] and all(r["first"])]
        fitted = [r["task"] for r in records if r["rule"]]
        report["splits"][split] = {"tasks": len(challenges), "same_shape": len(tasks), "rule_found": len(fitted),
                                   "solved_first": len(first), "solved_two": len(solved), "solved": solved,
                                   "records": records}
        print(f"{split}: {len(challenges)} tasks, {len(tasks)} same shape, a rule for {len(fitted)}, "
              f"solved {len(first)} at the first attempt and {len(solved)} in two", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
