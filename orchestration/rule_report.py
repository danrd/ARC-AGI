#!/usr/bin/env python3
"""What the rule probe's runs add up to: variants side by side, and the rules together with the LLM.

    python scripts/rule_report.py                   # every variant found under data/experiments/runs/kaggle
    python scripts/rule_report.py --gate            # the confidence gate: rules when cheap, the LLM otherwise

A variant is one rule_probe run, in one file or in shards of it (JOB_STEP_K.json from the Kaggle queue, or
data/experiments/rule_probe.json); the report merges the shards of a variant and counts, per split, the tasks it found a
rule for and solved at the first attempt and in two.

The gate asks what the rules and the LLM give together without knowing the answer: a task whose cheapest rule costs at
most C takes the rule's answer (right or wrong), any other takes the LLM's. The LLM's answers are the saved replies of
llm_only_gemma31b_0.jsonl, read with the lenient grader; they cover the evaluation split only.
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
RUNS = REPO_ROOT / "data" / "experiments" / "runs" / "kaggle"
LLM_RUN = RUNS / "llm_only_gemma31b_0.jsonl"
COSTS = (2.0, 3.0, 4.0, 5.0, 6.0, float("inf"))


def variants(paths):
    """label -> {split: {task: record}} merged over the shard files of each variant. A file's label is its name without
    the shard number: tpu-style JOB_STEP_K.json -> JOB_STEP; anything else its own stem."""
    found = defaultdict(lambda: defaultdict(dict))
    for path in sorted(paths):
        name = Path(path).stem
        label = re.sub(r"_\d+$", "", name)
        report = json.loads(Path(path).read_text())
        for split, body in report["splits"].items():
            for record in body["records"]:
                found[label][split][record["task"]] = record
    return found


def tally(records):
    """(tasks with a rule, solved at the first attempt, solved in two) over one split's records."""
    fitted = [r for r in records.values() if r.get("rule")]
    first = [r for r in fitted if r["first"] and all(r["first"])]
    two = [r for r in fitted if r["second"] and all(r["second"])]
    return len(fitted), len(first), len(two)


def gate(records, llm_solved, asked, costs=COSTS):
    """For each cost limit C: how many tasks are solved when a task whose cheapest rule costs at most C is answered by
    the rule and every other task by the LLM. `llm_solved` is the set of tasks the LLM got right, `asked` those it was
    asked. Returns [(C, taken_from_rules, rules_right, llm_right_on_the_rest, solved)]."""
    rows = []
    for limit in costs:
        confident = {t for t, r in records.items() if r.get("rule") and r.get("cost", float("inf")) <= limit}
        rule_right = {t for t in confident if records[t]["second"] and all(records[t]["second"])}
        llm_right = {t for t in asked - confident if t in llm_solved}
        rows.append((limit, len(confident), len(rule_right), len(llm_right), len(rule_right) + len(llm_right)))
    return rows


def llm_solutions(path=LLM_RUN):
    """({(split, task)} the LLM solved, {(split, task)} it was asked) from its saved replies."""
    import numpy as np
    from orchestration.llm_only import read_grid

    answers = {split: json.loads((REPO_ROOT / "data" / "datasets" / "ARC" / f"{split}_solutions.json").read_text())
               for split in ("training", "evaluation")}
    solved, asked = set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        asked.add((row["split"], row["task"]))
        grid = read_grid(row.get("reply") or "")
        if grid is not None and np.array_equal(np.asarray(grid), np.asarray(answers[row["split"]][row["task"]][0])):
            solved.add((row["split"], row["task"]))
    return solved, asked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", help="rule probe reports (default: every one found)")
    parser.add_argument("--gate", action="store_true", help="rules when cheap, the LLM otherwise")
    parser.add_argument("--label", default="rule_probe", help="the variant the gate uses")
    args = parser.parse_args()
    paths = args.files or [str(REPO_ROOT / "data" / "experiments" / "rule_probe.json"),
                           *glob.glob(str(RUNS / "*rules*_*.json"))]
    found = variants(p for p in paths if Path(p).exists())
    print(f"{'variant':40} {'split':11} {'tasks':>6} {'a rule':>7} {'first':>6} {'in two':>7}")
    for label, splits in sorted(found.items()):
        for split, records in splits.items():
            fitted, first, two = tally(records)
            print(f"{label:40} {split:11} {len(records):>6} {fitted:>7} {first:>6} {two:>7}")
    if args.gate:
        llm_solved, llm_asked = llm_solutions()
        records = found[args.label]["evaluation"]
        solved = {t for s, t in llm_solved if s == "evaluation"}
        asked = {t for s, t in llm_asked if s == "evaluation"}
        print(f"\nthe gate on evaluation ({len(asked)} tasks asked of the LLM, which solved {len(solved)}; "
              f"the rules alone solve {tally(records)[2]}):")
        print(f"{'cost <=':>9} {'by rule':>8} {'right':>6} {'LLM on the rest':>16} {'solved':>7}")
        for limit, taken, right, rest, total in gate(records, solved, asked):
            print(f"{limit:>9} {taken:>8} {right:>6} {rest:>16} {total:>7}")


if __name__ == "__main__":
    main()
