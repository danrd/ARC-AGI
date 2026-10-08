#!/usr/bin/env python3
"""How much a model-free check of an answer can tell, measured on ARC itself.

symbolic.invariants asks of an answer what every training pair does. Whether that
is worth anything is a question about numbers, so this asks it of answers whose
truth is known:

    true      the task's real test output. A check that refuses these is wrong, and
              the share refused (strong) is its false-refusal rate
    copy      the test input handed back unchanged - what a model that gives up writes
    other     the real output of another task, whatever shape it has
    nudged    the real output with 1-3 cells recoloured with a colour the output
              already has: a near miss, the hardest wrong answer to see without the rule

For each, the share the check refuses (a strong violation) and the share it at least
flags (any violation, weak ones too). Only tasks with a single test output are used.

    python scripts/check_invariants.py --split training
    python scripts/check_invariants.py --split evaluation --seed 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from symbolic.invariants import check, learn, refuses  # noqa: E402

DATA = ROOT / "data" / "datasets" / "ARC"


def tasks(split):
    challenges = json.loads((DATA / f"{split}_challenges.json").read_text())
    solutions = json.loads((DATA / f"{split}_solutions.json").read_text())
    for task_id in sorted(challenges):
        data = challenges[task_id]
        pairs = [(np.array(p["input"]), np.array(p["output"])) for p in data["train"]]
        for number, test in enumerate(data["test"]):
            yield task_id, pairs, np.array(test["input"]), np.array(solutions[task_id][number])


def nudged(answer, rng):
    """`answer` with 1-3 cells given another colour of its own palette (itself, when it has one colour)."""
    out = answer.copy()
    palette = [int(v) for v in np.unique(answer)]
    for _ in range(int(rng.integers(1, 4))):
        r, c = int(rng.integers(out.shape[0])), int(rng.integers(out.shape[1]))
        others = [v for v in palette if v != int(out[r, c])]
        if others:
            out[r, c] = others[int(rng.integers(len(others)))]
    return out


def measure(split, seed=0):
    rng = np.random.default_rng(seed)
    items = list(tasks(split))
    counts = {kind: [0, 0, 0] for kind in ("true", "copy", "other", "nudged")}      # n, refused, flagged
    for number, (task_id, pairs, test_input, truth) in enumerate(items):
        invariants = learn(pairs)
        other = items[(number + 1 + int(rng.integers(len(items) - 1))) % len(items)][3]
        for kind, answer in (("true", truth), ("copy", test_input), ("other", other), ("nudged", nudged(truth, rng))):
            found = check(invariants, test_input, answer)
            row = counts[kind]
            row[0] += 1
            row[1] += refuses(found)
            row[2] += bool(found)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["training", "evaluation"], default="training")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"{args.split}: share of answers refused (strong violation) and flagged (any)")
    for kind, (n, refused, flagged) in measure(args.split, args.seed).items():
        print(f"  {kind:<8}{n:>5} answers   refused {refused / n:6.1%}   flagged {flagged / n:6.1%}")


if __name__ == "__main__":
    main()
