#!/usr/bin/env python3
"""Which tasks each way of solving solves, and what the ways add to one another.

The symbolic solvers, the LLM, the search and the RL agents are measured by
separate scripts that write separate files, each on the tasks it was run on.
This reads those files as they are and puts them side by side over one split:

    solved by any way                    the ceiling if the right way could be picked
    solved only by one way               what each adds that no other does
    solved by two ways                   what they duplicate
    solved by nothing, by grade          where the work is

It runs nothing and changes no file, so the way each direction produces its
results can change without this being touched: a source is a small function
from a file to {attempted, solved, claimed-but-not-solved}, and a new source
is one more function.

What "solved" means differs by source, and this keeps the differences visible
instead of averaging over them:

    mixer, upscale_or_covering, color_restore   kept by checked_solve and equal
                                                to the test answer
    llm                                         a run whose grid equals the
                                                target (primary_score 1), in
                                                any run of the export
    rl                                          held_out 1.0 in any run
    search                                      only "explained": the search
                                                reproduced every training pair,
                                                which says nothing of the test
                                                pair, so it is never counted
                                                as solved

"Any run" is generous to the sources with several runs, and the union assumes
something that picks the right source for each task, which nothing does yet.
It is a ceiling.

    python scripts/solved_by_source.py --split evaluation \\
        --symbolic sym_eval.jsonl --llm wandb_export.csv --search hints.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "datasets" / "ARC"
GRADES = ("easy", "medium", "hard", "very_hard", "impossible", "symbolic", "oversized")


@dataclass
class Source:
    """One way of solving, as far as its file says.

    `attempted` is what it was run on, `solved` what it got right, `claimed`
    what it answered without being right (or cannot be shown to be right)."""
    name: str
    attempted: set = field(default_factory=set)
    solved: set = field(default_factory=set)
    claimed: set = field(default_factory=set)
    note: str = ""


# ---------------------------------------------------------------------------
# reading the files
# ---------------------------------------------------------------------------

def read_symbolic(path):
    """One Source per solver from scripts/symbolic_coverage.py's lines."""
    sources = {}
    for line in open(path):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        for name, flags in record["solvers"].items():
            source = sources.setdefault(name, Source(name, note="kept by checked_solve and correct"))
            source.attempted.add(record["task"])
            if flags["kept_correct"]:
                source.solved.add(record["task"])
            elif flags["claimed"]:
                source.claimed.add(record["task"])
    return list(sources.values())


def read_llm(path):
    """The wandb export: task_id, primary_score, one row per run. Solved when
    any run's grid matched the target exactly."""
    source = Source("llm", note="any run exactly right")
    rates = collections.defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rates[row["task_id"]].append(float(row["primary_score"]) == 1.0)
    for task, hits in rates.items():
        source.attempted.add(task)
        if any(hits):
            source.solved.add(task)
    return [source]


def read_rl(path):
    """JSON lines with `task` and `held_out`; solved when any run closed the
    held-out pair completely."""
    source = Source("rl", note="held-out pair closed in any run")
    for line in open(path):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        source.attempted.add(record["task"])
        if float(record["held_out"]) >= 1.0:
            source.solved.add(record["task"])
    return [source]


def read_search(path):
    """{task: verified hint text or null}: the search explained every
    training pair. Recorded as claimed, never as solved."""
    source = Source("search", note="explained the training pairs; not a test prediction")
    for task, text in json.load(open(path)).items():
        source.attempted.add(task)
        if text:
            source.claimed.add(task)
    return [source]


READERS = {"symbolic": read_symbolic, "llm": read_llm, "rl": read_rl, "search": read_search}


# ---------------------------------------------------------------------------
# the universe and the grades
# ---------------------------------------------------------------------------

def universe(split):
    with open(DATA / f"{split}_challenges.json") as handle:
        return list(json.load(handle))


def grades(split):
    """{task: grade} where the split has a hand grading, else {}."""
    path = DATA / f"{split}_difficulty.json"
    return json.load(open(path)) if path.exists() else {}


# ---------------------------------------------------------------------------
# putting them together
# ---------------------------------------------------------------------------

def combine(sources, tasks):
    """The numbers of a report over `tasks`, as plain data."""
    tasks = set(tasks)
    solved = {source.name: source.solved & tasks for source in sources}
    union = set().union(*solved.values()) if solved else set()
    only = {name: found - set().union(*(other for n, other in solved.items() if n != name))
            for name, found in solved.items()}
    overlap = {(a, b): len(solved[a] & solved[b]) for a, b in itertools.combinations(solved, 2)}
    return {"tasks": len(tasks),
            "attempted": {s.name: len(s.attempted & tasks) for s in sources},
            "solved": {name: len(found) for name, found in solved.items()},
            "claimed": {s.name: len(s.claimed & tasks) for s in sources},
            "union": len(union), "only": {name: sorted(found) for name, found in only.items()},
            "overlap": overlap, "unsolved": sorted(tasks - union), "solved_by_any": sorted(union)}


def by_grade(report, grade_of):
    """{grade: (tasks, solved by any)} - only when there is a grading."""
    if not grade_of:
        return {}
    solved = set(report["solved_by_any"])
    rows = collections.defaultdict(lambda: [0, 0])
    for task in set(report["unsolved"]) | solved:
        grade = grade_of.get(task, "?")
        rows[grade][0] += 1
        rows[grade][1] += task in solved
    order = [g for g in GRADES if g in rows] + sorted(g for g in rows if g not in GRADES)
    return {grade: tuple(rows[grade]) for grade in order}


def render(report, sources, grade_of, split):
    lines = [f"{split}: {report['tasks']} tasks", "",
             f"{'source':<22}{'attempted':>10}{'solved':>8}{'claimed':>9}{'only this':>11}  what solved means"]
    for source in sources:
        name = source.name
        lines.append(f"{name:<22}{report['attempted'][name]:>10}{report['solved'][name]:>8}"
                     f"{report['claimed'][name]:>9}{len(report['only'][name]):>11}  {source.note}")
    lines += ["", f"solved by at least one source: {report['union']} of {report['tasks']} "
                  f"({100 * report['union'] / max(report['tasks'], 1):.1f}%) - a ceiling, "
                  f"see the top of this script"]
    if report["overlap"]:
        lines += ["", "solved by both:"]
        lines += [f"  {a} & {b}: {n}" for (a, b), n in report["overlap"].items() if n]
    graded = by_grade(report, grade_of)
    if graded:
        lines += ["", f"{'grade':<12}{'tasks':>6}{'solved':>8}{'left':>6}"]
        lines += [f"{grade:<12}{total:>6}{done:>8}{total - done:>6}" for grade, (total, done) in graded.items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="evaluation", choices=("training", "evaluation"))
    for name in READERS:
        parser.add_argument(f"--{name}", help=f"the {name} results file")
    parser.add_argument("--json", help="also write the numbers to this file")
    args = parser.parse_args()
    sources = []
    for name, reader in READERS.items():
        path = getattr(args, name)
        if path:
            sources += reader(path)
    if not sources:
        parser.error("give at least one results file")
    tasks = universe(args.split)
    report = combine(sources, tasks)
    print(render(report, sources, grades(args.split), args.split))
    if args.json:
        report["overlap"] = {f"{a} & {b}": n for (a, b), n in report["overlap"].items()}
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
