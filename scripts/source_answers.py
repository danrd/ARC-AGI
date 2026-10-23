#!/usr/bin/env python3
"""The grid each source gives for each evaluation task, and what choosing among them gives (orchestration.source_selection).

The sources' saved runs hold only whether an answer was right, so the answers are made again: the symbolic modules that
were kept by scripts/symbolic_coverage.py (checked_solve), the cell rules behind a transformation where
scripts/transform_probe.py found one, and the language model's saved replies. They are written to a file once, then
the strategies are scored on it.

    python scripts/source_answers.py --workers 4                       # make the answers, score the strategies
    python scripts/source_answers.py --answers data/experiments/source_answers_eval.json --score-only
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from orchestration import source_selection as sel  # noqa: E402 - after the path is set

RUNS = REPO_ROOT / "data/experiments/runs/kaggle"
DATA = REPO_ROOT / "data/datasets/ARC"
MODULES = ("mixer", "upscale_or_covering", "color_restore")
SPLIT = "evaluation"


def module_answer(task_id, instance_of):
    """The grid of the first symbolic module kept for the task, made again."""
    from scripts.symbolic_coverage import load_task, solvers_for
    from symbolic.symbolic_module import checked_solve

    task = load_task(SPLIT, task_id)
    solvers = solvers_for(task)
    for name in MODULES:
        if name in instance_of:
            result = checked_solve(solvers[name][instance_of[name]], task)
            if result.success:
                return task_id, np.asarray(result.grid).tolist()
    return task_id, None


#: Memory a rule search may take before it is given up as a MemoryError, not left to the kernel to kill (a worker that
#: is killed takes the whole pool with it; one task whose search is too big for this machine has no answer here).
MEMORY_LIMIT = 4 * 2 ** 30


def rule_answer(task_id):
    """The first answer and the length of the cheapest rule behind a transformation (as scripts/transform_probe.py);
    None for a task whose search did not fit in memory."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    try:
        return _rule_answer(task_id)
    except MemoryError:
        return task_id, None


def _rule_answer(task_id):
    from symbolic import rule_search, transformed_rules

    rule_search.RELATIONAL = True
    challenge = json.loads((DATA / f"{SPLIT}_challenges.json").read_text())[task_id]
    found = transformed_rules.candidates(challenge["train"], np.asarray(challenge["test"][0]["input"]),
                                         ("core", "rich", "rich_ext"), 400.0)
    guesses = transformed_rules.attempts(found) if found else []
    return task_id, (guesses[0].tolist(), found[0].cost) if guesses else None


def llm_answers():
    from orchestration.llm_only import read_grid
    from orchestration.rule_report import LLM_RUN

    out = {}
    for line in Path(LLM_RUN).read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if row and row["split"] == SPLIT:
            grid = read_grid(row.get("reply") or "")
            if grid is not None:
                out[row["task"]] = np.asarray(grid).tolist()
    return out


def make_answers(workers):
    from scripts.rl_compare import read_lines

    kept, found = {}, []
    for path in glob.glob(str(RUNS / "symcov_a_*.jsonl")):
        for record in read_lines(path):
            if record["split"] == SPLIT:
                kept[record["task"]] = {n: f["instance"] for n, f in record["solvers"].items()
                                        if n in MODULES and f["kept"]}
    for path in glob.glob(str(RUNS / "transform_a_transform_*.json")):
        found += [r["task"] for r in json.loads(Path(path).read_text())["splits"][SPLIT]["records"] if r["rule"]]
    answers = {"modules": {}, "rules": {}, "llm": llm_answers()}
    with ProcessPoolExecutor(workers) as pool:
        for task, grid in pool.map(module_answer, [t for t, k in kept.items() if k], [k for k in kept.values() if k]):
            if grid is not None:
                answers["modules"][task] = grid
        for task, got in pool.map(rule_answer, found):
            if got is not None:
                answers["rules"][task] = got
    return answers


def per_task(answers):
    out = {}
    for source in sel.ORDER:
        for task, value in answers[source].items():
            grid, cost = (value if source == "rules" else (value, None))
            out.setdefault(task, {})[source] = (np.asarray(grid), cost)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default=str(REPO_ROOT / "data/experiments/source_answers_eval.json"))
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.score_only:
        Path(args.answers).write_text(json.dumps(make_answers(args.workers)) + "\n")
    answers = json.loads(Path(args.answers).read_text())
    solutions = {t: np.asarray(s[0]) for t, s in json.loads((DATA / f"{SPLIT}_solutions.json").read_text()).items()}
    tasks = per_task(answers)
    print({k: len(v) for k, v in answers.items()}, "tasks with any answer:", len(tasks), "of", len(solutions))
    for source in sel.ORDER:
        ok = sum(sel.same(np.asarray(v[source][0]), solutions[t]) for t, v in tasks.items() if source in v)
        print(f"  {source}: {ok} right of {len(answers[source])} answered")
    for limit in (2, 3, 4, float("inf")):
        for name in sel.STRATEGIES:
            r = sel.score(tasks, solutions, name, limit)
            print(f"rule length <= {limit}: {name:8} right {r['right']:3} of {r['answered']:3} answered  {r['by_source']}")


if __name__ == "__main__":
    main()
