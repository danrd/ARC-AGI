#!/usr/bin/env python3
"""Send the prompt variants of prompt_variants.py to a hosted model and keep the answers.

prompt_variants.py writes <dir>/<task>/<variant>[_perm].txt for a person to paste
into a model. This does the pasting: every prompt goes to each model through
OpenRouter, the reply is kept beside the prompt as
<variant>[_perm].<model>.response.txt for reading, and compared with
answer[_perm].txt. One line per reply goes to <dir>/results.jsonl and a table of
task by variant to <dir>/report.md. A reply that is already on disk is not asked
for again, so a run that stops is continued by running it again.

    python scripts/run_prompt_variants.py --out prompt_tests --models google/gemma-4-26b-a4b-it

The key is read from OPENROUTER_API_KEY. Prompts not yet written are written first,
with the colours as they are and shuffled. `solved` is the grid equal to the answer
and nothing else: a reply that does not parse is `parsed: false`, not a near miss.
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subsymbolic.utils import parse_llm_output  # noqa: E402


def slug(model):
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def read_answer(path):
    """The grid of an answer file: 'grid shape: n,m' and then 'row-number cells' lines."""
    lines = Path(path).read_text().strip().splitlines()
    return np.array([[int(c) for c in line.split()[1]] for line in lines[1:]])


def grade(reply, answer):
    """{'parsed', 'solved', 'shape_ok', 'cells_right'} of a reply against the answer grid."""
    grid = parse_llm_output(_without_thinking(reply)) if reply else ""
    if not isinstance(grid, np.ndarray):
        return {"parsed": False, "solved": False, "shape_ok": False, "cells_right": 0.0}
    same_shape = grid.shape == answer.shape
    return {"parsed": True, "solved": bool(same_shape and (grid == answer).all()), "shape_ok": same_shape,
            "cells_right": round(float((grid == answer).mean()), 4) if same_shape else 0.0}


def _without_thinking(reply):
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.S)


def prompts(directory, tasks=None, variants=None):
    """(task, variant, perm, prompt path, answer path) for every prompt on disk."""
    found = []
    for path in sorted(Path(directory).glob("*/*.txt")):
        name = path.stem
        if name.startswith("answer") or "." in name:
            continue
        perm = name.endswith("_perm")
        variant = name[:-5] if perm else name
        if (tasks and path.parent.name not in tasks) or (variants and variant not in variants):
            continue
        found.append((path.parent.name, variant, perm, path,
                      path.parent / ("answer_perm.txt" if perm else "answer.txt")))
    return found


def response_path(prompt_path, model):
    return prompt_path.with_name(f"{prompt_path.stem}.{slug(model)}.response.txt")


def run(directory, models, make_runner, tasks=None, variants=None, workers=4):
    """Ask every model every prompt that has no reply yet; returns the records of the new replies."""
    runners = {model: make_runner(model) for model in models}
    work = [(model, *item) for model in models for item in prompts(directory, tasks, variants)
            if not response_path(item[3], model).exists()]
    results = Path(directory) / "results.jsonl"

    def one(job):
        model, task, variant, perm, prompt_path, answer_path = job
        try:
            reply = runners[model].generate(prompt_path.read_text())
        except Exception as error:  # noqa: BLE001 - one failed request must not end the run
            print(f"{task} {variant} {model}: {error}", file=sys.stderr)
            return None
        response_path(prompt_path, model).write_text(reply or "")
        return {"task": task, "variant": variant, "perm": perm, "model": model,
                **grade(reply, read_answer(answer_path))}

    records = []
    with ThreadPoolExecutor(workers) as pool, open(results, "a") as handle:
        for record in pool.map(one, work):
            if record:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
    return records


def report(directory):
    """A table, task by variant, of solved (S), parsed but wrong (x) and unparsed (-) per model."""
    records = {}
    results = Path(directory) / "results.jsonl"
    for line in results.read_text().splitlines() if results.exists() else []:
        record = json.loads(line)
        records[(record["model"], record["task"], record["variant"] + ("_perm" if record["perm"] else ""))] = record
    out = []
    for model in sorted({key[0] for key in records}):
        columns = sorted({key[2] for key in records if key[0] == model})
        tasks = sorted({key[1] for key in records if key[0] == model})
        out += [f"## {model}", "", "| task | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
        for task in tasks:
            cells = []
            for column in columns:
                record = records.get((model, task, column))
                cells.append("" if record is None else "S" if record["solved"] else "x" if record["parsed"] else "-")
            out.append(f"| {task} | " + " | ".join(cells) + " |")
        solved = sum(r["solved"] for k, r in records.items() if k[0] == model)
        out += ["", f"{solved} of {sum(1 for k in records if k[0] == model)} solved", ""]
    (Path(directory) / "report.md").write_text("\n".join(out))
    return "\n".join(out)


def write_prompts(directory):
    for extra in ([], ["--permute"]):
        subprocess.run([sys.executable, str(ROOT / "scripts/prompt_variants.py"), "--out", str(directory), *extra],
                       check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="prompt_tests")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    from subsymbolic.llm_runtime import OpenRouterRunner

    def make_runner(model):
        return OpenRouterRunner([model], {"max_tokens": args.max_tokens, "temperature": 0}, timeout=300.0)

    if not prompts(args.out):
        write_prompts(args.out)
    run(args.out, args.models, make_runner, args.tasks, args.variants, args.workers)
    print(report(args.out))


if __name__ == "__main__":
    main()
