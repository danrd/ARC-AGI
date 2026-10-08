#!/usr/bin/env python3
"""Send the prompt variants of prompt_variants.py to a local model and keep the answers.

prompt_variants.py writes <dir>/<task>/<variant>[_perm].txt for a person to paste
into a model. This does the pasting, on the models the system itself uses: the
model is loaded the way the GPU notebooks load it (llama.cpp, every layer on the
cards, the answer held to the grid grammar, thinking off), every prompt goes to
it, the reply is kept beside the prompt as <variant>[_perm].<model>.response.txt
for reading, and compared with answer[_perm].txt. One line per reply goes to
<dir>/results.jsonl and a table of task by variant to <dir>/report.md. A reply that
is already on disk is not asked for again, so a run that stops is continued by
running it again.

    python scripts/run_prompt_variants.py --out prompt_tests \\
        --model unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_M.gguf --tokenizer Qwen/Qwen3.8-27B

`--model` is the GGUF repository and the quantisation file, joined by a colon.
Prompts not yet written are written first, with the colours as they are and
shuffled. `solved` is the grid equal to the answer and nothing else: a reply that
does not parse is `parsed: false`, not a near miss.
"""
import argparse
import json
import re
import subprocess
import sys
import time
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
    grid = last_grid(_without_thinking(reply)) if reply else ""
    if not isinstance(grid, np.ndarray):
        return {"parsed": False, "solved": False, "shape_ok": False, "cells_right": 0.0}
    same_shape = grid.shape == answer.shape
    return {"parsed": True, "solved": bool(same_shape and (grid == answer).all()), "shape_ok": same_shape,
            "cells_right": round(float((grid == answer).mean()), 4) if same_shape else 0.0}


def last_grid(text):
    """The last grid in the text in the asked-for format: a model that thinks aloud
    before it answers has its answer at the end, and a grid it drafted on the way is
    not it."""
    starts = [match.start() for match in re.finditer(r"^[ \t]*\d+,\d+:[ \t]*$", text, flags=re.M)]
    for start in reversed(starts):
        grid = parse_llm_output(text[start:].strip())
        if isinstance(grid, np.ndarray):
            return grid
    return parse_llm_output(text)


def _without_thinking(reply):
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.S)


def prompts(directory, tasks=None, variants=None, perm="both"):
    """(task, variant, perm, prompt path, answer path) for every prompt on disk."""
    found = []
    for path in sorted(Path(directory).glob("*/*.txt")):
        name = path.stem
        if name.startswith("answer") or "." in name:
            continue
        is_perm = perm_name = name.endswith("_perm")
        variant = name[:-5] if perm_name else name
        if (tasks and path.parent.name not in tasks) or (variants and variant not in variants):
            continue
        if (perm == "no" and is_perm) or (perm == "only" and not is_perm):
            continue
        found.append((path.parent.name, variant, perm_name, path,
                      path.parent / ("answer_perm.txt" if perm_name else "answer.txt")))
    return found


def response_path(prompt_path, model):
    return prompt_path.with_name(f"{prompt_path.stem}.{slug(model)}.response.txt")


def run(directory, models, make_runner, tasks=None, variants=None, workers=2, attempts=3, pause=30.0, perm="both"):
    """Ask every model every prompt that has no reply yet; returns the records of the new replies."""
    runners = {model: make_runner(model) for model in models}
    work = [(model, *item) for model in models for item in prompts(directory, tasks, variants, perm)
            if not response_path(item[3], model).exists()]
    results = Path(directory) / "results.jsonl"

    def one(job):
        model, task, variant, perm, prompt_path, answer_path = job
        for attempt in range(attempts):
            try:
                reply = runners[model].generate(prompt_path.read_text())
                if not reply or not reply.strip():
                    raise ValueError("an empty reply (a reasoning model out of tokens, most often)")
                break
            except Exception as error:  # noqa: BLE001 - one failed request must not end the run
                print(f"{task} {variant} {model}, try {attempt + 1}: {error}", file=sys.stderr)
                time.sleep(pause * (attempt + 1))
        else:
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
    parser.add_argument("--model", required=True, help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer", help="the model's own repository, for its chat template")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--perm", choices=["both", "no", "only"], default="both",
                        help="the prompts with the colours as they are, shuffled, or both")
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    from subsymbolic.llm_setup import build_runner
    from subsymbolic.local_config import local_model_config

    runner = build_runner(local_model_config(args.model, args.tokenizer, args.max_tokens))
    if not prompts(args.out):
        write_prompts(args.out)
    try:
        run(args.out, [args.model], lambda model: runner, args.tasks, args.variants, workers=1, perm=args.perm,
            attempts=1, pause=0.0)
    finally:
        runner.close()
    print(report(args.out))


if __name__ == "__main__":
    main()
