"""What the model asks to be told about a task, when it is asked.

The reasoning of a model that solves a puzzle shows what it works out by hand - copies the grid out with
row and column numbers, counts cells, lists which cells changed, finds objects. This asks it outright, in
two ways, to see what it wants before deciding what to give:

  wishes  the task as the model is always shown it, without the output format, and the question: what would
          you want a program to compute for you before you solve this? It answers in free text, a line a wish.
  menu    the model is offered a list of things it may ask for (`menu_tools`), up to three rounds, and the
          requests are answered for real: the symbolic analysis' own sections where it has them, and the
          plain computations (changed cells, colour counts, objects, the grids with their rows and columns
          numbered) where it has not. A few entries are on the list that nothing answers, to see whether they
          are wanted. Which it asks for, in which order, is the result; whether the answer then comes out
          right is kept beside it.

    python -m orchestration.data_requests --mode menu --model REPO:FILE --tokenizer REPO --limit 12 --out menu.jsonl

Neither is an evaluation of a way to prompt: the second asks for the thing that decides which ways are worth
the evaluation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from orchestration.tools import InfoTool, ToolUsingModule
from subsymbolic.prompt_builder import PromptBuilder

WISHES = """

---
You are not asked to solve this puzzle now. Suppose a program could compute anything about the grids for you \
before you start, from the examples and the test input - something you would otherwise have to work out by hand. \
What would you want it to tell you? Answer with at most 8 lines, one wish per line, each in the form

NAME | what it would print for one example pair, in a few words or one short sample | why you would want it

and nothing else."""


# ---------------------------------------------------------------------------
# what a request can be answered with
# ---------------------------------------------------------------------------

def _pairs(task):
    """(label, input grid, output grid or None) of every example and of the test input."""
    rows = [(f"Example {i + 1}", s.train_inp, s.train_out) for i, s in enumerate(task.subtasks)]
    return rows + [("Test", task.test_inp, None)]


def _grids(task):
    """(label, grid) of every grid the model is shown, the test output not among them."""
    out = []
    for label, grid_in, grid_out in _pairs(task):
        out.append((f"{label} input", np.asarray(grid_in)))
        if grid_out is not None:
            out.append((f"{label} output", np.asarray(grid_out)))
    return out


def indexed_grids(task) -> str:
    """Every grid with its columns and rows numbered from 1, as the answer's rows are, so that a cell is a coordinate and
    not a place to count to."""
    blocks = []
    for label, grid in _grids(task):
        height, width = grid.shape
        tens = "".join(str(c // 10) if c >= 10 else " " for c in range(1, width + 1))
        ones = "".join(str(c % 10) for c in range(1, width + 1))
        lines = [f"{label} ({height}x{width}), columns 1..{width}:"]
        if width > 10:
            lines.append("     " + tens)
        lines.append("     " + ones)
        lines += [f"r{r:<3} " + "".join(str(v) for v in row) for r, row in enumerate(grid, 1)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def cell_changes(task, limit: int = 40) -> str:
    """For each example, the cells that differ between input and output: where (rows and columns from 1, as in the
    answer), from what, to what."""
    lines = []
    for number, subtask in enumerate(task.subtasks, 1):
        grid_in, grid_out = np.asarray(subtask.train_inp), np.asarray(subtask.train_out)
        if grid_in.shape != grid_out.shape:
            lines.append(f"Example {number}: the size changes, {grid_in.shape[0]}x{grid_in.shape[1]} "
                         f"to {grid_out.shape[0]}x{grid_out.shape[1]}")
            continue
        rows, cols = np.nonzero(grid_in != grid_out)
        if not len(rows):
            lines.append(f"Example {number}: nothing changes")
            continue
        shown = ", ".join(f"(r{r + 1},c{c + 1}) {grid_in[r, c]}->{grid_out[r, c]}" for r, c in zip(rows[:limit], cols[:limit]))
        more = f", and {len(rows) - limit} more" if len(rows) > limit else ""
        kinds = {}
        for r, c in zip(rows, cols):
            kinds[(int(grid_in[r, c]), int(grid_out[r, c]))] = kinds.get((int(grid_in[r, c]), int(grid_out[r, c])), 0) + 1
        by_kind = "; ".join(f"{a}->{b} x{n}" for (a, b), n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        lines.append(f"Example {number}: {len(rows)} of {grid_in.size} cells change ({by_kind}): {shown}{more}")
    return "\n".join(lines)


def colour_counts(task) -> str:
    """How many cells of each colour every grid has."""
    lines = []
    for label, grid in _grids(task):
        values, counts = np.unique(grid, return_counts=True)
        lines.append(f"{label}: " + ", ".join(f"{v}: {n}" for v, n in zip(values.tolist(), counts.tolist())))
    return "\n".join(lines)


def objects(task, per_grid: int = 12) -> str:
    """The connected pieces of one colour in every grid, apart from its background: colour, size, rows and columns it
    spans (numbered from 1, as in the answer)."""
    from scipy import ndimage
    from symbolic.utils import resolve_background

    lines = []
    for label, grid in _grids(task):
        background = resolve_background(grid)
        if background is None:
            background = int(np.bincount(grid.ravel()).argmax())
        found = []
        for colour in np.unique(grid).tolist():
            if colour == background:
                continue
            labelled, count = ndimage.label(grid == colour)
            for index in range(1, count + 1):
                rows, cols = np.nonzero(labelled == index)
                found.append((colour, len(rows), rows.min(), rows.max(), cols.min(), cols.max()))
        found.sort(key=lambda o: (o[2], o[4]))
        shown = "; ".join(f"colour {c}, {n} cells, rows {r0 + 1}-{r1 + 1}, cols {c0 + 1}-{c1 + 1}" for c, n, r0, r1, c0, c1 in found[:per_grid])
        more = f"; and {len(found) - per_grid} more" if len(found) > per_grid else ""
        lines.append(f"{label} (background {background}): {len(found)} pieces" + (f": {shown}{more}" if found else ""))
    return "\n".join(lines)


def _findings_section(task, keep: str, builder) -> Optional[str]:
    """One section of the symbolic analysis' findings, rendered as the summary renders it."""
    from subsymbolic.arc_resolvers import _task_findings
    from symbolic.findings import render_findings

    findings = _task_findings(task)
    blank = {name: () for name in ("transformations", "invariants", "grid_observations") if name != keep}
    return render_findings(dataclasses.replace(findings, **blank), None, builder.count_tokens)


def _section(keep: str):
    return lambda task, builder: _findings_section(task, keep, builder)


#: What a run can put in the prompt without being asked for it: name -> (heading in the prompt, function of the task).
AIDS = {
    "numbered_grids": ("The grids again, with rows and columns numbered from 1", indexed_grids),
    "cell_changes": ("The cells that change between each example's input and output", cell_changes),
    "objects": ("The pieces of one colour in each grid", objects),
    "colour_counts": ("How many cells of each colour each grid has", colour_counts),
}


def aids_for(task, names: Sequence[str]) -> Dict[str, str]:
    """{heading: text} of the named aids for `task`, in the order named."""
    return {AIDS[name][0]: AIDS[name][1](task) for name in names}


def menu_tools() -> List[InfoTool]:
    """What the model may ask for. The first three are the symbolic analysis' own sections, the next four plain
    computations, the last three have nothing behind them."""
    return [
        InfoTool("changes", "what the symbolic analysis found changes between the inputs and the outputs",
                 _section("transformations")),
        InfoTool("kept", "what the symbolic analysis found stays the same between the inputs and the outputs",
                 _section("invariants")),
        InfoTool("grids_shown", "what the symbolic analysis found about the grids themselves: sizes, background, "
                                "colours that appear or disappear", _section("grid_observations")),
        InfoTool("cell_changes", "for each example, which cells differ between input and output: where, from which "
                                 "colour to which", lambda task, builder: cell_changes(task)),
        InfoTool("colour_counts", "how many cells of each colour every grid has",
                 lambda task, builder: colour_counts(task)),
        InfoTool("objects", "the connected pieces of one colour in every grid: colour, size, rows and columns it spans",
                 lambda task, builder: objects(task)),
        InfoTool("numbered_grids", "every grid again with its rows and columns numbered",
                 lambda task, builder: indexed_grids(task)),
        InfoTool("symmetry", "which grids and objects are symmetric, and about which axis", lambda task, builder: None),
        InfoTool("row_column_comparison", "for rows and columns of a grid, which are equal, which are complements",
                 lambda task, builder: None),
        InfoTool("object_relations", "for every pair of objects: which is above, left of, inside, touching the other",
                 lambda task, builder: None),
    ]


# ---------------------------------------------------------------------------
# asking
# ---------------------------------------------------------------------------

def wishes_prompt(builder: PromptBuilder, task) -> Optional[str]:
    """The task as the model is shown it, with the question where the output format would be."""
    prompt = builder.build(task, context={"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp},
                           assistant_prefix="")
    return None if prompt is None else prompt + WISHES


def wishes_builder(module) -> PromptBuilder:
    """A builder over the module's blocks without the one that demands a grid."""
    config = module.builder.config
    blocks = [b for b in config.blocks if "output_format" not in str(b)]
    return PromptBuilder(config.model_copy(update={"blocks": blocks}), module.builder.tokenizer,
                         resolver_registry=module.builder.resolver_registry, filter_registry=module.builder.filter_registry)


def parse_wishes(reply: str):
    """({name: {'prints', 'by_hand'}}, name after a `MOST:` line or None) of a reply in the form
    `NAME | what it prints | why`, one per line; the third column is kept as `by_hand`."""
    items, most = {}, None
    for line in reply.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        found = re.match(r"MOST\s*:\s*(.+)$", line, re.I)
        if found:
            most = found.group(1).strip(" *`.")
            continue
        parts = [part.strip(" *`") for part in line.split("|")]
        if len(parts) >= 2 and parts[0] and not set(parts[0]) <= set("-: "):
            items[parts[0]] = {"prints": parts[1], "by_hand": parts[2] if len(parts) > 2 else ""}
    return items, most


def ask_wishes(module, builder: PromptBuilder, task) -> Dict[str, Any]:
    """What the model would want computed for `task`: its reply, and the wishes read out of it."""
    prompt = wishes_prompt(builder, task)
    if prompt is None:
        return {"error": "prompt didn't fit token_limit", "reply": "", "wishes": {}}
    reply = module.runner.generate(prompt)
    reply = reply if isinstance(reply, str) else ""
    return {"reply": reply, "wishes": parse_wishes(reply)[0]}


def run_menu(module, task, tools: List[InfoTool], max_requests: int = 3) -> Dict[str, Any]:
    """The requests the model makes for `task`, in order, each marked answered or not, and its final answer."""
    asked = ToolUsingModule(module, tools, max_requests=max_requests)
    replies: List[str] = []
    original = asked._ask

    def recording(runner, prompt, name):
        reply = original(runner, prompt, name)
        replies.append(reply if isinstance(reply, str) else "")
        return reply

    asked._ask = recording
    context = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
    result = asked.solve(task, context)
    return {"solution": result.get("solution", ""), "calls": result.get("module_results", {}).get("tool_calls", []),
            "replies": replies}


def main():
    from orchestration.llm_only import (TASKS_FILE, grade, llm_only_params, shuffled_tasks, task_difficulties)
    from orchestration.run import load_task

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["wishes", "menu"], required=True)
    parser.add_argument("--model", required=True, help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-file", default=str(TASKS_FILE), help="JSON of {task id: difficulty}")
    parser.add_argument("--limit", type=int, help="only the first N of the shuffled tasks")
    parser.add_argument("--max-tokens", type=int, default=700, help="for the wishes; the menu's answer gets 1200")
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--out", default="data_requests.jsonl")
    args = parser.parse_args()

    from orchestration.configs import ExperimentConfig
    from subsymbolic.subsymbolic_module import SubsymbolicModule

    config = ExperimentConfig.from_dict(llm_only_params(args.model, args.tokenizer, args.max_tokens if args.mode == "wishes" else 1200))
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as error:  # noqa: BLE001 - counting tokens approximately is enough to run
        print(f"no tokenizer for {args.tokenizer} ({type(error).__name__}); counting tokens by whitespace", flush=True)
        from subsymbolic.prompt_builder import ApproxTokenizer
        tokenizer = ApproxTokenizer()
    difficulties = task_difficulties(args.tasks_file)
    module = SubsymbolicModule(config, tokenizer)
    builder = wishes_builder(module) if args.mode == "wishes" else None
    tools = menu_tools()
    done = {json.loads(line)["task"] for line in Path(args.out).read_text().splitlines()} if Path(args.out).exists() else set()
    try:
        for task_id in shuffled_tasks(difficulties, args.seed)[:args.limit]:
            if task_id in done:
                continue
            task = load_task(args.split, task_id)
            started = time.perf_counter()
            if args.mode == "wishes":
                record = ask_wishes(module, builder, task)
            else:
                record = run_menu(module, task, tools, args.max_requests)
                record.update(grade(task, record["solution"]))
                record.pop("solution")
            record = {"task": task_id, "difficulty": difficulties.get(task_id), "mode": args.mode, "model": args.model,
                      "seconds": round(time.perf_counter() - started, 1), **record}
            with open(args.out, "a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            shown = record.get("calls") if args.mode == "menu" else list(record.get("wishes", {}))
            print(f"{task_id}: {shown}", flush=True)
    finally:
        module.close()


if __name__ == "__main__":
    main()
