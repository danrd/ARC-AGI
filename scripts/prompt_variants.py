#!/usr/bin/env python3
"""Prompt files for testing by hand what a model needs to solve a task.

Each variant changes one thing from the baseline prompt (the examples and
the test input, no knowledge), so a difference in what a model does can be
laid at that one thing. The files are written for a person to paste into a
model and read the answer of - the analysis is by hand.

    P0   baseline
    P1   plus true but unhelpful facts about the puzzle: the effect of length
    P3   plus the rule, in a sentence, before the examples
    P3b  the same sentence after the test input: where in the prompt it sits
    P4   P3 plus what changes on the test input, region by region
    P5   plus a plausible but wrong rule: does prose beat the examples?
    P6   plus a draft answer that is half right, to be corrected
    P7   the examples as lists of changed cells instead of output grids
    P10a P10b P10c   the baseline with the grid written another way

P3 against P0 asks whether knowing the rule is enough. P4 against P3 asks how
much of applying it can be handed over. P5 asks whether a wrong hint is
followed. P1 keeps P3's gain from being only the added length.

The rules come from scripts/prompt_oracles.py, where each is held to every
pair of its task. `--permute` writes each task a second time with the colours
1-9 shuffled - the task is public and a model may know it - and the rules,
the draft and the answer follow the shuffle.

    python scripts/prompt_variants.py                       # first pass, every task
    python scripts/prompt_variants.py --tasks 6ea4a07e --variants p0 p3 p4
    python scripts/prompt_variants.py --permute --out prompt_tests

Writes <out>/<task>/<variant>[_perm].txt, the expected answer beside them as
answer[_perm].txt, and <out>/index.md to note results in. The baseline's block
list is BASE_BLOCKS; change it to the arm being compared against.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_oracles import ORACLES  # noqa: E402
from rl.arc_task import ARCSubtask, ARCTask  # noqa: E402
from subsymbolic.arc_grid_formatting import format_grid  # noqa: E402
from subsymbolic.prompt_builder import ApproxTokenizer, PromptBuilder, PromptingConfig  # noqa: E402
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY  # noqa: E402

DATA = REPO_ROOT / "data" / "datasets" / "ARC"

BASE_BLOCKS = ["role_instruction", "grid_description", "general_instruction", "examples_intro",
               "examples", "task_repr", "output_format"]
COLOUR_NAMES = "0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=sky, 9=brown"
FIRST_PASS = ("p0", "p1", "p3", "p4", "p5", "p6")
ALL_VARIANTS = ("p0", "p1", "p3", "p3b", "p4", "p5", "p6", "p7", "p10a", "p10b", "p10c")
GRID_TYPES = {"p10a": "ascii", "p10b": "color_text", "p10c": "text_ascii"}


# ---------------------------------------------------------------------------
# tasks, and the colours shuffled
# ---------------------------------------------------------------------------

def load_arrays(task_id):
    """(train pairs, test input, test output) of an evaluation task."""
    challenge = json.loads((DATA / "evaluation_challenges.json").read_text())[task_id]
    answer = json.loads((DATA / "evaluation_solutions.json").read_text())[task_id]
    train = [(np.array(p["input"]), np.array(p["output"])) for p in challenge["train"]]
    return train, np.array(challenge["test"][0]["input"]), np.array(answer[0])


def colour_map(task_id, permute):
    """{colour: colour it is shown as}; 0 stays, 1-9 shuffled by the task id."""
    if not permute:
        return {colour: colour for colour in range(10)}
    shuffled = list(range(1, 10))
    random.Random(task_id).shuffle(shuffled)
    return {0: 0, **{colour: shuffled[colour - 1] for colour in range(1, 10)}}


def recolour(grid, mapping):
    return np.vectorize(mapping.get)(grid) if grid.size else grid


def build_task(task_id, permute):
    """(ARCTask as shown, test answer as shown, the map)."""
    train, test_input, test_output = load_arrays(task_id)
    mapping = colour_map(task_id, permute)
    subtasks = [ARCSubtask(f"{task_id}_{index}", recolour(inp, mapping), recolour(out, mapping))
                for index, (inp, out) in enumerate(train)]
    task = ARCTask(label=task_id, subtasks=subtasks, test_inp=recolour(test_input, mapping),
                   test_out=recolour(test_output, mapping))
    return task, recolour(test_output, mapping), mapping


# ---------------------------------------------------------------------------
# what a variant adds
# ---------------------------------------------------------------------------

def plain(grid):
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def neutral_facts(task):
    """True, unhelpful statements about the puzzle: sizes and palettes."""
    pairs = [(np.asarray(s.train_inp), np.asarray(s.train_out)) for s in task.subtasks]
    lines = [f"There are {len(pairs)} example pairs."]
    if all(i.shape == o.shape for i, o in pairs):
        lines.append("In every example the output has the same size as the input.")
    shapes = sorted({tuple(int(v) for v in i.shape) for i, _ in pairs})
    lines.append("The input sizes are " + ", ".join(f"{r}x{c}" for r, c in shapes) + ".")
    palette = sorted({int(v) for i, o in pairs for v in np.unique(np.concatenate([i.ravel(), o.ravel()]))})
    lines.append("The colours used in the examples are " + ", ".join(map(str, palette)) + ".")
    test = np.asarray(task.test_subtask.train_inp)
    lines.append(f"The test input is {test.shape[0]}x{test.shape[1]}.")
    return "Facts about this puzzle:\n" + "\n".join(lines)


def change_regions(before, after):
    """Connected regions (8-neighbour) of cells that differ, top-left first."""
    labels, count = ndimage.label(before != after, structure=np.ones((3, 3)))
    regions = [labels == index for index in range(1, count + 1)]
    return sorted(regions, key=lambda mask: tuple(np.argwhere(mask)[0]))


def walkthrough(before, after):
    """What changes on the test input, one line per connected region."""
    lines = []
    for mask in change_regions(before, after):
        rows, cols = np.where(mask)
        moves = {}
        for r, c in zip(rows, cols):
            moves[(int(before[r, c]), int(after[r, c]))] = moves.get((int(before[r, c]), int(after[r, c])), 0) + 1
        what = ", ".join(f"{n} cell{'s' if n > 1 else ''} {a} -> {b}" for (a, b), n in sorted(moves.items()))
        lines.append(f"Rows {rows.min() + 1}-{rows.max() + 1}, columns {cols.min() + 1}-{cols.max() + 1}: {what}.")
    return "On the test input these cells change (rows and columns counted from 1):\n" + "\n".join(lines)


def draft(before, after):
    """The answer with about half of it left undone: every second changed
    region reverted, or half the cells of the only region there is."""
    out = after.copy()
    regions = change_regions(before, after)
    if len(regions) == 1:
        cells = np.argwhere(regions[0])
        for r, c in cells[len(cells) // 2:]:
            out[r, c] = before[r, c]
    else:
        for mask in regions[1::2]:
            out[mask] = before[mask]
    return out


def diff_examples(task):
    """The examples with each output given as a list of changed cells."""
    blocks = []
    for index, subtask in enumerate(task.subtasks, 1):
        inp, out = np.asarray(subtask.train_inp), np.asarray(subtask.train_out)
        changed = np.argwhere(inp != out)
        listing = "\n".join(f"({r + 1}, {c + 1}): {int(inp[r, c])} -> {int(out[r, c])}" for r, c in changed)
        blocks.append(f"Training example {index}:\nInput:\n{plain(inp)}\n"
                      f"The output has the same size as the input. Cells that change "
                      f"(row, column counted from 1): {int(len(changed))}\n{listing}")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# a prompt
# ---------------------------------------------------------------------------

def builder(blocks, join_format="xml"):
    config = PromptingConfig(blocks=blocks, join_format=join_format, token_limit=10 ** 6,
                             min_examples=1, filters=["grid"], resolvers=["examples"])
    return PromptBuilder(config, ApproxTokenizer(), resolver_registry=RESOLVER_REGISTRY,
                         filter_registry=FILTER_REGISTRY)


def _insert(blocks, name, after):
    at = blocks.index(after) + 1
    return blocks[:at] + [name] + blocks[at:]


def make_prompt(task_id, variant, permute=False, grid_type="concise", join_format="xml"):
    """(prompt, expected answer as text) for one task and variant."""
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(ALL_VARIANTS)}")
    oracle = ORACLES[task_id]
    task, answer, mapping = build_task(task_id, permute)
    shown = lambda colour: str(mapping[colour])  # noqa: E731
    test_input = np.asarray(task.test_subtask.train_inp)
    blocks, overrides = list(BASE_BLOCKS), {}
    grid_type = GRID_TYPES.get(variant, grid_type)

    if variant in ("p3", "p4", "p5"):
        text = oracle.wrong(shown) if variant == "p5" else oracle.rule(shown)
        if variant == "p4":
            text += "\n" + walkthrough(test_input, answer)
        blocks = _insert(blocks, "knowledge", "general_instruction")
        overrides["knowledge"] = "Hint about the transformation:\n" + text
    elif variant == "p3b":
        blocks = _insert(blocks, "knowledge", "task_repr")
        overrides["knowledge"] = "Hint about the transformation:\n" + oracle.rule(shown)
    elif variant == "p1":
        blocks = _insert(blocks, "knowledge", "general_instruction")
        overrides["knowledge"] = neutral_facts(task)
    elif variant == "p6":
        blocks = _insert(blocks, "knowledge", "task_repr")
        overrides["knowledge"] = ("A first attempt at the output for this input is below. It is partly "
                                  "right and partly wrong. Correct it and return the whole grid.\n"
                                  "First attempt:\n" + plain(draft(test_input, answer)))
    elif variant == "p7":
        overrides["examples"] = diff_examples(task)

    context = {"role_text": "You are participating in the ARC-AGI benchmark.",
               "color_mapping_text": COLOUR_NAMES, "grid_repr_type": grid_type,
               "test_input_grid": test_input}
    prompt = builder(blocks, join_format).build(task, context=context, overrides=overrides)
    return prompt, format_grid(answer, repr_type="concise")


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

def grade_of(task_id):
    path = DATA / "evaluation_difficulty.json"
    return json.loads(path.read_text()).get(task_id, "?") if path.exists() else "?"


def write_all(out, tasks, variants, permute_too, grid_type="concise", join_format="xml"):
    """Write every prompt and the index; returns how many prompts."""
    out = Path(out)
    written = 0
    for task_id in tasks:
        folder = out / task_id
        folder.mkdir(parents=True, exist_ok=True)
        for permuted in ((False, True) if permute_too else (False,)):
            suffix = "_perm" if permuted else ""
            answer = None
            for variant in variants:
                prompt, answer = make_prompt(task_id, variant, permuted, grid_type, join_format)
                (folder / f"{variant}{suffix}.txt").write_text(prompt)
                written += 1
            (folder / f"answer{suffix}.txt").write_text(answer or "")
    (out / "index.md").write_text(index_text(tasks, variants, permute_too))
    return written


def index_text(tasks, variants, permute_too):
    lines = ["# Prompt tests", "",
             "One task per row; write the outcome of each variant in its cell "
             "(solved / near / wrong, and what the answer got wrong).", "",
             "| task | grade | " + " | ".join(variants) + (" | (perm: same columns) |" if permute_too else " |"),
             "|---|---|" + "---|" * len(variants) + ("---|" if permute_too else "")]
    for task_id in tasks:
        lines.append(f"| {task_id} | {grade_of(task_id)} | " + " | ".join("" for _ in variants)
                     + (" | |" if permute_too else " |"))
    lines += ["", "## Rules", ""]
    for task_id in tasks:
        oracle = ORACLES[task_id]
        lines += [f"**{task_id}** - rule: {oracle.rule(str)}", "",
                  f"wrong rule (P5): {oracle.wrong(str)}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=sorted(ORACLES))
    parser.add_argument("--variants", nargs="*", default=list(FIRST_PASS), choices=ALL_VARIANTS)
    parser.add_argument("--permute", action="store_true", help="also write the colour-shuffled twin")
    parser.add_argument("--out", default="prompt_tests")
    parser.add_argument("--grid-type", default="concise")
    parser.add_argument("--join-format", default="xml", choices=("xml", "md", "plain"))
    args = parser.parse_args()
    count = write_all(args.out, args.tasks, args.variants, args.permute, args.grid_type, args.join_format)
    print(f"wrote {count} prompts for {len(args.tasks)} tasks to {args.out}/")


if __name__ == "__main__":
    main()
