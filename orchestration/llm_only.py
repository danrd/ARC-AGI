#!/usr/bin/env python3
"""What a model does on its own: the examples, the test input, and a free reply - no symbolic analysis, no search
hints, no grammar, no RL, no second model, one call per task.

It is the base the rest of the system is measured against. A grammar forces the reply into the shape of a grid, and
a summary or a hint tells the model something about the task; both are left out here, so what is solved is what the
model solves from the examples alone, and what the system adds over it is the system's.

    python -m orchestration.llm_only --model unsloth/gemma-4-31B-it-GGUF:gemma-4-31B-it-Q4_K_S.gguf \\
        --tokenizer google/gemma-4-31B-it --out llm_only.jsonl --hours 8

The tasks are the 370 the earlier model runs used (data/experiments/llm_tasks.json, each with its difficulty): the
evaluation split without the 27 the symbolic solvers solve, the one too large, and two whose prompt does not fit. They
are in a fixed shuffled order (`--seed`), so a run that is cut short by `--hours` is a random sample and not its easy
end, and the result is read by difficulty. A line is kept per task, with the reply as written; a run that stops is
continued by running it again, and what other notebooks have done is given as `--skip`.
"""
import argparse
import glob
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from orchestration.run import answer_of, load_task, read_lines
from subsymbolic.local_config import gpu_notebook_params

#: What the model is told: the instruction, the examples, the test input and the format. No `summary`, no `search_hints`.
BLOCKS = ["general_instruction", "examples_intro", "examples", "task_repr", "output_format"]
ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = ROOT / "data" / "experiments" / "llm_tasks.json"
ORDER = ("easy", "medium", "hard", "impossible", "very_hard")


def llm_only_params(model: str, tokenizer: Optional[str] = None, max_tokens: int = 1200,
                    thinking: bool = False, context: int = 14000, aids: bool = False,
                    prompt_lookup: Optional[int] = None, temperature: Optional[float] = None,
                    sampling_seed: Optional[int] = None) -> Dict[str, Any]:
    """The GPU notebooks' parameters for `model` with the grammar taken off and the prompt cut to BLOCKS.

    `thinking` lets the model reason before it answers (the notebooks switch that off, because a model that reasons can
    spend every token on it and never answer). The reasoning is part of the reply and counts against `max_tokens`, so
    the window grows to `context` and a prompt that would leave less than `max_tokens` of it is not asked.
    `aids` puts the auxiliary_info block, where `solve_one(aids=...)` puts what it computes, before the output format.
    `temperature` and `sampling_seed` make the run sample (the notebooks are greedy, seed 42) so that a repeat with another
    seed measures how much a result moves by chance alone.
    `prompt_lookup` turns on speculative decoding by prompt lookup with that many proposed tokens (LlmConfig's
    prompt_lookup_tokens)."""
    params = gpu_notebook_params(model, tokenizer, max_tokens)
    params["generation"] = {k: v for k, v in params["generation"].items() if k != "grammar"}
    if temperature is not None:                           # greedy unless asked: a repeat of a greedy run repeats it
        params["generation"]["temperature"] = temperature
    if sampling_seed is not None:
        params["base"] = {**params["base"], "seed": sampling_seed}
    blocks = list(BLOCKS)
    if aids:
        blocks.insert(blocks.index("output_format"), "auxiliary_info")
    params["prompt"] = {**params["prompt"], "blocks": blocks, "resolvers": ["examples"]}
    if prompt_lookup is not None:
        params["llm"] = {**params["llm"], "prompt_lookup_tokens": prompt_lookup}
    if thinking:
        params["generation"]["chat_template_kwargs"] = {"enable_thinking": True}
        params["llm"] = {**params["llm"], "max_context": context}
        params["prompt"]["token_limit"] = context - max_tokens
    return params


def task_difficulties(path=TASKS_FILE) -> Dict[str, str]:
    """{task id: difficulty} of the tasks the model runs use."""
    return json.loads(Path(path).read_text())


def shuffled_tasks(difficulties: Dict[str, str], seed: int = 0) -> List[str]:
    ids = sorted(difficulties)
    random.Random(seed).shuffle(ids)
    return ids


def task_size(task) -> tuple:
    """(examples, cells) of a task: how many training pairs it has and how many cells all the grids the model is shown
    hold - the examples' inputs and outputs and the test input."""
    cells = sum(s.train_inp.size + s.train_out.size for s in task.subtasks) + task.test_inp.size
    return len(task.subtasks), int(cells)


def small_tasks(ids: Iterable[str], count: int, max_examples: int = 3, split: str = "evaluation",
                loader=load_task) -> List[str]:
    """The `count` tasks of `ids` with the fewest cells among those that have at most `max_examples` examples: the
    ones a prompt, its aids and the model's reasoning all fit in the window for. Ties are broken by id."""
    sized = []
    for task_id in ids:
        examples, cells = task_size(loader(split, task_id))
        if examples <= max_examples:
            sized.append((cells, task_id))
    return [task_id for _, task_id in sorted(sized)[:count]]


#: (opening, closing) of the reasoning a model writes before its answer: Qwen's and Gemma 4's.
THINKING_MARKS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))


def final_answer(reply: str) -> str:
    """The reply without its reasoning. Reasoning that never closed (the token limit came first) leaves no answer:
    a grid written in the middle of it is a draft, not the model's reply."""
    for opening, closing in THINKING_MARKS:
        if closing in reply:
            return reply.rsplit(closing, 1)[1]
        if opening in reply:
            return ""
    return reply


HEADER = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*:?\s*$")


def grid_row(line: str, number: int, width: int) -> Optional[List[int]]:
    """One row of `width` cells from `line`, the `number`-th: as the prompt asks ("1 0 3 1", row number and spaced
    cells), without the number ("0 3 1"), or concatenated with or without it ("1 031", "031")."""
    parts = line.split()
    numbered = parts[:1] == [str(number)]
    if numbered and (len(parts) == width + 1 or (len(parts) == 2 and len(parts[1]) == width)):
        parts = parts[1:]
    if len(parts) == 1 and len(parts[0]) == width:
        parts = list(parts[0])
    if len(parts) != width or not all(len(c) == 1 and c.isdigit() for c in parts):
        return None
    return [int(c) for c in parts]


def read_grid(text: str) -> Optional[np.ndarray]:
    """The first grid in `text` led by its "n,m:" header, its rows in any of grid_row's forms, or None.

    The prompt's format is "1 x_1 ... x_m" - row numbers and spaced cells - and parse_llm_output, which the grammar
    runs are written for, reads only "1 x_1x_2...": a reply written exactly as asked was graded unreadable (think_small:
    8 of 20 Gemma replies held the right grid after their reasoning, and 1 was read)."""
    lines = [line for line in text.splitlines() if line.strip()]
    for at, line in enumerate(lines):
        header = HEADER.match(line)
        if not header:
            continue
        rows, cols = int(header.group(1)), int(header.group(2))
        if not (1 <= rows <= 30 and 1 <= cols <= 30):
            continue
        grid = [grid_row(row, i + 1, cols) for i, row in enumerate(lines[at + 1:at + 1 + rows])]
        if len(grid) == rows and all(row is not None for row in grid):
            return np.array(grid, dtype=int)
    return None


def grade(task, reply: Any) -> Dict[str, Any]:
    """Whether the reply holds a grid, of the right shape, and the task's answer; `format_ok` is whether it is also in
    the one form parse_llm_output reads (what a run under the grammar would have written)."""
    answer = final_answer(reply) if isinstance(reply, str) else ""
    grid = read_grid(answer)
    format_ok = answer_of({"solution": answer}) is not None if answer else False
    if grid is None:
        return {"parsed": False, "format_ok": False, "shape_ok": False, "solved": False, "cells_right": 0.0}
    same = grid.shape == task.test_out.shape
    return {"parsed": True, "format_ok": format_ok, "shape_ok": bool(same),
            "solved": bool(same and (grid == task.test_out).all()),
            "cells_right": round(float((grid == task.test_out).mean()), 4) if same else 0.0}


def solve_one(task, module, difficulty: Optional[str] = None, reply_limit: int = 6000,
              aids: Sequence[str] = ()) -> Dict[str, Any]:
    """`aids` are names from orchestration.data_requests.AIDS: computed from the task and put in the prompt. The blocks
    after the examples get no share of the budget, so a prompt that the aids make too long is not made (the module
    answers "prompt didn't fit token_limit" with no reply); the examples are never cut to make room for them."""
    context = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
    started = time.perf_counter()
    if aids:
        from orchestration.data_requests import aids_for
        context["auxiliary_info"] = aids_for(task, aids)
        from subsymbolic.prompt_builder import BlockSpec
        builder = getattr(module, "builder", None)
        if builder is not None and "auxiliary_info" not in [BlockSpec.parse(b).name for b in builder.config.blocks]:
            raise ValueError("aids were asked for and the module's prompt has no auxiliary_info block to put them in: "
                             "build its config with llm_only_params(aids=True)")
    result = module.solve(task, context)
    reply = result.get("solution", "")
    stats = result.get("module_results") or {}
    return {"task": task.label, "difficulty": difficulty, **grade(task, reply), "seconds": round(time.perf_counter() - started, 2),
            "prompt_tokens": stats.get("prompt_tokens"), "reply_tokens": stats.get("reply_tokens"),
            "error": stats.get("error"), "reply": reply[:reply_limit] if isinstance(reply, str) else ""}


def done_ids(paths: Iterable[str]) -> set:
    done = set()
    for pattern in paths:
        for path in glob.glob(pattern):
            done.update(r["task"] for r in read_lines(path) if "task" in r and "solved" in r)
    return done


def run(task_ids: List[str], module, out: str, split: str = "evaluation", hours: Optional[float] = None,
        skip: Iterable[str] = (), clock=time.time, loader=load_task,
        difficulties: Optional[Dict[str, str]] = None, reply_limit: int = 6000,
        aids: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Ask `module` about every task not in `out` or `skip`; stops starting new ones after `hours`."""
    finished = done_ids([out, *skip])
    todo = [t for t in task_ids if t not in finished]
    started, records = clock(), []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for task_id in todo:
        if hours is not None and clock() - started > hours * 3600:
            break
        record = solve_one(loader(split, task_id), module, (difficulties or {}).get(task_id), reply_limit, aids)
        records.append(record)
        with open(out, "a") as handle:
            handle.write(json.dumps({**record, "split": split}) + "\n")
        print(f"{task_id}: solved={record['solved']} parsed={record['parsed']} {record['seconds']}s", flush=True)
    return records


def summarise(lines: List[Dict[str, Any]]) -> str:
    """The totals, and solved of asked for each difficulty."""
    def counts(rows):
        return (f"solved {sum(r['solved'] for r in rows)} of {len(rows)}, "
                f"the right shape {sum(r['shape_ok'] for r in rows)}, holds a grid {sum(r['parsed'] for r in rows)}")

    out = [f"{counts(lines)}; {sum(r['seconds'] for r in lines) / 3600:.1f} h, "
           f"{sum(r.get('prompt_tokens') or 0 for r in lines)} tokens read, "
           f"{sum(r.get('reply_tokens') or 0 for r in lines)} written"]
    for level in ORDER:
        rows = [r for r in lines if r.get("difficulty") == level]
        if rows:
            out.append(f"  {level:<11}{counts(rows)}")
    return "\n".join(out)


def ids_from_args(args, difficulties: Dict[str, str], loader=load_task) -> List[str]:
    """The tasks the command line asks for: the smallest ones with --small, else the first of the shuffled order."""
    if args.small:
        return small_tasks(sorted(difficulties), args.small, args.max_examples, args.split, loader)
    return shuffled_tasks(difficulties, args.seed)[:args.limit]


def params_from_args(args) -> Dict[str, Any]:
    """The model parameters the command line asks for: the aids, if any, get their block."""
    return llm_only_params(args.model, args.tokenizer, args.max_tokens, args.thinking, args.context, bool(args.aids),
                           args.prompt_lookup, getattr(args, "temperature", None), getattr(args, "sampling_seed", None))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-file", default=str(TASKS_FILE), help="JSON of {task id: difficulty}")
    parser.add_argument("--limit", type=int, help="only the first N of the shuffled tasks")
    parser.add_argument("--small", type=int, help="instead of the shuffled order, the N tasks with the fewest cells among "
                        "those with at most --max-examples examples")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--thinking", action="store_true",
                        help="let the model reason first; --max-tokens then counts the reasoning too, and the whole "
                             "reply is kept (it is what shows how long the reasoning runs)")
    parser.add_argument("--context", type=int, default=14000, help="the window with --thinking")
    parser.add_argument("--prompt-lookup", type=int, help="speculative decoding by prompt lookup, this many tokens "
                        "proposed at a time (greedy output unchanged)")
    from orchestration.data_requests import AIDS
    parser.add_argument("--aids", nargs="*", default=[], choices=sorted(AIDS), help="what to compute from the task and put in the prompt: "
                        "names from orchestration.data_requests.AIDS; a task whose prompt it makes too long is skipped")
    parser.add_argument("--temperature", type=float, help="sample at this temperature (default: greedy)")
    parser.add_argument("--sampling-seed", type=int, help="the model's seed (default 42); with --temperature, another "
                        "seed is another sample")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--skip", nargs="*", default=[], help="result files of other runs: their tasks are done")
    parser.add_argument("--out", default="llm_only.jsonl")
    args = parser.parse_args()

    from orchestration.configs import ExperimentConfig
    from subsymbolic.subsymbolic_module import SubsymbolicModule

    config = ExperimentConfig.from_dict(params_from_args(args))
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as error:  # noqa: BLE001 - counting tokens approximately is enough to run
        print(f"no tokenizer for {args.tokenizer} ({type(error).__name__}); counting tokens by whitespace", flush=True)
        from subsymbolic.prompt_builder import ApproxTokenizer
        tokenizer = ApproxTokenizer()
    difficulties = task_difficulties(args.tasks_file)
    ids = ids_from_args(args, difficulties)
    module = SubsymbolicModule(config, tokenizer)
    try:
        run(ids, module, args.out, args.split, args.hours, args.skip, difficulties=difficulties,
            reply_limit=10 ** 6 if args.thinking else 6000, aids=args.aids)
    finally:
        module.close()
    print(summarise([r for r in read_lines(args.out) if "solved" in r]))


if __name__ == "__main__":
    main()
