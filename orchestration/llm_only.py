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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


from orchestration.run import answer_of, load_task, read_lines
from subsymbolic.local_config import gpu_notebook_params

#: What the model is told: the instruction, the examples, the test input and the format. No `summary`, no `search_hints`.
BLOCKS = ["general_instruction", "examples_intro", "examples", "task_repr", "output_format"]
ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = ROOT / "data" / "experiments" / "llm_tasks.json"
ORDER = ("easy", "medium", "hard", "impossible", "very_hard")


def llm_only_params(model: str, tokenizer: Optional[str] = None, max_tokens: int = 1200) -> Dict[str, Any]:
    """The GPU notebooks' parameters for `model` with the grammar taken off and the prompt cut to BLOCKS."""
    params = gpu_notebook_params(model, tokenizer, max_tokens)
    params["generation"] = {k: v for k, v in params["generation"].items() if k != "grammar"}
    params["prompt"] = {**params["prompt"], "blocks": list(BLOCKS), "resolvers": ["examples"]}
    return params


def task_difficulties(path=TASKS_FILE) -> Dict[str, str]:
    """{task id: difficulty} of the tasks the model runs use."""
    return json.loads(Path(path).read_text())


def shuffled_tasks(difficulties: Dict[str, str], seed: int = 0) -> List[str]:
    ids = sorted(difficulties)
    random.Random(seed).shuffle(ids)
    return ids


def grade(task, reply: Any) -> Dict[str, Any]:
    """Whether the reply holds a grid, of the right shape, and the task's answer."""
    grid = answer_of({"solution": reply}) if isinstance(reply, str) else None
    if grid is None:
        return {"parsed": False, "shape_ok": False, "solved": False, "cells_right": 0.0}
    same = grid.shape == task.test_out.shape
    return {"parsed": True, "shape_ok": bool(same), "solved": bool(same and (grid == task.test_out).all()),
            "cells_right": round(float((grid == task.test_out).mean()), 4) if same else 0.0}


def solve_one(task, module, difficulty: Optional[str] = None) -> Dict[str, Any]:
    context = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
    started = time.perf_counter()
    result = module.solve(task, context)
    reply = result.get("solution", "")
    stats = result.get("module_results") or {}
    return {"task": task.label, "difficulty": difficulty, **grade(task, reply), "seconds": round(time.perf_counter() - started, 2),
            "prompt_tokens": stats.get("prompt_tokens"), "reply_tokens": stats.get("reply_tokens"),
            "error": stats.get("error"), "reply": reply[:6000] if isinstance(reply, str) else ""}


def done_ids(paths: Iterable[str]) -> set:
    done = set()
    for pattern in paths:
        for path in glob.glob(pattern):
            done.update(r["task"] for r in read_lines(path) if "task" in r and "solved" in r)
    return done


def run(task_ids: List[str], module, out: str, split: str = "evaluation", hours: Optional[float] = None,
        skip: Iterable[str] = (), clock=time.time, loader=load_task,
        difficulties: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Ask `module` about every task not in `out` or `skip`; stops starting new ones after `hours`."""
    finished = done_ids([out, *skip])
    todo = [t for t in task_ids if t not in finished]
    started, records = clock(), []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for task_id in todo:
        if hours is not None and clock() - started > hours * 3600:
            break
        record = solve_one(loader(split, task_id), module, (difficulties or {}).get(task_id))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-file", default=str(TASKS_FILE), help="JSON of {task id: difficulty}")
    parser.add_argument("--limit", type=int, help="only the first N of the shuffled tasks")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--skip", nargs="*", default=[], help="result files of other runs: their tasks are done")
    parser.add_argument("--out", default="llm_only.jsonl")
    args = parser.parse_args()

    from orchestration.configs import ExperimentConfig
    from subsymbolic.subsymbolic_module import SubsymbolicModule

    config = ExperimentConfig.from_dict(llm_only_params(args.model, args.tokenizer, args.max_tokens))
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as error:  # noqa: BLE001 - counting tokens approximately is enough to run
        print(f"no tokenizer for {args.tokenizer} ({type(error).__name__}); counting tokens by whitespace", flush=True)
        from subsymbolic.prompt_builder import ApproxTokenizer
        tokenizer = ApproxTokenizer()
    difficulties = task_difficulties(args.tasks_file)
    ids = shuffled_tasks(difficulties, args.seed)[:args.limit]
    module = SubsymbolicModule(config, tokenizer)
    try:
        run(ids, module, args.out, args.split, args.hours, args.skip, difficulties=difficulties)
    finally:
        module.close()
    print(summarise([r for r in read_lines(args.out) if "solved" in r]))


if __name__ == "__main__":
    main()
