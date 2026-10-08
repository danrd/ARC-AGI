#!/usr/bin/env python3
"""Run the whole system on tasks and measure it: what it solves, which source answered, where the time went.

One model plays every role the system gives a model - it solves, it is the second model that checks
the answers, it makes the decisions the orchestrator hands to a model - loaded the way the GPU
notebooks load it. RL runs beside it in a background process, as in the graph. Each task is one line
of the output: the answer and whether it is the task's, the source that gave it, the options the run
had, the time and the tokens of every phase (orchestration.trace), what the model decided at each
step, and what each round of the refinement loop found wrong.

    python -m orchestration.run --tasks 009d5c81 00dbd492 --split evaluation \\
        --model unsloth/Qwen3.8-8B-GGUF:Qwen3.8-8B-Q4_K_M.gguf --tokenizer Qwen/Qwen3.8-8B \\
        --options '{"refine_rounds": 3, "feedback": true}' --out system.jsonl

`--options` are OrchestrationOptions' fields as JSON; none given is the system as it was. A line is
kept for each (task, options) and a run that stops is continued by running it again. `summary` reads
the lines back as a table of the phases and the solved share.
"""
import argparse
import copy
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from orchestration.assemble import assemble, solve_with_orchestration
from orchestration.configs import AgentRunConfig, OrchestrationOptions, SystemRunConfig
from orchestration.hierarchy import _read_grid
from orchestration.trace import Tracer, render
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.utils import parse_llm_output

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets" / "ARC"


def load_task(split: str, task_id: str) -> ARCTask:
    challenges = json.loads((DATA / f"{split}_challenges.json").read_text())
    solutions = json.loads((DATA / f"{split}_solutions.json").read_text())
    data = challenges[task_id]
    subtasks = [ARCSubtask(f"{task_id}_{i}", np.array(p["input"]), np.array(p["output"]))
                for i, p in enumerate(data["train"])]
    return ARCTask(label=task_id, subtasks=subtasks, test_inp=np.array(data["test"][0]["input"]),
                   test_out=np.array(solutions[task_id][0]))


def key_of(task_id: str, options: OrchestrationOptions, rl: bool) -> str:
    return json.dumps({"task": task_id, "options": vars(options), "rl": rl}, sort_keys=True)


def done_keys(path) -> set:
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                pass
    return keys


def answer_of(result: Dict[str, Any], parse=parse_llm_output) -> Optional[np.ndarray]:
    """The run's answer as a grid, whatever source gave it."""
    solution = result.get("solution")
    if isinstance(solution, np.ndarray):
        return solution if solution.ndim == 2 else None
    return _read_grid(parse, solution) if solution not in (None, "") else None


def run_task(task: ARCTask, options: OrchestrationOptions, module, verify, rl_start_fn, max_iterations: int = 3,
             rl_wait: float = 600.0, parse=parse_llm_output) -> Dict[str, Any]:
    """One task through the system; returns the record of the run (without its key)."""
    tracer = Tracer()
    run = assemble(options, module, verify, tracer=tracer, parse=parse, rl_start_fn=rl_start_fn)
    context = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
    started = time.perf_counter()
    result = solve_with_orchestration(task, run, auxiliary_info=context, system_run_config=SystemRunConfig(
        agent_run_config=AgentRunConfig(max_agent_iterations=max_iterations, rl_wait_timeout=rl_wait)))
    seconds = time.perf_counter() - started
    grid = answer_of(result, parse)
    refinements = [{"attempts": len(r.attempts), "accepted": r.accepted, "notes": [a.notes for a in r.attempts]}
                   for r in getattr(run.module, "runs", [])]
    return {"task": task.label, "solved": bool(grid is not None and grid.shape == task.test_out.shape
                                               and (grid == task.test_out).all()),
            "validated": bool(result.get("validated")), "source": result.get("accepted_source"),
            "iterations": result.get("iteration"), "seconds": round(seconds, 2),
            "trace": {phase: {k: round(v, 2) for k, v in row.items()} for phase, row in tracer.summary().items()},
            "decisions": run.decisions, "refinements": refinements}


def summarise(lines: List[Dict[str, Any]]) -> str:
    """Solved share by option set, then the phases summed over the runs."""
    from collections import defaultdict
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in lines:
        groups[json.dumps(line["options"], sort_keys=True)].append(line)
    out = []
    for options, runs in groups.items():
        solved = sum(r["solved"] for r in runs)
        sources = defaultdict(int)
        for r in runs:
            sources[r["source"] or "none"] += 1
        phases: Dict[str, Dict[str, float]] = {}
        for r in runs:
            for phase, row in r["trace"].items():
                total = phases.setdefault(phase, {"calls": 0, "seconds": 0.0, "longest": 0.0, "tokens_in": 0,
                                                  "tokens_out": 0, "errors": 0})
                for k in ("calls", "seconds", "tokens_in", "tokens_out", "errors"):
                    total[k] += row[k]
                total["longest"] = max(total["longest"], row["longest"])
        out += [f"options {options}", f"  solved {solved} of {len(runs)}; answered by {dict(sources)}",
                f"  {sum(r['seconds'] for r in runs):.0f} s in all, {sum(r['seconds'] for r in runs) / len(runs):.0f} s a task",
                *("  " + row for row in render(phases).splitlines()), ""]
    return "\n".join(out)


def read_lines(path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["run", "summary"], default="run")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--tasks-file", help="a file of task ids, whitespace-separated")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--model", help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer")
    parser.add_argument("--options", default="{}", help="OrchestrationOptions fields as JSON")
    parser.add_argument("--no-rl", action="store_true", help="no RL job beside the model")
    parser.add_argument("--rl-steps", type=int, default=50_000, help="training steps of the RL job")
    parser.add_argument("--rl-wait", type=float, default=600.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--out", default="system.jsonl")
    args = parser.parse_args()

    if args.command == "summary":
        print(summarise(read_lines(args.out)))
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    options = OrchestrationOptions(**json.loads(args.options))
    ids = list(args.tasks) + (Path(args.tasks_file).read_text().split() if args.tasks_file else [])
    finished = done_keys(args.out)
    todo = [t for t in ids if key_of(t, options, not args.no_rl) not in finished]
    if not todo:
        print("nothing to run")
        return

    from orchestration.configs import local_experiment
    from orchestration.tools import without_grammar
    from subsymbolic.answer_check import LlmVerifier
    from subsymbolic.subsymbolic_module import SubsymbolicModule

    config = local_experiment(args.model, args.tokenizer)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception:  # noqa: BLE001 - counting tokens approximately is enough to run
        from subsymbolic.prompt_builder import ApproxTokenizer
        tokenizer = ApproxTokenizer()
    module = SubsymbolicModule(config, tokenizer)
    verify = LlmVerifier(without_grammar(module.runner), tokenizer, prompt=config.prompt)
    if args.no_rl:
        rl_start = lambda task: None  # noqa: E731
    else:
        from data.configs.rl_configs import rl_config
        from rl.rl_job import RLJobHandle, _rl_training_worker
        settings = copy.deepcopy(rl_config)
        settings["total_steps"] = args.rl_steps
        rl_start = lambda task: RLJobHandle(task, partial(_rl_training_worker, rl_config=copy.deepcopy(settings)))  # noqa: E731
    try:
        for task_id in todo:
            record = run_task(load_task(args.split, task_id), options, module, verify, rl_start,
                              args.max_iterations, args.rl_wait)
            record.update({"key": key_of(task_id, options, not args.no_rl), "options": vars(options),
                           "rl": not args.no_rl, "model": args.model, "split": args.split})
            with open(args.out, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{task_id}: solved={record['solved']} source={record['source']} {record['seconds']}s", flush=True)
    finally:
        module.close()
    print(summarise([r for r in read_lines(args.out)]))


if __name__ == "__main__":
    main()
