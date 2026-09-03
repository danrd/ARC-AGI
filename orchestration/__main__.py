"""Runnable stand-in for a real experiment driver: loads one ARC task and
runs it through the full coordinator -> agent -> module graph
(orchestration.graph.solve_task), using only what works with no extra setup
- a single agent with the symbolic module available. Subsymbolic/RL aren't
wired in here (they need a tokenizer/inference backend or a training config
to be worth running as a demo) - pass a real module_dispatch_fn (see
orchestration.graph.make_module_dispatch_fn) and rl_start_fn to solve_task()
for those.

Usage: python -m orchestration
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orchestration.graph import AgentInvConfig, ModuleInvConfig, solve_task
from rl.arc_task import ARCSubtask, ARCTask

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_one_task() -> ARCTask:
    """One fixed ARC task, picked deterministically (first key, sorted) from
    the training set - same approach as tests/conftest.py's arc_task fixture."""
    challenges_path = REPO_ROOT / "data" / "datasets" / "ARC" / "training_challenges.json"
    solutions_path = REPO_ROOT / "data" / "datasets" / "ARC" / "training_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)

    task_id = sorted(challenges.keys())[0]
    task_data = challenges[task_id]

    subtasks = [
        ARCSubtask(label=f"{task_id}_{i}", train_inp=np.array(pair["input"]),
                   train_out=np.array(pair["output"]))
        for i, pair in enumerate(task_data["train"])
    ]
    test_inp = np.array(task_data["test"][0]["input"])
    test_out = np.array(solutions[task_id][0])
    return ARCTask(label=task_id, subtasks=subtasks, test_inp=test_inp, test_out=test_out)


def main() -> None:
    task = _load_one_task()
    agent = AgentInvConfig(
        agent_index=0, agent_name="default",
        initial_module=ModuleInvConfig(module_index=0, module_name="symbolic"),
        available_modules=[{"index": 0, "name": "symbolic"}],
    )
    result = solve_task(task, initial_agent=agent, available_agents=[{"index": 0, "name": "default"}])

    print(f"task: {task.label}")
    print(f"validated: {result.get('validated')}")
    print(f"solution:\n{result.get('solution')}")


if __name__ == "__main__":
    main()
