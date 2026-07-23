"""Aggregates the RL-training solving path: trains a PPO agent on a task
via rl.training.train_on_task.

Symmetric to symbolic.symbolic_module.SymbolicModule /
subsymbolic.subsymbolic_module.SubsymbolicModule: this is what actually
runs inside the background training subprocess started by
rl.rl_job.RLJobHandle — training an RL policy takes real wall-clock time,
unlike the other two modules' calls, which is exactly why it needs to run
in the background rather than block the agent-level graph.

For now: just train_on_task, the one entry point that already exists.
Loading a previously-trained policy and logging can layer on top later.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from rl.training import train_on_task


class RLModule:
    def __init__(self, rl_config: dict, ppo_config: Optional[dict] = None):
        self.rl_config = rl_config
        self.ppo_config = ppo_config

    def solve(self, task) -> Dict[str, Any]:
        accs, lens, agent, train_metrics = train_on_task(
            task=task, rl_config=self.rl_config, PPO_config=self.ppo_config,
        )
        self.agent = agent
        return {
            "solution": None,  # train_on_task trains a policy, not a grid prediction
            "module_results": {
                "accuracies": accs,
                "episode_lengths": lens,
                "train_metrics": train_metrics,
            },
        }
