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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RlConfig(BaseModel):
    """Env/reward/training settings for one RL run, by analogy with
    subsymbolic.llm_setup.LlmConfig / subsymbolic.llm_runtime.GenerationConfig.
    PPO hyperparameters stay a plain dict (see rl.rl_job.default_rl_start_fn) -
    they carry non-pydantic-friendly objects (a policy class, an nn.Module,
    a learning-rate schedule function), unlike these plain values."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)

    model_type: str = "PPO"
    total_steps: int = 1000000
    n_eval_episodes: int = 1
    n_envs: int = 1
    seed: int = 42
    eval_freq: int = 5
    log_path: str = ".data/logs/rl/"
    max_episode_len: int = 25
    right_placement_reward: float = 5.0
    action_penalty: float = 1.0
    repetitive_actions_penalty: float = 1.0
    font_color: float = 0.0
    padding: bool = False
    input_pattern: str = "start"
    milestones_rewards: List[int] = [1, 2, 3, 4]
    reward_approach: int = 3
    pad_val: int = 10
    feasible_actions: Dict[int, str] = Field(default_factory=lambda: {0: "submit"})
    repr_level: int = 1
    observation_space_elements: List[str] = ["objects_emb"]  # ["objects_emb", "relations_emb"]


class RLModule:
    def __init__(self, rl_config: RlConfig, ppo_config: Optional[dict] = None):
        self.rl_config = rl_config
        self.ppo_config = ppo_config

    def solve(self, task) -> Dict[str, Any]:
        from rl.training import train_on_task

        accs, lens, agent, train_metrics = train_on_task(
            task=task, rl_config=self.rl_config.model_dump(), PPO_config=self.ppo_config,
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
