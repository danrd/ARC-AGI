"""ARC/PPO-specific pieces for the Optuna hyperparameter search in
rl.optimization: the PPO hyperparameter search space, and the objective
function that trains one subtask with one sampled hyperparameter set and
scores it - the "what to search over" and "how to score a trial" the
generic search loop doesn't know about.

Known limitation: arc_ppo_objective drives create_vec_env/create_agent/
agent.learn() exactly as rl.training's own train_on_subtask does - it
doesn't work around any pre-existing bugs in that pipeline. As of this
writing, a live end-to-end run still hits an unrelated observation_space
mismatch inside vec_env.reset() (declared objects_emb shape vs. actual
embedding shape) and a missing vec_env.shapes_match attribute in
create_agent - both pre-existing, both out of scope here. tests/
test_rl_arc_hp_search.py verifies this module's own wiring/logic against
mocked training instead of a live run; re-verify against a real run once
those are fixed.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from data.configs.rl_configs import load_PPO_config
from rl.arc_task import ARCSubtask
from rl.optimization import OptunaPruningCallback
from rl.training import create_agent, create_vec_env

NET_ARCHS = {
    "small": [128, 128, 128],
    "large": [256, 256, 256],
    "small_deep": [128, 128, 128, 128],
    "large_deep": [256, 256, 256, 256],
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


def sample_ppo_hyperparameters(trial) -> dict:
    """The PPO hyperparameter search space. Architectures/activations are
    sampled by name and mapped to the real value afterwards - passing
    lists/classes directly as suggest_categorical choices works, but
    Optuna warns that non-(None/bool/int/float/str) choices aren't safe
    for persistent (SQLite) storage, which run_hyperparameter_search uses."""
    return {
        "gamma": trial.suggest_categorical("gamma", [0.7, 0.8, 0.9, 0.95, 0.99]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.7, 0.8, 0.9, 0.95]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048]),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "n_epochs": trial.suggest_categorical("n_epochs", [3, 4]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.6, 0.7, 0.8, 0.9]),
        "actor_arch": NET_ARCHS[trial.suggest_categorical("actor_arch_name", list(NET_ARCHS))],
        "critic_arch": NET_ARCHS[trial.suggest_categorical("critic_arch_name", list(NET_ARCHS))],
        "activation_fn": ACTIVATIONS[trial.suggest_categorical("activation_fn_name", list(ACTIVATIONS))],
    }


def arc_ppo_objective(trial, subtask: ARCSubtask, rl_config: dict,
                       warmup_fraction: float = 0.7, dead_epsilon: float = 1e-6,
                       report_freq: int = 1000) -> float:
    """Train PPO on `subtask` with one sampled hyperparameter set, prune
    dead runs past `warmup_fraction` of the budget, and score by mean
    episode reward over stable-baselines3's own ep_info_buffer (last 100
    episodes) - not evaluate_ARC_policy/MonitorCallback, which have their
    own unrelated, unfixed bugs; this keeps the search independent of
    those."""
    PPO_config = load_PPO_config()
    PPO_config.update(sample_ppo_hyperparameters(trial))

    vec_env = create_vec_env(
        [subtask], n_envs=rl_config["n_envs"], max_episode_len=rl_config["max_episode_len"],
        repr_level=rl_config["repr_level"], right_placement_reward=rl_config["right_placement_reward"],
        action_penalty=rl_config["action_penalty"], repetitive_actions_penalty=rl_config["repetitive_actions_penalty"],
        seed=rl_config["seed"], font_color=rl_config["font_color"], padding=rl_config["padding"],
        input_pattern=rl_config["input_pattern"], milestones_rewards=rl_config["milestones_rewards"],
        pad_val=rl_config["pad_val"], reward_approach=rl_config["reward_approach"],
        feasible_actions=rl_config["feasible_actions"], observation_space_elements=rl_config["observation_space_elements"],
    )
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config)

    pruning_callback = OptunaPruningCallback(
        trial, total_steps=rl_config["total_steps"], warmup_fraction=warmup_fraction,
        dead_epsilon=dead_epsilon, report_freq=report_freq,
    )
    agent.learn(rl_config["total_steps"], callback=pruning_callback)

    if not agent.ep_info_buffer:
        return float("-inf")  # never completed a single episode - as bad as it gets
    return float(np.mean([ep["r"] for ep in agent.ep_info_buffer]))
