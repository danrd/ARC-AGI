"""ARC-specific pieces for the Optuna search in rl.optimization: what to
vary, what to train, and what to call a good trial - the three things the
generic loop cannot know.

All three are arguments. `sample` is the search space, so the same
objective can tune PPO, the ARC-specific settings, or both at once;
`score` is the number Optuna maximises; `mode` is how the task is trained.
Nothing here decides for the caller which of those is interesting.

`score` matters more than it looks. The objective used to be the mean
episode reward out of stable-baselines3's own buffer, which is training
reward under whichever reward_approach the env was built with - not
comparable between reward settings, and not the thing the agent is finally
judged on. Those two have already been measured moving in opposite
directions: under reward_approach 2 the agent stopped giving up, its
reward rose, and the fraction of the distance it actually closed fell on
two of three tasks. A search optimising reward would have picked that.
"held_out" is the default for the same reason every sweep in this
repository reads it: it is the only number taken on a pair the policy
never trained on.

From a notebook, where Optuna wants a one-argument callable:

    from functools import partial
    from data.configs.rl_configs import rl_config
    from rl.arc_hp_search import arc_objective
    from rl.optimization import run_hyperparameter_search

    config = dict(rl_config, total_steps=150_000, seed=42,
                  feasible_actions=narrowed, max_objects=slots)
    study = run_hyperparameter_search(
        partial(arc_objective, task=task, config=config),
        n_trials=40, study_name="ppo_over_the_best_observation")
    study.best_params

`config` is the ARC half and is not mutated, so one dict serves every
trial. To search the ARC half instead, or both, pass a `sample` of your
own - it returns a flat dict and the keys are routed by which config
holds them:

    def sample(trial):
        return {"gamma": trial.suggest_categorical("gamma", [0.9, 0.99]),
                "action_penalty": trial.suggest_float("action_penalty", 0, 1)}

150_000 steps is the budget below which nothing is measurable - see
rl_config's own note on total_steps - so a 40-trial study is a day of
compute, not an afternoon. Pruning only ever drops a run that has shown
no movement at all, never one that is merely behind.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from typing import Callable

from data.configs.rl_configs import load_PPO_config, rl_config
from rl.optimization import OptunaPruningCallback
from rl.training import train_on_task

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


#: What a finished trial can be scored on. Each takes the four things
#: train_on_task hands back and returns one number to maximise.
#:
#: held_out        the fraction of the distance to the target closed on the
#:                 pair the policy never trained on - what every sweep in
#:                 this repository reports
#: mixed           the same fraction averaged over the training pairs. Says
#:                 whether the task was fitted, not whether the rule was
#: episode_reward  the mean episode reward SB3 itself tracked. Denominated
#:                 in whichever reward_approach the env used, so it cannot
#:                 compare two of them - and it is training reward
#: episode_len     mean episode length, the tell for giving up: under a
#:                 reward that pays for stopping, every episode is one step
#:                 long. Negated, so that maximising it means "keep acting"
SCORERS = {
    "held_out": lambda accuracies, lens, agent, metrics:
        float(metrics.get("test_acc", float("-inf"))),
    "mixed": lambda accuracies, lens, agent, metrics:
        float(np.mean(list(accuracies.values()))) if accuracies else float("-inf"),
    "episode_reward": lambda accuracies, lens, agent, metrics:
        float(np.mean([ep["r"] for ep in agent.ep_info_buffer]))
        if getattr(agent, "ep_info_buffer", None) else float("-inf"),
    "episode_len": lambda accuracies, lens, agent, metrics:
        float(np.mean(list(lens.values()))) if lens else float("-inf"),
}


def split_settings(sampled: dict) -> tuple:
    """A flat dict of sampled values, split into the PPO half and the ARC
    half by which config each key belongs to.

    Routed rather than declared, so a search space can name settings from
    either side without the caller having to say which is which - and a
    name belonging to neither raises instead of being silently dropped,
    which is how a sweep comes to measure the default under another name.
    """
    ppo_keys = set(load_PPO_config())
    arc_keys = set(rl_config)
    both = ppo_keys & arc_keys
    ppo, arc, unknown = {}, {}, []
    for key, value in sampled.items():
        if key in both:
            raise ValueError(f"{key!r} exists in both configs; the sampler "
                             f"cannot say which one it means")
        if key in ppo_keys:
            ppo[key] = value
        elif key in arc_keys:
            arc[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"sampled settings belong to no config: {unknown}")
    return ppo, arc


def arc_objective(trial, task, config: dict, *,
                  sample: Callable = sample_ppo_hyperparameters,
                  score="held_out", mode: str = "mixed",
                  warmup_fraction: float = 0.7, dead_epsilon: float = 1e-6,
                  report_freq: int = 1000) -> float:
    """One trial: sample, train the whole task, score it.

    `task` is a task and not one of its examples, because held-out is the
    point - a subtask has no pair left over to be scored on. `mode` is
    passed through to train_on_task, so 'sequential' is available for
    whoever wants the other experiment.

    `score` is a name from SCORERS or a callable taking the same four
    arguments. `sample` returns a flat dict of settings from either config;
    see split_settings.

    Dead runs are pruned past `warmup_fraction` of the budget - not ranked
    against other trials, see OptunaPruningCallback.
    """
    scorer = SCORERS[score] if isinstance(score, str) else score
    ppo_updates, arc_updates = split_settings(sample(trial))

    PPO_config = load_PPO_config()
    PPO_config.update(ppo_updates)
    run_config = dict(config, **arc_updates)

    pruning_callback = OptunaPruningCallback(
        trial, total_steps=run_config["total_steps"],
        warmup_fraction=warmup_fraction, dead_epsilon=dead_epsilon,
        report_freq=report_freq,
    )
    accuracies, lens, agent, metrics = train_on_task(
        task=task, rl_config=run_config, PPO_config=PPO_config, mode=mode,
        extra_callback=pruning_callback,
        # Not in a sweep: dozens of trials, each drawing every evaluation,
        # would bury the study's own output in figures.
        show_plots=False)
    return scorer(accuracies, lens, agent, metrics)
