"""Scoring a trained policy on the environments it was trained in.

Rewritten because the previous version could not have run: it read
`vec_env.test_env`, which `create_vec_env` never sets; it called `.append`
on the numpy array of dones; it iterated `range(n_envs + 1)` over arrays of
length `n_envs`; and it returned three lists of lists where every caller
unpacks three scalars and formats them with `:.2f`. The callers are the
specification here - `train_on_subtask` prints "Accuracy for X: {acc}" and
MonitorCallback stores one number per evaluation - so that is what this
returns.

Accuracy is the fraction of the distance closed, the same figure the search
is measured by:

    (max_int - base_int) / (target_int - base_int)

0.0 is the grid as it started and 1.0 is solved, which makes a trained
policy and a search comparable without either of them being scored in the
other's units.
"""
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv


def closed_fraction(env, grid=None) -> float:
    """How much of the distance to the target a grid has closed.

    Scored from `grid` rather than from the env's own max_int, because by
    the time a done is visible the vector has already reset that slot and
    its counters describe the next episode. The grid comes from the info's
    terminal_observation, which is the whole reason that key exists.

    Unwrapped first: what a vector holds is gymnasium's OrderEnforcing
    around the env, and the counters live on the env itself. base_int and
    target_int survive the reset - same env, same subtask - so only the
    intersection has to be recomputed.

    A subtask whose input already matches the target has no distance to
    close; it is scored 1.0 rather than dividing by zero.
    """
    env = getattr(env, "unwrapped", env)
    span = env.target_int - env.base_int
    if span <= 0:
        return 1.0
    reached = env.max_int if grid is None else env.maximal_intersection(grid)
    return float((reached - env.base_int) / span)


def evaluate_ARC_policy(
    model: "base_class.BaseAlgorithm",
    vec_env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> Tuple[float, float, Any]:
    """Run the policy until `n_eval_episodes` episodes have finished.

    Returns the mean accuracy over those episodes, their mean length, and
    the grid the last finished episode produced - what the caller prints
    and plots.

    `callback` is invoked after each step with the local scope, which is
    how MonitorCallback's success logging reads `reward`, `done` and
    `info`. Kept because that contract is used, odd as it is.
    """
    n_envs = vec_env.num_envs
    accuracies, lengths = [], []
    last_grid = None
    current_lengths = np.zeros(n_envs, dtype=int)
    observations = vec_env.reset()
    states = None

    while len(accuracies) < n_eval_episodes:
        actions, states = model.predict(observations, state=states,
                                        deterministic=deterministic)
        observations, rewards, dones, infos = vec_env.step(actions)
        current_lengths += 1
        for index in range(n_envs):
            reward, done, info = rewards[index], dones[index], infos[index]
            if callback is not None:
                callback(locals(), globals())
            if not done:
                continue
            terminal = info.get("terminal_observation") or {}
            grid = terminal.get("grid")
            if grid is not None:
                last_grid = grid
            accuracies.append(closed_fraction(vec_env.envs[index], grid))
            lengths.append(int(current_lengths[index]))
            current_lengths[index] = 0
            if len(accuracies) >= n_eval_episodes:
                break

    return (float(np.mean(accuracies)) if accuracies else 0.0,
            float(np.mean(lengths)) if lengths else 0.0,
            last_grid)
