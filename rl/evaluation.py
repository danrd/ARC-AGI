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

from data.configs.env_configs import COLORS_MAPPING


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
    # real_grid because what comes out of an observation may be padded to a
    # common shape for the buffer's sake (see ARCGridWorld.observed_grid),
    # and scoring that against the target compares two different shapes.
    reached = env.max_int if grid is None else \
        env.maximal_intersection(env.real_grid(grid))
    return float((reached - env.base_int) / span)


def describe_slot(objects, index) -> str:
    """One object as a reader can find it on the grid: its colours, its
    size, and the rows and columns it spans."""
    if index >= len(objects):
        return "empty slot"
    obj = objects[index]
    colours = "/".join(COLORS_MAPPING.get(int(c), str(int(c))) for c in obj.color_numbers)
    return (f"[{colours}: {obj.size} cells, rows {obj.min_i}-{obj.max_i}, "
            f"cols {obj.min_j}-{obj.max_j}]")


def step_label(env, action) -> str:
    """One step of an episode in words a reader can check against the grid.

    Read before the step is taken: an object action names slots, and the
    objects are rebuilt after every step, so slot 2 at step three is not the
    object slot 2 named at step one. A coordinate action names its two
    cells directly, as (row, column). The action itself is its vocabulary
    name with the underscores dropped - "red fill", "black contour
    connection yellow" - a leading colour being the one it paints with.
    """
    env = getattr(env, "unwrapped", env)
    # Through the whitelist when there is one: the policy then picks an
    # index into it, and [index, 0, 0] is not the triple that gets applied.
    action = [int(value) for value in np.asarray(env.resolved_action(action)).reshape(-1)]
    name = env.actions_dict.get(action[0], str(action[0]))
    if name == "submit":
        return "submit"
    verb = name.replace("_", " ")
    if getattr(env, "addressing", "objects") == "coordinates":
        return f"{verb} ({action[1]},{action[2]})-({action[3]},{action[4]})"
    objects = env.objects
    first = describe_slot(objects, action[1])
    if action[2] == action[1]:
        return f"{verb} {first}"
    return f"{verb} {first} & {describe_slot(objects, action[2])}"


def evaluate_ARC_policy(
    model: "base_class.BaseAlgorithm",
    vec_env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Any]:
    """Run the policy until `n_eval_episodes` episodes have finished.

    Returns the mean accuracy over those episodes, their mean length, and
    the grid the last finished episode produced - what the caller prints
    and plots.

    `callback` is invoked after each step with the local scope, which is
    how MonitorCallback's success logging reads `reward`, `done` and
    `info`. Kept because that contract is used, odd as it is.

    `trace`, when a dict is passed, is filled with the episode the returned
    grid came from - what rl.plotting.plot_evaluation draws - so an
    accuracy can be read against what the policy actually did:

      start     the grid the episode began on
      steps     one (label, grid after it, reward) per step, the label
                from step_label
      target    the grid it was meant to reach
      accuracy  its closed fraction, the number this function averages
      subtask   which example it was
    """
    n_envs = vec_env.num_envs
    accuracies, lengths = [], []
    last_grid = None
    current_lengths = np.zeros(n_envs, dtype=int)
    observations = vec_env.reset()
    states = None
    # Per env, the episode in progress - only kept when a trace is wanted.
    episodes = [None] * n_envs
    if trace is not None:
        episodes = [{"start": np.array(observations["grid"][index]), "steps": []}
                    for index in range(n_envs)]

    while len(accuracies) < n_eval_episodes:
        actions, states = model.predict(observations, state=states,
                                        deterministic=deterministic)
        labels = ([step_label(vec_env.envs[index], actions[index]) for index in range(n_envs)]
                  if trace is not None else None)
        observations, rewards, dones, infos = vec_env.step(actions)
        current_lengths += 1
        for index in range(n_envs):
            reward, done, info = rewards[index], dones[index], infos[index]
            if callback is not None:
                callback(locals(), globals())
            if trace is not None:
                after = ((info.get("terminal_observation") or {}).get("grid")
                         if done else observations["grid"][index])
                episodes[index]["steps"].append((labels[index], np.array(after),
                                                 float(reward)))
            if not done:
                continue
            terminal = info.get("terminal_observation") or {}
            grid = terminal.get("grid")
            if grid is not None:
                last_grid = grid
            accuracies.append(closed_fraction(vec_env.envs[index], grid))
            lengths.append(int(current_lengths[index]))
            current_lengths[index] = 0
            if trace is not None:
                env = getattr(vec_env.envs[index], "unwrapped", vec_env.envs[index])
                trace.clear()
                trace.update(episodes[index], target=np.array(env.train_out),
                             accuracy=accuracies[-1],
                             subtask=getattr(env, "subtask_label", None))
                episodes[index] = {"start": np.array(observations["grid"][index]),
                                   "steps": []}
            if len(accuracies) >= n_eval_episodes:
                break

    return (float(np.mean(accuracies)) if accuracies else 0.0,
            float(np.mean(lengths)) if lengths else 0.0,
            last_grid)
