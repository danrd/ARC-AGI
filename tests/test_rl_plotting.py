"""Smoke tests for rl/plotting.py: build a small rollout on the real
environment (rl/arc_env.py) and confirm the plotting functions run without
raising. Not checking pixel output - just that these functions still work
against the current env/rollout data shapes.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display needed to run these tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rl.arc_env import ARCGridWorld
from rl.plotting import plot_grids_comparison, plot_rollout_grid_trace

SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}


@pytest.fixture
def rollout_and_env(arc_task):
    """A short real rollout collected by stepping the actual environment -
    same observation/info shape the real training loop would produce."""
    env = ARCGridWorld(max_episode_len=3, feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(arc_task.subtasks[0])
    obs, _ = env.reset()

    rollout = {"observations": [], "actions": [], "rewards": [], "dones": [], "infos": []}
    for _ in range(2):
        action = np.array([1, 0, 0])  # rotate90
        rollout["observations"].append(obs)
        obs, reward, done, truncated, info = env.step(action)
        rollout["actions"].append(action)
        rollout["rewards"].append(reward)
        rollout["dones"].append(done)
        rollout["infos"].append(info)

    return rollout, env


def test_plot_rollout_grid_trace_does_not_crash(rollout_and_env):
    rollout, env = rollout_and_env

    fig = plot_rollout_grid_trace(rollout, action_mapping=env.actions_dict)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rollout_grid_trace_without_descriptions(rollout_and_env):
    """include_descriptions=False skips the get_step_description() path -
    covered separately since it reads a different set of rollout fields."""
    rollout, env = rollout_and_env

    fig = plot_rollout_grid_trace(rollout, action_mapping=env.actions_dict,
                                   include_descriptions=False)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_grids_comparison_does_not_crash(arc_task):
    subtask = arc_task.subtasks[0]

    # plot_grids_comparison calls plt.show() and doesn't return the figure -
    # just check it runs, then grab whatever got drawn via gcf().
    plot_grids_comparison(subtask.train_inp, subtask.train_out)

    assert plt.gcf().get_axes()
    plt.close("all")


def test_plot_grids_comparison_with_target_grid(arc_task):
    subtask = arc_task.subtasks[0]

    plot_grids_comparison(subtask.train_inp, subtask.train_out, target_grid=subtask.train_out)

    assert plt.gcf().get_axes()
    plt.close("all")
