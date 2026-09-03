"""Tests for rl/plotting.py. Two groups:

- Smoke tests against a real rollout collected on the actual environment
  (rl/arc_env.py) - not checking pixel output, just that these functions
  still work against the current env/rollout data shapes.
- Functional tests for plot_rollout_grid_trace's trajectory-review
  features (eventful-step prioritization, changed-cell highlighting,
  target-match %, the reward-per-step summary row) against a small
  synthetic rollout built by hand, where the exact shape of each step is
  known - these check specific properties (titles, patch counts), not
  just "doesn't crash".
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


def test_plot_rollout_grid_trace_missing_infos_does_not_crash():
    """Regression test: `infos` used to default to {} when the key was
    absent from the rollout dict, then get indexed with an int (infos[step_idx])
    - always a KeyError the moment include_descriptions=True (the default)."""
    rollout = {
        "observations": [{"grid": np.zeros((3, 3), dtype=int)}, {"grid": np.zeros((3, 3), dtype=int)}],
        "actions": [np.array([1, 0, 0]), np.array([1, 0, 0])],
        "rewards": [0.1, 0.2],
        # no "infos" key
    }

    fig = plot_rollout_grid_trace(rollout, action_mapping={0: "submit", 1: "rotate90"})

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rollout_grid_trace_empty_rollout_does_not_crash():
    """Regression test: an empty rollout used to raise ZeroDivisionError
    (n_cols computed as 0) and then a matplotlib GridSpec ValueError even
    after that - both fixed by an explicit early return."""
    rollout = {"observations": [], "actions": [], "rewards": [], "infos": []}

    fig = plot_rollout_grid_trace(rollout)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# -- trajectory-review features: synthetic rollout, exact shapes known ------

def _synthetic_rollout(n_steps, grid_shape=(3, 3)):
    """A rollout where step i's grid differs from step i-1's by exactly one
    cell (i % grid_shape[0], 0) - lets tests assert exact diff/match counts
    instead of just "did it crash"."""
    grids = []
    base = np.zeros(grid_shape, dtype=int)
    for i in range(n_steps):
        g = base.copy()
        g[i % grid_shape[0], 0] = 1
        grids.append(g)
        base = g
    rollout = {
        "observations": [{"grid": g} for g in grids],
        "actions": [np.array([1, 0, 0])] * n_steps,
        "rewards": [0.0] * n_steps,
        "infos": [{} for _ in range(n_steps)],
    }
    return rollout, grids


def _grid_axes(fig):
    return [ax for ax in fig.axes if ax.get_title().startswith("Step")]


def test_prioritize_eventful_steps_keeps_the_big_reward_step():
    """The whole point of prioritize_eventful_steps: a uniform sample over
    a long, mostly-flat episode can step right over the one step with a
    real reward swing. spike_step is deliberately NOT on the plain
    np.linspace(0, 19, 5) sample (see the sanity check below) - it must
    still show up once prioritization is on."""
    n_steps = 20
    rollout, _ = _synthetic_rollout(n_steps)
    spike_step = 7
    rollout["rewards"][spike_step] = 100.0

    fig = plot_rollout_grid_trace(rollout, num_steps_to_plot=5, include_descriptions=False,
                                   prioritize_eventful_steps=True)

    titles = [ax.get_title() for ax in _grid_axes(fig)]
    assert any(f"Step {spike_step}" in t for t in titles)
    plt.close(fig)


def test_uniform_sampling_would_have_missed_the_spike_step():
    """Sanity check for the test above: confirms step 7 genuinely isn't in
    the plain uniform np.linspace(0, 19, 5) sample, so that test is
    actually exercising the prioritization logic, not passing by luck."""
    uniform = set(np.linspace(0, 19, 5, dtype=int))
    assert 7 not in uniform


def test_highlight_changes_outlines_exactly_the_changed_cell():
    rollout, _ = _synthetic_rollout(3)

    fig = plot_rollout_grid_trace(rollout, include_descriptions=False, highlight_changes=True)

    grid_axes = _grid_axes(fig)
    assert len(grid_axes[0].patches) == 0  # first step: nothing to diff against
    assert len(grid_axes[1].patches) == 1  # each later step changes exactly 1 cell
    assert len(grid_axes[2].patches) == 1
    plt.close(fig)


def test_highlight_changes_disabled_adds_no_patches():
    rollout, _ = _synthetic_rollout(3)

    fig = plot_rollout_grid_trace(rollout, include_descriptions=False, highlight_changes=False)

    assert all(len(ax.patches) == 0 for ax in _grid_axes(fig))
    plt.close(fig)


def test_target_grid_shows_match_percentage_in_title():
    rollout, grids = _synthetic_rollout(3)
    target = grids[-1]  # the last step's grid is exactly the target -> 100% match

    fig = plot_rollout_grid_trace(rollout, include_descriptions=False, target_grid=target)

    assert "match: 100%" in _grid_axes(fig)[-1].get_title()
    plt.close(fig)


def test_reward_summary_row_present_when_rewards_exist():
    rollout, _ = _synthetic_rollout(3)
    rollout["rewards"] = [0.1, 0.2, 0.3]

    fig = plot_rollout_grid_trace(rollout, include_descriptions=False)

    assert any("Reward per step" in ax.get_title() for ax in fig.axes)
    plt.close(fig)


def test_reward_summary_row_absent_when_no_rewards():
    rollout = {
        "observations": [{"grid": np.zeros((3, 3), dtype=int)}],
        "actions": [np.array([0, 0, 0])],
        "rewards": [],
        "infos": [{}],
    }

    fig = plot_rollout_grid_trace(rollout, include_descriptions=False)

    assert not any("Reward per step" in ax.get_title() for ax in fig.axes)
    plt.close(fig)
