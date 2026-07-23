"""Tests for the ARCGridWorld environment (rl/arc_env.py) - environment
init, basic lifecycle methods, and dispatching transformations through
World. Part 1 of the RL test plan (environment); MCTS rollout collection
and plotting are covered in their own test modules.

Same philosophy as the LLM smoke test: a handful of exact tests where the
right answer is known by construction, plus broader "does it crash" smoke
tests for everything else - this environment's surface (reward shaping,
observation assembly, transformation dispatch through World) is too large
to hand-verify exhaustively for every code path.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl.arc_env import ARCGridWorld

SUBMIT_ONLY = {0: "submit"}
SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}


@pytest.fixture
def subtask(arc_task):
    return arc_task.subtasks[0]


def make_env(**kwargs) -> ARCGridWorld:
    kwargs.setdefault("max_episode_len", 5)
    kwargs.setdefault("feasible_actions", SUBMIT_ONLY)
    return ARCGridWorld(**kwargs)


# -- exact tests: known answer by construction -------------------------------

def test_maximal_intersection_exact():
    """Matching cells count positive, mismatching cells count negative,
    cells padded in either grid are excluded entirely."""
    env = make_env(pad_val=10)
    env.train_out = np.array([[1, 2, 10], [3, 10, 10]])
    grid = np.array([[1, 9, 10], [3, 10, 10]])
    # matches: (0,0), (1,0) = 2; mismatches: (0,1) = 1 (both non-pad);
    # rest excluded (train_out is padded there) -> 2 - 1 = 1
    assert env.maximal_intersection(grid) == 1


def test_step_intersection_tracks_delta():
    env = make_env(pad_val=10)
    env.train_out = np.array([[1, 1], [1, 1]])
    env.max_int = 0
    env.target_int = 4
    grid = np.array([[1, 1], [1, 1]])

    right_placement, done = env.step_intersection(grid)

    assert right_placement == 4  # went from 0 matches to 4
    assert bool(done)  # max_int reached target_int
    assert env.max_int == 4


# -- lifecycle: set_subtask / reset ------------------------------------------

def test_set_subtask_and_reset_produces_valid_observation(subtask):
    env = make_env()
    env.set_subtask(subtask)
    obs, info = env.reset()

    assert {"grid", "action_space"} <= obs.keys()
    assert obs["grid"].shape == subtask.train_out_shape
    assert len(env.objects) > 0
    assert env.action_space.nvec[1] == len(env.objects)
    assert env.action_space.nvec[2] == len(env.objects)


def test_reset_returns_to_the_same_starting_state(subtask):
    """reset() should bring episode-local state back to the same starting
    point every time, not accumulate state across resets."""
    env = make_env()
    env.set_subtask(subtask)
    obs1, _ = env.reset()
    env.step(np.array([0, 0, 0]))  # submit, ends the episode
    obs2, _ = env.reset()

    assert np.array_equal(obs1["grid"], obs2["grid"])
    assert env.step_no == 0


# -- basic step / submit -----------------------------------------------------

def test_submit_action_terminates_immediately(subtask):
    env = make_env()
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([0, 0, 0]))

    assert done is True
    assert isinstance(reward, (int, float, np.integer, np.floating))


def test_episode_terminates_at_max_episode_len(subtask):
    """Without ever submitting, the episode still ends once
    max_episode_len steps have been taken."""
    env = make_env(max_episode_len=3, feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    done = False
    steps = 0
    for _ in range(10):  # safety bound well above max_episode_len
        _, _, done, _, _ = env.step(np.array([1, 0, 0]))  # rotate90, never submits
        steps += 1
        if done:
            break

    assert done is True
    assert steps == 3


# -- calling a real transformation through World -----------------------------

def test_step_dispatches_a_real_transformation(subtask):
    """A non-submit action should route through World.step ->
    arc_transformators and come back with a well-formed observation, not
    just the submit shortcut."""
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([1, 0, 0]))

    assert obs["grid"].shape == env.grid.shape
    assert "change_of_grid" in info
    assert isinstance(reward, (int, float, np.integer, np.floating))


# -- state save/restore -------------------------------------------------------

def test_get_set_state_roundtrip(subtask):
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()
    env.step(np.array([1, 0, 0]))

    state = env.get_state()
    grid_before = env.grid.copy()
    step_no_before = env.step_no

    env.step(np.array([1, 0, 0]))
    env.set_state(state)

    assert np.array_equal(env.grid, grid_before)
    assert env.step_no == step_no_before


# -- full random episode: crash-or-not smoke test -----------------------------

def test_full_random_episode_does_not_crash(subtask):
    env = make_env(max_episode_len=8, feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    done = False
    for _ in range(env.max_episode_len + 2):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        if done:
            break

    assert done is True


# -- reward_approach: sweep the working ones, track the broken one -----------

@pytest.mark.parametrize("reward_approach", [1, 2, 3])
def test_reward_approach_submit_does_not_crash(subtask, reward_approach):
    env = make_env(reward_approach=reward_approach)
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([0, 0, 0]))

    assert done is True


def test_reward_approach_4_is_currently_broken(subtask):
    """Regression tracker, not desired behavior: reward_approach == 4
    reads self.max_reward_base, which is never set anywhere in
    ARCGridWorld. If this starts passing, the bug's been fixed - update or
    remove this test rather than leaving it pinned to the old behavior."""
    env = make_env(reward_approach=4)
    env.set_subtask(subtask)
    env.reset()

    with pytest.raises(AttributeError):
        env.step(np.array([0, 0, 0]))
