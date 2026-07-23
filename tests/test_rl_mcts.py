"""Tests for the experience-collection utilities in rl/mcts.py
(collect_random_rollouts, MCTSNode, EnvironmentSimulator) against the real
environment. Crash-or-not smoke tests, in the same spirit as
test_rl_env.py - these functions explore/search over the environment
rather than compute a single well-defined answer, so there's no simple
"known right result" to assert against.
"""
from __future__ import annotations

import numpy as np
import pytest

import rl.mcts as mcts
from rl.ARC_env import ARCGridWorld

SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}


@pytest.fixture
def env(arc_task):
    e = ARCGridWorld(max_episode_len=4, feasible_actions=SUBMIT_AND_ROTATE)
    e.set_subtask(arc_task.subtasks[0])
    e.reset()
    return e


def test_collect_random_rollouts_does_not_crash(env):
    rollouts = mcts.collect_random_rollouts(
        env, promising_actions=[], n_rollouts=3, max_episode_len=4,
    )
    assert isinstance(rollouts, list)
    for rollout in rollouts:
        assert rollout["total_reward"] > 0  # collect_random_rollouts only keeps these
        assert len(rollout["actions"]) == rollout["length"]
        assert len(rollout["observations"]) == len(rollout["actions"])


def test_environment_simulator_sample_and_step(env):
    simulator = mcts.EnvironmentSimulator(env)
    action = simulator.sample_action()

    assert env.action_space.contains(np.array(action))

    obs, reward, done, truncated, info = simulator.simulate_step(env.reset()[0], action)
    assert isinstance(obs, dict)


def test_mcts_node_expand_and_simulate_does_not_crash(env):
    simulator = mcts.EnvironmentSimulator(env)
    root = mcts.MCTSNode(observation=env.reset()[0])

    child = root.expand(simulator)
    assert child is not None
    assert child.parent is root

    reward = child.simulate(simulator, max_depth=3)
    assert isinstance(reward, (int, float))

    child.backpropagate(reward)
    assert child.visits >= 1


def test_mcts_search_does_not_crash(env):
    search = mcts.MCTS(env, max_iterations=5, max_depth=3)
    root_obs = env.reset()[0]

    root = search.search(root_obs)

    best_action = search.get_best_action(root)
    assert env.action_space.contains(np.array(best_action))
