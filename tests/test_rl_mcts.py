"""Tests for the experience-collection utilities in rl/mcts.py
(collect_random_rollouts, MCTSNode, EnvironmentSimulator) against the real
environment. Mostly crash-or-not smoke tests, in the same spirit as
test_rl_env.py - these functions explore/search over the environment
rather than compute a single well-defined answer, so there's no simple
"known right result" to assert against for most of them.

One exception: test_environment_simulator_does_not_mutate_real_env is an
exact regression test for the bug this round's rework fixed - MCTS search
used to run real env.step() calls and try to undo them via get_state()/
set_state(), which never captured env.objects, so every simulated node
permanently corrupted the objects the real rollout depends on.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import rl.mcts as mcts
from rl.arc_env import ARCGridWorld

from .resource_utils import resource_budget

SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}


@pytest.fixture
def env(arc_task):
    e = ARCGridWorld(max_episode_len=4, feasible_actions=SUBMIT_AND_ROTATE)
    e.set_subtask(arc_task.subtasks[0])
    e.reset()
    return e


def test_the_search_does_not_enumerate_empty_object_slots(env):
    """The action space is padded to max_objects so every subtask has the
    same one - but indices past this subtask's objects name nothing, and
    the env scores them all as the same do-nothing action. Enumerating them
    costs (max_objects / n)^2: for a two-object grid at max_objects=16 that
    is 64 identical no-ops for every real action, and a search that tries
    them spends nearly all of its budget rediscovering that.
    """
    visible = env.visible_object_count()
    assert visible < env.max_objects, "fixture task should not fill every slot"

    actions = mcts.enumerate_actions(env)

    assert len(actions) == len(env.actions_dict) * visible * visible
    assert all(a[1] < visible and a[2] < visible for a in actions)
    # Still real actions, not a narrower space of their own.
    assert all(env.action_space.contains(np.array(a)) for a in actions)


def test_the_tree_expands_over_the_same_actions_it_tests(env):
    """Trimming one enumeration and not the other would leave the search
    exploring slots that phase 1 already knows are empty."""
    simulator = mcts.EnvironmentSimulator(env)
    search = mcts.MCTS(env, max_iterations=2, max_depth=2)

    assert [list(a) for a in search.all_actions] == mcts.enumerate_actions(env)
    assert simulator.all_actions == mcts.enumerate_actions(env)


def test_collect_random_rollouts_does_not_crash(env):
    # Also a resource guard: rollout collection deep-copies objects per
    # step (see EnvironmentSimulator's "copy only touched objects" design,
    # motivated by real memory constraints) - a regression back to copying
    # everything would still pass functionally but blow past this budget.
    with resource_budget(max_seconds=5.0, max_memory_mb=50.0):
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

    state = mcts.env_state_snapshot(env)
    next_state, reward, done, truncated, info = simulator.simulate_step(state, action)
    assert isinstance(next_state, dict)
    assert set(next_state.keys()) == {"grid", "objects", "max_int", "prev_action"}


def test_environment_simulator_does_not_mutate_real_env(env):
    """Regression test: simulate_step used to run a real env.step() and try
    to undo it via get_state()/set_state(), which never captured
    env.objects - so every simulated node permanently mutated the real
    objects. simulate_step must now leave env.objects/env.grid/env.max_int
    untouched no matter how many simulated steps it runs."""
    simulator = mcts.EnvironmentSimulator(env)
    action = np.array([1, 0, 0])  # rotate90 on object 0
    assert env.action_space.contains(action)

    objects_before = [deepcopy(obj) for obj in env.objects]
    grid_before = env.grid.copy()
    max_int_before = env.max_int

    state = mcts.env_state_snapshot(env)
    for _ in range(5):  # several simulated steps in a row, as expand()/simulate() would do
        state, reward, done, truncated, info = simulator.simulate_step(state, action)
        if done or truncated:
            break

    assert len(env.objects) == len(objects_before)
    for obj, obj_before in zip(env.objects, objects_before):
        assert tuple(obj.coords) == tuple(obj_before.coords)
    assert np.array_equal(env.grid, grid_before)
    assert env.max_int == max_int_before


def test_mcts_node_expand_and_simulate_does_not_crash(env):
    simulator = mcts.EnvironmentSimulator(env)
    root = mcts.MCTSNode(state=mcts.env_state_snapshot(env))

    child = root.expand(simulator)
    assert child is not None
    assert child.parent is root

    reward = child.simulate(simulator, max_depth=3)
    assert isinstance(reward, (int, float))

    child.backpropagate(reward)
    assert child.visits >= 1


def test_mcts_node_expand_children_have_independent_untried_actions(env):
    """Regression test: expand() used to hand each child the SAME
    untried_actions list the parent held, so popping an action for one
    child's own later expansion silently removed it from the parent's
    (and every sibling's) list too."""
    simulator = mcts.EnvironmentSimulator(env)
    root = mcts.MCTSNode(state=mcts.env_state_snapshot(env))
    root.is_fully_expanded(simulator)  # lazily initializes root.untried_actions as a side effect

    remaining_before = len(root.untried_actions)
    child = root.expand(simulator)
    assert child is not None
    assert child.untried_actions is not root.untried_actions
    assert len(root.untried_actions) == remaining_before - 1

    child.expand(simulator)  # pop from the child's own list
    assert len(root.untried_actions) == remaining_before - 1  # unaffected by the child's pop


def test_actions_exploration_runs_without_a_policy(arc_task):
    """The entry point the analyst will read from, and the one nothing
    covered. It passed the PPO agent where rollout_preparation wanted an
    environment - `AttributeError: 'PPO' object has no attribute 'reset'`,
    nine lines after a line that happened to work because an agent has an
    action_space too.

    Building that agent was also the only reason this path touched
    rl.policy, whose action_heads=3 branch raises IndexError before any
    search can start. MCTS reads the environment directly, so neither is
    needed: no policy, no training, nothing to fail before the search.
    """
    from data.configs.rl_configs import rl_config
    from rl.training import actions_exploration

    config = dict(rl_config)
    config.update(feasible_actions=SUBMIT_AND_ROTATE, max_episode_len=4,
                   observation_space_elements=["objects_emb"])

    promising = actions_exploration(arc_task.subtasks[0], config,
                                     n_rollouts=2, mcts_iterations=2, top_k=2)

    assert isinstance(promising, list)
    assert all(name in SUBMIT_AND_ROTATE.values() for name in promising)


def test_mcts_search_does_not_crash(env):
    # Same rationale as test_collect_random_rollouts_does_not_crash above:
    # tree expansion/simulation is where a "copy only touched objects"
    # regression would actually show up as runaway memory.
    search = mcts.MCTS(env, max_iterations=5, max_depth=3)
    root_state = mcts.env_state_snapshot(env)

    with resource_budget(max_seconds=5.0, max_memory_mb=50.0):
        root = search.search(root_state)

    best_action = search.get_best_action(root)
    assert env.action_space.contains(np.array(best_action))
