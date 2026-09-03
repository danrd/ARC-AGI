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

import collections
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

    actions = mcts.enumerate_actions(env, include_submit=True)

    assert len(actions) == len(env.actions_dict) * visible * visible
    assert all(a[1] < visible and a[2] < visible for a in actions)
    # Still real actions, not a narrower space of their own.
    assert all(env.action_space.contains(np.array(a)) for a in actions)


class TestSubmitIsNotSearched:
    """A search knows when it has reached the target, so it never needs to
    decide to submit - and the action can only end an episode early. Over
    four measured configurations it ended 19-26% of them, each cut short of
    the cap by an action that cannot improve the grid, while also putting a
    terminal child under every node whose negative value dragged the
    branch's UCB down.
    """

    def test_submit_is_absent_from_the_pool(self, env):
        actions = mcts.enumerate_actions(env)
        index = mcts.submit_index(env)

        assert index is not None, "fixture vocabulary should contain submit"
        assert all(a[0] != index for a in actions)

    def test_dropping_it_removes_exactly_one_action_per_object_pair(self, env):
        visible = env.visible_object_count()

        assert len(mcts.enumerate_actions(env)) == (
            len(env.actions_dict) - 1) * visible * visible

    def test_it_can_be_asked_for_when_the_submit_reward_is_the_subject(self, env):
        """Measuring the submit reward needs submit in the tree: that is the
        only route by which it reaches a node value at all."""
        index = mcts.submit_index(env)

        assert any(a[0] == index
                   for a in mcts.enumerate_actions(env, include_submit=True))

    def test_a_vocabulary_without_submit_is_left_alone(self, arc_task):
        """Nothing to drop is not an error - and must not silently drop
        index 0, which in such a vocabulary names a real transform."""
        e = ARCGridWorld(max_episode_len=4, feasible_actions={0: "rotate90"})
        e.set_subtask(arc_task.subtasks[0])
        e.reset()

        assert mcts.submit_index(e) is None
        assert mcts.enumerate_actions(e) == mcts.enumerate_actions(e, include_submit=True)
        assert any(a[0] == 0 for a in mcts.enumerate_actions(e))


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
    # Every rollout comes back, whatever it scored - selecting on
    # total_reward > 0 here discarded the whole run, since on this reward
    # scale a total is 0.0 or slightly negative almost always.
    assert len(rollouts) == 3
    for rollout in rollouts:
        assert len(rollout["actions"]) == rollout["length"]
        assert len(rollout["observations"]) == len(rollout["actions"])
        # Not any(dones): submitting ends the episode with done=True
        # whatever was submitted, so that would call every rollout solved.
        assert rollout["solved"] == (env.max_int == env.target_int)


@pytest.mark.parametrize("reward_approach", [1, 2, 3])
def test_a_rollout_records_progress_and_not_only_reward(env, reward_approach):
    """The one part of the record two reward approaches can be compared on.

    total_reward is denominated in whichever approach the env was built
    with, and they do not share a scale - approach 1 pays -4..+4 across the
    milestones where approach 2 pays -4..+10 - so a search doing better can
    score lower, and comparing the two on reward answers nothing. The
    intersection is the same count of cells in every approach.
    """
    env.reward_approach = reward_approach
    env.reset()

    rollouts = mcts.collect_random_rollouts(
        env, promising_actions=[], n_rollouts=3, max_episode_len=4)

    for rollout in rollouts:
        assert rollout["base_int"] <= rollout["target_int"]
        assert rollout["max_int"] <= rollout["target_int"]
        assert rollout["solved"] == (rollout["max_int"] == rollout["target_int"])


# -- selection: what survives the filters ------------------------------------

def _rollout(total_reward, length, solved=False, action_types=()):
    return {
        "total_reward": total_reward,
        "length": length,
        "solved": solved,
        "dones": [False] * (length - 1) + [solved] if length else [],
        "actions": [np.array([a, 0, 0]) for a in action_types],
    }


def test_a_solved_rollout_is_never_dropped_for_being_short():
    """An episode ends the moment the intersection reaches the target, so a
    solution is short *because* it worked. The old min_len=5 cut exactly
    those and kept the wandering."""
    solution = _rollout(total_reward=3.0, length=2, solved=True)
    wandering = _rollout(total_reward=-1.0, length=20)

    selected = mcts.select_best_rollouts([wandering, solution], top_k=5, min_len=5)

    assert solution in selected


def test_a_solved_rollout_outranks_a_higher_scoring_unsolved_one():
    """Reward alone can rank a long partial-credit rollout above a finished
    one, and the caller reads the list top-down."""
    solution = _rollout(total_reward=1.0, length=2, solved=True)
    lucky = _rollout(total_reward=9.0, length=20)

    selected = mcts.select_best_rollouts([lucky, solution], top_k=5)

    assert selected[0] is solution


def test_selection_keeps_negative_rollouts():
    """There is usually nothing else: on this reward scale a total is 0.0 or
    slightly negative almost always, so anything that only kept positives
    returned an empty list every time."""
    rollouts = [_rollout(-1.0, 3), _rollout(-4.0, 6), _rollout(0.0, 2)]

    selected = mcts.select_best_rollouts(rollouts, top_k=2)

    assert [r["total_reward"] for r in selected] == [0.0, -1.0]


def test_promising_actions_fall_back_to_the_best_when_none_clears_the_bar():
    results = {(1, 0, 0): {"reward": -1.0}, (2, 0, 0): {"reward": -0.2},
               (3, 0, 0): {"reward": -5.0}}

    assert mcts.identify_promising_actions(results, reward_threshold=0.0) == []
    fallback = mcts.identify_promising_actions(results, reward_threshold=0.0, keep_best=2)

    assert fallback == [(2, 0, 0), (1, 0, 0)]  # least bad first


def test_extract_promising_actions_names_the_transforms_by_use():
    """It read an 'action_mapping' key out of infos that ARCGridWorld has
    never produced, and iterated feasible_actions as if it held names rather
    than {index: name} - comparing an int against a string. Neither could
    run; the empty-rollouts filter upstream meant neither ever did."""
    feasible = {0: "submit", 1: "rotate90", 2: "fliplr"}
    rollouts = [_rollout(1.0, 3, action_types=(1, 2, 1)),
                _rollout(0.5, 2, action_types=(1, 0))]

    names = mcts.extract_promising_actions(rollouts, feasible)

    assert names[0] == "rotate90"          # used three times across the two
    assert set(names) == {"rotate90", "fliplr", "submit"}


def test_extract_promising_actions_on_nothing_returns_nothing():
    assert mcts.extract_promising_actions([], {0: "submit"}) == []


# -- iterative pruning --------------------------------------------------------

def _rollout_with(actions, total_reward, solved=False):
    return {"total_reward": total_reward, "length": len(actions), "solved": solved,
            "dones": [False] * (len(actions) - 1) + [solved],
            "actions": [np.array(a) for a in actions]}


class TestPruning:
    POOL = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]

    def test_an_action_is_scored_by_the_mean_of_its_rollouts_not_the_sum(self):
        """An action tried twenty times would otherwise outrank a better one
        tried twice purely by turning up more often."""
        rollouts = [_rollout_with([[1, 0, 0]], 1.0), _rollout_with([[1, 0, 0]], 1.0),
                    _rollout_with([[2, 0, 0]], 5.0)]

        stats = mcts.action_statistics(rollouts)

        assert stats[(1, 0, 0)]["mean_reward"] == 1.0
        assert stats[(2, 0, 0)]["mean_reward"] == 5.0
        assert stats[(1, 0, 0)]["uses"] == 2

    def test_repeated_use_within_one_rollout_counts_once_toward_its_score(self):
        rollouts = [_rollout_with([[1, 0, 0], [1, 0, 0], [1, 0, 0]], 3.0)]

        stats = mcts.action_statistics(rollouts)

        assert stats[(1, 0, 0)]["uses"] == 3
        assert stats[(1, 0, 0)]["rollouts"] == 1
        assert stats[(1, 0, 0)]["mean_reward"] == 3.0

    def test_the_pool_shrinks_by_the_fraction(self):
        rollouts = [_rollout_with([[i, 0, 0]], float(i)) for i in range(6)]

        kept = mcts.prune_actions(self.POOL, rollouts, keep_fraction=0.5, min_pool=2)

        assert len(kept) == 3
        assert [5, 0, 0] in kept and [0, 0, 0] not in kept

    def test_an_action_that_solved_survives_the_fraction(self):
        """The same mistake the length filter used to make: honouring a ratio
        by discarding something that finished the task."""
        rollouts = [_rollout_with([[0, 0, 0]], -9.0, solved=True)]
        rollouts += [_rollout_with([[i, 0, 0]], float(i)) for i in range(1, 6)]

        kept = mcts.prune_actions(self.POOL, rollouts, keep_fraction=0.34, min_pool=2)

        assert [0, 0, 0] in kept

    def test_an_action_no_rollout_reached_for_is_dropped_first(self):
        """The search had the chance and did not take it - that is weaker
        evidence than a bad score, not stronger."""
        rollouts = [_rollout_with([[1, 0, 0]], -5.0), _rollout_with([[2, 0, 0]], -6.0)]

        kept = mcts.prune_actions(self.POOL, rollouts, keep_fraction=0.5, min_pool=2)

        assert kept == [[1, 0, 0], [2, 0, 0]]

    def test_a_small_pool_is_left_alone(self):
        small = self.POOL[:3]

        assert mcts.prune_actions(small, [_rollout_with([[0, 0, 0]], 1.0)],
                                   keep_fraction=0.5, min_pool=4) == small

    def test_pruning_preserves_the_pools_order_and_shape(self):
        rollouts = [_rollout_with([[i, 0, 0]], float(i)) for i in range(6)]

        kept = mcts.prune_actions(self.POOL, rollouts, keep_fraction=0.5, min_pool=2)

        assert all(action in self.POOL for action in kept)
        assert kept == [a for a in self.POOL if a in kept]  # original order


@pytest.fixture
def wide_env():
    """Several objects and several real transform names, so there is a pool
    to prune. The default fixture's task has one object and two actions -
    a pool of two, which prune_actions leaves alone by design."""
    from rl.arc_task import ARCSubtask

    grid = np.zeros((8, 8), dtype=int)
    for k, (i, j) in enumerate([(1, 1), (1, 5), (5, 1), (5, 5)]):
        grid[i, j] = k + 1
    out = grid.copy()
    out[1, 1] = 9

    actions = {0: "submit", 1: "rotate90", 2: "fliplr", 3: "flipud",
               4: "color_inversion", 5: "x_alignment", 6: "y_alignment"}
    e = ARCGridWorld(max_episode_len=4, feasible_actions=actions, repr_level=1,
                     input_pattern="start", observation_space_elements=["objects_emb"])
    e.set_subtask(ARCSubtask("wide", grid, out))
    e.reset()
    return e


def test_rounds_narrow_the_pool_the_search_runs_over(wide_env, capsys):
    """The point of the loop: each round re-searches only what survived, so
    the depth reachable within one iteration budget grows as branching drops."""
    mcts.rollout_preparation(wide_env, method="mcts", n_initial_rollouts=2,
                              mcts_iterations=2, top_k=2, n_rounds=3, keep_fraction=0.5,
                              min_pool=2)

    sizes = [int(line.split("over ")[1].split(" actions")[0])
             for line in capsys.readouterr().out.splitlines() if "Phase 3." in line]

    assert len(sizes) >= 2, "expected more than one round"
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] < sizes[0]


def test_environment_simulator_sample_and_step(env):
    simulator = mcts.EnvironmentSimulator(env)
    action = simulator.sample_action()

    assert env.action_space.contains(np.array(action))

    state = mcts.env_state_snapshot(env)
    next_state, reward, done, truncated, info = simulator.simulate_step(state, action)
    assert isinstance(next_state, dict)
    assert set(next_state.keys()) == {"grid", "objects", "max_int", "prev_action"}


class TestCapturingASolutionAPlayoutFound:
    """A playout that matches the target exactly used to return a number and
    drop the actions that got it there. The rollouts a search returns are
    built from the tree's chosen action at each real step and never see
    inside a playout, so the find had nowhere to go.

    Rare - twice over eight tasks - but twice was the whole of what the
    search had to show, and a trace corpus that starts empty stays empty.
    """

    def _solving_env(self):
        """A task one action finishes, so a playout reaches the target
        quickly enough for a small search to find it."""
        from rl.arc_task import ARCSubtask

        grid = np.zeros((4, 4), dtype=int)
        grid[1, 1] = 3
        out = grid.copy()
        out[1, 1] = 5
        env = ARCGridWorld(max_episode_len=6,
                           feasible_actions={0: "submit", 1: "gray_recolor"},
                           repr_level=1, input_pattern="start",
                           observation_space_elements=["objects_emb"])
        env.set_subtask(ARCSubtask("one_step", grid, out))
        env.reset()
        return env

    def test_the_actions_that_reached_the_target_are_kept(self):
        env = self._solving_env()
        search = mcts.MCTS(env, max_iterations=20, max_depth=4)

        search.search(mcts.env_state_snapshot(env))

        assert search.env_simulator.solutions, "the playout solved it and said nothing"

    def test_a_kept_solution_replays_into_a_solved_rollout(self):
        env = self._solving_env()
        search = mcts.MCTS(env, max_iterations=20, max_depth=4)
        search.search(mcts.env_state_snapshot(env))

        rollout = mcts.replay_solution(env, search.env_simulator.solutions[0])

        assert rollout is not None
        assert rollout["solved"] is True
        assert rollout["max_int"] == rollout["target_int"]
        assert len(rollout["observations"]) == len(rollout["actions"]) - 1

    def test_the_trace_ends_on_submit(self):
        """Appended, not found. The env ends the episode the moment the
        intersection reaches the target, so no search would ever reach the
        action - and a trace meant to be imitated needs an episode that
        ends the way the agent should end one."""
        env = self._solving_env()
        search = mcts.MCTS(env, max_iterations=20, max_depth=4)
        search.search(mcts.env_state_snapshot(env))

        rollout = mcts.replay_solution(env, search.env_simulator.solutions[0])

        assert rollout["actions"][-1][0] == 0
        assert rollout["dones"][-1] is True
        assert rollout["infos"][-1].get("appended_submit") is True

    def test_a_candidate_that_does_not_solve_is_refused(self):
        """The simulator is not the env, which is why it exists. A sequence
        that only solves there is a bug to fail on, not an answer to keep."""
        env = self._solving_env()

        assert mcts.replay_solution(env, [[1, 0, 0]] * 3) is None or \
               env.max_int == env.target_int

        env2 = self._solving_env()
        assert mcts.replay_solution(env2, [[0, 0, 0]]) is None

    def test_shorter_solutions_win_and_duplicates_are_dropped(self):
        simulator = mcts.EnvironmentSimulator(self._solving_env())

        simulator.record_solution([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        simulator.record_solution([[1, 0, 0]])
        simulator.record_solution([[1, 0, 0]])

        assert len(simulator.solutions) == 2
        assert simulator.solutions[0] == [[1, 0, 0]]

    def test_the_search_returns_the_solution_it_found(self):
        """End to end, and the half that matters: a solution captured but
        never handed back is as lost as one never captured. The search's own
        rollouts follow the tree's chosen action at each real step, so they
        cannot contain it - it has to be put there."""
        env = self._solving_env()

        rollouts = mcts.collect_mcts_rollouts(env, n_rollouts=1, mcts_iterations=20,
                                               max_episode_len=4)

        solved = [r for r in rollouts if r["solved"]]
        assert solved, "the search solved it and returned nothing that says so"
        assert solved[0]["actions"][-1][0] == 0, "the trace should end on submit"

    def test_an_action_path_walks_back_to_the_root(self, arc_task):
        # Two transforms, not _solving_env's one: a child inherits the
        # actions left after its parent popped one, so a single-action pool
        # leaves nothing to expand a grandchild from.
        env = ARCGridWorld(max_episode_len=4,
                           feasible_actions={0: "submit", 1: "rotate90", 2: "rotate180"})
        env.set_subtask(arc_task.subtasks[0])
        env.reset()
        simulator = mcts.EnvironmentSimulator(env)
        root = mcts.MCTSNode(mcts.env_state_snapshot(env),
                             untried_actions=list(simulator.all_actions))
        child = root.expand(simulator)
        grandchild = child.expand(simulator)

        assert root.action_path() == []
        assert grandchild.action_path() == [list(child.action), list(grandchild.action)]


class TestPlayoutPolicy:
    """The two terms, checked apart from each other and apart from a search.

    Both are easy to get backwards, and a search would not say so - it
    would just be slightly worse, on a signal that is already faint.
    """

    ACTIONS = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]

    def test_an_action_that_moves_the_intersection_is_drawn_more_often(self):
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.0)
        for _ in range(20):
            policy.record([1, 0, 0], +4)
            policy.record([2, 0, 0], -4)

        weights = policy.weights()

        assert weights[0] > weights[1]

    def test_an_untried_action_outranks_one_known_to_do_nothing(self):
        """A mean of zero from no tries and a mean of zero from fifty are
        the same number and different states of knowledge. The value term
        cannot tell them apart, which is what the fruitless count is for."""
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.0)
        for _ in range(50):
            policy.record([1, 0, 0], 0)

        weights = policy.weights()

        assert weights[1] > weights[0], "the untried action should be preferred"

    def test_a_fruitless_run_lowers_a_weight_gradually(self):
        """Over tries + 1, so one quiet try counts for little - an action
        can be wrong about a grid it has not seen yet."""
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.0)
        policy.record([1, 0, 0], 0)
        after_one = policy.weights()[0]
        for _ in range(50):
            policy.record([1, 0, 0], 0)
        after_many = policy.weights()[0]

        assert after_many < after_one

    def test_an_action_that_pays_is_not_penalised_for_being_used(self):
        """The count is of fruitless tries, not of tries. Only 3.1% of
        simulated steps move anything, so a penalty on use as such would
        push down the few actions that work along with the rest."""
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.0)
        for _ in range(50):
            policy.record([1, 0, 0], +4)
        policy.record([2, 0, 0], +4)

        weights = policy.weights()

        assert weights[0] >= weights[1] * 0.9, (
            "an action used often and productively should keep its weight")

    def test_nothing_reaches_probability_zero(self):
        """129 of the 20,383 actions that went twenty tries without effect
        did eventually have one. A weight that hits zero cannot come back,
        and the pool is searched again after every round."""
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.1)
        for _ in range(500):
            policy.record([1, 0, 0], 0)
            policy.record([2, 0, 0], -50)

        weights = policy.weights()

        assert (weights > 0).all()
        assert weights.min() >= 0.1 / len(self.ACTIONS) * 0.99

    def test_the_weights_are_a_distribution(self):
        policy = mcts.PlayoutPolicy(self.ACTIONS)
        policy.record([1, 0, 0], +2)

        assert np.isclose(policy.weights().sum(), 1.0)

    def test_sampling_returns_actions_from_the_pool(self):
        policy = mcts.PlayoutPolicy(self.ACTIONS)

        drawn = [policy.sample() for _ in range(50)]

        assert all(action in self.ACTIONS for action in drawn)

    def test_sampling_follows_the_weights(self):
        """The table is rebuilt every refresh_every records, so a policy
        told the same thing repeatedly has to actually act on it."""
        policy = mcts.PlayoutPolicy(self.ACTIONS, floor=0.0, refresh_every=1)
        for _ in range(50):
            policy.record([1, 0, 0], +8)
            policy.record([2, 0, 0], -8)

        drawn = collections.Counter(tuple(policy.sample()) for _ in range(2000))

        assert drawn[(1, 0, 0)] > drawn[(2, 0, 0)] * 2

    def test_an_action_outside_the_pool_is_ignored(self):
        """The tree expands over the pool, but a caller can hand a simulator
        an action from elsewhere - recording it would index nothing."""
        policy = mcts.PlayoutPolicy(self.ACTIONS)

        policy.record([99, 9, 9], +5)

        assert np.isclose(policy.weights().sum(), 1.0)


def test_a_simulator_with_a_policy_learns_from_its_own_steps(env):
    """The wiring: simulate_step feeds the policy the change in
    intersection, so a search's own rollouts are what shape the playout."""
    policy = mcts.PlayoutPolicy(mcts.enumerate_actions(env), refresh_every=1)
    simulator = mcts.EnvironmentSimulator(env, policy=policy)
    state = mcts.env_state_snapshot(env)

    for _ in range(10):
        state, _r, done, truncated, _i = simulator.simulate_step(state, np.array([1, 0, 0]))
        if done or truncated:
            break

    assert policy._tried[policy._index[(1, 0, 0)]] > 0
    assert simulator.sample_action() in policy.actions


def test_a_simulator_without_a_policy_samples_the_raw_space(env):
    """Unchanged by default: the policy is opt-in, and the two draw from
    different sets - the raw space includes the padded slots."""
    simulator = mcts.EnvironmentSimulator(env)

    action = simulator.sample_action()

    assert env.action_space.contains(np.array(action))


class TestWhatGetsCopiedBeforeASimulatedStep:
    """_copy_touched_objects exists so a simulated step can mutate objects
    without disturbing the state the tree branches from again. What it must
    not do is pay for that when no transform will run at all - and most
    sampled actions are exactly that case: the action space carries
    max_objects slots whatever the subtask holds, so on a four-object grid
    at max_objects=16, 94% of sampled index pairs name nothing.

    Identity assertions rather than timings: "the same list came back" is
    exactly the claim, and it does not depend on how fast the machine is.
    """

    def _simulator_and_objects(self, wide_env):
        simulator = mcts.EnvironmentSimulator(wide_env)
        return simulator, list(wide_env.objects)

    def test_an_index_naming_no_object_copies_nothing(self, wide_env):
        simulator, objects = self._simulator_and_objects(wide_env)
        assert len(objects) < 16, "the fixture is meant to leave empty slots"

        returned = simulator._copy_touched_objects(objects, np.array([1, 15, 15]))

        assert returned is objects

    def test_submit_copies_nothing(self, wide_env):
        """simulate_action returns before touching an object, whatever the
        indices alongside it happen to be."""
        simulator, objects = self._simulator_and_objects(wide_env)

        returned = simulator._copy_touched_objects(objects, np.array([0, 0, 1]))

        assert returned is objects

    def test_a_real_action_still_copies_what_it_names(self, wide_env):
        simulator, objects = self._simulator_and_objects(wide_env)

        returned = simulator._copy_touched_objects(objects, np.array([1, 0, 2]))

        assert returned is not objects
        assert returned[0] is not objects[0]
        assert returned[2] is not objects[2]
        # And leaves the ones it does not name alone, which is the point of
        # copying per action rather than the whole list.
        assert returned[1] is objects[1]

    def test_object_recolor_copies_everything_it_could_reach(self, wide_env):
        """That variant recolors a third object it looks up through cell2obj,
        so the two named indices do not bound what it can mutate."""
        actions = {0: "submit", 1: "red_emission_with_blue_object_recolor_N"}
        env = ARCGridWorld(max_episode_len=4, feasible_actions=actions, repr_level=1,
                           input_pattern="start",
                           observation_space_elements=["objects_emb"])
        env.set_subtask(wide_env.subtask)
        env.reset()
        simulator = mcts.EnvironmentSimulator(env)
        objects = list(env.objects)

        returned = simulator._copy_touched_objects(objects, np.array([1, 0, 1]))

        assert all(a is not b for a, b in zip(returned, objects))

    def test_object_recolor_copies_nothing_when_the_indices_name_nothing(self, wide_env):
        """The case the ordering is for. Reached with an empty slot, that
        branch copied every object on the grid and then ran no transform."""
        actions = {0: "submit", 1: "red_emission_with_blue_object_recolor_N"}
        env = ARCGridWorld(max_episode_len=4, feasible_actions=actions, repr_level=1,
                           input_pattern="start",
                           observation_space_elements=["objects_emb"])
        env.set_subtask(wide_env.subtask)
        env.reset()
        simulator = mcts.EnvironmentSimulator(env)
        objects = list(env.objects)

        returned = simulator._copy_touched_objects(objects, np.array([1, 15, 15]))

        assert returned is objects

    def test_a_skipped_copy_still_leaves_the_step_correct(self, wide_env):
        """Returning the caller's own list is only safe because nothing
        downstream writes to it. Stepping repeatedly on empty slots must
        leave the objects the state started with untouched."""
        simulator = mcts.EnvironmentSimulator(wide_env)
        state = mcts.env_state_snapshot(wide_env)
        before = [tuple(obj.coords) for obj in state["objects"]]

        for _ in range(5):
            state, _reward, done, truncated, _info = simulator.simulate_step(
                state, np.array([1, 15, 15]))
            if done or truncated:
                break

        assert [tuple(obj.coords) for obj in state["objects"]] == before
        assert [tuple(obj.coords) for obj in wide_env.objects] == before


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


class TestTracesAreCutAtTheirPeak:
    """A rollout is kept for imitation, so what it ends on is what it
    teaches. Measured over four configurations the searches ended a mean
    0.04-0.21 of the gap *below* where they started while their peaks stood
    at +0.25 - the whole of that spread is the search walking back downhill
    after its best state, and cloning the trace whole teaches both the half
    that found something and the half that undid it.
    """

    @staticmethod
    def _env_with_walk(walk):
        """An env whose real steps trace `walk` as the intersection, so the
        shape of the rollout can be checked against a known peak rather
        than against whatever a real search happens to find."""
        from rl.arc_task import ARCSubtask

        grid = np.zeros((4, 4), dtype=int)
        grid[1, 1] = 3
        out = grid.copy()
        out[1, 1] = 5
        env = ARCGridWorld(max_episode_len=len(walk),
                           feasible_actions={0: "submit", 1: "gray_recolor"},
                           repr_level=1, input_pattern="start",
                           observation_space_elements=["objects_emb"])
        env.set_subtask(ARCSubtask("walk", grid, out))
        env.reset()

        steps = iter(walk)
        real_step, real_reset = env.step, env.reset

        def resetting(*args, **kwargs):
            result = real_reset(*args, **kwargs)
            # After reset, not before: reset recomputes both from the grid.
            env.base_int, env.target_int = 0, 10
            return result

        env.reset = resetting
        env.reset()

        def stepping(action):
            observation, reward, _done, _truncated, info = real_step(action)
            env.max_int = next(steps, env.max_int)
            # done/truncated forced off: this fixture is about where the
            # trace is cut, and letting the real env end the episode early
            # would decide that instead.
            return observation, reward, False, False, info

        env.step = stepping
        return env

    def _collect(self, walk, monkeypatch):
        env = self._env_with_walk(walk)
        monkeypatch.setattr(mcts.MCTS, "search", lambda self, state: None)
        monkeypatch.setattr(mcts.MCTS, "get_best_action",
                            lambda self, root: np.array([1, 0, 0]))
        return mcts.collect_mcts_rollouts(env, n_rollouts=1,
                                          mcts_iterations=1,
                                          max_episode_len=len(walk))[0]

    def test_the_trace_stops_where_the_grid_was_best(self, monkeypatch):
        rollout = self._collect([2, 5, 3, 1], monkeypatch)

        assert rollout["length"] == 2, "should stop on the step that reached 5"
        assert len(rollout["actions"]) == 2
        assert rollout["max_int"] == 5
        assert rollout["truncated_at_peak"] is True

    def test_every_parallel_list_is_cut_to_the_same_length(self, monkeypatch):
        rollout = self._collect([2, 5, 3, 1], monkeypatch)

        for key in ("observations", "actions", "rewards", "dones", "infos"):
            assert len(rollout[key]) == 2, key

    def test_a_rollout_that_only_improves_is_left_whole(self, monkeypatch):
        rollout = self._collect([1, 2, 3, 4], monkeypatch)

        assert rollout["length"] == 4
        assert rollout["truncated_at_peak"] is False
        assert rollout["max_int"] == 4

    def test_a_rollout_that_never_improves_is_left_whole(self, monkeypatch):
        """Nothing to cut to. Such a trace is not a demonstration at all and
        the caller drops it on max_int == base_int - truncating it to one
        arbitrary step would disguise that."""
        rollout = self._collect([0, 0, 0, 0], monkeypatch)

        assert rollout["length"] == 4
        assert rollout["truncated_at_peak"] is False

    def test_reaching_the_target_and_moving_off_it_still_counts_as_solved(self, monkeypatch):
        """The regression: `solved` read env.max_int after the loop, which
        is the final state, so a rollout that passed through the answer and
        then walked away from it was recorded as a failure."""
        rollout = self._collect([4, 10, 6], monkeypatch)

        assert rollout["solved"] is True
        assert rollout["max_int"] == rollout["target_int"] == 10
        assert rollout["length"] == 2

    def test_the_reward_total_is_recomputed_for_the_kept_steps(self, monkeypatch):
        rollout = self._collect([2, 5, 3, 1], monkeypatch)

        assert rollout["total_reward"] == sum(rollout["rewards"])
