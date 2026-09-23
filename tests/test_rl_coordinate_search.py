"""MCTS and the replay under coordinate addressing.

The search enumerates actions itself rather than sampling the action
space, and the product of the four coordinate axes names each rectangle
eight ways and every cell past the grid besides. coordinate_actions lists
one action per distinct stroke; replay_solution appends a submit as wide as
the action space. Both were written for three-wide object actions first.
"""
from __future__ import annotations

import numpy as np

from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.mcts import coordinate_actions, enumerate_actions, replay_solution
from rl.utils import get_action_description

ACTIONS = {0: "submit", 1: "red_fill", 2: "red_line", 3: "red_triangle"}


def env_on(inp, out, shape):
    env = ARCGridWorld(max_episode_len=5, feasible_actions=ACTIONS, reward_approach=3,
                       repr_level=1, input_pattern="start", addressing="coordinates",
                       coordinate_shape=shape, observation_space_elements=["delta_input"])
    env.set_subtask(ARCSubtask("case", inp, out))
    env.reset()
    return env


def by_name(actions):
    counts = {}
    for action in actions:
        name = ACTIONS[action[0]]
        counts[name] = counts.get(name, 0) + 1
    return counts


class TestTheStrokesASearchTries:
    def test_one_per_distinct_stroke_on_a_ten_by_ten_grid(self):
        grid = np.zeros((10, 10), dtype=int)
        env = env_on(grid, grid + 1, (10, 10))
        assert by_name(coordinate_actions(env)) == {
            "red_fill": 55 * 55, "red_line": 1570, "red_triangle": 100 * 81}

    def test_the_line_count_is_the_straight_segments(self):
        """Counted another way: every unordered pair of distinct cells on
        a row, a column or a 45-degree diagonal, plus the single cells."""
        grid = np.zeros((10, 10), dtype=int)
        env = env_on(grid, grid + 1, (10, 10))
        cells = [(i, j) for i in range(10) for j in range(10)]
        straight = sum(1 for a in range(100) for b in range(a + 1, 100)
                       if (cells[a][0] == cells[b][0] or cells[a][1] == cells[b][1]
                           or abs(cells[a][0] - cells[b][0]) == abs(cells[a][1] - cells[b][1])))
        assert by_name(coordinate_actions(env))["red_line"] == straight + 100

    def test_a_fill_is_named_once_by_its_top_left_and_bottom_right(self):
        grid = np.zeros((4, 4), dtype=int)
        env = env_on(grid, grid + 1, (4, 4))
        fills = [a for a in coordinate_actions(env) if a[0] == 1]
        assert all(a[1] <= a[3] and a[2] <= a[4] for a in fills)
        assert len({tuple(a) for a in fills}) == len(fills)

    def test_nothing_past_the_grid_the_env_holds(self):
        """The space is sized for the task's largest grid; the search is
        only offered cells of this one."""
        grid = np.zeros((3, 4), dtype=int)
        env = env_on(grid, grid + 1, (6, 6))
        actions = coordinate_actions(env)
        assert max(max(a[1], a[3]) for a in actions) == 2
        assert max(max(a[2], a[4]) for a in actions) == 3

    def test_submit_only_when_asked_for_and_five_wide(self):
        grid = np.zeros((3, 3), dtype=int)
        env = env_on(grid, grid + 1, (3, 3))
        assert all(a[0] != 0 for a in coordinate_actions(env))
        with_submit = coordinate_actions(env, include_submit=True)
        assert [a for a in with_submit if a[0] == 0] == [[0, 0, 0, 0, 0]]

    def test_enumerate_actions_hands_over_to_it(self):
        grid = np.zeros((3, 3), dtype=int)
        env = env_on(grid, grid + 1, (3, 3))
        assert enumerate_actions(env) == coordinate_actions(env)


class TestReplayingACoordinateSolution:
    def test_the_appended_submit_is_as_wide_as_the_action_space(self):
        inp = np.zeros((5, 5), dtype=int)
        out = inp.copy()
        out[1:3, 1:4] = 2
        env = env_on(inp, out, (5, 5))
        rollout = replay_solution(env, [[1, 1, 1, 2, 3]])
        assert rollout is not None
        assert rollout["actions"] == [[1, 1, 1, 2, 3], [0, 0, 0, 0, 0]]
        assert env.action_space.contains(np.array(rollout["actions"][-1]))

    def test_a_sequence_that_does_not_solve_it_is_not_a_solution(self):
        inp = np.zeros((5, 5), dtype=int)
        out = inp.copy()
        out[1:3, 1:4] = 2
        env = env_on(inp, out, (5, 5))
        assert replay_solution(env, [[1, 1, 1, 1, 3]]) is None


class TestDescribingIt:
    def test_both_cells_are_named(self):
        assert (get_action_description([1, 2, 0, 4, 2], ACTIONS)
                == "red_fill from (2, 0) to (4, 2)")

    def test_object_actions_read_as_before(self):
        assert get_action_description([1, 3, 3], ACTIONS) == "red_fill for 3"
        assert get_action_description([1, 3, 4], ACTIONS) == "red_fill between 3 and 4"
