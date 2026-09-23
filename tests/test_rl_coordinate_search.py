"""MCTS and the replay under coordinate addressing.

The search enumerates actions itself rather than sampling the action
space, and the product of the four coordinate axes names each rectangle
eight ways and every cell past the grid besides. coordinate_actions lists
one action per distinct stroke; replay_solution appends a submit as wide as
the action space. Both were written for three-wide object actions first.
"""
from __future__ import annotations

import numpy as np
import pytest

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


class TestStrokesReadOffTheAnswer:
    """rl.coordinate_search: the pool is safe strokes no other safe stroke
    contains, read off the pair's output."""

    def test_an_l_is_two_maximal_rectangles(self):
        from rl.coordinate_search import maximal_rectangles

        allowed = np.zeros((4, 4), dtype=bool)
        allowed[0:4, 0] = True
        allowed[3, 0:4] = True
        assert sorted(maximal_rectangles(allowed)) == [(0, 0, 3, 0), (3, 0, 3, 3)]

    def test_a_full_grid_is_one(self):
        from rl.coordinate_search import maximal_rectangles

        assert maximal_rectangles(np.ones((3, 5), dtype=bool)) == [(0, 0, 2, 4)]

    def test_a_diagonal_is_one_line(self):
        from rl.coordinate_search import diagonal_runs

        allowed = np.eye(4, dtype=bool)[::-1]
        assert diagonal_runs(allowed) == [((0, 3), (3, 0))]

    @pytest.mark.parametrize("down,across", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    def test_a_triangle_is_named_so_the_env_paints_it(self, down, across):
        """The corner and legs this finds, turned into fill_triangle's two
        cells, paint exactly the triangle - the right angle where it was
        found, not at some other corner of the box."""
        from rl.coordinate_search import largest_triangles, stroke_mask

        corner = (0 if down > 0 else 3, 0 if across > 0 else 3)
        wanted = np.zeros((4, 4), dtype=bool)
        for a in range(4):
            for b in range(4 - a):
                wanted[corner[0] + down * a, corner[1] + across * b] = True
        found = [pair for pair in largest_triangles(wanted)
                 if stroke_mask((4, 4), "triangle", *pair).sum() == 10]
        assert found
        assert all(np.array_equal(stroke_mask((4, 4), "triangle", *pair), wanted)
                   for pair in found)

    def test_every_candidate_is_safe_and_useful(self):
        from rl.coordinate_search import candidate_strokes, stroke_mask

        rng = np.random.default_rng(0)
        grid = rng.integers(0, 3, size=(8, 8))
        target = rng.integers(0, 3, size=(8, 8))
        strokes = candidate_strokes(grid, target)
        assert strokes
        for colour, transform, first, second in strokes:
            mask = stroke_mask(target.shape, transform, first, second)
            assert (target[mask] == colour).all()
            assert (grid[mask] != colour).any()

    def test_no_candidate_is_contained_in_another(self):
        from rl.coordinate_search import candidate_strokes, stroke_mask

        grid = np.zeros((6, 6), dtype=int)
        target = grid.copy()
        target[1:5, 1:5] = 2
        target[0, :] = 3
        strokes = candidate_strokes(grid, target)
        fixes = [stroke_mask(target.shape, t, a, b) & (grid != target)
                 for _c, t, a, b in strokes]
        for i, one in enumerate(fixes):
            for j, other in enumerate(fixes):
                if i != j:
                    assert (one & ~other).any(), (strokes[i], strokes[j])

    def test_a_block_and_a_bar_are_two_fills(self):
        from rl.coordinate_search import candidate_strokes

        grid = np.zeros((6, 6), dtype=int)
        target = grid.copy()
        target[1:5, 1:5] = 2
        target[0, :] = 3
        assert sorted(candidate_strokes(grid, target)) == [
            (2, "fill", (1, 1), (4, 4)), (3, "fill", (0, 0), (0, 5))]


def vocabulary(*colours):
    from rl.search_hints import coordinate_vocabulary
    return coordinate_vocabulary([COLOUR_NAMES[c] for c in colours])


from data.configs.env_configs import COLORS_MAPPING as COLOUR_NAMES  # noqa: E402


class TestSearchingAPair:
    def test_a_block_is_one_stroke(self):
        from rl.coordinate_search import search_pair

        inp = np.zeros((6, 6), dtype=int)
        out = inp.copy()
        out[1:4, 2:5] = 2
        found = search_pair(ARCSubtask("block", inp, out), vocabulary(0, 2))
        assert len(found["solution"]) == 1

    def test_what_it_returns_solves_the_pair(self):
        from rl.coordinate_search import pair_env, search_pair, CoordinateSearchSettings

        inp = np.zeros((7, 7), dtype=int)
        out = inp.copy()
        out[0:3, 0:3] = 2
        for k in range(7):
            out[6 - k, k] = 4
        out[5:7, 4:7] = 1
        actions = vocabulary(0, 1, 2, 4)
        found = search_pair(ARCSubtask("mixed", inp, out), actions)
        env = pair_env(ARCSubtask("mixed", inp, out), actions, CoordinateSearchSettings())
        assert found["solution"]
        assert replay_solution(env, found["solution"]) is not None

    def test_a_pair_with_nothing_to_repaint_is_solved_by_nothing(self):
        from rl.coordinate_search import search_pair

        grid = np.ones((4, 4), dtype=int)
        assert search_pair(ARCSubtask("same", grid, grid.copy()), vocabulary(1))["solution"] == []

    def test_a_square_with_a_dot_is_covered_around_the_dot(self):
        """The limit the module docstring names: red over the whole square
        and then blue on the dot is two strokes, and the first of them is
        not safe. The safe cover takes the red round the dot in pieces -
        three here, with a triangle among them - and then the dot."""
        from rl.coordinate_search import search_pair

        inp = np.zeros((5, 5), dtype=int)
        out = np.full((5, 5), 2)
        out[2, 2] = 1
        found = search_pair(ARCSubtask("dot", inp, out), vocabulary(1, 2))
        assert len(found["solution"]) > 2


class TestTheVocabularyItKeeps:
    def test_nothing_found_keeps_everything(self):
        """A vocabulary without the colour the output needs: nothing to
        paint with, so nothing is narrowed away."""
        from rl.arc_task import ARCTask
        from rl.coordinate_search import feasible_from_coordinate_search

        inp = np.zeros((4, 4), dtype=int)
        out = inp.copy()
        out[0, 0] = 3
        task = ARCTask(label="t", subtasks=[ARCSubtask("t_0", inp, out)],
                       test_inp=inp, test_out=out)
        actions = vocabulary(2)
        kept, _found = feasible_from_coordinate_search(task, actions)
        assert kept == actions

    def test_every_pair_counts(self):
        from rl.arc_task import ARCTask
        from rl.coordinate_search import feasible_from_coordinate_search

        first_in, second_in = np.zeros((5, 5), dtype=int), np.zeros((5, 5), dtype=int)
        first_out = first_in.copy()
        first_out[0:2, 0:2] = 2
        second_out = second_in.copy()
        second_out[4, 0], second_out[3, 1], second_out[2, 2] = 1, 1, 1
        task = ARCTask(label="t", subtasks=[ARCSubtask("t_0", first_in, first_out),
                                            ARCSubtask("t_1", second_in, second_out)],
                       test_inp=first_in, test_out=first_out)
        kept, found = feasible_from_coordinate_search(task, vocabulary(0, 1, 2))
        assert set(kept.values()) == {"submit", "red_fill", "blue_line"}
        assert set(found) == {"t_0", "t_1"}


class TestTheWeightedPlayout:
    def test_it_can_be_asked_for_by_name(self):
        from rl.mcts import MCTS, PlayoutPolicy

        grid = np.zeros((3, 3), dtype=int)
        env = env_on(grid, grid + 2, (3, 3))
        assert isinstance(MCTS(env, playout="weighted").env_simulator.policy, PlayoutPolicy)
        assert MCTS(env).env_simulator.policy is None

    def test_an_unknown_one_is_refused(self):
        from rl.mcts import MCTS

        grid = np.zeros((3, 3), dtype=int)
        with pytest.raises(ValueError, match="playout"):
            MCTS(env_on(grid, grid + 2, (3, 3)), playout="sideways")
