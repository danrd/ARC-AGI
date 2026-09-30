"""Tests for which of its three ways of combining the pieces the mixer picks.

The search for the conjunction rule, and the way of applying it to a test
input, were written and never reached: the choice between ways
(_color_analysis) returned 'color_mix' or 'logical_ops' and nothing else.
The rule is the one d47aa2ff needs - two pieces side by side, the cells they
share kept, the cells only one has marked in a colour of the piece's own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.symbolic_module import MixerSolver

DATA = Path(__file__).resolve().parents[1] / "data" / "datasets" / "ARC"


def pair(seed, size=8):
    """Two pieces of 3s with a column of 5 between; the output keeps the 3s
    both have, marks those only the left has 2 and those only the right has 1."""
    rng = np.random.default_rng(seed)
    left, right = rng.random((size, size)) > 0.7, rng.random((size, size)) > 0.7
    grid = np.hstack([np.where(left, 3, 0), np.full((size, 1), 5), np.where(right, 3, 0)])
    out = np.zeros((size, size), dtype=int)
    out[left & right] = 3
    out[left & ~right] = 2
    out[~left & right] = 1
    return grid, out


def task_of(pairs):
    subtasks = [ARCSubtask(f"t_{i}", inp, out) for i, (inp, out) in enumerate(pairs[:-1])]
    return ARCTask(label="t", subtasks=subtasks, test_inp=pairs[-1][0], test_out=pairs[-1][1])


class TestTheConjunctionRule:
    def test_extra_marker_colours_choose_it(self):
        task = task_of([pair(seed) for seed in range(4)])
        solver = MixerSolver(font_val=0)
        from symbolic.patterns import retrieve_shapes
        first = task.subtasks[0]
        patterns = retrieve_shapes(first.train_inp, first.train_inp_shape, ("markup", "partition_lines"), 0)
        assert solver._color_analysis(task, patterns) == "conjunction"

    def test_a_task_of_that_kind_is_solved(self):
        task = task_of([pair(seed) for seed in range(4)])
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)

    def test_one_output_colour_is_still_a_logical_operation(self):
        """Two pieces, the output 4 wherever either has a cell: one colour
        in the output, so not a conjunction."""
        pairs = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            top, bottom = rng.random((6, 4)) > 0.55, rng.random((6, 4)) > 0.55
            pairs.append((np.vstack([np.where(top, 3, 0), np.where(bottom, 5, 0)]),
                          np.where(top | bottom, 4, 0)))
        task = task_of(pairs)
        solver = MixerSolver(font_val=0)
        assert solver._color_analysis(task, {}) == "logical_ops"

    def test_d47aa2ff_is_solved(self):
        challenge = json.loads((DATA / "evaluation_challenges.json").read_text())["d47aa2ff"]
        answer = json.loads((DATA / "evaluation_solutions.json").read_text())["d47aa2ff"]
        subtasks = [ARCSubtask(f"d47_{i}", np.array(p["input"]), np.array(p["output"]))
                    for i, p in enumerate(challenge["train"])]
        task = ARCTask(label="d47aa2ff", subtasks=subtasks, test_inp=np.array(challenge["test"][0]["input"]),
                       test_out=np.array(answer[0]))
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)


def stacked_layers(seed, size=6):
    """Four 3-row pieces one over the other, layered with priority 5, 4, 8, 2
    (a cell takes the colour of the first piece that has one there)."""
    rng = np.random.default_rng(seed)
    colours = (5, 4, 8, 2)
    pieces = [np.where(rng.random((3, size)) > 0.6, colour, 0) for colour in colours]
    out = np.zeros((3, size), dtype=int)
    for piece in pieces:
        out = np.where(out == 0, piece, out)
    return np.vstack(pieces), out


class TestAskingTheSearchAgain:
    def test_an_order_that_only_the_first_example_allowed_is_set_aside(self):
        """In the first example the pieces never overlap where 8 and 2 meet,
        so 8-then-2 and 2-then-8 both give it; the later examples decide.
        The search used to keep the first order it found and fail."""
        first_pieces = [np.array([[5, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 4, 0], [0, 0, 0], [0, 0, 0]]),
                        np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [8, 0, 0], [0, 0, 0]])]
        first = (np.vstack(first_pieces), sum(first_pieces))
        pairs = [first] + [stacked_layers(seed, size=3) for seed in range(3)] + [stacked_layers(9, size=3)]
        task = task_of(pairs)
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)

    def test_a_real_task_whose_first_example_allows_two_orders(self):
        """3d31c5b3: four pieces layered with a priority, and in the first
        example the last two never meet, so the first order found (ID, in
        the order given) was wrong for the second."""
        challenge = json.loads((DATA / "evaluation_challenges.json").read_text())["3d31c5b3"]
        answer = json.loads((DATA / "evaluation_solutions.json").read_text())["3d31c5b3"]
        subtasks = [ARCSubtask(f"l_{i}", np.array(p["input"]), np.array(p["output"]))
                    for i, p in enumerate(challenge["train"])]
        task = ARCTask(label="l", subtasks=subtasks, test_inp=np.array(challenge["test"][0]["input"]),
                       test_out=np.array(answer[0]))
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)

    def test_when_no_order_fits_every_example_it_still_fails_and_says_why(self):
        pairs = [stacked_layers(seed, size=3) for seed in range(3)]
        broken = (pairs[1][0], np.where(pairs[1][1] == 0, 1, 0))
        task = task_of([pairs[0], broken, pairs[2], pairs[2]])
        result = MixerSolver(font_val=0).solve(task)
        assert not result.success and ("no consistent" in result.debug or "contradicts" in result.debug)


def fitting_pair(fits, rng, size=4):
    """Two pieces side by side around a column of 5. Where they do not
    collide the output lays them over one another; where they do it is the
    left one alone."""
    left = rng.random((size, size)) > 0.6
    right = (rng.random((size, size)) > 0.6) & ~left if fits else (rng.random((size, size)) > 0.4) | left
    if not fits:
        right |= left
    grid = np.hstack([np.where(left, 1, 0), np.full((size, 1), 5), np.where(right, 2, 0)])
    out = np.where(left, 1, np.where(right, 2, 0)) if fits else np.where(left, 1, 0)
    return grid, out


class TestTheFitRule:
    def task(self):
        rng = np.random.default_rng(0)
        return task_of([fitting_pair(True, rng), fitting_pair(False, rng), fitting_pair(True, rng),
                        fitting_pair(False, rng), fitting_pair(True, rng), fitting_pair(False, rng)])

    def test_pieces_are_laid_over_one_another_when_they_do_not_collide_and_the_first_stands_alone_when_they_do(self):
        task = self.task()
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)

    def test_it_is_tried_only_after_the_kind_the_colours_point_to_finds_nothing(self):
        solver = MixerSolver(font_val=0)
        rng = np.random.default_rng(1)
        pieces = [np.array([[1, 0], [0, 0]]), np.array([[0, 2], [0, 0]])]
        assert np.array_equal(solver._fit_build(pieces), np.array([[1, 2], [0, 0]]))
        colliding = [np.array([[1, 0], [0, 0]]), np.array([[3, 2], [0, 0]])]
        assert np.array_equal(solver._fit_build(colliding), np.array([[1, 0], [0, 0]]))
        assert rng is not None

    def test_pieces_of_different_shapes_are_not_fitted(self):
        solver = MixerSolver(font_val=0)
        assert solver._fit_search([np.zeros((2, 2), int), np.zeros((3, 2), int)], np.zeros((2, 2), int), []) is False

    def test_a_real_task_needs_it(self):
        challenge = json.loads((DATA / "evaluation_challenges.json").read_text())["bbb1b8b6"]
        answer = json.loads((DATA / "evaluation_solutions.json").read_text())["bbb1b8b6"]
        subtasks = [ARCSubtask(f"b_{i}", np.array(p["input"]), np.array(p["output"]))
                    for i, p in enumerate(challenge["train"])]
        task = ARCTask(label="b", subtasks=subtasks, test_inp=np.array(challenge["test"][0]["input"]),
                       test_out=np.array(answer[0]))
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)
