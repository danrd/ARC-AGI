"""Tests for the colour the symbolic solvers take a task's missing or
background cells to be.

Every solver in symbolic/symbolic_module.py is built around one such colour
and it was left at 0. Where a symmetric picture has a patch of 8 laid over it
that finds nothing; told 8, ColorRestoreSolver restores it. Measured on the
evaluation tasks meant for the symbolic modules, a fixed 0 solved 14 of 30
and reading the colour off the training pairs seven more, with the mixer's
strip split (below) two of those.

The second thing pinned is the mixer's split of a grid where nothing marks
one: it took the finest split whose pieces have two colours, which cut a
12x4 input that is two 6x4 pieces into six 2x4 strips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.symbolic_module import MixerSolver, SymbolicModule, checked_solve, font_value_candidates

DATA = Path(__file__).resolve().parents[1] / "data" / "datasets" / "ARC"


def task_of(pairs, test=None):
    subtasks = [ARCSubtask(f"t_{i}", np.array(a), np.array(b)) for i, (a, b) in enumerate(pairs)]
    test = test or pairs[-1]
    return ARCTask(label="t", subtasks=subtasks, test_inp=np.array(test[0]), test_out=np.array(test[1]))


def real_task(task_id):
    challenge = json.loads((DATA / "evaluation_challenges.json").read_text())[task_id]
    answer = json.loads((DATA / "evaluation_solutions.json").read_text())[task_id]
    subtasks = [ARCSubtask(f"{task_id}_{i}", np.array(p["input"]), np.array(p["output"]))
                for i, p in enumerate(challenge["train"])]
    return ARCTask(label=task_id, subtasks=subtasks, test_inp=np.array(challenge["test"][0]["input"]),
                   test_out=np.array(answer[0]))


class TestTheCandidates:
    def test_the_colour_a_restoration_removes_comes_first(self):
        """Same-size pairs where the cells that change are all 7."""
        pairs = [([[1, 2, 7], [2, 1, 2]], [[1, 2, 1], [2, 1, 2]]),
                 ([[7, 7, 3], [3, 4, 4]], [[3, 4, 3], [3, 4, 4]])]
        assert font_value_candidates(task_of(pairs))[0] == 7

    def test_the_colour_a_crop_cuts_out_comes_first(self):
        """The inputs share an 8 that no output has."""
        pairs = [([[8, 8, 1], [8, 8, 2]], [[1], [2]]), ([[8, 3, 8], [8, 4, 8]], [[3], [4]])]
        assert font_value_candidates(task_of(pairs))[0] == 8

    def test_zero_is_always_a_candidate_when_there_is_room(self):
        pairs = [([[5, 1]], [[1, 1]])]
        assert 0 in font_value_candidates(task_of(pairs))

    def test_no_colour_is_named_twice_and_the_list_has_a_limit(self):
        pairs = [([[0, 1, 0]], [[1, 1, 1]])]
        candidates = font_value_candidates(task_of(pairs))
        assert len(candidates) == len(set(candidates)) and candidates[0] == 0
        assert len(font_value_candidates(task_of([([[1, 2, 3, 4]], [[4, 3, 2, 1]])]), limit=1)) == 1

    def test_a_task_with_no_examples_falls_back_to_zero(self):
        empty = ARCTask(label="t", subtasks=[], test_inp=np.zeros((2, 2), int), test_out=np.zeros((2, 2), int))
        assert font_value_candidates(empty) == [0]

    def test_one_module_per_candidate(self):
        task = task_of([([[1, 2, 7]], [[1, 2, 1]])])
        modules = SymbolicModule.for_task(task)
        assert [module.font_val for module in modules] == font_value_candidates(task)
        assert modules[0].color_restore.font_val == modules[0].font_val


@pytest.mark.parametrize("task_id", ["67b4a34d", "0934a4d8", "929ab4e9"])
def test_a_restoration_painted_over_in_another_colour_is_solved_with_the_colour_read_off(task_id):
    """The three differ in the colour of the patch (3, 8, 2) and one of them
    is a crop rather than a same-size restoration."""
    task = real_task(task_id)
    answer = np.asarray(task.test_subtask.train_out)
    solved = [checked_solve(module.color_restore, task) for module in SymbolicModule.for_task(task)]
    assert any(result.success and np.array_equal(result.grid, answer) for result in solved)


def test_the_orchestrator_tries_the_colours_the_task_suggests():
    from orchestration.graph import _dispatch_symbolic
    task = real_task("67b4a34d")
    out = _dispatch_symbolic(task)
    assert np.array_equal(out["solution"], task.test_subtask.train_out)


def test_a_module_the_caller_fixed_is_used_as_it_is():
    from orchestration.graph import _dispatch_symbolic
    task = real_task("67b4a34d")
    fixed = SymbolicModule(font_val=5)
    out = _dispatch_symbolic(task, symbolic_module=fixed)
    assert not (isinstance(out["solution"], np.ndarray) and np.array_equal(out["solution"], task.test_subtask.train_out))


class TestTheMixersStripSplit:
    @staticmethod
    def two_halves(seed):
        """Two 6x4 pieces one over the other, 3s above and 5s below, and the
        output is 4 wherever either has a cell."""
        rng = np.random.default_rng(seed)
        top, bottom = rng.random((6, 4)) > 0.55, rng.random((6, 4)) > 0.55
        grid = np.vstack([np.where(top, 3, 0), np.where(bottom, 5, 0)])
        return grid.tolist(), np.where(top | bottom, 4, 0).tolist()

    def task(self):
        pairs = [self.two_halves(seed) for seed in range(4)]
        return task_of(pairs[:3], test=pairs[3])

    def test_two_large_pieces_are_not_cut_into_six_strips(self):
        task = self.task()
        result = MixerSolver(font_val=0).solve(task)
        assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)

    def test_the_pieces_follow_the_size_of_the_output(self):
        solver = MixerSolver(font_val=0)
        grid = np.array(self.two_halves(0)[0])
        strips = solver._segments_from_heuristic(grid, grid.shape, out_shape=(6, 4))
        assert [strip.shape for strip in strips] == [(6, 4), (6, 4)]
        without_output = solver._segments_from_heuristic(grid, grid.shape)
        assert len(without_output) > 2

    def test_a_test_input_is_cut_as_many_ways_as_the_examples_were(self):
        solver = MixerSolver(font_val=0)
        grid = np.array(self.two_halves(1)[0])
        assert [s.shape for s in solver._segments_from_heuristic(grid, grid.shape, count=2)] == [(6, 4), (6, 4)]

    def test_a_split_that_does_not_fit_falls_back_to_the_old_one(self):
        solver = MixerSolver(font_val=0)
        grid = np.array(self.two_halves(2)[0])
        fitted = solver._segments_from_heuristic(grid, grid.shape, out_shape=(5, 4))
        plain = solver._segments_from_heuristic(grid, grid.shape)
        assert len(fitted) == len(plain) and all(np.array_equal(a, b) for a, b in zip(fitted, plain))
