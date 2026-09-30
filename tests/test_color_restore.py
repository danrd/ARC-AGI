"""Tests for ColorRestoreSolver's restoration from symmetries read off the
grid itself.

It used to fix the symmetric region from the last training output and reuse
that region on the test input, which cannot follow a picture whose axis sits
somewhere else in each example, and it reflected whole strips, rewriting cells
that were right wherever the picture stopped being symmetric. Now the mirrors
are found in the grid in hand, cells are filled only from mirrors that are
exact, and a restoration that would touch a cell that was not missing is
refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.symbolic_module import ColorRestoreSolver, SymbolicModule, checked_solve

DATA = Path(__file__).resolve().parents[1] / "data" / "datasets" / "ARC"
FONT = 7


def symmetric_picture(size=12, offset=2, seed=0):
    """A grid whose part from `offset` on is mirrored left-right and up-down
    about its own centre. The `offset` rows above it and columns beside it
    are mirrored too, along the picture's axes, and the corner where they
    meet has no image - as in the tasks this is built for."""
    rng = np.random.default_rng(seed)
    inner = size - offset
    quarter = rng.integers(1, 6, (inner // 2, inner // 2))
    top = np.hstack([quarter, np.fliplr(quarter)])
    grid = rng.integers(1, 6, (size, size))
    grid[offset:, offset:] = np.vstack([top, np.flipud(top)])
    if offset:
        strip = rng.integers(1, 6, (offset, inner // 2))
        grid[:offset, offset:] = np.hstack([strip, np.fliplr(strip)])
        side = rng.integers(1, 6, (inner // 2, offset))
        grid[offset:, :offset] = np.vstack([side, np.flipud(side)])
    return grid


def real_task(task_id):
    challenge = json.loads((DATA / "evaluation_challenges.json").read_text())[task_id]
    answer = json.loads((DATA / "evaluation_solutions.json").read_text())[task_id]
    subtasks = [ARCSubtask(f"{task_id}_{i}", np.array(p["input"]), np.array(p["output"]))
                for i, p in enumerate(challenge["train"])]
    return ARCTask(label=task_id, subtasks=subtasks, test_inp=np.array(challenge["test"][0]["input"]),
                   test_out=np.array(answer[0]))


class TestRestoringFromTheGridSMirrors:
    def solver(self):
        return ColorRestoreSolver(font_val=FONT)

    def test_a_patch_over_a_symmetric_picture_is_filled_from_its_mirror_images(self):
        truth = symmetric_picture()
        grid = truth.copy()
        grid[4:7, 6:9] = FONT
        assert np.array_equal(self.solver()._restore_by_inferred_symmetries(grid), truth)

    def test_the_axis_may_be_anywhere_in_the_grid(self):
        for offset in (0, 2, 4):
            truth = symmetric_picture(size=14, offset=offset, seed=offset)
            grid = truth.copy()
            grid[offset + 2:offset + 4, offset + 1:offset + 4] = FONT
            assert np.array_equal(self.solver()._restore_by_inferred_symmetries(grid), truth), offset

    def test_a_patch_whose_mirror_images_are_all_missing_is_not_invented(self):
        truth = symmetric_picture()
        grid = truth.copy()
        # the same cell in all four quarters: nothing left to read it from
        for i, j in ((4, 4), (4, 9), (9, 4), (9, 9)):
            grid[i, j] = FONT
        assert self.solver()._restore_by_inferred_symmetries(grid) is None

    def test_a_grid_with_nothing_missing_or_nothing_visible_is_left_alone(self):
        assert self.solver()._restore_by_inferred_symmetries(symmetric_picture()) is None
        assert self.solver()._restore_by_inferred_symmetries(np.full((6, 6), FONT)) is None

    def test_a_grid_with_no_symmetry_is_not_restored(self):
        rng = np.random.default_rng(3)
        grid = rng.integers(1, 6, (12, 12))
        grid[3:5, 3:5] = FONT
        assert self.solver()._restore_by_inferred_symmetries(grid) is None

    def test_only_cells_that_were_missing_change(self):
        truth = symmetric_picture()
        grid = truth.copy()
        grid[4:7, 6:9] = FONT
        restored = self.solver()._restore_by_inferred_symmetries(grid)
        assert np.array_equal(restored[grid != FONT], grid[grid != FONT])


class TestTheFrameAroundAPicture:
    def test_a_strip_is_filled_from_a_mirror_that_is_exact_along_it(self):
        """Row 0 is the transpose of column 0, which no picture-wide mirror
        says; the cells missing from it are read off the column."""
        rng = np.random.default_rng(1)
        truth = rng.integers(1, 6, (12, 12))
        truth[0, 1:] = truth[1:, 0]
        grid = truth.copy()
        grid[0, 4:7] = FONT
        missing = grid == FONT
        filled, still = ColorRestoreSolver(font_val=FONT)._restore_by_local_mirrors(grid, missing)
        assert not still.any() and np.array_equal(filled, truth)

    def test_a_mirror_that_disagrees_somewhere_along_the_strip_is_not_trusted(self):
        rng = np.random.default_rng(1)
        truth = rng.integers(1, 6, (12, 12))
        truth[0, 1:] = truth[1:, 0]
        truth[0, 9] = (truth[9, 0] % 5) + 1  # one pair along the row that disagrees
        grid = truth.copy()
        grid[0, 4:7] = FONT
        _, still = ColorRestoreSolver(font_val=FONT)._restore_by_local_mirrors(grid, grid == FONT)
        assert still.any()

    def test_a_strip_with_too_few_pairs_is_not_trusted(self):
        rng = np.random.default_rng(2)
        truth = rng.integers(1, 6, (6, 6))
        truth[0, 1:] = truth[1:, 0]
        grid = truth.copy()
        grid[0, 2] = FONT
        _, still = ColorRestoreSolver(font_val=FONT)._restore_by_local_mirrors(grid, grid == FONT)
        assert still.any()

    def test_quarter_turns_are_about_the_centre_the_two_mirrors_share(self):
        turns = ColorRestoreSolver()._quarter_turns(None, [("lr", 31), ("ud", 31)])
        assert len(turns) == 2
        assert {turn(0, 5) for turn in turns} == {(5, 31), (26, 0)}


class TestOnRealTasks:
    def test_each_is_restored_and_survives_the_held_out_check(self):
        for task_id, font in (("47996f11", 6), ("981571dc", 0), ("de493100", 7)):
            task = real_task(task_id)
            solver = SymbolicModule(font_val=font).color_restore
            result = checked_solve(solver, task)
            assert result.success, task_id
            assert np.array_equal(result.grid, task.test_subtask.train_out), task_id

    def test_a_picture_it_cannot_restore_is_declined_not_rewritten(self):
        """f9d67f8b: the patch reaches the frame, where no mirror says what
        was there. The region-based restoration used to answer anyway,
        rewriting 20 cells that were right, and the answer passed the check."""
        result = SymbolicModule(font_val=9).color_restore.solve(real_task("f9d67f8b"))
        assert not result.success and "not missing" in result.debug


class TestTheFinalCheck:
    def picture(self):
        """Every row a palindrome, so left-right about the middle is exact,
        with the two middle cells of row 0 - each other's mirror image -
        missing."""
        rng = np.random.default_rng(5)
        grid = rng.integers(1, 6, (12, 12))
        for row in range(12):
            grid[row, 6:] = grid[row, :6][::-1]
        picture = grid.copy()
        picture[0, 5:7] = FONT
        return grid, picture

    def test_a_strip_is_filled_the_way_the_picture_wide_mirror_says(self):
        """Neither middle cell has a visible image under left-right, so they
        are read from elsewhere - and come out equal, as the mirror needs."""
        solver = ColorRestoreSolver(font_val=FONT)
        _, picture = self.picture()
        restored = solver._restore_by_inferred_symmetries(picture)
        assert restored is None or restored[0, 5] == restored[0, 6]

    def test_a_fill_that_breaks_a_picture_wide_mirror_is_refused(self, monkeypatch):
        """Whatever fills what the mirrors could not reach, the result has to
        keep every mirror it was filled by exact."""
        solver = ColorRestoreSolver(font_val=FONT)
        truth, picture = self.picture()

        def fills_unequal(grid, missing, extra=()):
            filled = grid.copy()
            filled[0, 5], filled[0, 6] = 1, 2
            return filled, np.zeros_like(missing)

        monkeypatch.setattr(solver, "_restore_by_local_mirrors", fills_unequal)
        assert solver._restore_by_inferred_symmetries(picture) is None
