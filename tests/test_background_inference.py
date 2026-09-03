"""Tests for symbolic/utils.py's infer_background.

The behaviour worth pinning down is when it *declines* to answer. Naming
the wrong colour is worse than naming none: every consumer downstream
treats that colour as "not an object", so a wrong guess deletes real
content from the analysis. Declining says the colour is unidentified,
never that the grid has no canvas - every grid sits on one, and a canvas
painted over completely just leaves nothing to read off that grid. The thresholds come from measurements over
ARC-AGI-2's training set - see the function's docstring - so the tests here
cover the decision boundary rather than re-deriving the statistics.
"""
from __future__ import annotations

import numpy as np

from symbolic.utils import (
    BACKGROUND_MIN_BORDER_PURITY,
    BACKGROUND_MIN_COVERED_SHARE,
    BACKGROUND_MIN_DOMINANCE,
    _border_cells,
    infer_background,
    infer_task_canvas,
    resolve_background,
)


def _grid(rows):
    """rows: list of digit strings, e.g. ['000', '010']."""
    return np.array([[int(c) for c in row] for row in rows])


def _dominance(grid) -> float:
    """Share of the grid held by its most common colour - recomputed here
    so the boundary tests can assert which clause they're exercising
    instead of trusting the name."""
    _, counts = np.unique(grid, return_counts=True)
    return counts.max() / grid.size


def _border_purity(grid) -> float:
    border = _border_cells(grid)
    _, counts = np.unique(border, return_counts=True)
    return counts.max() / border.size


class TestFindsAnObviousBackground:
    @staticmethod
    def test_a_few_objects_on_a_large_black_field():
        grid = np.zeros((10, 10), dtype=int)
        grid[2, 2] = 3
        grid[7, 8] = 4

        assert infer_background(grid) == 0

    @staticmethod
    def test_the_background_does_not_have_to_be_black():
        """Hardcoding 0 - what this replaces - is wrong for 22.8% of tasks."""
        grid = np.full((10, 10), 7, dtype=int)
        grid[3, 3] = 1

        assert infer_background(grid) == 7

    @staticmethod
    def test_a_uniform_grid_is_all_background():
        assert infer_background(np.full((5, 5), 8, dtype=int)) == 8


class TestDeclinesWhenTheColourIsNotIdentifiable:
    @staticmethod
    def test_a_colour_map_on_a_tiny_grid_yields_none():
        """Three colours, three cells each - the most common one isn't the
        canvas, it's just the one that happens to be first."""
        grid = _grid(["123", "123", "123"])

        assert infer_background(grid) is None

    @staticmethod
    def test_a_dense_two_colour_checkerboard_yields_none():
        grid = np.indices((8, 8)).sum(axis=0) % 2

        assert infer_background(grid) is None

    @staticmethod
    def test_the_two_signals_must_agree():
        """A field of one colour inside a frame of another: the most common
        colour and the border colour name different things, so neither is
        established. Refusing beats picking whichever won by count."""
        grid = np.full((7, 7), 3, dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 5

        assert infer_background(grid) is None


class TestTheDecisionBoundary:
    @staticmethod
    def test_dominance_alone_is_enough_when_the_border_is_not_uniform():
        """Isolates the dominance clause: the border is only 0.79 one
        colour, under its bar, so this can only be accepted on dominance
        (0.82). Deleting the dominance clause would break this test."""
        grid = np.zeros((12, 12), dtype=int)
        grid[1:5, 1:5] = 3
        grid[0, 1:6] = 4  # breaks up the border without taking it over
        grid[-1, 1:6] = 4

        assert _dominance(grid) >= BACKGROUND_MIN_DOMINANCE
        assert _border_purity(grid) < BACKGROUND_MIN_BORDER_PURITY
        assert infer_background(grid) == 0

    @staticmethod
    def test_a_uniform_border_carries_a_grid_dominance_alone_would_not():
        """Isolates the border clause: dominance is 0.55, under its bar,
        and the foreground is split across two colours so neither of them
        takes over. Only the untouched border makes 0 a background here."""
        grid = np.zeros((10, 10), dtype=int)
        interior = [(i, j) for i in range(1, 9) for j in range(1, 9)]
        for k, (i, j) in enumerate(interior[:45]):
            grid[i, j] = 1 if k % 2 == 0 else 2

        assert _dominance(grid) < BACKGROUND_MIN_DOMINANCE
        assert _border_purity(grid) >= BACKGROUND_MIN_BORDER_PURITY
        assert infer_background(grid) == 0

    @staticmethod
    def test_neither_signal_clearing_its_bar_yields_none():
        """Just over half the cells and a border that is only two-thirds
        one colour - below both bars, so nothing is claimed."""
        grid = _grid([
            "001",
            "011",
            "100",
        ])
        assert infer_background(grid) is None


class TestEdgeCases:
    @staticmethod
    def test_a_single_cell_grid():
        assert infer_background(_grid(["5"])) == 5

    @staticmethod
    def test_a_single_row_grid():
        assert infer_background(_grid(["0000010"])) == 0

    @staticmethod
    def test_an_empty_grid_yields_none_rather_than_raising():
        assert infer_background(np.zeros((0, 0), dtype=int)) is None

    @staticmethod
    def test_accepts_a_plain_nested_list():
        assert infer_background([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 0


# -- task level ---------------------------------------------------------------

class TestBackgroundSummary:
    """TaskAnalysis.background reports what the examples showed, including
    that they disagreed - measured over ARC-AGI-2's training set, 7.1% of
    tasks use different backgrounds across their own examples and 19.4%
    establish none anywhere, so neither case is exotic enough to paper over."""

    @staticmethod
    def _task(pairs):
        from rl.arc_task import ARCSubtask, ARCTask

        subtasks = [ARCSubtask(f"ex_{i}", np.array(inp), np.array(out))
                    for i, (inp, out) in enumerate(pairs)]
        return ARCTask("t", subtasks, np.array(pairs[0][0]), np.array(pairs[0][1]))

    @staticmethod
    def _plain(color, mark=9):
        grid = np.full((8, 8), color, dtype=int)
        grid[3, 3] = mark
        return grid

    @classmethod
    def _analyze(cls, pairs):
        from symbolic.analyzer import SymbolicAnalyzer

        return SymbolicAnalyzer().analyze_task(cls._task(pairs)).background

    def test_a_consistent_non_black_background_is_reported_as_such(self):
        """The case a hardcoded 0 got wrong: every example is on orange."""
        bg = self._analyze([(self._plain(7), self._plain(7)),
                            (self._plain(7), self._plain(7))])

        assert bg.consistent_color == 7
        assert bg.varies_across_examples is False
        assert bg.preserved_by_transformation is True

    def test_examples_disagreeing_is_reported_not_resolved(self):
        bg = self._analyze([(self._plain(7), self._plain(7)),
                            (self._plain(1), self._plain(1))])

        assert bg.varies_across_examples is True
        assert bg.consistent_color is None  # no winner picked
        assert set(bg.per_example_input) == {7, 1}

    def test_a_transformation_that_repaints_the_background_is_visible(self):
        """Input and output are inferred separately, so a recolour shows up
        instead of the output being read against the input's colour."""
        bg = self._analyze([(self._plain(7), self._plain(1)),
                            (self._plain(7), self._plain(1))])

        assert bg.per_example_input == (7, 7)
        assert bg.per_example_output == (1, 1)
        assert bg.preserved_by_transformation is False

    def test_no_background_anywhere_claims_nothing(self):
        """all([]) is True - so "preserved" had to be made None here, or a
        task with nothing to compare would report that the transformation
        preserved a background it never had."""
        dense = _grid(["123", "231", "312"])
        bg = self._analyze([(dense, dense), (dense, dense)])

        assert bg.established_anywhere is False
        assert bg.consistent_color is None
        assert bg.varies_across_examples is False
        assert bg.preserved_by_transformation is None

    def test_an_explicit_font_color_still_overrides_inference(self):
        """A caller who already knows the background keeps saying so."""
        from symbolic.analyzer import SymbolicAnalyzer

        task = self._task([(self._plain(7), self._plain(7))])
        bg = SymbolicAnalyzer(font_color=3).analyze_task(task).background

        assert bg.per_example_input == (3,)
        assert bg.consistent_color == 3


# -- the canvas a task's other grids reveal ------------------------------------

class TestTaskCanvas:
    """A grid whose own signals are inconclusive is often the canvas painted
    over, and the task's other grids still show which colour that is - of the
    2100 grids infer_background declines across ARC-AGI-2's training set,
    31.4% belong to a task that establishes a canvas elsewhere."""

    @staticmethod
    def _clear(color):
        grid = np.full((8, 8), color, dtype=int)
        grid[3, 3] = 9
        return grid

    @staticmethod
    def _covered(color, other, share_of_canvas):
        """A grid where `color` survives on roughly `share_of_canvas` of the
        cells and `other` covers the rest - a canvas painted over."""
        grid = np.full((10, 10), other, dtype=int)
        keep = int(round(100 * share_of_canvas))
        grid.flat[:keep] = color
        return grid

    def test_a_majority_of_grids_agreeing_names_the_canvas(self):
        grids = [self._clear(7), self._clear(7), self._covered(7, 2, 0.45)]

        assert infer_task_canvas(grids) == 7

    def test_no_agreement_means_no_canvas(self):
        """Two examples on different backgrounds: neither is the task's
        canvas, and declaring one would flatten a real difference."""
        grids = [self._clear(7), self._clear(1)]

        assert infer_task_canvas(grids) is None

    def test_grids_that_never_identify_one_give_no_canvas(self):
        dense = _grid(["123", "231", "312"])

        assert infer_task_canvas([dense, dense, dense]) is None

    def test_a_minority_is_not_enough(self):
        """One clear grid out of four doesn't make its colour the task's
        canvas - the majority bar exists so a single example can't decide."""
        dense = _grid(["123", "231", "312"])
        grids = [self._clear(7), dense, dense, dense]

        assert infer_task_canvas(grids) is None


class TestResolveBackground:
    @staticmethod
    def _covered(color, other, share_of_canvas):
        grid = np.full((10, 10), other, dtype=int)
        grid.flat[:int(round(100 * share_of_canvas))] = color
        return grid

    def test_the_grids_own_reading_wins_over_the_task_canvas(self):
        """What keeps a task whose examples use different backgrounds from
        being flattened: a grid that shows its own colour keeps it."""
        grid = np.full((8, 8), 1, dtype=int)
        grid[3, 3] = 9

        assert resolve_background(grid, task_canvas=7) == 1

    def test_a_covered_canvas_is_recovered_from_the_task(self):
        """The case a per-grid rule gets wrong: the canvas is still there,
        just painted over, and the task's other grids name it."""
        grid = self._covered(color=7, other=2, share_of_canvas=0.45)

        assert infer_background(grid) is None       # this grid alone can't say
        assert resolve_background(grid, task_canvas=7) == 7

    def test_a_colour_too_thin_to_be_a_covered_canvas_is_refused(self):
        """Below the bar it's content that happens to share the canvas's
        colour; excluding it would delete real objects. The grid has to be
        one the per-grid rule declines, or its own reading answers first
        and the canvas branch is never reached."""
        grid = _grid(["712", "121", "212"])

        assert infer_background(grid) is None                      # inconclusive on its own
        assert (grid == 7).mean() < BACKGROUND_MIN_COVERED_SHARE   # one cell of 7
        assert resolve_background(grid, task_canvas=7) is None

    def test_a_fully_covered_canvas_stays_unidentified(self):
        """Nothing left to observe. Excluding a colour with no cells would
        be a no-op anyway, so nothing is lost by not claiming it."""
        grid = _grid(["123", "231", "312"])

        assert infer_background(grid) is None
        assert (grid == 7).sum() == 0
        assert resolve_background(grid, task_canvas=7) is None

    def test_without_a_task_canvas_it_matches_the_per_grid_rule(self):
        dense = _grid(["123", "231", "312"])

        assert resolve_background(dense, task_canvas=None) is infer_background(dense)
