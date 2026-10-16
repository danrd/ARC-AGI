"""scripts/stroke_census.py: how many cell strokes a task's pairs need."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stroke_census as strokes  # noqa: E402


def test_a_pair_one_rectangle_away_takes_one_stroke():
    grid = np.zeros((4, 4), int)
    target = grid.copy()
    target[1:3, 1:3] = 2
    assert strokes.strokes_for(grid, target, ["black", "red"], timeout=10) == (1, 4)


def test_two_colours_take_a_stroke_each_and_nothing_to_do_takes_none():
    grid = np.zeros((3, 5), int)
    target = grid.copy()
    target[0, :] = 1
    target[2, :] = 3
    assert strokes.strokes_for(grid, target, ["black", "blue", "green"], timeout=10) == (2, 10)
    assert strokes.strokes_for(grid, grid, ["black"], timeout=10) == (0, 0)


def test_a_task_whose_grids_change_size_is_skipped_and_one_that_does_not_has_a_line_per_pair():
    small, big = np.zeros((2, 2), int), np.zeros((3, 3), int)
    assert "skipped" in strokes.stroke_task("t", [("p", small, big)], hybrid=False)
    target = small.copy()
    target[0, 0] = 2
    record = strokes.stroke_task("t", [("p0", small, target), ("p1", small, small)], hybrid=False)
    assert record["pairs"] == [{"strokes": 1, "wrong": 1}, {"strokes": 0, "wrong": 0}] and "hybrid" not in record
