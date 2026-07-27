"""Regression tests for bugs found and fixed in utils/plotting.py:
plot_shape (independent axis mins), plot_intersection (single-shape vs
list-of-shapes detection), plot_multiple_tasks (broken dataset indexing),
plot_task (subplots squeeze on a 1-column grid). Crash-or-not, in the same
spirit as tests/test_rl_plotting.py - these are visualization helpers, the
"right answer" is that they run without raising on the inputs their own
type hints promise to accept.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless: no display needed to run these tests
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import plot_intersection, plot_multiple_tasks, plot_shape, plot_task


def test_plot_shape_with_independent_axis_extents_does_not_crash():
    """Regression test: a single combined min(min(i), min(j)) used to shift
    both axes by the same amount, instead of each by its own min - crashes
    (or silently mis-plots) whenever the i-range and j-range don't share a
    minimum, e.g. a diagonal shape."""
    shape = [(5, 10), (6, 11), (7, 12)]

    plot_shape(shape)

    assert plt.gcf().get_axes()
    plt.close("all")


def test_plot_intersection_single_shape_does_not_crash():
    """Regression test: isinstance(shape, list) can't distinguish a single
    shape (List[tuple]) from a list of shapes (List[List[tuple]]) - both
    are `list` at the top level - so a single shape used to get "flattened"
    into its own individual coordinate numbers and crash inside
    coords_transform."""
    grid = np.zeros((5, 5), dtype=int)

    plot_intersection(grid, [(1, 2), (3, 4)])

    assert plt.gcf().get_axes()
    plt.close("all")


def test_plot_intersection_list_of_shapes_does_not_crash():
    grid = np.zeros((5, 5), dtype=int)

    plot_intersection(grid, [[(1, 2), (2, 2)], [(3, 4)]])

    assert plt.gcf().get_axes()
    plt.close("all")


def _fake_dataset(task_id: str):
    """Minimal stand-in for ARCDataset - only what plot_task actually
    reads (.training_challenges/.training_solutions) - real ARCDataset
    construction is heavyweight and unrelated to what's under test here."""
    challenge = {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 1], [0, 0]]}],
    }
    return SimpleNamespace(
        training_challenges={task_id: challenge},
        training_solutions={task_id: [[[0, 0], [1, 1]]]},
    )


def test_plot_task_single_column_does_not_crash():
    """Regression test: plt.subplots(2, w) squeezes away a dimension of
    size 1, so a task with exactly 1 train example and 0 test examples (or
    vice versa) made axs 1D and every axs[row, col] index below raise."""
    task_id = "faketask"
    dataset = _fake_dataset(task_id)
    dataset.training_challenges[task_id]["test"] = []  # w = 1 (train only)

    plot_task(task_id, dataset)

    assert plt.gcf().get_axes()
    plt.close("all")


def test_plot_multiple_tasks_with_string_ids_does_not_crash():
    """Regression test: plot_multiple_tasks indexed dataset.tasks[task_id]
    (a list) by a string task_id - matching its own List[str] type hint
    would always TypeError. It should just forward task_id straight to
    plot_task, which already expects a string label."""
    task_id = "faketask"
    dataset = _fake_dataset(task_id)

    plot_multiple_tasks([task_id], dataset)

    assert plt.gcf().get_axes()
    plt.close("all")
