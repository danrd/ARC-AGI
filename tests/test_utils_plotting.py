"""Tests for utils/plotting.py: regression tests for bugs found and fixed
(plot_shape's independent axis mins, plot_intersection's single-shape vs
list-of-shapes detection, plot_multiple_tasks' broken dataset indexing,
plot_task's subplots squeeze on a 1-column grid) plus functional tests for
the composability/correctness-overlay rework (plot_grid(ax=...), the real
per-cell diff in plot_grids_comparison, plot_multiple_grids' one-figure-
per-grid fix, plot_task_with_prediction). Where there's a checkable
property (a returned figure, an axis title, a distinct-figures count) the
tests check it; everything else stays crash-or-not, in the same spirit as
tests/test_rl_plotting.py - these are visualization helpers, not something
worth asserting exact pixel output against.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # headless: no display needed to run these tests
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import (
    plot_grid,
    plot_grids_comparison,
    plot_intersection,
    plot_multiple_grids,
    plot_multiple_tasks,
    plot_shape,
    plot_task,
    plot_task_result,
    plot_task_with_prediction,
)


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


def _title_texts(fig):
    return [ax.get_title() for ax in fig.axes]


def test_plot_grid_draws_on_the_given_ax():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    grid = np.array([[1, 2], [3, 4]])

    returned = plot_grid(grid, ax=ax1)

    assert returned is ax1
    assert ax1.images  # something was actually drawn on ax1...
    assert not ax2.images  # ...and ax2 was left untouched
    plt.close(fig)


def test_plot_grids_comparison_overlay_reports_a_match():
    grid = np.array([[1, 2], [3, 4]])

    fig = plot_grids_comparison(grid, grid.copy())

    assert any("MATCH" in t for t in _title_texts(fig))
    plt.close(fig)


def test_plot_grids_comparison_overlay_reports_a_mismatch():
    """Regression test: the old diff panel compared np.setdiff1d(grid_2,
    grid_1) - the SET of colors used, not per-cell positions. Two grids
    using the same colors in a completely different arrangement (a wrong
    prediction, essentially) used to show an empty "no new cells" diff.

    grid_2 is the one checked against target_grid (see plot_grids_comparison's
    docstring), so `predicted` has to be passed as grid_2, not grid_1."""
    predicted = np.array([[1, 2], [3, 9]])  # one cell wrong vs target
    target = np.array([[1, 2], [3, 4]])

    fig = plot_grids_comparison(target, predicted, target_grid=target)

    assert any("mismatch" in t for t in _title_texts(fig))
    plt.close(fig)


def test_plot_grids_comparison_shape_mismatch_does_not_crash():
    predicted = np.zeros((3, 3), dtype=int)
    target = np.zeros((4, 4), dtype=int)

    fig = plot_grids_comparison(target, predicted, target_grid=target)

    assert any("mismatch" in t for t in _title_texts(fig))
    plt.close(fig)


def test_plot_multiple_grids_returns_one_distinct_figure_per_grid():
    """Regression test: plot_multiple_grids used to call plot_grid(grid) in
    a loop without ever creating a new figure - every grid landed on
    whatever the CURRENT axes happened to be, so only the last one ended
    up actually visible."""
    grids = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.zeros((2, 2), dtype=int)]

    figs = plot_multiple_grids(grids)

    assert len(figs) == len(grids)
    assert len({id(f) for f in figs}) == len(grids)
    for fig in figs:
        plt.close(fig)


def test_plot_task_returns_the_figure():
    task_id = "faketask"
    dataset = _fake_dataset(task_id)

    fig = plot_task(task_id, dataset)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_task_with_prediction_correct_and_incorrect():
    task_id = "faketask"
    dataset = _fake_dataset(task_id)
    correct_answer = np.array(dataset.training_solutions[task_id][0])

    fig_task, fig_cmp_correct = plot_task_with_prediction(task_id, dataset, correct_answer)
    assert isinstance(fig_task, plt.Figure)
    assert any("MATCH" in t for t in _title_texts(fig_cmp_correct))

    wrong_answer = 1 - correct_answer  # flips every 0/1 cell in this fixture's solution
    _, fig_cmp_wrong = plot_task_with_prediction(task_id, dataset, wrong_answer)
    assert any("mismatch" in t for t in _title_texts(fig_cmp_wrong))

    plt.close(fig_task)
    plt.close(fig_cmp_correct)
    plt.close(fig_cmp_wrong)


def _fake_task(n_train: int = 2):
    """Minimal stand-in for rl.arc_task.ARCTask - only what plot_task_result
    actually reads (task.subtasks[i].train_inp/.train_out, task.test_subtask,
    task.id)."""
    subtasks = [
        SimpleNamespace(train_inp=np.array([[1, 0], [0, 1]]), train_out=np.array([[0, 1], [1, 0]]))
        for _ in range(n_train)
    ]
    test_subtask = SimpleNamespace(train_inp=np.array([[1, 1], [0, 0]]), train_out=np.array([[0, 0], [1, 1]]))
    return SimpleNamespace(subtasks=subtasks, test_subtask=test_subtask, id="faketask")


def test_plot_task_result_shows_every_train_pair_and_the_task_id():
    task = _fake_task(n_train=3)

    fig = plot_task_result(task, predicted_grid=task.test_subtask.train_out,
                            eval_result=SimpleNamespace(primary_score=1.0, solved=True))

    titles = _title_texts(fig)
    assert sum(t.startswith("Train") and t.endswith("input") for t in titles) == 3
    assert sum(t.startswith("Train") and t.endswith("output") for t in titles) == 3
    assert "faketask" in fig.get_suptitle()
    assert "solved=True" in fig.get_suptitle()
    plt.close(fig)


def test_plot_task_result_reports_match_for_a_correct_prediction():
    task = _fake_task()

    fig = plot_task_result(task, predicted_grid=task.test_subtask.train_out,
                            eval_result=SimpleNamespace(primary_score=1.0, solved=True))

    assert any("MATCH" in t for t in _title_texts(fig))
    plt.close(fig)


def test_plot_task_result_reports_shape_mismatch_with_dimensions():
    task = _fake_task()
    wrong_shape_prediction = np.zeros((5, 5), dtype=int)

    fig = plot_task_result(task, predicted_grid=wrong_shape_prediction,
                            eval_result=SimpleNamespace(primary_score=0.0, solved=False))

    assert any("shape mismatch" in t and "5x5" in t for t in _title_texts(fig))
    plt.close(fig)


def test_plot_task_result_handles_a_prediction_that_never_parsed():
    """predicted_grid=None (arc_result_plotter's signal that generated_text
    didn't parse into a grid at all) must not crash - there's nothing to
    compare cell-by-cell, so the prediction/diff panels are just left
    empty instead."""
    task = _fake_task()

    fig = plot_task_result(task, predicted_grid=None,
                            eval_result=SimpleNamespace(primary_score=0.0, solved=False))

    assert any("didn't parse" in t for t in _title_texts(fig))
    plt.close(fig)
