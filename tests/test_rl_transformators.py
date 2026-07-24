"""Tests for the object-transform functions integrated into
rl/arc_transformators.py this round: symmetry_reflection,
symmetric_restoration, color_swap, shape_swap, color_copy, shape_copy,
dense_outer_contour - plus their wiring into rl.arc_world.World.apply_transform.

Same philosophy as the rest of the RL test suite: a handful of exact tests
built by hand where the right answer is known, plus smoke tests (crash-or-not
+ "no duplicate coordinates", the specific bug class two of these functions
had before this round) for the rest.
"""
from __future__ import annotations

import numpy as np

from data.configs.env_configs import COLORS_MAPPING
from rl.arc_transformators import (
    color_copy, color_swap, dense_outer_contour, shape_copy, shape_swap,
    symmetric_restoration, symmetry_reflection,
)
from rl.arc_world import World
from symbolic.objects_analysis import GridObject


def make_object(coords, color, grid, label="obj"):
    return GridObject("test", coords, [color], label, grid.shape, 0, grid)


def assert_no_duplicate_coords(obj):
    assert len(obj.coords) == len(set(obj.coords)), f"duplicate coords in {obj.coords}"


# -- color_swap / color_copy -------------------------------------------------

def test_color_swap_exact():
    grid = np.zeros((3, 3), dtype=int)
    grid[0, 0] = 1
    grid[2, 2] = 2
    obj1 = make_object([(0, 0)], 1, grid, "obj1")
    obj2 = make_object([(2, 2)], 2, grid, "obj2")

    new_grid = color_swap(grid, obj1, obj2, font_color=0)

    assert new_grid[0, 0] == 2
    assert new_grid[2, 2] == 1
    assert obj1.color_numbers == (2,)
    assert obj2.color_numbers == (1,)


def test_color_copy_exact():
    grid = np.zeros((3, 3), dtype=int)
    grid[0, 0] = 1
    grid[2, 2] = 2
    obj1 = make_object([(0, 0)], 1, grid, "obj1")
    obj2 = make_object([(2, 2)], 2, grid, "obj2")

    new_grid = color_copy(grid, obj1, obj2, font_color=0)

    assert new_grid[0, 0] == 2
    assert obj1.color_numbers == (2,)
    assert obj1.colors == (COLORS_MAPPING[2],)


# -- shape_copy / shape_swap --------------------------------------------------

def test_shape_copy_exact():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3
    l_shape = [(4, 4), (4, 5), (5, 4)]
    for x, y in l_shape:
        grid[x, y] = 5
    obj1 = make_object([(0, 0)], 3, grid, "obj1")
    obj2 = make_object(l_shape, 5, grid, "obj2")

    new_grid = shape_copy(grid, obj1, obj2, font_color=0)

    assert set(obj1.coords) == {(0, 0), (0, 1), (1, 0)}
    assert new_grid[0, 0] == 3 and new_grid[0, 1] == 3 and new_grid[1, 0] == 3
    assert obj1.color_numbers == (3,)


def test_shape_swap_exact():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3
    l_shape = [(4, 4), (4, 5), (5, 4)]
    for x, y in l_shape:
        grid[x, y] = 5
    obj1 = make_object([(0, 0)], 3, grid, "obj1")
    obj2 = make_object(l_shape, 5, grid, "obj2")

    new_grid = shape_swap(grid, obj1, obj2, font_color=0)

    # obj1 takes obj2's shape around its own former center, keeps its own color
    assert set(obj1.coords) == {(0, 0), (0, 1), (1, 0)}
    assert obj1.color_numbers == (3,)
    # obj2 takes obj1's (single-cell) shape around its own former center, keeps its own color
    assert set(obj2.coords) == {(4, 4)}
    assert obj2.color_numbers == (5,)
    assert new_grid[4, 4] == 5


# -- symmetry_reflection / symmetric_restoration -----------------------------

def test_symmetry_reflection_exact():
    """An L-tromino reflected across its own row-center completes a solid
    2x2 square (compactness 1.0, the maximum possible) - the 'horizontal'
    direction is tried first and reaches that maximum, so it must win."""
    grid = np.zeros((6, 6), dtype=int)
    coords = [(2, 2), (2, 3), (3, 2)]
    for x, y in coords:
        grid[x, y] = 8
    obj1 = make_object(coords, 8, grid, "obj1")

    new_grid = symmetry_reflection(grid, obj1, font_color=0)

    assert set(obj1.coords) == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert new_grid[3, 3] == 8
    assert obj1.color_numbers == (8,)
    assert_no_duplicate_coords(obj1)


def test_symmetric_restoration_exact():
    """A 3-cell L-shape (missing one corner of its own bounding box) gets
    completed into a full 2x2 square by mirroring across its own center."""
    grid = np.zeros((5, 5), dtype=int)
    coords = [(0, 0), (0, 1), (1, 0)]
    for x, y in coords:
        grid[x, y] = 7
    obj1 = make_object(coords, 7, grid, "obj1")

    new_grid = symmetric_restoration(grid, obj1, font_color=0)

    assert set(obj1.coords) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert new_grid[1, 1] == 7
    assert_no_duplicate_coords(obj1)


# -- dense_outer_contour ------------------------------------------------------

def test_dense_outer_contour_exact():
    """Regression test: a single-row object (min_i == max_i) used to have
    every contour cell double-counted, since the "top edge" and "bottom
    edge" loops both scan the same row when the object is exactly 1 row
    tall - inflating obj1.coords with duplicates."""
    grid = np.zeros((6, 6), dtype=int)
    coords = [(1, 1), (1, 4)]
    for x, y in coords:
        grid[x, y] = 9
    obj1 = make_object(coords, 9, grid, "obj1")

    new_grid = dense_outer_contour(grid, obj1, color=3, font_color=0)

    assert new_grid[1, 2] == 3 and new_grid[1, 3] == 3
    assert set(obj1.coords) == {(1, 1), (1, 2), (1, 3), (1, 4)}
    assert_no_duplicate_coords(obj1)


# -- smoke tests: assorted shapes, crash-or-not + no-duplicate-coords --------

SMOKE_SHAPES = [
    [(0, 0)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(2, 0), (3, 0), (2, 1)],
]


def test_symmetry_reflection_smoke():
    for coords in SMOKE_SHAPES:
        grid = np.zeros((8, 8), dtype=int)
        for x, y in coords:
            grid[x, y] = 4
        obj1 = make_object(coords, 4, grid, "obj1")
        symmetry_reflection(grid, obj1, font_color=0)
        assert_no_duplicate_coords(obj1)


def test_symmetric_restoration_smoke():
    for coords in SMOKE_SHAPES:
        grid = np.zeros((8, 8), dtype=int)
        for x, y in coords:
            grid[x, y] = 4
        obj1 = make_object(coords, 4, grid, "obj1")
        symmetric_restoration(grid, obj1, font_color=0)
        assert_no_duplicate_coords(obj1)


def test_dense_outer_contour_smoke():
    for coords in SMOKE_SHAPES:
        grid = np.zeros((8, 8), dtype=int)
        for x, y in coords:
            grid[x, y] = 4
        obj1 = make_object(coords, 4, grid, "obj1")
        dense_outer_contour(grid, obj1, color=6, font_color=0)
        assert_no_duplicate_coords(obj1)


# -- wiring: World.apply_transform dispatches to the new functions -----------

def _world():
    return World(objects=[], actions_dict={}, font_color=0)


def test_world_dispatches_color_swap():
    grid = np.zeros((3, 3), dtype=int)
    grid[0, 0] = 1
    grid[2, 2] = 2
    obj1 = make_object([(0, 0)], 1, grid, "obj1")
    obj2 = make_object([(2, 2)], 2, grid, "obj2")

    new_grid = _world().apply_transform(-1, "color_swap", obj1, obj2, grid, [obj1, obj2], {})

    assert new_grid[0, 0] == 2 and new_grid[2, 2] == 1


def test_world_dispatches_dense_outer_contour():
    grid = np.zeros((6, 6), dtype=int)
    coords = [(1, 1), (1, 4)]
    for x, y in coords:
        grid[x, y] = 9
    obj1 = make_object(coords, 9, grid, "obj1")

    new_grid = _world().apply_transform(3, "dense_outer_contour", obj1, obj1, grid, [obj1], {})

    assert new_grid[1, 2] == 3 and new_grid[1, 3] == 3
