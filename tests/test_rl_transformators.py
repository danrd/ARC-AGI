"""Tests for the object-transform functions in rl/arc_transformators.py:
symmetry_reflection, symmetric_restoration, color_swap, shape_swap,
color_copy, shape_copy, dense_outer_contour - plus their wiring into
rl.arc_world.World.apply_transform.

Same philosophy as the rest of the RL test suite: a handful of exact tests
built by hand where the right answer is known, plus smoke tests for the rest
- crash-or-not, and "no duplicate coordinates", which is the failure these
functions are prone to: several of them write coordinates computed two ways
into one object, and a coordinate emitted twice corrupts every size and
compactness read off it afterwards.
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


# -- the shortest path: any colour, never padding ------------------------------

class TestTheShortestPath:
    """find_shortest_path used to treat colour 1 as a wall - a leftover of
    grids padded with 1. With padding at 10 it walled off blue cells alone:
    a path bent round them, never reached a blue shape, and gravity could
    not move anything to or from one."""

    @staticmethod
    def _row(middle, ends=4):
        grid = np.zeros((3, 7), dtype=int)
        grid[1, 0] = grid[1, 6] = ends
        grid[1, 3] = middle
        return grid

    def test_a_blue_cell_is_crossed_like_any_other_colour(self):
        from rl.arc_transformators import find_shortest_path
        for middle in (1, 2):
            path = find_shortest_path(self._row(middle), (1, 0), (1, 6))
            assert path == [(1, y) for y in range(7)], f"bent round colour {middle}"

    def test_blue_shapes_can_be_joined(self):
        from rl.arc_transformators import find_shortest_path
        assert len(find_shortest_path(self._row(0, ends=1), (1, 0), (1, 6))) == 7

    def test_padding_is_never_entered(self):
        from rl.arc_transformators import find_shortest_path
        grid = self._row(0)
        grid[:, 3] = 10
        grid[2, 3] = 0
        path = find_shortest_path(grid, (1, 0), (1, 6))
        assert path and all(grid[x, y] != 10 for x, y in path)
        assert (2, 3) in path, "round the padding, through the one gap"

    def test_gravity_moves_a_blue_shape(self):
        from rl.arc_transformators import gravity
        grid = np.zeros((1, 7), dtype=int)
        grid[0, 0], grid[0, 6] = 1, 1
        anchor = make_object([(0, 0)], 1, grid, "anchor")
        mover = make_object([(0, 6)], 1, grid, "mover")

        new_grid = gravity(grid, anchor, mover, 0)

        assert new_grid.tolist() == [[1, 1, 0, 0, 0, 0, 0]]

    def test_world_joins_two_blue_shapes(self):
        grid = np.zeros((1, 5), dtype=int)
        grid[0, 0], grid[0, 4] = 1, 1
        obj1 = make_object([(0, 0)], 1, grid, "obj1")
        obj2 = make_object([(0, 4)], 1, grid, "obj2")

        new_grid = _world().apply_transform(3, "shortest_path", obj1, obj2, grid, [obj1, obj2], {})

        assert new_grid.tolist() == [[1, 3, 3, 3, 1]]


# -- directed gravity -----------------------------------------------------------

class TestDirectedGravity:
    """Push a shape one way until it touches something: another shape, or
    the edge of the grid."""

    @staticmethod
    def _push(grid, cells, direction):
        from rl.arc_transformators import directed_gravity
        obj = make_object(cells, int(grid[cells[0]]), grid, "pushed")
        return directed_gravity(grid.copy(), obj, direction, 0), obj

    def test_it_stops_against_another_shape(self):
        grid = np.array([[2, 2, 0, 0, 0, 3]])
        new_grid, obj = self._push(grid, [(0, 0), (0, 1)], "E")
        assert new_grid.tolist() == [[0, 0, 0, 2, 2, 3]]
        assert sorted(obj.coords) == [(0, 3), (0, 4)], "the object moves with its cells"

    def test_with_nothing_in_the_way_it_stops_at_the_edge(self):
        grid = np.zeros((5, 3), dtype=int)
        grid[1, 1] = 4
        new_grid, _ = self._push(grid, [(1, 1)], "S")
        assert new_grid[4, 1] == 4 and new_grid.sum() == 4

    def test_a_shape_already_touching_stays(self):
        grid = np.array([[0, 2, 3]])
        new_grid, _ = self._push(grid, [(0, 1)], "E")
        assert new_grid.tolist() == [[0, 2, 3]]

    def test_the_whole_shape_stops_when_any_cell_would_touch(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[0, 0] = grid[1, 0] = 5
        grid[3, 1] = 7
        new_grid, _ = self._push(grid, [(0, 0), (1, 0)], "E")
        assert new_grid[0].tolist() == [0, 0, 0, 5] and new_grid[1].tolist() == [0, 0, 0, 5]
        grid[1, 3] = 7
        new_grid, _ = self._push(grid, [(0, 0), (1, 0)], "E")
        assert new_grid[0].tolist() == [0, 0, 5, 0], "row 1 hits the 7 first"

    def test_padding_stops_it_like_the_edge(self):
        grid = np.array([[2, 0, 0, 10, 10]])
        new_grid, _ = self._push(grid, [(0, 0)], "E")
        assert new_grid.tolist() == [[0, 0, 2, 10, 10]]

    def test_diagonally(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[0, 0] = 6
        new_grid, _ = self._push(grid, [(0, 0)], "SE")
        assert new_grid[3, 3] == 6 and new_grid.sum() == 6

    def test_the_world_dispatches_it_by_name(self):
        from rl.utils import define_feasible_actions
        from data.configs.env_configs import (COLOR_DEPENDENT_ACTIONS, DIRECTION_DEPENDENT_ACTIONS,
                                              DOUBLE_COLOR_DEPENDENT_ACTIONS)
        names = define_feasible_actions(["directed_gravity"], ["red"], ["N", "W"],
                                        COLOR_DEPENDENT_ACTIONS, DOUBLE_COLOR_DEPENDENT_ACTIONS,
                                        DIRECTION_DEPENDENT_ACTIONS)
        assert set(names.values()) == {"directed_gravity_N", "directed_gravity_W", "submit"}
        grid = np.array([[0, 0, 9]])
        obj = make_object([(0, 2)], 9, grid, "obj")
        new_grid = _world().apply_transform(-1, "directed_gravity_W", obj, obj, grid, [obj], {})
        assert new_grid.tolist() == [[9, 0, 0]]


# -- part_recolor ----------------------------------------------------------------

class TestPartRecolor:
    """Repaint one colour's cells inside a shape, leaving its other colours:
    the part a task recolours is often a colour of a many-coloured shape,
    and recolor repaints the whole of it."""

    @staticmethod
    def _shape():
        grid = np.array([[2, 2, 5], [5, 2, 0]])
        return grid, make_object([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], 2, grid, "shape")

    def test_only_that_colour_changes(self):
        from rl.arc_transformators import part_recolor
        grid, obj = self._shape()
        new_grid = part_recolor(grid.copy(), obj, 1, 5)
        assert new_grid.tolist() == [[2, 2, 1], [1, 2, 0]]
        assert set(obj.color_numbers) == {1, 2}, "the shape knows its new colours"

    def test_a_colour_the_shape_lacks_or_itself_changes_nothing(self):
        from rl.arc_transformators import part_recolor
        grid, obj = self._shape()
        assert part_recolor(grid.copy(), obj, 1, 7).tolist() == grid.tolist()
        assert part_recolor(grid.copy(), obj, 5, 5).tolist() == grid.tolist()

    def test_the_world_reads_both_colours_off_the_name(self):
        grid, obj = self._shape()
        world = World(objects=[obj], actions_dict={1: "blue_part_recolor_gray"}, font_color=0)
        add, transform = world.parse_action([1, 0, 0])
        new_grid = world.apply_transform(add, transform, obj, obj, grid, [obj], {})
        assert new_grid.tolist() == [[2, 2, 1], [1, 2, 0]]
