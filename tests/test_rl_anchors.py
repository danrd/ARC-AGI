"""Addressable coordinates - what a slot names when it does not name an
object.

The module exists because the first version of this did not: every anchor
was a one-cell GridObject so the env's slot machinery would take it, and
of OBJECT_SCHEMA's 25 fields 8 were then constant across every anchor of
every grid, describing the shape of a 1x1 cell. These tests hold the
separation in place - that an anchor is points and a schema of its own,
and that nothing here builds an object.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from rl.anchors import (ANCHOR_DIM, ANCHOR_SCHEMA, anchor_embeddings,
                        anchor_points, boundary_lines)


def _banded():
    """One change down and one across, so the boundary lines are something
    other than the four edges."""
    grid = np.zeros((6, 6), dtype=int)
    grid[3, 2] = 1
    return grid


class TestWhichLinesAnAnchorCanSitOn:

    def test_a_uniform_grid_offers_only_its_edges(self):
        grid = np.zeros((6, 6), dtype=int)
        assert boundary_lines(grid, 0) == [0, 5]
        assert boundary_lines(grid, 1) == [0, 5]

    def test_both_sides_of_a_change_are_kept(self):
        """A rectangle that stops before a new colour begins ends on the
        line before it; one that starts with it begins on the line itself.
        Keeping one of the two makes half the rectangles inexpressible."""
        assert boundary_lines(_banded(), 0) == [0, 2, 3, 4, 5]

    def test_a_one_line_grid_does_not_fall_over(self):
        assert boundary_lines(np.zeros((1, 4), dtype=int), 0) == [0]

    def test_the_points_are_the_product_of_the_two_axes(self):
        grid = _banded()
        points = anchor_points(grid)
        assert len(points) == len(boundary_lines(grid, 0)) * len(boundary_lines(grid, 1))
        assert points.shape[1] == 2
        assert len(set(map(tuple, points))) == len(points), "a point twice"

    def test_the_order_is_stable(self):
        """A slot index has to mean the same point every time it is read."""
        grid = _banded()
        assert np.array_equal(anchor_points(grid), anchor_points(grid))


class TestWhatAnAnchorCarries:

    def test_the_width_is_the_schema(self):
        assert ANCHOR_DIM == sum(width for _name, width in ANCHOR_SCHEMA)
        block = anchor_embeddings(_banded(), anchor_points(_banded()))
        assert block.shape[1] == ANCHOR_DIM

    def test_nothing_in_it_describes_a_shape(self):
        """The point of the separation. An anchor has no size, symmetry,
        compactness, closure or holes - a 1x1 cell has the same answer for
        all of them, which is why carrying OBJECT_SCHEMA here left 8 fields
        constant on every grid."""
        names = {name for name, _width in ANCHOR_SCHEMA}
        assert not names & {"size", "hor_size", "vert_size", "symmetry_type",
                            "compactness", "closure", "inner_holes",
                            "outer_holes", "inner_holes_share"}

    def test_the_position_is_normalised_to_the_grid(self):
        grid = np.zeros((5, 9), dtype=int)
        block = anchor_embeddings(grid, np.array([[0, 0], [4, 8]]))
        assert block[0, 0] == 0.0 and block[0, 1] == 0.0
        assert block[1, 0] == 1.0 and block[1, 1] == 1.0

    def test_the_colour_is_the_one_under_the_point(self):
        grid = _banded()
        block = anchor_embeddings(grid, np.array([[3, 2], [0, 0]]))
        assert block[0, 2 + 1] == 1.0, "the ink cell holds colour 1"
        assert block[1, 2 + 0] == 1.0, "the background cell holds colour 0"
        assert block[:, 2:12].sum(axis=1).tolist() == [1.0, 1.0], "one colour each"

    def test_it_reports_the_grid_it_is_given_not_the_one_it_came_from(self):
        """The points are fixed for an episode, what is under them is not -
        an agent that paints has to see what it painted."""
        grid = _banded()
        painted = grid.copy()
        painted[0, 0] = 4
        first = anchor_embeddings(grid, np.array([[0, 0]]))
        second = anchor_embeddings(painted, np.array([[0, 0]]))
        assert first[0, 2 + 0] == 1.0
        assert second[0, 2 + 4] == 1.0

    def test_the_edge_flags_say_which_edge(self):
        block = anchor_embeddings(np.zeros((5, 5), dtype=int),
                                  np.array([[0, 0], [4, 4], [2, 2]]))
        assert block[0, 16] == 1.0 and block[0, 18] == 1.0   # top, left
        assert block[1, 17] == 1.0 and block[1, 19] == 1.0   # bottom, right
        assert block[2, 16:20].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_the_boundary_flags_mark_where_the_grid_changes(self):
        """The field that says why this point is an anchor at all."""
        grid = _banded()                      # row 3 differs from row 2
        block = anchor_embeddings(grid, np.array([[3, 0], [0, 0]]))
        assert block[0, 12] == 1.0, "row 3 differs from the row above it"
        assert block[1, 12] == 0.0, "row 0 has no row above it"

    def test_every_value_is_in_the_box_the_env_declares(self):
        grid = _banded()
        block = anchor_embeddings(grid, anchor_points(grid))
        assert block.min() >= 0.0 and block.max() <= 1.0

    def test_the_module_never_reaches_for_the_symbolic_layer(self):
        """The regression this module is. Anchors used to be one-cell
        GridObjects, which is how a corner came to carry symmetry,
        compactness and hole counts.

        Checked on the imports rather than by patching GridObject: this
        module does not import it, so a patch cannot fail and a test built
        on one passes whatever the module does.
        """
        import rl.anchors as anchors

        tree = ast.parse(Path(anchors.__file__).read_text())
        modules = {node.module for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module}
        modules |= {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names}
        assert not [m for m in modules if m.split(".")[0] in ("symbolic", "rl")], modules
