"""Tests for the relation vocabulary added to symbolic/summaries.py:
real containment, adjacency, and the size ratio.

Containment is the one with history. It used to be decided by bounding
boxes alone, which - measured across real tasks - was right 49% of the
time: it fires for 3.59% of object pairs while only 1.76% are genuinely
contained. It never missed a real one, though, which is why it survives as
the prefilter in front of the flood fill rather than being replaced.
"""
from __future__ import annotations

import numpy as np

from symbolic.objects_analysis import GridObject
from symbolic.summaries import (
    RELATION_FEATURE_NAMES,
    RelationAnalyzer,
)


def _object(coords, grid_shape, color=3, label="complex_0"):
    grid = np.zeros(grid_shape, dtype=int)
    for i, j in coords:
        grid[i, j] = color
    return GridObject("complex", coords, [color], label, grid_shape, 0, grid)


def _ring(size=7):
    """A hollow square frame with a one-cell wall."""
    return [(i, j) for i in range(size) for j in range(size)
            if i in (0, size - 1) or j in (0, size - 1)]


class TestContainment:
    GRID = (9, 9)

    def test_a_dot_inside_a_ring_is_contained(self):
        ring = _object([(i + 1, j + 1) for i, j in _ring(7)], self.GRID, label="complex_0")
        dot = _object([(4, 4)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.in_contour(ring, dot, self.GRID) == "object_2"

    def test_the_bounding_box_alone_is_not_containment(self):
        """The regression this exists for: an L-shape's bounding box holds
        a second object that is nowhere near enclosed - the old rule called
        that containment, and half its firings were of this kind."""
        l_shape = _object([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)],
                          self.GRID)
        inside_bbox = _object([(2, 3)], self.GRID, color=2, label="complex_1")

        # the bounding-box test still says yes...
        assert RelationAnalyzer._bbox_inside(inside_bbox, l_shape) is True
        # ...and the confirmed relation says no
        assert RelationAnalyzer.in_contour(l_shape, inside_bbox, self.GRID) is None

    def test_a_ring_with_a_gap_does_not_contain(self):
        """One missing wall cell and the dot can walk out - the difference
        a flood fill sees and a bounding box cannot."""
        wall = [(i + 1, j + 1) for i, j in _ring(7)]
        wall.remove((1, 4))  # punch a hole in the top edge
        ring = _object(wall, self.GRID)
        dot = _object([(4, 4)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.in_contour(ring, dot, self.GRID) is None

    def test_containment_is_reported_in_both_directions(self):
        ring = _object([(i + 1, j + 1) for i, j in _ring(7)], self.GRID, label="complex_0")
        dot = _object([(4, 4)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.in_contour(dot, ring, self.GRID) == "object_1"

    def test_side_by_side_objects_are_not_contained(self):
        left = _object([(2, 2), (3, 2)], self.GRID)
        right = _object([(2, 6), (3, 6)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.in_contour(left, right, self.GRID) is None

    def test_without_a_grid_shape_the_old_answer_is_returned(self):
        """Older callers pass no shape. Returning the unconfirmed
        bounding-box verdict keeps them working exactly as before, rather
        than silently reporting nothing."""
        l_shape = _object([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)],
                          self.GRID)
        inside_bbox = _object([(2, 3)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.in_contour(l_shape, inside_bbox) == "object_2"
        assert RelationAnalyzer.in_contour(l_shape, inside_bbox, self.GRID) is None


class TestTouches:
    GRID = (8, 8)

    def test_objects_sharing_an_edge_touch(self):
        a = _object([(2, 2), (3, 2)], self.GRID)
        b = _object([(2, 3), (3, 3)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.touches(a, b) is True

    def test_objects_meeting_only_at_a_corner_do_not(self):
        """Edge adjacency only - the same 4-connectivity that kept these
        two separate objects in the first place."""
        a = _object([(2, 2)], self.GRID)
        b = _object([(3, 3)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.touches(a, b) is False

    def test_distant_objects_do_not_touch(self):
        a = _object([(1, 1)], self.GRID)
        b = _object([(6, 6)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.touches(a, b) is False

    def test_touching_is_independent_of_how_far_apart_the_centres_are(self):
        """Why this isn't covered by the distance features: two long bars
        can run against each other for their whole length and still have
        centres far apart."""
        top = _object([(1, j) for j in range(1, 7)], self.GRID)
        bottom = _object([(2, j) for j in range(1, 7)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.touches(top, bottom) is True


class TestSizeRatio:
    GRID = (8, 8)

    def test_equal_objects_give_one(self):
        a = _object([(1, 1), (1, 2)], self.GRID)
        b = _object([(5, 5), (5, 6)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.size_ratio(a, b) == 1.0

    def test_a_whole_multiple_is_reported_as_such(self):
        """What `same_size` collapsed into "not equal": 28.3% of real pairs
        sit on a whole 2x-5x multiple."""
        small = _object([(1, 1)], self.GRID)
        big = _object([(5, 1), (5, 2), (5, 3)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.size_ratio(small, big) == 3.0

    def test_the_ratio_does_not_depend_on_argument_order(self):
        small = _object([(1, 1)], self.GRID)
        big = _object([(5, 1), (5, 2), (5, 3)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.size_ratio(small, big) == RelationAnalyzer.size_ratio(big, small)


class TestSchemaIsTheSourceOfTruth:
    def test_every_new_feature_has_a_slot(self):
        for name in ("in_contour", "touches", "size_ratio"):
            assert name in RELATION_FEATURE_NAMES

    def test_the_embedding_is_laid_out_by_name_not_by_position(self):
        """Regression guard for the pattern this replaced: the vector used
        to be filled by a running index that had to match the schema by
        hand, so inserting a feature mid-schema shifted every later field
        while keeping the vector's width - nothing failed, the fields just
        stopped meaning what they said."""
        import inspect

        from symbolic.summaries import GridSummary

        source = inspect.getsource(GridSummary._create_embedding)

        assert "RELATION_FEATURE_NAMES" in source, "layout must come from the schema"
        assert "idx += 1" not in source, "positional filling is what broke before"


class TestTheRelationsReachTheAnalysis:
    """Testing the predicate alone leaves the wiring uncovered - a relation
    that never reaches the triples or the tallies is invisible to every
    consumer no matter how correct the predicate is."""

    @staticmethod
    def _stats(grid):
        from symbolic.summaries import GridSummary

        return GridSummary(grid=grid, shape=grid.shape, font_color=0,
                            levels=[2]).repr_levels[2].relation_statistics

    def test_touching_objects_are_tallied(self):
        grid = np.zeros((6, 6), dtype=int)
        grid[2, 2] = 1
        grid[2, 3] = 2  # edge-adjacent, different colour so they stay separate objects

        assert self._stats(grid).touches == 1

    def test_separated_objects_are_not_tallied_as_touching(self):
        grid = np.zeros((6, 6), dtype=int)
        grid[1, 1] = 1
        grid[4, 4] = 2

        assert self._stats(grid).touches == 0

    def test_a_contained_object_is_tallied(self):
        grid = np.zeros((9, 9), dtype=int)
        for i, j in _ring(7):
            grid[i + 1, j + 1] = 3
        grid[4, 4] = 2

        assert self._stats(grid).in_contour == 1

    def test_a_bounding_box_overlap_is_not_tallied_as_containment(self):
        """The measured half of old in_contour firings that were wrong."""
        grid = np.zeros((9, 9), dtype=int)
        for cell in [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]:
            grid[cell] = 3
        grid[2, 3] = 2  # inside the L's bounding box, not inside the L

        assert self._stats(grid).in_contour == 0


class TestEdgeAlignment:
    """x_alignment/y_alignment demand that both edges of an axis coincide.
    Measured on real tasks that holds for 7.63% of pairs, while another
    14.81% share a single edge and were reported as nothing - the larger
    half of the signal."""

    GRID = (10, 10)

    def test_a_shared_top_edge_is_reported(self):
        """Neither object could ever satisfy strict alignment - they have
        different heights - yet they hang from the same line."""
        short = _object([(2, 1)], self.GRID)
        tall = _object([(2, 5), (3, 5), (4, 5)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.x_alignment(short, tall) is False
        assert "aligned_top" in RelationAnalyzer.edge_alignments(short, tall)

    def test_a_shared_bottom_edge_is_distinguished_from_a_top_one(self):
        """Objects standing on a line and objects hanging from one are
        arranged differently, so they can't collapse into one flag."""
        short = _object([(4, 1)], self.GRID)
        tall = _object([(2, 5), (3, 5), (4, 5)], self.GRID, color=2, label="complex_1")

        alignments = RelationAnalyzer.edge_alignments(short, tall)

        assert "aligned_bottom" in alignments
        assert "aligned_top" not in alignments

    def test_shared_left_and_right_edges_are_reported(self):
        narrow = _object([(1, 3)], self.GRID)
        wide = _object([(6, 3), (6, 4), (6, 5)], self.GRID, color=2, label="complex_1")

        assert "aligned_left" in RelationAnalyzer.edge_alignments(narrow, wide)
        assert "aligned_right" not in RelationAnalyzer.edge_alignments(narrow, wide)

    def test_strictly_aligned_objects_share_both_edges(self):
        """These fire independently of the strict relation. Suppressing
        them would make aligned_top mean "shares a top edge but not a
        bottom one", which is a stranger thing to reason about."""
        a = _object([(2, 1), (3, 1)], self.GRID)
        b = _object([(2, 6), (3, 6)], self.GRID, color=2, label="complex_1")

        alignments = RelationAnalyzer.edge_alignments(a, b)

        assert RelationAnalyzer.x_alignment(a, b) is True
        assert "aligned_top" in alignments and "aligned_bottom" in alignments

    def test_objects_sharing_no_edge_report_nothing(self):
        a = _object([(1, 1)], self.GRID)
        b = _object([(5, 6), (6, 6)], self.GRID, color=2, label="complex_1")

        assert RelationAnalyzer.edge_alignments(a, b) == ()

    def test_centre_alignment_is_left_to_in_line(self):
        """Two objects whose centres share a row are already reported by
        in_line; a fifth alignment flag would say it twice."""
        from symbolic.summaries import RELATION_FEATURE_NAMES

        assert "in_line" in RELATION_FEATURE_NAMES
        assert not any(name.startswith("aligned_centre") for name in RELATION_FEATURE_NAMES)


class TestEdgeAlignmentReachesTheAnalysis:
    @staticmethod
    def _stats(grid):
        from symbolic.summaries import GridSummary

        return GridSummary(grid=grid, shape=grid.shape, font_color=0,
                            levels=[2]).repr_levels[2].relation_statistics

    def test_a_shared_top_edge_is_tallied(self):
        grid = np.zeros((8, 8), dtype=int)
        grid[2, 1] = 1                      # one cell
        grid[2, 5] = grid[3, 5] = grid[4, 5] = 2   # three cells, same top row

        stats = self._stats(grid)

        assert stats.aligned_top == 1
        assert stats.x_aligned_with == 0    # different heights, so never strictly aligned

    def test_unaligned_objects_are_not_tallied(self):
        grid = np.zeros((8, 8), dtype=int)
        grid[1, 1] = 1
        grid[5, 6] = 2

        stats = self._stats(grid)

        assert stats.aligned_top == 0
        assert stats.aligned_bottom == 0
        assert stats.aligned_left == 0
        assert stats.aligned_right == 0
