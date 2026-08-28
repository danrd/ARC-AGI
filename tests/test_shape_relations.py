"""Tests for how shape congruence is wired into real GridObject relations:
calculate_shape_similarity and RelationAnalyzer.rotation_symmetry.

Both used to detect rotation/reflection through separate, narrower ad-hoc
comparisons - calculate_shape_similarity only ever cropped at two fixed
alignments without trying any rotation, and rotation_symmetry recomputed
np.rot90/flipud/fliplr for every pair regardless of whether the objects
shared anything at all. Both now gate on the objects' cached congruence_key
first. What's pinned here is behavioural parity plus the two things that
actually changed: shape_similarity now recognises a rotated duplicate, and
identity still earns no rotation/symmetry flag (that's same_shape's job,
untouched by this change).
"""
from __future__ import annotations

import numpy as np

from symbolic.objects_analysis import GridObject
from symbolic.summaries import RelationAnalyzer, calculate_shape_similarity


def _object(coords, label="obj", grid_shape=(10, 10), color=1):
    grid = np.zeros(grid_shape, dtype=int)
    for i, j in coords:
        grid[i, j] = color
    # shape="complex" matches how real objects are actually constructed
    # (retrieve_connected_components_homo etc.) and triggers classify_shape,
    # which expects label to look like "prefix_id" - real construction
    # always supplies that shape ("complex_0", "complex_1", ...).
    return GridObject("complex", coords, [color], f"{label}_0", grid_shape, 0, grid)


L_COORDS = [(0, 0), (1, 0), (2, 0), (2, 1)]
T_COORDS = [(0, 0), (0, 1), (0, 2), (1, 1)]


def _rotate_90(coords):
    """Coordinates of the same shape turned 90 degrees, placed back at the
    origin - independent of GridObject's own machinery, so the test doesn't
    validate the code against itself."""
    max_i = max(i for i, _ in coords)
    return [(j, max_i - i) for i, j in coords]


class TestShapeSimilarityRecognisesCongruence:
    @staticmethod
    def test_identical_objects_score_1():
        assert calculate_shape_similarity(_object(L_COORDS), _object(L_COORDS)) == 1.0

    @staticmethod
    def test_rotated_duplicate_scores_1():
        """The regression this exists to fix: the old crop-based overlap
        never tried a rotation, so a shape and its 90-degree turn scored
        near 0 despite being the same object rotated in place."""
        rotated = _rotate_90(L_COORDS)
        assert calculate_shape_similarity(_object(L_COORDS), _object(rotated)) == 1.0

    @staticmethod
    def test_unrelated_shapes_do_not_score_1():
        assert calculate_shape_similarity(_object(L_COORDS), _object(T_COORDS)) != 1.0

    @staticmethod
    def test_missing_congruence_key_falls_back_without_crashing():
        """Objects built for 'inner_hole'/'outer_hole' shapes never get a
        congruence_key (GridObject skips that construction branch for them)
        - the gate has to degrade to the old overlap-ratio path, not raise."""
        obj1, obj2 = _object(L_COORDS), _object(L_COORDS)
        del obj1.congruence_key
        del obj2.congruence_key

        result = calculate_shape_similarity(obj1, obj2)

        assert isinstance(result, float)


class TestRotationSymmetryMatchesRealPairs:
    @staticmethod
    def test_rotated_pair_is_flagged_rotation():
        rotated = _rotate_90(L_COORDS)
        obj1, obj2 = _object(L_COORDS, "a"), _object(rotated, "b")

        assert "rotation" in RelationAnalyzer.rotation_symmetry(obj1, obj2)

    @staticmethod
    def test_identical_pair_is_not_flagged_as_a_symmetry_relation():
        """Identity carries no rotation/reflection flag: two identical
        objects are related by same_shape, and saying it a second time here
        would double-count the one fact across two relations."""
        obj1, obj2 = _object(L_COORDS, "a"), _object(list(L_COORDS), "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []

    @staticmethod
    def test_unrelated_shapes_get_no_flags():
        obj1, obj2 = _object(L_COORDS, "a"), _object(T_COORDS, "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []

    @staticmethod
    def test_single_cells_are_excluded_regardless_of_congruence():
        """Two single cells are trivially congruent (both 1x1) but carry no
        rotation relation - preserves the original len(coords) > 1 guard,
        which existed to keep this relation meaningful only for objects with
        actual shape."""
        obj1, obj2 = _object([(0, 0)], "a"), _object([(5, 5)], "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []

    @staticmethod
    def test_a_reflection_symmetric_object_can_earn_two_flags_at_once():
        """T (its own mirror image left-right) rotated 180 degrees is
        simultaneously related to the original by that rotation AND by a
        top-bottom flip - both legitimately hold at once, and the old
        per-flag checks (run independently) would set both. Using
        relating_transform here instead of matching_transforms would report
        only one, which is exactly the behaviour change this test guards
        against."""
        t_shape = [(0, 0), (0, 1), (0, 2), (1, 1)]
        turned = _rotate_90(_rotate_90(t_shape))  # 180 degrees
        obj1, obj2 = _object(t_shape, "a"), _object(turned, "b")

        flags = RelationAnalyzer.rotation_symmetry(obj1, obj2)

        assert "rotation" in flags
        assert "horizontal_symmetry" in flags

    @staticmethod
    def test_missing_congruence_key_does_not_crash():
        obj1, obj2 = _object(L_COORDS, "a"), _object(_rotate_90(L_COORDS), "b")
        del obj1.congruence_key
        del obj2.congruence_key

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []


class TestIdenticalSelfSymmetricObjectsEarnNoFlag:
    """The intent "identity carries no rotation/reflection flag" was only
    ever enforced for shapes with no symmetry of their own, where the
    identity is the sole matching transform. For a mask that is its own
    mirror the flip lands back on the original, so the flip was reported
    too - and since most ARC objects are rectangles, bars and squares, that
    was 96.5% of horizontal_symmetry's firings on real level-1 pairs, 97.0%
    of vertical_symmetry's and 91.2% of rotation's. Three features almost
    entirely restating same_shape, which is why the first two correlated at
    +0.979 across 46422 pairs.
    """

    RECTANGLE = [(0, 0), (0, 1), (1, 0), (1, 1)]
    BAR = [(0, 0), (0, 1), (0, 2)]

    def test_two_identical_squares_are_not_called_symmetric(self):
        obj1 = _object(self.RECTANGLE, "a")
        obj2 = _object([(i + 5, j + 5) for i, j in self.RECTANGLE], "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []

    def test_two_identical_bars_are_not_called_symmetric(self):
        obj1 = _object(self.BAR, "a")
        obj2 = _object([(i + 4, j + 4) for i, j in self.BAR], "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == []

    def test_a_genuine_mirror_still_earns_exactly_the_right_flag(self):
        """What the guard must not cost: an asymmetric shape and its
        left-right mirror are related by a flip and by nothing else, and
        that is the case the relation exists for."""
        mirrored = [(i, 2 - j) for i, j in L_COORDS]
        obj1, obj2 = _object(L_COORDS, "a"), _object(mirrored, "b")

        flags = RelationAnalyzer.rotation_symmetry(obj1, obj2)

        assert flags == ["vertical_symmetry"]

    def test_a_genuine_top_bottom_mirror_is_told_apart_from_a_left_right_one(self):
        """The two flags are distinguishable, whatever the correlation on
        real data says - flipud and fliplr are separate transforms and the
        0.23% of real pairs where exactly one holds are asymmetric shapes
        like this one."""
        flipped = [(2 - i, j) for i, j in L_COORDS]
        obj1, obj2 = _object(L_COORDS, "a"), _object(flipped, "b")

        assert RelationAnalyzer.rotation_symmetry(obj1, obj2) == ["horizontal_symmetry"]


class TestTouchesCannotHoldAtLevelOne:
    """Not a defect in `touches` - measured over 5279 real level-1 pairs it
    is 0.00%, and the reason is structural rather than statistical: level 1
    objects are colour-agnostic 8-connected components, so two objects that
    touch would have been extracted as one. Pinned because the RL
    observation runs at repr_level=1 and is therefore carrying a
    constant-zero channel, which is a fact about the schema worth failing a
    test if it ever silently changes.
    """

    @staticmethod
    def _objects(grid, level):
        from symbolic.summaries import GridSummary

        return GridSummary(grid=grid, shape=grid.shape, font_color=0,
                            levels=[level]).repr_levels[level].objects

    @staticmethod
    def _touching_pairs(objects):
        return sum(1 for i, a in enumerate(objects) for b in objects[i + 1:]
                   if RelationAnalyzer.touches(a, b))

    ADJACENT = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    def test_adjacent_colours_are_one_object_at_level_one(self):
        objects = self._objects(self.ADJACENT, level=1)

        assert len(objects) == 1
        assert self._touching_pairs(objects) == 0

    def test_the_same_grid_has_two_touching_objects_at_level_two(self):
        objects = self._objects(self.ADJACENT, level=2)

        assert len(objects) == 2
        assert self._touching_pairs(objects) == 1
