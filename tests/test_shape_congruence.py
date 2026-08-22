"""Tests for symbolic/objects_analysis.py's shape-congruence primitives:
congruence_key, relating_transform, matching_transforms.

Whether two objects "have the same shape" is a question about pairs, not a
category one object gets assigned - these exist so that question can be
answered exactly (no thresholds) and cheaply (compare two cached keys rather
than compare masks pairwise). Correctness is what's tested here; the cost
claim and the wiring into calculate_shape_similarity / rotation_symmetry are
tested in test_symbolic.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from symbolic.objects_analysis import congruence_key, matching_transforms, relating_transform


def _mask(rows):
    """rows: list of '01' strings, e.g. ['110', '010']."""
    return np.array([[int(c) for c in row] for row in rows])


L = _mask(["10", "10", "11"])       # 3-cell L, no symmetry of its own
T = _mask(["111", "010"])           # T-tromino
SQUARE = _mask(["111", "111", "111"])
RECT_2X4 = _mask(["1111", "1111"])
CROSS = _mask(["010", "111", "010"])  # plus sign - symmetric under everything


class TestCongruenceKey:
    @staticmethod
    def test_identical_masks_share_a_key():
        assert congruence_key(L) == congruence_key(L.copy())

    @staticmethod
    def test_rotated_mask_shares_the_key():
        assert congruence_key(L) == congruence_key(np.rot90(L, 1))
        assert congruence_key(L) == congruence_key(np.rot90(L, 2))
        assert congruence_key(L) == congruence_key(np.rot90(L, 3))

    @staticmethod
    def test_reflected_mask_shares_the_key():
        assert congruence_key(L) == congruence_key(np.flipud(L))
        assert congruence_key(L) == congruence_key(np.fliplr(L))

    @staticmethod
    def test_non_congruent_masks_do_not_share_a_key():
        """L and T are both trominoes (3 cells, 2x3 bbox) - close enough that
        a size- or bbox-based shortcut would conflate them; the key must not."""
        assert congruence_key(L) != congruence_key(T)

    @staticmethod
    def test_a_shape_with_full_symmetry_still_produces_a_key():
        """The plus sign is invariant under all 8 transforms - the min() over
        8 identical stamps must not raise or behave differently from the
        asymmetric case."""
        assert congruence_key(CROSS) == congruence_key(np.rot90(CROSS, 1))

    @staticmethod
    def test_different_sized_masks_never_collide():
        """A rotation can swap height and width - the key has to keep shape
        and content bound together, or two differently-proportioned masks
        with the same flattened bytes could be mistaken for congruent."""
        wide = _mask(["11"])    # 1x2, two cells
        tall = _mask(["1", "1"])  # 2x1, two cells - literally L's rotation
        assert congruence_key(wide) == congruence_key(tall)  # these ARE congruent (rotation)

        different_content = _mask(["10"])  # 1x2, one cell - different content, would-be false positive if shape were ignored
        assert congruence_key(wide) != congruence_key(different_content)


class TestRelatingTransform:
    @staticmethod
    def test_identity_is_preferred_over_a_coinciding_symmetry():
        """The cross is invariant under every transform - relating_transform
        must still report 'identity' for two identical copies, not whichever
        symmetric rotation happens to sort first."""
        assert relating_transform(CROSS, CROSS.copy()) == "identity"

    @staticmethod
    def test_reports_the_specific_rotation():
        assert relating_transform(L, np.rot90(L, 1)) == "rot90"
        assert relating_transform(L, np.rot90(L, 2)) == "rot180"
        assert relating_transform(L, np.rot90(L, 3)) == "rot270"

    @staticmethod
    def test_reports_none_for_non_congruent_masks():
        assert relating_transform(L, T) is None

    @staticmethod
    def test_reports_a_reflection():
        assert relating_transform(L, np.flipud(L)) == "horizontal_flip"
        assert relating_transform(L, np.fliplr(L)) == "vertical_flip"


class TestMatchingTransforms:
    @staticmethod
    def test_asymmetric_shape_matches_exactly_one_transform_per_target():
        """L has no self-symmetry, so relating it to one specific rotated
        copy should name only that rotation, not also claim it's related by
        some other transform that happens to coincide."""
        assert matching_transforms(L, np.rot90(L, 1)) == {"rot90"}

    @staticmethod
    def test_symmetric_shape_can_match_multiple_transforms_at_once():
        """Regression case for the switch from relating_transform (picks one
        winner) to matching_transforms (reports all): the old per-flag
        detection checked rotation and each reflection independently and
        could set more than one flag for the same pair - a highly symmetric
        shape has to keep doing that, not silently drop to a single label."""
        matches = matching_transforms(RECT_2X4, RECT_2X4.copy())
        assert "identity" in matches
        assert "rot180" in matches          # a solid rectangle is its own 180-rotation
        assert "horizontal_flip" in matches
        assert "vertical_flip" in matches

    @staticmethod
    def test_no_matches_for_incongruent_masks():
        assert matching_transforms(L, T) == frozenset()
