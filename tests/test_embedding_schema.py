"""Tests for the object/relation embedding schemas.

The vector is a contract between symbolic/ and rl/, and it used to be an
implicit one: composition was implied by a hand-written flattening sequence in
create_embedding, while consumers kept their own copies of the resulting
sizes and offsets. Copies drift - rl.features carried `object_dim=25` beside a
comment saying 32, and sliced `[19:32]` out of a 25-wide vector, which only
worked because Python clamps a slice past the end.

So what is pinned here is agreement: the declared schema, the vector actually
produced, and the indices handed to consumers all have to describe the same
thing, and any future edit that breaks that has to fail here rather than
quietly feed a model the wrong columns.
"""
from __future__ import annotations

import numpy as np
import pytest

from symbolic.objects_analysis import (
    COLOR,
    EXTERNAL_DIM,
    EXTERNAL_GROUPS,
    INTERNAL_DIM,
    INTERNAL_GROUPS,
    OBJECT_DIM,
    OBJECT_SCHEMA,
    POSITION,
    SIZE,
    TOPOLOGY,
    GridObject,
    group_indices,
)
from symbolic.summaries import (
    RELATION_DIM,
    RELATION_FEATURE_NAMES,
    RELATION_SCHEMA,
    SHAPE_REL,
    SIMILARITY_REL,
    SPATIAL_REL,
    relation_group_indices,
)


def _object():
    """A small L-shaped object on a 5x5 grid - enough for every branch of
    create_embedding to produce a real value."""
    grid = np.zeros((5, 5), dtype=int)
    coords = [(1, 1), (2, 1), (3, 1), (3, 2)]
    for i, j in coords:
        grid[i, j] = 1
    return GridObject('test', coords, [1], 'obj_1', grid.shape, 0, grid)


# ---------------------------------------------------------------------------
# object schema
# ---------------------------------------------------------------------------

class TestObjectSchema:
    @staticmethod
    def test_declared_width_matches_the_vector_actually_produced():
        assert len(_object().create_embedding()) == OBJECT_DIM

    @staticmethod
    def test_every_declared_field_is_one_the_builder_actually_fills():
        obj = _object()
        obj.create_embedding()

        assert {name for name, _g, _a in OBJECT_SCHEMA} <= set(obj.embedding_dict)

    @staticmethod
    def test_multi_slot_field_declares_its_true_width():
        """color_shares is the only field wider than one slot; a mismatch
        between its real length and the schema must raise rather than shift
        every field after it."""
        obj = _object()
        obj.create_embedding()

        assert len(obj.embedding_dict["color_shares"]) == 10

    @staticmethod
    def test_every_field_lands_at_the_position_the_schema_declares():
        """The contract in one assertion: reading slot i of the vector gives
        the field the schema says lives at slot i. Everything else here is
        about widths and groups; this is about the values themselves."""
        obj = _object()
        vector = obj.create_embedding()

        position = 0
        for name, _group, arity in OBJECT_SCHEMA:
            actual = list(vector[position:position + arity])
            declared = list(obj.embedding_dict[name]) if arity > 1 else [obj.embedding_dict[name]]
            assert actual == declared, f"{name} is not at slot {position}"
            position += arity
        assert position == OBJECT_DIM

    @staticmethod
    def test_groups_partition_the_vector_exactly():
        every_group = group_indices(COLOR, SIZE, TOPOLOGY, POSITION)

        assert every_group == tuple(range(OBJECT_DIM))

    @staticmethod
    def test_internal_and_external_split_the_vector_without_overlap():
        internal = set(group_indices(*INTERNAL_GROUPS))
        external = set(group_indices(*EXTERNAL_GROUPS))

        assert not internal & external
        assert internal | external == set(range(OBJECT_DIM))
        assert len(internal) == INTERNAL_DIM
        assert len(external) == EXTERNAL_DIM

    @staticmethod
    def test_position_is_the_only_thing_outside_the_intrinsic_block():
        """The point of the split: what stays the same if the object were
        moved elsewhere on the grid, versus where it happens to sit."""
        assert set(group_indices(*EXTERNAL_GROUPS)) == set(group_indices(POSITION))

    @staticmethod
    def test_indices_come_back_in_vector_order():
        indices = group_indices(TOPOLOGY, COLOR)

        assert list(indices) == sorted(indices)

    @staticmethod
    def test_unknown_group_is_rejected_rather_than_silently_empty():
        with pytest.raises(ValueError):
            group_indices("nonexistent")

    @staticmethod
    def test_moving_an_object_changes_only_the_external_block():
        """Translation invariance is the property the split exists to make
        available, so it is worth checking rather than assuming."""
        grid_a = np.zeros((7, 7), dtype=int)
        coords_a = [(0, 0), (1, 0), (1, 1)]
        for i, j in coords_a:
            grid_a[i, j] = 1
        first = np.array(GridObject('t', coords_a, [1], 'a', grid_a.shape, 0, grid_a).create_embedding())

        grid_b = np.zeros((7, 7), dtype=int)
        coords_b = [(4, 3), (5, 3), (5, 4)]
        for i, j in coords_b:
            grid_b[i, j] = 1
        second = np.array(GridObject('t', coords_b, [1], 'b', grid_b.shape, 0, grid_b).create_embedding())

        internal = list(group_indices(*INTERNAL_GROUPS))
        external = list(group_indices(*EXTERNAL_GROUPS))

        assert np.allclose(first[internal], second[internal])
        assert not np.allclose(first[external], second[external])


# ---------------------------------------------------------------------------
# relation schema
# ---------------------------------------------------------------------------

class TestRelationSchema:
    @staticmethod
    def test_feature_names_are_derived_from_the_schema_not_restated():
        assert RELATION_FEATURE_NAMES == tuple(name for name, _g in RELATION_SCHEMA)
        assert RELATION_DIM == len(RELATION_FEATURE_NAMES)

    @staticmethod
    def test_groups_partition_the_relation_vector_exactly():
        every_group = relation_group_indices(SIMILARITY_REL, SHAPE_REL, SPATIAL_REL)

        assert every_group == tuple(range(RELATION_DIM))

    @staticmethod
    def test_groups_do_not_overlap():
        groups = [set(relation_group_indices(g)) for g in (SIMILARITY_REL, SHAPE_REL, SPATIAL_REL)]

        assert sum(len(g) for g in groups) == RELATION_DIM
        assert set.union(*groups) == set(range(RELATION_DIM))

    @staticmethod
    def test_unknown_group_is_rejected():
        with pytest.raises(ValueError):
            relation_group_indices("nonexistent")
