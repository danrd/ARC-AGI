"""Tests for rl/features.py's graph_inputs.

The env packs relations as one row per object - that object's vector
against each other object, laid end to end with its own slot skipped. The
extractor used to read that matrix as one row per pair, slice it by pair
count, and hand rows of `(max_objects - 1) * RELATION_DIM` values to an
edge encoder expecting RELATION_DIM; it also built edge indices out of
padded object slots, which do not index the node tensor once the padding
is dropped. None of that raised: the paths it broke on were reached only
when a grid had more than one object.

So these pin the decoding itself - which block belongs to which pair, what
indexes what, and what an absent relation looks like.
"""
from __future__ import annotations

import torch

from rl.features import graph_inputs
from symbolic.objects_analysis import OBJECT_DIM
from symbolic.summaries import RELATION_DIM

SLOTS = 4


def _objects(*filled):
    """Object embeddings with `filled` slots occupied, the rest padding."""
    objects = torch.zeros((SLOTS, OBJECT_DIM))
    for slot in filled:
        objects[slot] = 1.0
    return objects


def _relations():
    return torch.zeros((SLOTS, (SLOTS - 1) * RELATION_DIM))


def _set(relations, source, target, value):
    """Write one pair's vector the way the env packs it."""
    block = target - (1 if target > source else 0)
    relations[source, block * RELATION_DIM:(block + 1) * RELATION_DIM] = value


class TestWhichBlockBelongsToWhichPair:
    def test_the_vector_written_for_a_pair_is_the_one_read_back(self):
        relations = _relations()
        _set(relations, 0, 2, 5.0)

        _, edges, attributes = graph_inputs(_objects(0, 2), relations)

        assert edges == [[0, 1]]
        assert attributes.shape == (1, RELATION_DIM)
        assert float(attributes[0, 0]) == 5.0

    def test_a_rows_own_slot_is_skipped_when_counting_blocks(self):
        """Object 2's row holds 0, 1 and 3 - so 3 sits in the third block,
        not the fourth. Off by one here silently reads a neighbour's
        relation."""
        relations = _relations()
        _set(relations, 2, 3, 7.0)

        _, edges, attributes = graph_inputs(_objects(2, 3), relations)

        assert edges == [[0, 1]]
        assert float(attributes[0, 0]) == 7.0

    def test_both_directions_are_kept_when_both_are_recorded(self):
        """i against j and j against i are different rows and mean
        different things."""
        relations = _relations()
        _set(relations, 0, 1, 3.0)
        _set(relations, 1, 0, 4.0)

        _, edges, attributes = graph_inputs(_objects(0, 1), relations)

        assert sorted(edges) == [[0, 1], [1, 0]]
        assert sorted(float(row[0]) for row in attributes) == [3.0, 4.0]


class TestWhatTheEdgesIndex:
    def test_edges_index_the_nodes_that_are_there(self):
        """Not the padded slots: after padding is dropped, slot 3 is node 1,
        and an edge naming 3 would be out of range."""
        relations = _relations()
        _set(relations, 1, 3, 2.0)

        nodes, edges, _ = graph_inputs(_objects(1, 3), relations)

        assert nodes.shape == (2, OBJECT_DIM)
        assert all(0 <= index < nodes.shape[0] for edge in edges for index in edge)


class TestWhatIsLeftOut:
    def test_an_unrecorded_pair_is_not_an_edge(self):
        nodes, edges, attributes = graph_inputs(_objects(0, 1), _relations())

        assert nodes.shape == (2, OBJECT_DIM)
        assert edges == []
        assert attributes.shape == (0, RELATION_DIM)

    def test_a_single_object_has_no_edges_but_is_still_a_node(self):
        nodes, edges, attributes = graph_inputs(_objects(0), _relations())

        assert nodes.shape == (1, OBJECT_DIM)
        assert edges == [] and attributes.shape == (0, RELATION_DIM)

    def test_an_empty_observation_still_yields_one_node(self):
        """A graph with no nodes has nothing to pool over; one zero node
        keeps the batch shapes valid."""
        nodes, edges, attributes = graph_inputs(torch.zeros((SLOTS, OBJECT_DIM)),
                                                _relations())

        assert nodes.shape == (1, OBJECT_DIM)
        assert edges == [] and attributes.shape == (0, RELATION_DIM)


class TestTheWidthsComeFromTheSchema:
    def test_edge_features_are_relation_dim_wide(self):
        relations = _relations()
        _set(relations, 0, 1, 1.0)

        _, _, attributes = graph_inputs(_objects(0, 1), relations)

        assert attributes.shape[1] == RELATION_DIM

    def test_nodes_are_object_dim_wide(self):
        nodes, _, _ = graph_inputs(_objects(0, 1), _relations())

        assert nodes.shape[1] == OBJECT_DIM
