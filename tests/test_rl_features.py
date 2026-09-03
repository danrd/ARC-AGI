"""Tests for rl/features.py's ARCCombinedExtractor - specifically its
per-key dispatch in __init__, which used to have two problems: raising a
plain string instead of an exception for an unrecognized key, and treating
'action_space' (which ARCGridWorld always includes in every observation -
see rl/arc_env.py's set_subtask) as unrecognized, when it's simply not a
feature the extractor is meant to embed (forward() never reads it, same
as ARCGNNExtractor/ARCSeparateExtractor).
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from rl.features import ARCCombinedExtractor
from symbolic.objects_analysis import OBJECT_DIM


def test_action_space_key_is_skipped_not_raised():
    """Regression test: __init__ iterates every observation_space key and
    used to hit its `else: raise(f'Unknown feature: {key}')` branch for
    'action_space' - itself broken (raises a str, not an exception:
    `TypeError: exceptions must derive from BaseException`), on top of
    'action_space' not actually being unrecognized-and-fatal at all."""
    observation_space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
        "action_space": spaces.Box(low=0, high=900, shape=(1, 3), dtype=np.int64),
    })

    extractor = ARCCombinedExtractor(observation_space)

    assert "action_space" not in extractor.extractors
    assert "grid" in extractor.extractors


def test_unknown_feature_key_raises_a_real_exception():
    observation_space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
        "mystery_key": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
    })

    with pytest.raises(ValueError, match="mystery_key"):
        ARCCombinedExtractor(observation_space)


# ---------------------------------------------------------------------------
# embedding schema is honoured, not restated
# ---------------------------------------------------------------------------

def test_object_feature_groups_cover_the_whole_object_vector():
    """Regression test: the group heads used to slice fixed ranges - the last
    one `x[:, :, 19:32]` against a 25-wide vector. That only appeared to work
    because Python clamps an over-long slice, so the head silently received 6
    columns instead of the 13 it asked for, and would have started reading
    different fields the moment the vector's width changed."""

    from rl.features import ObjectProcessor
    from symbolic.objects_analysis import OBJECT_DIM

    extractor = ObjectProcessor()
    covered = (
        extractor.color_index.tolist()
        + extractor.spatial_index.tolist()
        + extractor.shape_index.tolist()
    )

    assert sorted(covered) == list(range(OBJECT_DIM))
    assert extractor.color_dim + extractor.spatial_dim + extractor.shape_dim == OBJECT_DIM


def test_object_feature_extractor_accepts_a_schema_width_vector():
    import torch

    from rl.features import ObjectProcessor
    from symbolic.objects_analysis import OBJECT_DIM

    extractor = ObjectProcessor()

    out = extractor(torch.zeros(2, 3, OBJECT_DIM))

    assert out.shape[0] == 2
    assert out.shape[1] == 3


def test_relation_feature_groups_cover_the_whole_relation_vector():
    from rl.features import RelationProcessor
    from symbolic.summaries import RELATION_DIM

    processor = RelationProcessor()
    covered = (
        processor.similarity_index.tolist()
        + processor.shape_rel_index.tolist()
        + processor.spatial_rel_index.tolist()
    )

    assert sorted(covered) == list(range(RELATION_DIM))


def test_each_relation_head_is_built_at_the_width_of_the_group_it_names():
    """A head's name, the width it is built at, and the features forward()
    passes it are three separate lines of code that have to agree. Two of
    them were crossed - the head named for spatial relations was built at the
    shape group's width and fed the shape group - which the totals hid,
    because they were crossed consistently. Nothing here checks the totals;
    it checks that each name means what it says.
    """
    from rl.features import RelationProcessor

    processor = RelationProcessor()
    expected = {
        "similarity_processor": processor.similarity_dim,
        "shape_processor": processor.shape_rel_dim,
        "spatial_processor": processor.spatial_rel_dim,
    }

    for name, width in expected.items():
        first_linear = getattr(processor, name)[0]
        assert first_linear.in_features == width, (
            f"{name} is built at width {first_linear.in_features}, "
            f"but the group it names is {width} wide"
        )


# -- a grid with no objects at all -------------------------------------------
#
# All-background grids exist in ARC, and an observation of one has every
# object slot padded. Attention over a fully padded row masks every key,
# and a softmax over nothing but -inf is NaN - which spreads through the
# shared layers to the whole batch and surfaces much later as PPO's
# "Expected parameter logits ... found invalid values", naming neither the
# grid nor the layer. ARCSeparateExtractor guards the same thing per batch
# (`if obj_mask.any()`), which is the wrong grain: one empty row among
# several still goes through masked.

def _processor():
    import torch
    from rl.features import ObjectSetProcessor

    torch.manual_seed(0)
    return ObjectSetProcessor(embedding_dim=OBJECT_DIM).eval()


def _slots(filled=()):
    """Object embeddings for one observation: `filled` slots occupied."""
    import torch

    slots = torch.zeros((1, 4, OBJECT_DIM))
    for slot in filled:
        slots[0, slot] = 0.5
    return slots


def test_an_object_less_observation_does_not_produce_nan():
    import torch

    embeddings = _slots()

    with torch.no_grad():
        output = _processor()(embeddings, mask=(embeddings.sum(dim=-1) != 0))

    assert not torch.isnan(output).any()


def test_an_empty_row_beside_a_filled_one_does_not_produce_nan():
    """The per-row case a per-batch guard misses, and the one a mixed
    rollout actually produces."""
    import torch

    embeddings = torch.cat([_slots((0,)), _slots()])  # one row filled, one empty

    with torch.no_grad():
        output = _processor()(embeddings, mask=(embeddings.sum(dim=-1) != 0))

    assert not torch.isnan(output).any()


def test_an_object_less_row_contributes_nothing_rather_than_noise():
    """Its features are zeros: the row attends over a padding slot only so
    that softmax has something to normalise, and the masked mean multiplies
    the result away again."""
    import torch

    empty = _slots()
    together = torch.cat([_slots(), _slots((0,))])

    with torch.no_grad():
        alone = _processor()(empty, mask=(empty.sum(dim=-1) != 0))
        beside = _processor()(together, mask=(together.sum(dim=-1) != 0))

    assert torch.allclose(alone[0], beside[0], atol=1e-6)
