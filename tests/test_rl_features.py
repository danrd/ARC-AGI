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
    import torch

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
