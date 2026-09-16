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
import torch
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


# ---------------------------------------------------------------------------
# an extractor owns the weights it trains
# ---------------------------------------------------------------------------
#
# data.configs.rl_configs builds its grid encoder once at import (lin_arch =
# lin()) and hands that one module to every agent through
# features_extractor_kwargs. The extractor used to adopt it as-is, so all
# the agents built in one process shared a single set of nn.Conv2d weights:
# an agent began where the previous one's training had left the encoder, and
# two agents alive at once had their optimisers writing to the same tensors.
# Every in-process comparison of two configurations was measuring the second
# one on top of the first.

def _grid_space():
    return spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
    })


def test_two_extractors_built_from_one_architecture_train_separately():
    import torch
    from rl.features import default_grid_arch

    shared = default_grid_arch()

    first = ARCCombinedExtractor(_grid_space(), extr_arch=shared)
    second = ARCCombinedExtractor(_grid_space(), extr_arch=shared)

    before = second.extractors["grid"][0].weight.detach().clone()
    with torch.no_grad():
        first.extractors["grid"][0].weight.add_(1.0)

    assert torch.equal(second.extractors["grid"][0].weight, before)


def test_an_architecture_can_be_given_as_something_that_builds_one():
    from rl.features import default_grid_arch, grid_arch_width

    extractor = ARCCombinedExtractor(_grid_space(), extr_arch=default_grid_arch)

    # Asked of the architecture rather than written out: the number moved
    # from 16 to 144 when the pooling went from 1x1 to 3x3, and a test that
    # restates it only records which day it was written.
    assert extractor.features_dim == grid_arch_width(default_grid_arch())


def test_the_default_encoder_keeps_some_notion_of_where():
    """The measured property, not the number: a grid encoder that ends in a
    global mean returns one value per channel whatever the grid holds, and
    it scored zero on the held-out pair in all six runs it was measured in
    where every spatially-pooled encoder scored above zero somewhere."""
    from rl.features import default_grid_arch
    from data.configs.rl_configs import lin

    for build in (default_grid_arch, lin):
        arch = build()
        with torch.no_grad():
            wide = arch(torch.zeros(1, 10, 20, 20)).shape[1]
        channels = [layer.out_channels for layer in arch
                    if isinstance(layer, torch.nn.Conv2d)][-1]
        assert wide > channels, (
            f"{build.__name__} returns {wide} features for {channels} "
            "channels, so it pools every position into one")


# ---------------------------------------------------------------------------
# the other grids in an observation
# ---------------------------------------------------------------------------
#
# ARCGridWorld puts the example's own input in the observation under
# 'input_pattern' when rl_config asks for input_pattern='separate', and the
# wanted output under 'target' when observation_space_elements names it.
# Both are declared in its observation space and filled in reset() and
# step(); the extractor rejected them as unknown, so either setting killed
# the run with `ValueError: Unknown feature` before its first step.

@pytest.mark.parametrize("key", ["input_pattern", "target"])
def test_the_other_grids_are_encoded_rather_than_rejected(key):
    import torch

    observation_space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
        key: spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
    })

    extractor = ARCCombinedExtractor(observation_space)
    features = extractor({
        "grid": torch.zeros((2, 9, 9), dtype=torch.int64),
        key: torch.zeros((2, 9, 9), dtype=torch.int64),
    })

    assert set(extractor.extractors) == {"grid", key}
    assert features.shape == (2, extractor.features_dim)
    # Both grids through the same architecture, so twice its width - asked
    # of the architecture rather than written out, because the width moved
    # from 16 to 144 when the default pooling went from 1x1 to 3x3.
    from rl.features import default_grid_arch, grid_arch_width
    assert extractor.features_dim == 2 * grid_arch_width(default_grid_arch())


def test_a_second_grid_is_read_rather_than_ignored():
    """Encoded through its own weights, so changing it changes the features
    - a branch that quietly returned zeros would also make the shapes fit."""
    import torch

    observation_space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
        "input_pattern": spaces.Box(low=0, high=10, shape=(9, 9), dtype=np.int64),
    })
    extractor = ARCCombinedExtractor(observation_space)
    grid = torch.zeros((1, 9, 9), dtype=torch.int64)
    other = torch.zeros((1, 9, 9), dtype=torch.int64)
    other[0, 4, 4] = 3

    with torch.no_grad():
        unchanged = extractor({"grid": grid, "input_pattern": grid})
        changed = extractor({"grid": grid, "input_pattern": other})

    assert not torch.allclose(unchanged, changed)


# ---------------------------------------------------------------------------
# the three extractors are buildable, and comparable
# ---------------------------------------------------------------------------
#
# rl.training.create_agent always passes features_extractor_kwargs={'extr_arch':
# ...}, and neither ARCGNNExtractor nor ARCSeparateExtractor took such an
# argument - so both raised TypeError in the constructor and no run through
# rl.training ever reached its first step with either. They also encoded the
# grid differently from ARCCombinedExtractor: a single channel of colour
# numbers, where colour 9 is nine times colour 1 to a convolution and the
# difference between two colours is a distance. Sweeping architectures that
# disagree about what a colour is would not compare what the names suggest.

def _relation_space(slots=4):
    from symbolic.summaries import RELATION_DIM

    return spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(6, 6), dtype=np.int64),
        "objects_emb": spaces.Box(low=0, high=1, shape=(slots, OBJECT_DIM),
                                  dtype=np.float32),
        "relations_emb": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(slots, (slots - 1) * RELATION_DIM), dtype=np.float32),
    })


def _extractor_classes():
    from rl.features import (ARCCombinedExtractor, ARCGNNExtractor,
                             ARCSeparateExtractor)

    return [ARCCombinedExtractor, ARCGNNExtractor, ARCSeparateExtractor]


@pytest.mark.parametrize("extractor_class", _extractor_classes())
def test_every_extractor_accepts_the_architecture_the_config_names(
        extractor_class):
    from rl.features import default_grid_arch

    extractor = extractor_class(_relation_space(), extr_arch=default_grid_arch())

    assert extractor.features_dim > 0


@pytest.mark.parametrize("extractor_class", _extractor_classes())
def test_every_extractor_encodes_a_colour_as_a_name_not_a_quantity(
        extractor_class):
    """Swapping two colours should not move features by an amount that
    depends on how far apart their numbers are: with one-hot planes the two
    swaps below are the same permutation of channels, and with a single
    channel of colour numbers they are not."""
    import torch
    from symbolic.summaries import RELATION_DIM

    torch.manual_seed(0)
    extractor = extractor_class(_relation_space()).eval()

    def features(first, second):
        grid = torch.zeros((1, 6, 6), dtype=torch.int64)
        grid[0, :3] = first
        grid[0, 3:] = second
        with torch.no_grad():
            return extractor({
                "grid": grid,
                "objects_emb": torch.zeros((1, 4, OBJECT_DIM)),
                "relations_emb": torch.zeros((1, 4, 3 * RELATION_DIM)),
            })

    near = features(1, 2)
    far = features(1, 9)

    assert not torch.allclose(near, far), "the grid is not read at all"
    swapped_near = features(2, 1)
    swapped_far = features(9, 1)
    assert (torch.allclose(near, swapped_near)
            == torch.allclose(far, swapped_far)), \
        "how much a colour swap moves the features depends on the numbers"


@pytest.mark.parametrize("extractor_class", _extractor_classes()[1:])
def test_an_observation_without_relations_says_which_setting_to_change(
        extractor_class):
    """rl_config ships observation_space_elements = ["objects_emb"], and
    both of these read relations - without this the failure is a KeyError
    from inside forward, naming a dict key rather than the setting."""
    space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(6, 6), dtype=np.int64),
        "objects_emb": spaces.Box(low=0, high=1, shape=(4, OBJECT_DIM),
                                  dtype=np.float32),
    })

    with pytest.raises(ValueError, match="observation_space_elements"):
        extractor_class(space)


def test_the_combined_extractor_still_runs_without_relations():
    """It is the one that does not need them, and the shipped config does
    not provide them."""
    space = spaces.Dict({
        "grid": spaces.Box(low=0, high=10, shape=(6, 6), dtype=np.int64),
        "objects_emb": spaces.Box(low=0, high=1, shape=(4, OBJECT_DIM),
                                  dtype=np.float32),
    })

    assert ARCCombinedExtractor(space).features_dim > 0


# -- the embeddings the pooling averages away ------------------------------

class TestWhatThePoolingThrowsAway:
    """An action names an object slot, and the logit for slot i is built by
    ARCCustomNetwork from a vector in which slot i's embedding has been
    averaged with every other one. So the only thing an object head can
    learn is a prior over slot numbers - which is why a whitelist of slot
    indices helped on the examples it came from and moved the held-out pair
    by zero on every task in the ablation. A head that scored each slot
    from its own embedding would need those embeddings to still exist."""

    def _processor(self, slots=4, width=OBJECT_DIM):
        from rl.features import OptimalObjectExtractor
        return OptimalObjectExtractor((slots, width), output_dim=64)

    def test_the_per_object_embeddings_survive_the_pooling(self):
        extractor = self._processor(slots=4)
        objects = torch.zeros(2, 4, OBJECT_DIM)
        objects[:, :3] = torch.rand(2, 3, OBJECT_DIM)

        pooled = extractor(objects.reshape(2, -1))

        assert pooled.shape == (2, 64)
        assert extractor.per_object.shape[:2] == (2, 4), (
            f"per-object embeddings are {tuple(extractor.per_object.shape)}, "
            "expected one row per slot")
        assert extractor.per_object_mask.shape == (2, 4)

    def test_the_mask_marks_the_slots_that_hold_an_object(self):
        extractor = self._processor(slots=5)
        objects = torch.zeros(1, 5, OBJECT_DIM)
        objects[0, :2] = torch.rand(2, OBJECT_DIM)

        extractor(objects.reshape(1, -1))

        assert extractor.per_object_mask[0].tolist() == [True, True,
                                                         False, False, False]

    def test_two_different_objects_get_two_different_embeddings(self):
        """The property a slot-scoring head rests on: if every slot came out
        the same, scoring them separately would buy nothing."""
        extractor = self._processor(slots=3)
        objects = torch.zeros(1, 3, OBJECT_DIM)
        objects[0, 0] = torch.rand(OBJECT_DIM)
        objects[0, 1] = torch.rand(OBJECT_DIM)

        extractor(objects.reshape(1, -1))
        rows = extractor.per_object[0]

        assert not torch.allclose(rows[0], rows[1], atol=1e-6)

    def test_the_embeddings_carry_a_gradient_back_to_the_encoder(self):
        """Read off an attribute rather than returned, so it is worth
        pinning that they are still part of the graph - a head training on
        a detached copy would learn from a fixed encoder."""
        extractor = self._processor(slots=3)
        objects = torch.zeros(1, 3, OBJECT_DIM)
        objects[0, :2] = torch.rand(2, OBJECT_DIM)

        extractor(objects.reshape(1, -1))
        extractor.per_object.sum().backward()

        grads = [p.grad for p in extractor.parameters() if p.grad is not None]
        assert grads, "nothing in the encoder received a gradient"

    def test_the_pooled_output_is_what_it_always_was(self):
        """Characterisation: exposing the embeddings must not move the
        vector every existing policy reads.

        In eval mode, because ObjectSetProcessor's attention carries
        dropout=0.1 - in train mode the same observation gives a different
        feature vector on every forward pass, which is what dropout is for
        and is worth knowing is there.
        """
        torch.manual_seed(0)
        extractor = self._processor(slots=4).eval()
        objects = torch.zeros(2, 4, OBJECT_DIM)
        objects[:, :3] = torch.rand(2, 3, OBJECT_DIM)
        flat = objects.reshape(2, -1)

        once = extractor(flat)
        twice = extractor(flat)

        assert torch.allclose(once, twice)
        assert once.shape == (2, 64)

    def test_the_encoder_is_stochastic_while_it_trains(self):
        """The reason the test above says eval(): dropout is on in train
        mode, so two passes over one observation differ. Recorded because a
        head added later will be compared against these features and a
        difference that is really dropout is easy to mistake for one that
        is not."""
        torch.manual_seed(0)
        extractor = self._processor(slots=4).train()
        objects = torch.zeros(2, 4, OBJECT_DIM)
        objects[:, :3] = torch.rand(2, 3, OBJECT_DIM)
        flat = objects.reshape(2, -1)

        assert not torch.allclose(extractor(flat), extractor(flat))


# -- a delta read as a map ---------------------------------------------------

class TestReadingADeltaAsAMap:
    """A mean over a delta is the fraction of cells that differ, which is
    max_int arrived at expensively, and it divides by the grid area: on a
    20x20 plane one differing cell reads 0.0004 through the colour encoder.
    DeltaReadout takes three readings instead, because how much, whether
    anything, and where are three questions."""

    def plane(self, cells, rows=20, cols=20):
        grid = torch.zeros(1, 1, rows, cols)
        for row, col in cells:
            grid[0, 0, row, col] = 1.0
        return grid

    def parts(self, readout, out):
        width = readout.channels
        return (out[:, :width], out[:, width:2 * width],
                out[:, 2 * width:3 * width], out[:, 3 * width:])

    def test_the_centre_of_mass_finds_the_corner_the_difference_is_in(self):
        from rl.features import DeltaReadout
        torch.manual_seed(0)
        readout = DeltaReadout().eval()

        with torch.no_grad():
            corners = {where: self.parts(readout, readout(self.plane([where])))
                       for where in [(1, 1), (18, 18), (1, 18)]}

        def centre(where):
            _mass, _peak, down, across = corners[where]
            return float(down.max()), float(across.max())

        top_left, bottom_right, top_right = (centre((1, 1)), centre((18, 18)),
                                             centre((1, 18)))
        assert top_left[0] < 0.4 and top_left[1] < 0.4, top_left
        assert bottom_right[0] > 0.6 and bottom_right[1] > 0.6, bottom_right
        # The row and the column are read separately, so a difference in the
        # top right has to come out low on one and high on the other.
        assert top_right[0] < 0.4 < top_right[1], top_right

    def test_an_empty_plane_has_no_peak_and_no_centre(self):
        """Nothing to locate reads as (0, 0), which is also a real corner -
        so the peak is what tells them apart, and it is zero here and not
        there."""
        from rl.features import DeltaReadout
        torch.manual_seed(0)
        readout = DeltaReadout().eval()

        with torch.no_grad():
            _mass, peak, down, across = self.parts(readout,
                                                   readout(self.plane([])))
            _m2, corner_peak, _d2, _a2 = self.parts(
                readout, readout(self.plane([(0, 0)])))

        assert float(peak.max()) == 0.0
        assert float(down.abs().max()) == 0.0 and float(across.abs().max()) == 0.0
        assert float(corner_peak.max()) > 0.0

    def test_the_mass_counts_and_the_peak_does_not_vanish(self):
        """The two readings the pooled encoder had to choose between: a mean
        scales with how many cells differ, a max stays large when only one
        does."""
        from rl.features import DeltaReadout
        torch.manual_seed(0)
        readout = DeltaReadout().eval()
        few = [(1, 1)]
        many = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3)]

        with torch.no_grad():
            mass_few, peak_few, *_ = self.parts(readout, readout(self.plane(few)))
            mass_many, peak_many, *_ = self.parts(readout, readout(self.plane(many)))

        assert float(mass_many.max()) > float(mass_few.max())
        assert float(peak_few.max()) > 10 * float(mass_few.max()), (
            "the peak is meant to survive a sparse plane that the mean divides "
            "away")

    def test_the_convolutions_carry_no_bias(self):
        """The property the centre of mass rests on: with a bias, ReLU fires
        on empty cells too, every channel gets a constant background over the
        whole plane, and the expected position of background-plus-a-speck is
        the middle of the grid whatever the delta holds. Measured that way
        before the fix, a cell at (1,1) and a cell at (18,18) both read a
        centre of (0.524, 0.475)."""
        from rl.features import DeltaReadout
        readout = DeltaReadout()

        for layer in readout.convs:
            if isinstance(layer, torch.nn.Conv2d):
                assert layer.bias is None, "a bias puts a floor under every cell"

    def test_the_readout_is_four_numbers_per_channel(self):
        from rl.features import DeltaReadout
        readout = DeltaReadout(channels=8)

        with torch.no_grad():
            out = readout(self.plane([(2, 3)], rows=7, cols=9))

        assert readout.out_dim == 32
        assert out.shape == (1, 32)

    def test_the_width_does_not_follow_the_grid_size(self):
        """Every other grid-shaped key has to produce a fixed width whatever
        the example's size - rl.features.unpadded_grid_features writes them
        all into one array."""
        from rl.features import DeltaReadout
        readout = DeltaReadout().eval()

        with torch.no_grad():
            small = readout(self.plane([(1, 1)], rows=4, cols=5))
            large = readout(self.plane([(1, 1)], rows=28, cols=30))

        assert small.shape == large.shape == (1, readout.out_dim)
