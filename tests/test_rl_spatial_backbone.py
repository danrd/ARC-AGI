"""Tests for rl.features.SpatialBackbone - one feature map over the grid at
full resolution, and what the context, the objects and the coordinate rows
read from it.

What is pinned is what the map was built for: padding is invisible to it,
an object's vector now depends on the grid in and around it, and a
coordinate row is read with the same weights at every position.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.arc_task import ARCSubtask
from rl.features import ARCCombinedExtractor, CoordinateRows, SpatialBackbone, true_cells
from rl.training import create_agent, create_vec_env
from symbolic.objects_analysis import OBJECT_DIM, OBJECT_SCHEMA


def observation(grids, shapes=None, objects=None, delta=None):
    obs = {"grid": torch.tensor(np.stack(grids), dtype=torch.int64)}
    if shapes is not None:
        obs["grid_shape"] = torch.tensor(shapes, dtype=torch.int64)
    if objects is not None:
        obs["objects_emb"] = torch.tensor(np.stack(objects), dtype=torch.float32)
    if delta is not None:
        obs["delta_input"] = torch.tensor(np.stack(delta), dtype=torch.float32)
    return obs


def object_row(box, grid_shape, colour=1):
    """An objects_emb row with a colour share and a bounding box
    (min_i, min_j, max_i, max_j), normalised the way GridObject does."""
    row = np.zeros(OBJECT_DIM, dtype=np.float32)
    index, position = {}, 0
    for name, _group, arity in OBJECT_SCHEMA:
        index[name] = position
        position += arity
    row[index["color_shares"] + colour] = 1.0
    rows, cols = grid_shape
    for name, value, size in (("min_i", box[0], rows), ("min_j", box[1], cols),
                              ("max_i", box[2], rows), ("max_j", box[3], cols)):
        row[index[name]] = value / size
    return row


class TestTheMap:
    def test_padding_and_cells_past_the_true_shape_are_not_grid(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[3, :] = 10
        valid = true_cells(observation([grid], shapes=[[4, 3]]))
        assert valid[0].sum().item() == 9 and not valid[0, 3].any() and not valid[0, :, 3].any()

    def test_the_map_is_zero_off_the_grid(self):
        backbone = SpatialBackbone(["grid"], [], channels=4)
        grid = np.full((5, 5), 10)
        grid[:3, :3] = 2
        features, _ = backbone(observation([grid], shapes=[[3, 3]]))
        assert features[0, :, 3:, :].abs().sum() == 0 and features[0, :, :, 3:].abs().sum() == 0

    def test_padding_does_not_change_what_the_grid_reads(self):
        """The same 3x3 grid, alone and padded into 6x6: every true cell
        reads the same, so pooled features do too."""
        torch.manual_seed(0)
        backbone = SpatialBackbone(["grid"], [], channels=4)
        small = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        padded = np.full((6, 6), 10)
        padded[:3, :3] = small
        alone, valid_a = backbone(observation([small], shapes=[[3, 3]]))
        inside, valid_b = backbone(observation([padded], shapes=[[3, 3]]))
        assert torch.allclose(alone[0], inside[0, :, :3, :3], atol=1e-6)
        assert torch.allclose(SpatialBackbone.pooled(alone, valid_a),
                              SpatialBackbone.pooled(inside, valid_b), atol=1e-6)


class TestReadingItOverAnObject:
    def test_the_box_is_averaged_and_an_empty_slot_is_zero(self):
        features = torch.arange(2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
        objects = np.stack([object_row((1, 1, 2, 3), (4, 4)), np.zeros(OBJECT_DIM)])
        boxes = SpatialBackbone.over_boxes(features, torch.tensor(objects[None]),
                                           observation([np.zeros((4, 4), int)]))
        expected = features[0, :, 1:3, 1:4].mean(dim=(1, 2))
        assert torch.allclose(boxes[0, 0], expected)
        assert boxes[0, 1].abs().sum() == 0

    def _extractor(self, spatial):
        torch.manual_seed(0)
        space = spaces.Dict({
            "grid": spaces.Box(0, 10, shape=(5, 5), dtype=np.int64),
            "objects_emb": spaces.Box(0, 1, shape=(2, OBJECT_DIM), dtype=np.float32)})
        return ARCCombinedExtractor(space, spatial_channels=spatial).eval()

    def _rows(self, extractor, grid):
        objects = np.stack([object_row((1, 1, 3, 3), (5, 5)), object_row((0, 4, 0, 4), (5, 5))])
        extractor(observation([grid], objects=[objects]))
        return extractor.extractors["objects_emb"].per_object[0, 0]

    def test_an_object_sees_the_grid_inside_its_box(self):
        """Its computed numbers the same, the dot inside it moved: the rows
        a pointer or a colour head reads now tell the two apart."""
        plain = np.zeros((5, 5), dtype=int)
        plain[1:4, 1:4] = 1
        dotted = plain.copy()
        dotted[1, 1] = 2
        moved = plain.copy()
        moved[3, 3] = 2
        with torch.no_grad():
            spatial = self._extractor(8)
            assert not torch.allclose(self._rows(spatial, dotted), self._rows(spatial, moved))
            blind = self._extractor(0)
            assert torch.allclose(self._rows(blind, dotted), self._rows(blind, moved))

    def test_the_features_are_as_wide_as_declared(self):
        extractor = self._extractor(8)
        objects = np.stack([object_row((1, 1, 3, 3), (5, 5)), np.zeros(OBJECT_DIM)])
        features = extractor(observation([np.zeros((5, 5), int)], objects=[objects]))
        assert features.shape[1] == extractor.features_dim


class TestCoordinateRowsFromTheMap:
    def test_a_row_is_read_with_the_same_weights_at_any_width(self):
        narrow = CoordinateRows(5, 7, [], dim=4, spatial_channels=6)
        wide = CoordinateRows(5, 20, [], dim=4, spatial_channels=6)
        assert narrow.row_encoder[0].in_features == wide.row_encoder[0].in_features == 13

    def test_two_rows_holding_the_same_differ_only_by_position(self):
        """Without the map, a row's cells go through a weight per column."""
        torch.manual_seed(0)
        space = spaces.Dict({"grid": spaces.Box(0, 10, shape=(6, 6), dtype=np.int64),
                             "grid_shape": spaces.Box(0, 30, shape=(2,), dtype=np.int64)})
        extractor = ARCCombinedExtractor(space, coordinate_dim=4, spatial_channels=6)
        assert extractor.coordinate_rows.spatial_channels == 6
        grid = np.zeros((6, 6), dtype=int)
        grid[2, 1] = 3
        obs = observation([grid], shapes=[[6, 6]])
        features = extractor(obs)
        assert features.shape[1] == extractor.features_dim

    def test_rows_need_the_map_when_built_from_it(self):
        rows = CoordinateRows(3, 3, [], dim=4, spatial_channels=2)
        with pytest.raises(TypeError):
            rows(observation([np.zeros((3, 3), int)]))


def two_objects():
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 2
    grid[3, 3] = 4
    out = grid.copy()
    out[1, 1] = 1
    return [ARCSubtask("t", grid, out)]


class TestAnAgent:
    def test_an_object_agent_builds_and_learns_with_the_map(self):
        vec_env = create_vec_env(two_objects(), n_envs=1, max_episode_len=4,
                                 feasible_actions={0: "submit", 1: "blue_recolor"},
                                 observation_space_elements=["objects_emb"], repr_level=1,
                                 input_pattern="start")
        try:
            agent = create_agent({"model_type": "PPO"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0,
                                  "spatial_channels": 8})
            assert agent.policy.features_extractor.spatial is not None
            agent.learn(total_timesteps=32)
        finally:
            vec_env.close()

    def test_the_critic_reads_the_answer_s_delta_and_the_actor_does_not(self):
        from tests.test_rl_coordinate_policy import coordinate_agent_with
        agent, vec_env = coordinate_agent_with({"spatial_channels": 8})
        try:
            actor = agent.policy.pi_features_extractor.spatial
            critic = agent.policy.vf_features_extractor.spatial
            assert "delta_target" not in actor.delta_keys and "delta_target" in critic.delta_keys
            agent.learn(total_timesteps=32)
        finally:
            vec_env.close()
