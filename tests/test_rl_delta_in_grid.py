"""Tests for the deltas as planes of the grid encoder (delta_in_grid): one
stack of convolutions reads a cell's colour and whether it differs from the
input together, where the default gives each delta a DeltaReadout of its own
and the two meet only in the concatenated vector.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from data.configs.rl_configs import lin
from rl.arc_task import ARCSubtask
from rl.features import (ARCCombinedExtractor, DeltaReadout, build_grid_arch,
                         default_grid_arch, grid_arch_channels, grid_arch_width)
from rl.training import create_agent, create_vec_env

GRID_WIDTH = 144


def space(shape=(6, 6), deltas=("delta_input",)):
    entries = {"grid": spaces.Box(0, 10, shape=shape, dtype=np.int64)}
    entries["grid_shape"] = spaces.Box(1, 30, shape=(2,), dtype=np.int64)
    for key in deltas:
        entries[key] = spaces.Box(0, 1, shape=shape, dtype=np.int64)
    return spaces.Dict(entries)


def observation(grid, delta, shape=None):
    grid = np.asarray(grid)
    shape = shape or grid.shape
    return {"grid": torch.tensor(grid[None], dtype=torch.int64),
            "grid_shape": torch.tensor([shape], dtype=torch.int64),
            "delta_input": torch.tensor(np.asarray(delta)[None], dtype=torch.int64)}


def extractor(delta_in_grid, **kwargs):
    torch.manual_seed(0)
    return ARCCombinedExtractor(space(), delta_in_grid=delta_in_grid, **kwargs).eval()


class TestTheEncoders:
    def test_the_grid_encoder_reads_the_delta_as_a_plane_of_its_own(self):
        built = extractor(True)
        assert grid_arch_channels(built.extractors["grid"]) == 11
        assert "delta_input" not in built.extractors

    def test_by_default_a_delta_has_a_readout_and_the_grid_ten_planes(self):
        built = extractor(False)
        assert grid_arch_channels(built.extractors["grid"]) == 10
        assert isinstance(built.extractors["delta_input"], DeltaReadout)

    def test_the_features_are_as_wide_as_declared(self):
        for delta_in_grid in (True, False):
            built = extractor(delta_in_grid)
            out = built(observation(np.zeros((6, 6), int), np.zeros((6, 6), int)))
            assert out.shape[1] == built.features_dim
        assert extractor(True).features_dim == GRID_WIDTH

    def test_a_factory_and_the_default_are_asked_for_the_planes_they_must_read(self):
        assert grid_arch_channels(build_grid_arch(None, 12)) == 12
        assert grid_arch_channels(build_grid_arch(lin, 11)) == 11
        assert grid_arch_channels(lin(in_channels=11, widths=(32, 32))) == 11
        assert grid_arch_width(default_grid_arch(in_channels=11)) == GRID_WIDTH
        assert grid_arch_width(default_grid_arch(widths=(32, 32))) == 32 * 9


class TestWhatItReads:
    def grid(self):
        grid = np.zeros((6, 6), dtype=int)
        grid[2, 2] = 2
        return grid

    def delta_at(self, i, j):
        delta = np.zeros((6, 6), dtype=int)
        delta[i, j] = 1
        return delta

    def test_the_delta_reaches_the_grid_encoder(self):
        with torch.no_grad():
            built = extractor(True)
            near = built(observation(self.grid(), self.delta_at(2, 3)))
            far = built(observation(self.grid(), self.delta_at(5, 5)))
            same = built(observation(self.grid(), self.delta_at(2, 3)))
        assert torch.allclose(near, same)
        assert not torch.allclose(near, far)

    def test_without_it_the_grid_encoder_never_sees_the_delta(self):
        with torch.no_grad():
            built = extractor(False)
            # The grid's block of the vector: a Dict space is ordered by key,
            # so the delta's readout comes first and the grid's 144 last.
            assert list(built.extractors) == ["delta_input", "grid"]
            near = built(observation(self.grid(), self.delta_at(2, 3)))[:, -GRID_WIDTH:]
            far = built(observation(self.grid(), self.delta_at(5, 5)))[:, -GRID_WIDTH:]
        assert torch.allclose(near, far)

    def test_padding_does_not_change_what_it_reads(self):
        """The same 3x3 grid and delta, alone and inside a 6x6 observation."""
        grid = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        delta = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
        padded_grid = np.full((6, 6), 10)
        padded_grid[:3, :3] = grid
        padded_delta = np.zeros((6, 6), dtype=int)
        padded_delta[:3, :3] = delta
        with torch.no_grad():
            built = extractor(True)
            padded = built(observation(padded_grid, padded_delta, shape=(3, 3)))
            alone = built(observation(grid, delta))
        assert torch.allclose(padded, alone, atol=1e-6)


def two_objects():
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 2
    grid[3, 3] = 4
    out = grid.copy()
    out[1, 1] = 1
    return [ARCSubtask("t", grid, out)]


class TestAnAgent:
    def test_an_agent_learns_with_the_delta_in_the_grid_and_the_critic_keeps_its_own(self):
        vec_env = create_vec_env(two_objects(), n_envs=1, max_episode_len=4,
                                 feasible_actions={0: "submit", 1: "blue_recolor"},
                                 observation_space_elements=["objects_emb", "delta_input",
                                                             "delta_target"],
                                 repr_level=1, input_pattern="start")
        try:
            agent = create_agent({"model_type": "PPO"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0,
                                  "delta_in_grid": True})
            actor = agent.policy.features_extractor
            assert actor.delta_in_grid and "delta_input" not in actor.extractors
            critic = agent.policy.vf_features_extractor.own
            assert isinstance(critic.extractors["delta_target"], DeltaReadout)
            agent.learn(total_timesteps=32)
        finally:
            vec_env.close()
