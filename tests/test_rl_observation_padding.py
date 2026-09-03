"""Observation padding: for the buffer, undone in the policy.

stable-baselines3 wants one array shape per observation key across a whole
vector of envs, and half the shape-preserving training split shows its rule
at several grid sizes - so one agent per task needs the observation to have
a fixed grid shape. Padding the env's grid itself is not the answer: it
moves the objects, the intersection and every reward derived from them, and
the signal is thin enough already.

So `observation_grid_shape` pads nothing but the array handed over, carries
the true shape alongside as `grid_shape`, and the extractor crops it back
off before the conv ever sees it. These pin the two halves of that: the env
computes on the real grid throughout, and the features come out identical
to the ones the unpadded grid would have produced.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from data.configs.rl_configs import lin
from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.features import ARCCombinedExtractor, unpadded_grid_features
from rl.training import create_vec_env

SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}
OBSERVED = (10, 10)


def subtask_of(size, fill=1):
    grid = np.zeros((size, size), dtype=int)
    grid[0, 0] = fill
    out = np.zeros((size, size), dtype=int)
    out[0, 1] = fill
    return ARCSubtask(f"s_{size}", grid, out)


def env_for(size, observed=OBSERVED, **kwargs):
    env = ARCGridWorld(max_episode_len=5, feasible_actions=SUBMIT_AND_ROTATE,
                       observation_space_elements=["objects_emb"],
                       observation_grid_shape=observed, **kwargs)
    env.set_subtask(subtask_of(size))
    return env


class TestWhatTheObservationCarries:
    def test_the_grid_is_padded_to_the_observation_shape(self):
        env = env_for(4)
        obs, _ = env.reset()

        assert obs["grid"].shape == OBSERVED

    def test_the_padding_is_top_left_aligned(self):
        """So grid_shape alone names the valid region - centred padding
        would need the offsets carried too."""
        env = env_for(4)
        obs, _ = env.reset()

        assert np.array_equal(obs["grid"][:4, :4], env.grid)
        assert (obs["grid"][4:, :] == env.pad_val).all()
        assert (obs["grid"][:, 4:] == env.pad_val).all()

    def test_the_true_shape_rides_along(self):
        env = env_for(4)
        obs, _ = env.reset()

        assert tuple(obs["grid_shape"]) == (4, 4)

    def test_every_observation_falls_inside_the_declared_space(self):
        env = env_for(4)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        obs, _, _, _, _ = env.step(np.array([1, 0, 0]))
        assert env.observation_space.contains(obs)

        obs, _, _, _, _ = env.step(np.array([0, 0, 0]))  # submit
        assert env.observation_space.contains(obs)

    def test_a_grid_too_large_for_the_shape_says_so(self):
        with pytest.raises(ValueError, match=r"\(12, 12\).*\(10, 10\)"):
            env_for(12)

    def test_without_the_setting_nothing_is_padded(self):
        env = env_for(4, observed=None)
        obs, _ = env.reset()

        assert obs["grid"].shape == (4, 4)
        assert "grid_shape" not in obs
        assert "grid_shape" not in env.observation_space.spaces


class TestTheEnvStillComputesOnTheRealGrid:
    """The whole reason not to pad the grid itself. Every one of these is
    the same number with the setting on and off, or the padding has moved
    into the reward."""

    @pytest.mark.parametrize("attribute", ["max_int", "target_int", "base_int",
                                           "max_reward"])
    def test_the_counters_are_untouched(self, attribute):
        plain, padded = env_for(4, observed=None), env_for(4)
        plain.reset(), padded.reset()

        assert getattr(plain, attribute) == getattr(padded, attribute)

    def test_the_grid_itself_is_untouched(self):
        plain, padded = env_for(4, observed=None), env_for(4)
        plain.reset(), padded.reset()

        assert np.array_equal(plain.grid, padded.grid)

    def test_a_step_pays_the_same(self):
        plain, padded = env_for(4, observed=None), env_for(4)
        plain.reset(), padded.reset()
        action = np.array([1, 0, 0])

        _, plain_reward, _, _, _ = plain.step(action)
        _, padded_reward, _, _, _ = padded.step(action)

        assert plain_reward == padded_reward

    def test_the_object_count_is_untouched(self):
        """Padding the grid itself would add a region of pad_val cells,
        which the component retrieval reads as another object."""
        plain, padded = env_for(4, observed=None), env_for(4)
        plain.reset(), padded.reset()

        assert len(plain.objects) == len(padded.objects)


class TestUndoingItInThePolicy:
    def _extractor(self, env):
        torch.manual_seed(0)
        extractor = ARCCombinedExtractor(env.observation_space, extr_arch=lin())
        return extractor.eval()

    def _observation(self, env):
        obs, _ = env.reset()
        return {key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
                for key, value in obs.items()}

    def test_the_features_are_the_ones_the_real_grid_gives(self):
        """Not approximately: the padded run crops to the same array before
        the first convolution, so it is the same computation."""
        plain, padded = env_for(4, observed=None), env_for(4)

        with torch.no_grad():
            from_plain = self._extractor(plain)(self._observation(plain))
            from_padded = self._extractor(padded)(self._observation(padded))

        assert torch.allclose(from_plain, from_padded, atol=1e-6)

    def test_grid_shape_is_not_encoded_as_a_feature(self):
        """It is there to undo the padding, not to be embedded - and a
        feature width that depended on it would differ per subtask."""
        plain, padded = env_for(4, observed=None), env_for(4)

        assert self._extractor(padded)._features_dim == \
            self._extractor(plain)._features_dim

    def test_grids_of_several_sizes_go_through_in_one_batch(self):
        """What a mixed rollout hands the extractor: one padded array, one
        row per env, and the envs are different sizes."""
        small, large = env_for(3), env_for(6)
        extractor = self._extractor(small)
        batch = {}
        for key in ("grid", "grid_shape", "objects_emb", "action_space"):
            rows = [np.asarray(self._observation(env)[key][0]) for env in (small, large)]
            batch[key] = torch.as_tensor(np.stack(rows))

        with torch.no_grad():
            batched = extractor(batch)
            one_at_a_time = torch.cat([extractor(self._observation(env))
                                       for env in (small, large)])

        assert batched.shape[0] == 2
        assert torch.allclose(batched, one_at_a_time, atol=1e-6)

    def test_shapes_of_none_is_the_unpadded_path(self):
        grids = torch.randint(0, 10, (2, 5, 5))
        extractor = torch.nn.Sequential(torch.nn.Flatten())

        assert unpadded_grid_features(extractor, grids).shape == (2, 25)


class TestWhatItBuys:
    def test_a_vec_env_spans_subtasks_of_different_sizes(self):
        """The point of the whole exercise: 128 of the 262 shape-preserving
        training tasks show their rule at more than one grid size, and this
        is what lets one agent train on all the examples of one."""
        vec_env = create_vec_env([subtask_of(4), subtask_of(7)], n_envs=1,
                                  max_episode_len=5,
                                  feasible_actions=SUBMIT_AND_ROTATE,
                                  observation_space_elements=["objects_emb"],
                                  observation_grid_shape=OBSERVED)
        try:
            obs = vec_env.reset()

            assert obs["grid"].shape == (2, *OBSERVED)
            assert [tuple(row) for row in obs["grid_shape"]] == [(4, 4), (7, 7)]
        finally:
            vec_env.close()

    def test_without_it_they_cannot_share_a_vector(self):
        vec_env = None
        try:
            with pytest.raises(ValueError, match="broadcast"):
                vec_env = create_vec_env([subtask_of(4), subtask_of(7)], n_envs=1,
                                          max_episode_len=5,
                                          feasible_actions=SUBMIT_AND_ROTATE,
                                          observation_space_elements=["objects_emb"])
                vec_env.reset()
        finally:
            if vec_env is not None:
                vec_env.close()


class TestScoringAPaddedObservation:
    """Everything outside the policy that is handed an observation rather
    than the env's own grid has to undo the padding too. Scoring a terminal
    observation was the one that broke: maximal_intersection compared an
    (8, 8) padded array against a (2, 2) target and raised "operands could
    not be broadcast together"."""

    def test_the_padding_comes_back_off(self):
        env = env_for(4)
        obs, _ = env.reset()

        assert np.array_equal(env.real_grid(obs["grid"]), env.grid)

    def test_an_unpadded_grid_is_returned_as_it_is(self):
        env = env_for(4, observed=None)
        obs, _ = env.reset()

        assert np.array_equal(env.real_grid(obs["grid"]), env.grid)

    def test_an_observation_scores_the_same_as_the_env_grid(self):
        from rl.evaluation import closed_fraction

        env = env_for(4)
        obs, _ = env.reset()

        assert closed_fraction(env, obs["grid"]) == closed_fraction(env)
