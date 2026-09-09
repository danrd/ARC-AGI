"""Tests for rl/policy.py's argument wiring.

Constructing a real ARCGNNPolicy/ARCSeparatePolicy needs SB3 machinery plus
a live observation/action space, and the thing under test here is purely
which arguments these subclasses forward to their parent - so the parent
__init__ is patched to record what it receives, rather than building the
whole policy.
"""
from __future__ import annotations

import pytest

from rl.policy import ARCCustomActorCriticPolicy, ARCGNNPolicy, ARCSeparatePolicy


@pytest.fixture
def recorded_parent_kwargs(monkeypatch):
    """Patches ARCCustomActorCriticPolicy.__init__ (the parent of both
    policies under test) to record its kwargs instead of running."""
    recorded = {}

    def fake_init(self, *args, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ARCCustomActorCriticPolicy, "__init__", fake_init)
    return recorded


@pytest.mark.parametrize("policy_cls", [ARCGNNPolicy, ARCSeparatePolicy])
def test_features_extractor_kwargs_reach_the_parent(policy_cls, recorded_parent_kwargs):
    """Regression test: both policies did
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        super().__init__(..., features_extractor_kwargs=kwargs.get('features_extractor_kwargs', {}))
    - reading the key back out of kwargs with .get() AFTER .pop() had already
    removed it, so the parent always received {} and whatever rl.training's
    create_agent passed (extr_arch, shapes_match) was silently dropped."""
    extractor_kwargs = {"extr_arch": "sentinel-arch", "shapes_match": True}

    policy_cls(observation_space=None, action_space=None, lr_schedule=None,
               features_extractor_kwargs=extractor_kwargs)

    assert recorded_parent_kwargs["features_extractor_kwargs"] == extractor_kwargs


@pytest.mark.parametrize("policy_cls", [ARCGNNPolicy, ARCSeparatePolicy])
def test_features_extractor_kwargs_default_to_empty(policy_cls, recorded_parent_kwargs):
    policy_cls(observation_space=None, action_space=None, lr_schedule=None)

    assert recorded_parent_kwargs["features_extractor_kwargs"] == {}


@pytest.mark.parametrize("policy_cls", [ARCGNNPolicy, ARCSeparatePolicy])
def test_features_extractor_kwargs_not_passed_twice(policy_cls, recorded_parent_kwargs):
    """The key must be pop()ed rather than read in place: leaving it in
    kwargs would also send it through **kwargs and collide with the
    explicit keyword argument (TypeError: multiple values)."""
    policy_cls(observation_space=None, action_space=None, lr_schedule=None,
               features_extractor_kwargs={"extr_arch": "sentinel-arch"}, ortho_init=False)

    # reached the parent at all, and the unrelated kwarg still came through
    assert recorded_parent_kwargs["features_extractor_kwargs"] == {"extr_arch": "sentinel-arch"}
    assert recorded_parent_kwargs["ortho_init"] is False


class TestActionHeads:
    """Four groupings of the same three action dimensions - transform,
    object, object. Whatever the grouping, the concatenated logits are
    split by MultiCategoricalDistribution against action_dims, so what has
    to hold is that they concatenate to sum(action_dims) in order.

    The three-head branch used to index action_dims[3] and [4], written for
    a five-dimensional action space nothing builds - and three is what the
    shipped config asks for, so every training run died there before its
    first step."""

    DIMS = [89, 16, 16]

    @staticmethod
    def _net(heads, dims=None):
        from rl.policy import ARCCustomNetwork

        return ARCCustomNetwork(feature_dim=8, action_dims=dims or [89, 16, 16],
                                net_arch={"pi": [16], "vf": [16]},
                                action_heads=heads)

    @pytest.mark.parametrize("heads,widths", [
        (1, [121]),
        (2, [89, 32]),
        (3, [89, 16, 16]),
        (5, [89, 16, 16]),
    ])
    def test_the_heads_cover_the_action_space_exactly(self, heads, widths):
        import torch

        latent, _ = self._net(heads)(torch.zeros(2, 8))

        assert [tensor.shape[1] for tensor in latent] == widths
        assert sum(tensor.shape[1] for tensor in latent) == sum(self.DIMS)

    def test_three_heads_is_one_per_dimension(self):
        import torch

        latent, _ = self._net(3)(torch.zeros(2, 8))

        assert len(latent) == 3

    def test_three_heads_refuses_a_space_that_is_not_three_dimensional(self):
        """Rather than indexing past the end of action_dims, which is how
        this branch failed before."""
        with pytest.raises(ValueError, match="three-dimensional"):
            self._net(3, dims=[89, 16])

    def test_an_unsupported_count_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported"):
            self._net(4)


class TestWhatTheCriticMaySeeAndTheActorMayNot:
    """The critic runs only during training: it turns returns into
    advantages, and nothing calls it at inference. So it may read what will
    not exist at test time, while the actor - all that runs on a held-out
    pair - must not. 'target' is the case this exists for: the wanted output
    is known for every training example and unknown for the test one.

    stable-baselines3's share_features_extractor=False does not do this. It
    builds two extractors over the same observation (ActorCriticPolicy.
    extract_features), so both halves still see every key. Routing keys is
    what these pin.
    """

    @staticmethod
    def _space():
        import numpy as np
        from gymnasium import spaces
        from symbolic.objects_analysis import OBJECT_DIM

        return spaces.Dict({
            "grid": spaces.Box(low=0, high=10, shape=(5, 5), dtype=np.int64),
            "target": spaces.Box(low=0, high=10, shape=(5, 5), dtype=np.int64),
            "objects_emb": spaces.Box(low=0, high=1, shape=(4, OBJECT_DIM),
                                      dtype=np.float32),
        })

    @staticmethod
    def _policy(critic_only_keys):
        import numpy as np
        from gymnasium import spaces

        return ARCCustomActorCriticPolicy(
            TestWhatTheCriticMaySeeAndTheActorMayNot._space(),
            spaces.MultiDiscrete(np.array([3, 4, 4])),
            lambda _progress: 3e-4,
            net_arch={"pi": [16], "vf": [16]},
            action_heads=3,
            critic_only_keys=critic_only_keys,
        )

    @staticmethod
    def _observation():
        import torch
        from symbolic.objects_analysis import OBJECT_DIM

        return {"grid": torch.zeros((2, 5, 5), dtype=torch.int64),
                "target": torch.zeros((2, 5, 5), dtype=torch.int64),
                "objects_emb": torch.zeros((2, 4, OBJECT_DIM))}

    def test_the_actor_is_not_given_the_key_at_all(self):
        policy = self._policy(("target",))

        assert "target" not in policy.pi_features_extractor.extractors
        assert "target" in policy.vf_features_extractor.extractors

    def test_naming_a_critic_only_key_stops_the_extractor_being_shared(self):
        """Two extractors, or there is nothing to route between."""
        assert self._policy(("target",)).share_features_extractor is False

    def test_by_default_both_halves_see_the_same_observation(self):
        policy = self._policy(())

        assert policy.share_features_extractor is True
        assert "target" in policy.features_extractor.extractors

    def test_the_critic_reads_the_key_and_the_actor_does_not(self):
        """The behavioural statement, not the structural one: changing only
        the target must move the value and leave the action distribution
        exactly where it was."""
        import torch

        policy = self._policy(("target",))
        obs = self._observation()
        altered = {key: value.clone() for key, value in obs.items()}
        altered["target"] = torch.full_like(altered["target"], 3)

        with torch.no_grad():
            before = policy.get_distribution(obs).distribution[0].probs
            after = policy.get_distribution(altered).distribution[0].probs
            value_before = policy.predict_values(obs)
            value_after = policy.predict_values(altered)

        assert torch.allclose(before, after), "the actor saw the target"
        assert not torch.allclose(value_before, value_after), \
            "the critic did not read the target"

    def test_the_two_halves_may_be_different_widths(self):
        """The critic's extractor is wider by whatever the extra keys add,
        and the value network has to be built at that width rather than the
        actor's."""
        policy = self._policy(("target",))

        assert (policy.vf_features_extractor.features_dim
                > policy.pi_features_extractor.features_dim)
        assert (policy.mlp_extractor.value_net[0].in_features
                == policy.vf_features_extractor.features_dim)
        assert (policy.mlp_extractor.shared_net[0].in_features
                == policy.pi_features_extractor.features_dim)

    def test_the_actor_s_extractor_is_still_trained(self):
        """It is rebuilt inside _build_mlp_extractor, which runs before the
        optimiser is created - a rebuild after that point would leave the
        actor's weights receiving no updates at all."""
        policy = self._policy(("target",))
        optimised = {id(p) for group in policy.optimizer.param_groups
                     for p in group["params"]}

        assert all(id(p) in optimised
                   for p in policy.pi_features_extractor.parameters())

    def test_a_forward_pass_returns_actions_and_values(self):
        import torch

        policy = self._policy(("target",))

        actions, values, log_prob = policy(self._observation())

        assert actions.shape == (2, 3)
        assert values.shape == (2, 1)
        assert log_prob.shape == (2,)
        assert not torch.isnan(values).any()

    def test_evaluate_actions_works_with_the_split(self):
        """PPO's update calls it every epoch, and it unpacks the tuple
        extract_features returns."""
        import torch

        policy = self._policy(("target",))
        actions, _values, _log_prob = policy(self._observation())

        values, log_prob, entropy = policy.evaluate_actions(
            self._observation(), actions)

        assert values.shape == (2, 1)
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(entropy).any()
