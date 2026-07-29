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
