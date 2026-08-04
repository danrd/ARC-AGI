"""Tests for rl/arc_hp_search.py.

sample_ppo_hyperparameters is tested against a real optuna Trial (cheap,
no env needed) - including a persistent-storage smoke test, since the
whole point of sampling architectures/activations by name (see that
function's docstring) is to avoid the "choices aren't safe for
persistent storage" warning a naive list/class choice would trigger.

arc_ppo_objective's own training call (create_vec_env -> create_agent ->
agent.learn()) is exercised against monkeypatched fakes, not a live ARC
env - see rl/arc_hp_search.py's module docstring for why a live run
can't complete right now (unrelated, pre-existing pipeline bugs). This
tests arc_ppo_objective's own wiring/scoring logic in isolation.
"""
from __future__ import annotations

import warnings

import numpy as np
import optuna
import pytest

import rl.arc_hp_search as arc_hp_search
from rl.arc_hp_search import ACTIVATIONS, NET_ARCHS, arc_ppo_objective, sample_ppo_hyperparameters


def _ask() -> optuna.trial.Trial:
    return optuna.create_study().ask()


def test_sample_ppo_hyperparameters_returns_the_expected_keys_and_types():
    params = sample_ppo_hyperparameters(_ask())

    assert params["actor_arch"] in NET_ARCHS.values()
    assert params["critic_arch"] in NET_ARCHS.values()
    assert params["activation_fn"] in ACTIVATIONS.values()
    assert isinstance(params["learning_rate"], float)
    assert 1e-4 <= params["learning_rate"] <= 1e-1


def test_sample_ppo_hyperparameters_actor_and_critic_sampled_independently():
    """Regression guard for the bug in the original (deleted) version of
    this search space: critic_arch's suggest_categorical reused the
    'actor_arch' trial param name, so both were always driven by the same
    sampled value instead of being independent search dimensions."""
    seen_pairs = set()
    for _ in range(30):
        params = sample_ppo_hyperparameters(_ask())
        seen_pairs.add((tuple(params["actor_arch"]), tuple(params["critic_arch"])))
    distinct_actor = {pair[0] for pair in seen_pairs}
    distinct_critic = {pair[1] for pair in seen_pairs}
    mismatched = {pair for pair in seen_pairs if pair[0] != pair[1]}
    assert len(distinct_actor) > 1 and len(distinct_critic) > 1
    assert mismatched, "actor_arch and critic_arch never differed across 30 samples - looks coupled"


def test_sample_ppo_hyperparameters_choices_are_safe_for_persistent_storage():
    """Architectures/activations are sampled by name specifically so the
    stored choices are plain strings, not lists/classes - verify no
    "unsafe for persistent storage" warning fires."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_ppo_hyperparameters(_ask())
    persistent_storage_warnings = [w for w in caught if "persistent storage" in str(w.message)]
    assert not persistent_storage_warnings


class FakeAgent:
    def __init__(self, ep_info_buffer):
        self.ep_info_buffer = ep_info_buffer
        self.learn_calls = []

    def learn(self, total_steps, callback=None):
        self.learn_calls.append((total_steps, callback))


def test_arc_ppo_objective_scores_by_mean_ep_info_buffer_reward(monkeypatch):
    fake_agent = FakeAgent(ep_info_buffer=[{"r": 1.0}, {"r": 3.0}, {"r": 5.0}])
    monkeypatch.setattr(arc_hp_search, "create_vec_env", lambda *a, **kw: "fake-vec-env")
    monkeypatch.setattr(arc_hp_search, "create_agent", lambda *a, **kw: fake_agent)

    rl_config = {
        "n_envs": 1, "max_episode_len": 5, "repr_level": 1, "right_placement_reward": 5.0,
        "action_penalty": 1.0, "repetitive_actions_penalty": 1.0, "seed": 42, "font_color": 0,
        "padding": False, "input_pattern": False, "milestones_rewards": (1, 2, 3, 4), "pad_val": 10,
        "reward_approach": 1, "feasible_actions": {0: "submit"}, "observation_space_elements": ["objects_emb"],
        "total_steps": 1000,
    }
    score = arc_ppo_objective(_ask(), subtask="fake-subtask", rl_config=rl_config)

    assert score == pytest.approx(np.mean([1.0, 3.0, 5.0]))
    assert len(fake_agent.learn_calls) == 1
    total_steps, callback = fake_agent.learn_calls[0]
    assert total_steps == 1000
    assert callback.total_steps == 1000  # OptunaPruningCallback wired with the right budget


def test_arc_ppo_objective_returns_negative_infinity_when_no_episode_completed(monkeypatch):
    fake_agent = FakeAgent(ep_info_buffer=[])
    monkeypatch.setattr(arc_hp_search, "create_vec_env", lambda *a, **kw: "fake-vec-env")
    monkeypatch.setattr(arc_hp_search, "create_agent", lambda *a, **kw: fake_agent)

    rl_config = {
        "n_envs": 1, "max_episode_len": 5, "repr_level": 1, "right_placement_reward": 5.0,
        "action_penalty": 1.0, "repetitive_actions_penalty": 1.0, "seed": 42, "font_color": 0,
        "padding": False, "input_pattern": False, "milestones_rewards": (1, 2, 3, 4), "pad_val": 10,
        "reward_approach": 1, "feasible_actions": {0: "submit"}, "observation_space_elements": ["objects_emb"],
        "total_steps": 1000,
    }
    score = arc_ppo_objective(_ask(), subtask="fake-subtask", rl_config=rl_config)

    assert score == float("-inf")
