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

import os
from pathlib import Path
from types import SimpleNamespace

from rl.arc_hp_search import ACTIVATIONS, NET_ARCHS, sample_ppo_hyperparameters


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


class TestTheObjectiveIsTheCallersChoice:
    """What to vary, what to train, what to call good - all three are
    arguments now. The objective used to hardcode the third: mean episode
    reward out of SB3's buffer.

    That is training reward under whichever reward_approach the env was
    built with, so it cannot compare two of them, and it is not the number
    the agent is finally judged on. The two have been measured moving
    apart: under reward_approach 2 the agent stopped giving up, reward
    rose, and the fraction of the distance closed fell on two tasks of
    three. A search maximising reward would have chosen that.
    """

    @staticmethod
    def _outcome():
        accuracies = {"t_0": 0.25, "t_1": 0.75}
        lens = {"t_0": 4.0, "t_1": 6.0}
        agent = SimpleNamespace(ep_info_buffer=[{"r": 1.0}, {"r": 3.0}])
        metrics = {"test_acc": 0.5, "test_len": 5.0}
        return accuracies, lens, agent, metrics

    @pytest.mark.parametrize("name,expected", [
        ("held_out", 0.5),
        ("mixed", 0.5),
        ("episode_reward", 2.0),
        ("episode_len", 5.0),
    ])
    def test_each_scorer_reads_what_it_names(self, name, expected):
        from rl.arc_hp_search import SCORERS

        assert SCORERS[name](*self._outcome()) == pytest.approx(expected)

    def test_held_out_and_reward_can_disagree(self):
        """The reason the default changed, as a test rather than a story:
        a run whose training reward is high and whose held-out pair is
        untouched scores well on one and worst on the other."""
        from rl.arc_hp_search import SCORERS

        accuracies = {"t_0": 0.9}
        lens = {"t_0": 25.0}
        agent = SimpleNamespace(ep_info_buffer=[{"r": 8.0}])
        metrics = {"test_acc": -1.0}

        assert SCORERS["episode_reward"](accuracies, lens, agent, metrics) > 0
        assert SCORERS["held_out"](accuracies, lens, agent, metrics) < 0

    def test_a_sampled_setting_is_routed_to_the_config_it_belongs_to(self):
        from rl.arc_hp_search import split_settings

        ppo, arc = split_settings({"gamma": 0.95, "action_penalty": 0.0,
                                   "n_steps": 512, "reward_approach": 4})

        assert ppo == {"gamma": 0.95, "n_steps": 512}
        assert arc == {"action_penalty": 0.0, "reward_approach": 4}

    def test_a_setting_belonging_to_neither_config_is_refused(self):
        """Silently dropping it is how a sweep comes to measure the default
        under the name of something else - the same failure an arm that
        built itself wrong would have."""
        from rl.arc_hp_search import split_settings

        with pytest.raises(ValueError, match="no config"):
            split_settings({"gamma": 0.9, "learning_rat": 3e-4})

    def test_the_search_space_is_an_argument(self, monkeypatch):
        """So the same objective can tune the ARC-specific half, which is
        the half with the most written about it and the least measured."""
        import rl.arc_hp_search as hp

        seen = {}

        def fake_train(task, rl_config, PPO_config, mode, extra_callback):
            seen["arc"] = rl_config
            seen["ppo"] = PPO_config
            return {"t": 0.4}, {"t": 3.0}, SimpleNamespace(ep_info_buffer=[]), \
                {"test_acc": 0.7}

        monkeypatch.setattr(hp, "train_on_task", fake_train)
        score = hp.arc_objective(
            _ask(), task="fake-task",
            config={"total_steps": 10, "action_penalty": 1.0},
            sample=lambda trial: {"action_penalty": 0.0, "gamma": 0.8})

        assert score == pytest.approx(0.7)
        assert seen["arc"]["action_penalty"] == 0.0
        assert seen["ppo"]["gamma"] == 0.8

    def test_the_metric_is_an_argument(self, monkeypatch):
        import rl.arc_hp_search as hp

        def fake_train(task, rl_config, PPO_config, mode, extra_callback):
            return {"t": 0.4}, {"t": 3.0}, SimpleNamespace(ep_info_buffer=[]), \
                {"test_acc": 0.7}

        monkeypatch.setattr(hp, "train_on_task", fake_train)
        held = hp.arc_objective(_ask(), task="t", config={"total_steps": 10},
                                sample=lambda trial: {})
        mixed = hp.arc_objective(_ask(), task="t", config={"total_steps": 10},
                                 sample=lambda trial: {}, score="mixed")
        custom = hp.arc_objective(_ask(), task="t", config={"total_steps": 10},
                                  sample=lambda trial: {},
                                  score=lambda accuracies, lens, agent, metrics: 42.0)

        assert (held, mixed, custom) == (pytest.approx(0.7),
                                         pytest.approx(0.4),
                                         pytest.approx(42.0))

    def test_the_config_handed_in_is_not_mutated(self, monkeypatch):
        """A study runs many trials off one config; a sampled setting
        written into it would leak into every trial after."""
        import rl.arc_hp_search as hp

        monkeypatch.setattr(hp, "train_on_task",
                            lambda **kwargs: ({}, {}, SimpleNamespace(
                                ep_info_buffer=[]), {"test_acc": 0.0}))
        config = {"total_steps": 10, "action_penalty": 1.0}
        hp.arc_objective(_ask(), task="t", config=config,
                         sample=lambda trial: {"action_penalty": 0.0})

        assert config["action_penalty"] == 1.0


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("ARC_TEST_LIVE") != "1",
                    reason="trains an agent; set ARC_TEST_LIVE=1")
def test_one_trial_runs_end_to_end():
    """The claim this module made about itself for a long time, and never
    checked: "a live end-to-end run still hits an unrelated
    observation_space mismatch inside vec_env.reset()". Its tests ran
    against mocked training, so nothing would have noticed the day that
    stopped being true - which it has.

    Opt-in because it trains. A few thousand steps is far below the budget
    anything learns at; what is being tested is that the wiring holds, not
    that the agent gets anywhere.
    """
    import json

    import optuna

    from data.configs.rl_configs import rl_config
    from rl.arc_hp_search import arc_objective
    from rl.arc_task import ARCSubtask, ARCTask

    root = Path(__file__).resolve().parents[1] / "data" / "datasets" / "ARC"
    challenges = json.loads((root / "training_challenges.json").read_text())
    solutions = json.loads((root / "training_solutions.json").read_text())
    label = "a48eeaf7"
    entry = challenges[label]
    task = ARCTask(
        label=label,
        subtasks=[ARCSubtask(f"{label}_{i}", np.array(pair["input"]),
                             np.array(pair["output"]))
                  for i, pair in enumerate(entry["train"])],
        test_inp=np.array(entry["test"][0]["input"]),
        test_out=np.array(solutions[label][0]))
    config = dict(rl_config, total_steps=2000, eval_freq=20_000, seed=42,
                  feasible_actions={0: "submit", 1: "gravity"})

    score = arc_objective(optuna.create_study().ask(), task, config)

    assert isinstance(score, float)
    assert score == score, "the objective came back NaN"
