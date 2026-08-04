"""Tests for the task-agnostic Optuna scaffolding in rl/optimization.py.

OptunaPruningCallback is exercised directly (setting .model/.num_timesteps
by hand, same trick SB3 itself uses internally) rather than through a real
agent.learn() call - no live ARC env is needed for what this module does,
which is purely "watch a logger metric, report it, maybe prune". A fake
trial records .report() calls instead of a real optuna.Trial.

run_hyperparameter_search is exercised for real (in-memory/tmp-file
SQLite, a trivial objective) since optuna itself is cheap and fast here -
the point under test is the resume-from-storage behavior, which needs a
real Study/storage round-trip to mean anything.
"""
from __future__ import annotations

from types import SimpleNamespace

import optuna
import pytest

from rl.optimization import OptunaPruningCallback, run_hyperparameter_search


class FakeTrial:
    def __init__(self):
        self.reports: list[tuple[float, int]] = []

    def report(self, value, step):
        self.reports.append((value, step))


def make_callback(**kwargs) -> tuple[OptunaPruningCallback, FakeTrial]:
    trial = FakeTrial()
    defaults = dict(total_steps=1000, warmup_fraction=0.7, dead_epsilon=1e-6, report_freq=100)
    defaults.update(kwargs)
    callback = OptunaPruningCallback(trial, **defaults)
    callback.model = SimpleNamespace(logger=SimpleNamespace(name_to_value={}))
    return callback, trial


def step(callback: OptunaPruningCallback, num_timesteps: int, metric_value):
    callback.num_timesteps = num_timesteps
    callback.model.logger.name_to_value["rollout/ep_rew_mean"] = metric_value
    return callback._on_step()


def test_reports_at_report_freq_cadence():
    callback, trial = make_callback(report_freq=100)
    for t in range(0, 1000, 10):  # finer-grained than report_freq; metric moves so warmup pruning can't interfere
        step(callback, t, metric_value=t / 1000.0)
    # Only every 100 steps should have produced a report, starting at step 0
    assert [s for _, s in trial.reports] == list(range(0, 1000, 100))


def test_no_report_when_metric_not_yet_logged():
    callback, trial = make_callback()
    callback.num_timesteps = 500
    callback.model.logger.name_to_value.clear()  # nothing logged yet
    assert callback._on_step() is True
    assert trial.reports == []


def test_does_not_prune_before_warmup_even_if_flat():
    callback, trial = make_callback(total_steps=1000, warmup_fraction=0.7, report_freq=100)
    for t in range(0, 700, 100):  # up to but not past 70% of 1000
        step(callback, t, metric_value=0.0)  # perfectly flat, i.e. "dead"
    assert len(trial.reports) > 0  # confirms reporting itself did happen


def test_prunes_a_dead_run_past_warmup():
    callback, trial = make_callback(total_steps=1000, warmup_fraction=0.7, report_freq=100, dead_epsilon=1e-6)
    with pytest.raises(optuna.TrialPruned):
        for t in range(0, 1000, 100):
            step(callback, t, metric_value=0.0)  # flat the entire way


def test_does_not_prune_a_moving_metric_past_warmup():
    """The core design point: a run that's still improving (even slowly,
    even after a long plateau) must never be pruned just for being past
    warmup - only a run with zero movement in its whole history."""
    callback, trial = make_callback(total_steps=1000, warmup_fraction=0.7, report_freq=100, dead_epsilon=1e-6)
    values = [0.0] * 7 + [0.1, 0.5, 1.0]  # plateau, then a late breakthrough
    for t, v in zip(range(0, 1000, 100), values):
        step(callback, t, metric_value=v)  # should not raise
    assert len(trial.reports) == len(values)


def test_run_hyperparameter_search_is_resumable_via_storage(tmp_path):
    storage_path = f"sqlite:///{tmp_path / 'study.db'}"

    def objective(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    study1 = run_hyperparameter_search(objective, n_trials=2, study_name="resume-check",
                                        storage_path=storage_path)
    assert len(study1.trials) == 2

    # Re-running against the same study_name/storage_path should pick up
    # where it left off (3 total), not start a fresh study (2 total).
    study2 = run_hyperparameter_search(objective, n_trials=1, study_name="resume-check",
                                        storage_path=storage_path)
    assert len(study2.trials) == 3


def test_run_hyperparameter_search_uses_nop_pruner(tmp_path):
    """Comparative pruners (Median/SuccessiveHalving/Hyperband) rank trials
    against each other - exactly what OptunaPruningCallback is meant to
    replace for this project's RL trials (see rl/optimization.py's module
    docstring). The study itself must not layer that back in."""
    storage_path = f"sqlite:///{tmp_path / 'study.db'}"
    study = run_hyperparameter_search(lambda trial: trial.suggest_float("x", 0, 1),
                                       n_trials=1, study_name="pruner-check", storage_path=storage_path)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
