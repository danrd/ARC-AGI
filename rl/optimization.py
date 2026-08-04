"""Task-agnostic Optuna hyperparameter-search scaffolding: study/storage
setup, the search loop, and a training callback that reports an SB3
logger metric to a trial as training progresses.

Nothing here knows about ARC, PPO, or which hyperparameters exist -
callers supply an `objective_fn(trial) -> float` that builds, trains, and
scores one trial. The ARC/PPO-specific pieces (hyperparameter search
space, how a trial's score is computed, what counts as "this run is
dead") live in rl.arc_hp_search instead.

Pruning philosophy (deliberately not the default MedianPruner/
SuccessiveHalvingPruner/Hyperband): RL learning curves plateau for long
stretches before a late breakthrough far more often than supervised-
learning loss curves do, especially with sparse/milestone-shaped rewards.
"Behind the other trials so far" is therefore a weak signal here - it
would prune runs that are about to pay off. OptunaPruningCallback below
only ever prunes a run that's shown *zero* movement on its own tracked
metric for its entire history so far, and only after `warmup_fraction` of
the training budget has elapsed - a dead-run safety net, not comparative
ranking. run_hyperparameter_search therefore attaches NopPruner to the
study; all real pruning decisions are made by the callback itself.
"""
from __future__ import annotations

from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback


def run_hyperparameter_search(objective_fn: Callable, n_trials: int, study_name: str,
                               storage_path: Optional[str] = None, direction: str = "maximize",
                               n_jobs: int = 1, sampler=None):
    """Run an Optuna study over `objective_fn`.

    storage_path (default: f"sqlite:///{study_name}.db" in the cwd) makes
    the study resumable: re-running with the same study_name/storage_path
    picks up where it left off (load_if_exists=True) instead of losing
    every trial run so far to a crash or interruption.
    """
    import optuna

    storage = storage_path or f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    try:
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print(f"Hyperparameter search interrupted - study state saved at: {storage}")
    return study


class OptunaPruningCallback(BaseCallback):
    """Reports `metric_key` (an SB3 logger metric, e.g. the default
    "rollout/ep_rew_mean") to `trial` roughly every `report_freq` steps,
    and prunes the trial only if, past `warmup_fraction` of `total_steps`,
    that metric has shown no movement at all since training started -
    see this module's docstring for why "dead", not "behind", is the bar.
    """

    def __init__(self, trial, total_steps: int, metric_key: str = "rollout/ep_rew_mean",
                 warmup_fraction: float = 0.7, dead_epsilon: float = 1e-6, report_freq: int = 1000):
        super().__init__()
        self.trial = trial
        self.total_steps = total_steps
        self.metric_key = metric_key
        self.warmup_fraction = warmup_fraction
        self.dead_epsilon = dead_epsilon
        self.report_freq = report_freq
        self.history: list[float] = []
        self._last_report_step = -report_freq  # so the very first eligible step reports

    def _on_step(self) -> bool:
        import optuna

        if self.num_timesteps - self._last_report_step < self.report_freq:
            return True
        value = self.model.logger.name_to_value.get(self.metric_key)
        if value is None:
            return True  # nothing logged yet (no episode has finished) - nothing to report

        self._last_report_step = self.num_timesteps
        self.history.append(value)
        self.trial.report(value, self.num_timesteps)

        past_warmup = self.num_timesteps >= self.warmup_fraction * self.total_steps
        run_is_dead = (max(self.history) - min(self.history)) <= self.dead_epsilon
        if past_warmup and run_is_dead:
            raise optuna.TrialPruned(
                f"{self.metric_key} showed no movement "
                f"({min(self.history):.4g}..{max(self.history):.4g}) "
                f"past {self.warmup_fraction:.0%} of the training budget."
            )
        return True
