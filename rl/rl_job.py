"""Background RL-training job, run as a separate OS process (spawned via
multiprocessing, not forked — safe even once the worker touches torch/CUDA
elsewhere in the process tree), with non-blocking polling and clean
cancellation (terminate -> kill).

This exists because the RL module actually TRAINS a policy for the specific
task (via ARCGridWorld + PPO) rather than just running inference — that can
take a while, so the agent-level graph needs to be able to: start it, keep
doing other things (call the LLM) without blocking, check in on it
non-destructively, and kill it outright if its result ends up not needed.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, Callable, Dict, Optional


def _rl_worker_entrypoint(worker_fn: Callable[[Any], Dict[str, Any]], task: Any,
                           result_queue: "mp.Queue") -> None:
    """Runs in the child process. Never lets an exception escape silently —
    puts an error result on the queue instead, so the parent always gets
    something to poll for."""
    try:
        result = worker_fn(task)
        result_queue.put({"status": "ok", **result})
    except Exception as e:  # noqa: BLE001 - must not crash silently in the child
        result_queue.put({"status": "error", "debug": f"{type(e).__name__}: {e}"})


class RLJobHandle:
    """Wraps one background RL-training run.

    `worker_fn(task) -> dict` does the actual training/search and returns a
    result dict (e.g. {"solution": grid, "debug": "..."}) — this is the seam
    a project wires its real RL training call into; see default_rl_start_fn
    below for the not-wired-up-yet placeholder.
    """

    def __init__(self, task: Any, worker_fn: Callable[[Any], Dict[str, Any]]):
        ctx = mp.get_context("spawn")
        self.result_queue: "mp.Queue" = ctx.Queue()
        self.process = ctx.Process(
            target=_rl_worker_entrypoint, args=(worker_fn, task, self.result_queue), daemon=True,
        )
        self.process.start()
        self._result: Optional[Dict[str, Any]] = None

    def poll(self) -> Optional[Dict[str, Any]]:
        """Non-blocking: returns the result dict once available, else None.
        Safe to call repeatedly — caches the result after the first hit."""
        if self._result is not None:
            return self._result
        try:
            self._result = self.result_queue.get_nowait()
        except queue.Empty:
            return None
        return self._result

    def wait(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Blocking, but bounded: waits up to `timeout` seconds for a
        result, returns None if it doesn't arrive in time (job keeps
        running — call cancel() if you're giving up on it)."""
        if self._result is not None:
            return self._result
        try:
            self._result = self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return self._result

    @property
    def done(self) -> bool:
        return self.poll() is not None

    def cancel(self, timeout: float = 5.0) -> None:
        """Clean shutdown: terminate, then kill if it doesn't die in time.
        Safe to call on an already-finished or already-cancelled job."""
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


def with_searched_actions(task: Any, rl_config: Dict[str, Any],
                          settings: Any = None) -> Dict[str, Any]:
    """`rl_config` with feasible_actions narrowed to what a search can use.

    The shipped config carries {0: 'submit'} - a placeholder, not a
    vocabulary - so a run started from it trains an agent whose only move is
    to give up. Every measurement in this repository built the action set by
    hand instead, which meant the thing being measured was never what the
    pipeline ran.

    The search is what narrows it: 137 to 607 actions in a task's generated
    vocabulary against 20 to 135 the search ever moved the grid with, and 9
    to 13 once intersected with the roster of the agent a task is labelled
    with. `task.agent` supplies that label when it has one; without it the
    search's own set stands.

    A search that fails or finds nothing leaves the config untouched rather
    than taking the pipeline down with it - training on a wider space is a
    worse run, not a broken one. It costs 0.7 to 116 seconds per task
    (median around 16), against the minutes a training run takes, and
    SearchSettings.timeout bounds the tail.
    """
    from rl.search_hints import SearchSettings, feasible_from_search

    try:
        actions = feasible_from_search(task, settings or SearchSettings(),
                                       agent=getattr(task, "agent", None))
    except Exception:  # noqa: BLE001 - a failed search must not fail the run
        return rl_config
    return {**rl_config, "feasible_actions": actions}


def _rl_training_worker(task: Any) -> Dict[str, Any]:
    """Runs in the child process. Must be a module-level function (not
    nested) since the spawn context pickles the target. Imports
    rl.rl_module lazily, here rather than at module level, so that
    importing rl_job.py itself (e.g. from orchestration) never requires
    torch/stable-baselines3 just to manage a subprocess handle."""
    from data.configs.rl_configs import rl_config, load_PPO_config
    from rl.rl_module import RLModule, RlConfig

    return RLModule(RlConfig(**with_searched_actions(task, rl_config)),
                    load_PPO_config()).solve(task)


def default_rl_start_fn(task: Any) -> RLJobHandle:
    """Starts RL training (rl.rl_module.RLModule, i.e. rl.training.train_on_task)
    for `task` as a background subprocess, using the default rl_config /
    PPO config. Pass a different rl_start_fn to solve_task() to use a
    non-default config or a previously-trained policy instead."""
    return RLJobHandle(task, _rl_training_worker)
