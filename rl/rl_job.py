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


def object_slots(task: Any, repr_level: int) -> int:
    """How many object slots this task's grids ever fill.

    The action space is (transform, object, object) over max_objects slots
    whatever the task holds, so a slot past the objects a grid has is a
    legal action that does nothing. Measured over the 262 shape-preserving
    training tasks at repr_level 1: the median task holds 3 objects against
    16 slots, 251 of them fewer than 16, and only 3.5% of the (object,
    object) pairs name two real objects on the median task. Sizing the slots
    to the task shrinks that pair space by a median factor of 28.

    Counted the way ARCGridWorld counts them - GridSummary over the example
    input at the configured repr_level - and over the held-out input too,
    since one agent is scored on that grid as well.
    """
    from symbolic.summaries import GridSummary

    grids = [subtask.train_inp for subtask in task.subtasks]
    test_subtask = getattr(task, "test_subtask", None)
    if test_subtask is not None:
        grids.append(test_subtask.train_inp)
    return max(len(GridSummary(grid=grid, shape=grid.shape,
                               levels=[repr_level])
                   .repr_levels[repr_level].objects)
               for grid in grids)


def observation_shape(task: Any):
    """The shape an observation has to be padded to, or None.

    None when every example is already one size: there is nothing to pad,
    and a shape here would only make the observation bigger than the task.
    Otherwise the largest of the task's own grids, which is what it takes to
    put them in one rollout buffer - not ARC's 30x30 maximum, which is 7
    times the median task's need.

    Only the observation is padded; the env's grid is untouched and
    rl.features.unpadded_grid_features crops each observation back to its
    true shape before the policy sees it.
    """
    shapes = [subtask.train_inp_shape for subtask in task.subtasks]
    test_subtask = getattr(task, "test_subtask", None)
    if test_subtask is not None:
        shapes.append(test_subtask.train_inp_shape)
    if len(set(shapes)) == 1:
        return None
    return (max(shape[0] for shape in shapes),
            max(shape[1] for shape in shapes))


def narrowed_for_task(task: Any, rl_config: Dict[str, Any],
                      settings: Any = None) -> Dict[str, Any]:
    """`rl_config` cut down to the space this task actually needs.

    The shipped config is written to be task-independent, and each of those
    defaults is a placeholder that costs a run:

    - feasible_actions is {0: 'submit'}, so a run started from it trains an
      agent whose only move is to give up. The search narrows it: 137 to 607
      actions in a task's generated vocabulary, 20 to 135 the search ever
      moved the grid with, 9 to 13 once intersected with the roster of the
      agent the task is labelled with (`task.agent`, when it has one).
    - max_objects is 16 for every task, and the median task holds 3.
    - observation_grid_shape is None, which refuses any task whose examples
      differ in size - 133 of the 262 shape-preserving training tasks, so
      more than half of them never reached their first step.

    Every measurement in this repository set these by hand instead, which is
    why a script's numbers and the pipeline's were never about the same
    thing.

    A search that fails leaves the actions alone rather than taking the
    pipeline down with it - training over a wider space is a worse run, not
    a broken one. The slots and the shape are read off the task and cannot
    fail that way. The search costs 0.7 to 116 seconds per task (median
    around 16) against the minutes a training run takes, and
    SearchSettings.timeout bounds the tail.
    """
    from rl.search_hints import SearchSettings, feasible_from_search

    narrowed = dict(rl_config)
    narrowed["max_objects"] = object_slots(
        task, rl_config.get("repr_level", 1))
    narrowed["observation_grid_shape"] = observation_shape(task)
    try:
        narrowed["feasible_actions"] = feasible_from_search(
            task, settings or SearchSettings(),
            agent=getattr(task, "agent", None))
    except Exception:  # noqa: BLE001 - a failed search must not fail the run
        pass
    return narrowed


def _rl_training_worker(task: Any) -> Dict[str, Any]:
    """Runs in the child process. Must be a module-level function (not
    nested) since the spawn context pickles the target. Imports
    rl.rl_module lazily, here rather than at module level, so that
    importing rl_job.py itself (e.g. from orchestration) never requires
    torch/stable-baselines3 just to manage a subprocess handle."""
    from data.configs.rl_configs import rl_config, load_PPO_config
    from rl.rl_module import RLModule, RlConfig

    return RLModule(RlConfig(**narrowed_for_task(task, rl_config)),
                    load_PPO_config()).solve(task)


def default_rl_start_fn(task: Any) -> RLJobHandle:
    """Starts RL training (rl.rl_module.RLModule, i.e. rl.training.train_on_task)
    for `task` as a background subprocess, using the default rl_config /
    PPO config. Pass a different rl_start_fn to solve_task() to use a
    non-default config or a previously-trained policy instead."""
    return RLJobHandle(task, _rl_training_worker)
