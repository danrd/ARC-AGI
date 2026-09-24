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

import numpy as np


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


def slots_for_grids(grids, repr_level: int = 1) -> int:
    """How many object slots these grids need, counted the way ARCGridWorld
    counts them - a GridSummary over each grid at the configured level.

    Separate from object_slots because the count is a property of grids, not
    of an ARCTask: search_hints builds an env around a single (input,
    output) pair and has no task to ask. Measured over the 400 training
    tasks at repr_level 1, the busiest grid of a task holds a median of 2
    objects against the fixed 16, so sizing to the grids shrinks the
    (object, object) half of the action space by a median factor of 64.

    At least two, because both the action space and the relation block are
    defined over pairs. An action is (transform, object, object), and the
    relations an observation carries are shaped
    (slots, (slots - 1) * RELATION_DIM) - which at one slot is a block of
    width zero, and every reader of it fails: measured over the arm sweep,
    all eight tasks whose grids hold a single object crashed in all three
    arms that carry relations, and only those. The spare slot names no
    object, so an action reaching for it does nothing, exactly as a slot
    past the objects always has.
    """
    from symbolic.summaries import GridSummary

    return max(2, max(len(GridSummary(grid=grid, shape=grid.shape,
                                      levels=[repr_level])
                          .repr_levels[repr_level].objects)
                      for grid in grids))


def object_slots(task: Any, repr_level: int) -> int:
    """How many object slots this task's grids ever fill.

    The action space is (transform, object, object) over max_objects slots
    whatever the task holds, so a slot past the objects a grid has is a
    legal action that does nothing. Measured over the 262 shape-preserving
    training tasks at repr_level 1: the median task holds 3 objects against
    16 slots, 251 of them fewer than 16, and only 3.5% of the (object,
    object) pairs name two real objects on the median task. Sizing the slots
    to the task shrinks that pair space by a median factor of 28.

    Counted over the example inputs at the configured repr_level - and over
    the held-out input too, since one agent is scored on that grid as well.
    The max over the task's grids rather than per grid, because one agent
    trains across all of them and a gymnasium space cannot change shape
    between subtasks. Measured over the 400 training tasks, the busiest
    grid of a task holds a median of 1 object more than its emptiest, so
    the padding a per-task size leaves behind is small.
    """
    grids = [subtask.train_inp for subtask in task.subtasks]
    test_subtask = getattr(task, "test_subtask", None)
    if test_subtask is not None:
        grids.append(test_subtask.train_inp)
    return slots_for_grids(grids, repr_level)


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


#: What the observation of a coordinate-addressed env carries besides the
#: grid. delta_input is what the agent has painted so far, legal at
#: inference; delta_target is what is still wrong - the answer, which only
#: the critic may read (ARCCustomActorCriticPolicy.critic_only_keys, and
#: rl.training.answer_keys enforces it whatever the config says).
COORDINATE_OBSERVATION = ("delta_input", "delta_target")

#: The observation keys that describe objects, and so mean nothing where
#: there are no objects to describe.
OBJECT_KEYS = ("objects_emb", "relations_emb")


def observation_for(elements, addressing: str) -> list:
    """The observation keys that go with this addressing.

    Object addressing keeps what the caller asked for. Coordinate
    addressing drops the object blocks - no action names an object, and the
    coordinate heads read rows and columns of the grid instead - and adds
    the two deltas, which are what makes a row or a column worth choosing.
    Everything else the caller asked for is left as it is.
    """
    if addressing != "coordinates":
        return list(elements)
    kept = [key for key in elements if key not in OBJECT_KEYS]
    return kept + [key for key in COORDINATE_OBSERVATION if key not in kept]


def addressing_for(agent) -> str:
    """What this task's actions address: 'objects' or 'coordinates'.

    Read off the agent's label rather than off the task, because the label
    already says what kind of change the task makes and the two
    vocabularies do not overlap - see AGENT2ADDRESSING for the measurement
    behind each entry. An unlabelled task, or one whose agent is not in the
    map, gets objects: what every task got before coordinates existed.
    """
    from data.configs.env_configs import AGENT2ADDRESSING

    return AGENT2ADDRESSING.get(agent, "objects")


def coordinate_shape(task: Any):
    """(rows, cols) the coordinate dimensions of the action space span.

    The largest grid any subtask works on. The working grid is the input
    padded out to the output's size where the output is larger (see
    ARCGridWorld.initialize_observation_space), so both shapes of every
    pair count - and the held-out pair's, since one agent is scored on it
    too. One action space has to serve all of them; a cell past the grid in
    hand names nothing and is scored as an action that changes nothing.
    """
    shapes = []
    for subtask in list(task.subtasks) + [getattr(task, "test_subtask", None)]:
        if subtask is None:
            continue
        shapes.append(np.asarray(subtask.train_inp).shape)
        if getattr(subtask, "train_out", None) is not None:
            shapes.append(np.asarray(subtask.train_out).shape)
    return (max(shape[0] for shape in shapes), max(shape[1] for shape in shapes))


#: The feasible_actions a config carries when nobody chose any: submit
#: alone, which trains an agent whose only move is to give up.
PLACEHOLDER_ACTIONS = {0: "submit"}


def _chosen(rl_config: Dict[str, Any], key: str, placeholder=None) -> bool:
    """Whether the caller set `key` to something of their own, rather than
    leaving it to be decided per task."""
    value = rl_config.get(key)
    return value is not None and value != placeholder


def narrowed_for_task(task: Any, rl_config: Dict[str, Any],
                      settings: Any = None) -> Dict[str, Any]:
    """`rl_config` with what it leaves open decided for this task.

    Only what it leaves open. A key the caller set is kept as set - an
    override in a notebook or in rl_configs.py itself is a decision, and
    this used to overwrite every one of them, so a run with max_objects,
    addressing or an action list of its own trained on the search's instead
    and said nothing. Left open means:

    - addressing None: read off the agent's label (addressing_for).
    - feasible_actions {0: 'submit'}: the search's. The shipped placeholder,
      which trains an agent whose only move is to give up. The object
      search narrows the action types - 137 to 607 names in a task's
      generated vocabulary, the ones that ever moved the grid, intersected
      with the roster of the agent the task is labelled with - and never
      the directions (feasible_from_search); the coordinate search narrows
      colours and strokes (rl.coordinate_search).
    - max_objects None: the slots the task's grids fill, a median 3
      against the 16 the action space would otherwise carry.
    - observation_grid_shape None: the task's largest grid when its
      examples differ in size, which a fixed-shape buffer needs - 133 of the
      262 shape-preserving training tasks never reached their first step
      without it.
    - coordinate_shape None: the task's largest grid, under coordinates.

    observation_space_elements is always passed through observation_for,
    which only drops what the addressing cannot use and adds what it needs;
    and under coordinates the observation is padded to coordinate_shape
    whatever observation_grid_shape says, because the coordinate heads
    score the observed grid's rows and columns, which have to be the action
    space's.

    A search that fails leaves the placeholder's vocabulary as the full one
    rather than taking the pipeline down - training over a wider space is a
    worse run, not a broken one. The search costs seconds to a couple of
    minutes a task against the minutes training takes, and its settings'
    timeout bounds the tail.
    """
    from data.configs.env_configs import MAIN_DIRECTIONS
    from rl.search_hints import (SearchSettings, coordinate_vocabulary,
                                 feasible_from_search, output_colours)

    agent = getattr(task, "agent", None)
    narrowed = dict(rl_config)
    narrowed["addressing"] = (rl_config["addressing"] if _chosen(rl_config, "addressing")
                              else addressing_for(agent))
    narrowed["observation_space_elements"] = observation_for(
        rl_config.get("observation_space_elements") or [],
        narrowed["addressing"])
    search_actions = not _chosen(rl_config, "feasible_actions", PLACEHOLDER_ACTIONS)
    if narrowed["addressing"] == "coordinates":
        if search_actions:
            # Not the object search: its findings are object actions, and
            # none of them belongs to this vocabulary. The coordinate
            # search reads strokes off each training pair's answer and
            # keeps the colours and strokes its covers used.
            from rl.coordinate_search import (CoordinateSearchSettings,
                                              feasible_from_coordinate_search)

            vocabulary = coordinate_vocabulary(
                output_colours(*[subtask.train_out for subtask in task.subtasks]))
            narrowed["feasible_actions"] = vocabulary
            try:
                coordinate_settings = (settings if isinstance(settings, CoordinateSearchSettings)
                                       else CoordinateSearchSettings())
                narrowed["feasible_actions"], _found = feasible_from_coordinate_search(
                    task, vocabulary, coordinate_settings)
            except Exception:  # noqa: BLE001 - a failed search must not fail the run
                pass
        if not _chosen(rl_config, "coordinate_shape"):
            narrowed["coordinate_shape"] = coordinate_shape(task)
        narrowed["observation_grid_shape"] = tuple(narrowed["coordinate_shape"])
        return narrowed
    if not _chosen(rl_config, "max_objects"):
        narrowed["max_objects"] = object_slots(task, rl_config.get("repr_level", 1))
    if not _chosen(rl_config, "observation_grid_shape"):
        narrowed["observation_grid_shape"] = observation_shape(task)
    if search_actions:
        # All four main directions for the search itself: it decides which
        # action types move the grid, and a type that only works southwards
        # is invisible to a search that only tries north and east - the
        # scan's default, which SearchSettings keeps for the hints.
        settings = settings or SearchSettings(directions=tuple(MAIN_DIRECTIONS))
        try:
            narrowed["feasible_actions"] = feasible_from_search(
                task, settings, agent=agent)
        except Exception:  # noqa: BLE001 - a failed search must not fail the run
            pass
    return narrowed


def _rl_training_worker(task: Any, rl_config: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Runs in the child process. Must be a module-level function (not
    nested) since the spawn context pickles the target. Imports
    rl.rl_module lazily, here rather than at module level, so that
    importing rl_job.py itself (e.g. from orchestration) never requires
    torch/stable-baselines3 just to manage a subprocess handle.

    `rl_config` is the parent's, handed over by default_rl_start_fn: a
    spawned child imports data.configs.rl_configs afresh, so reading it
    here gets the file's values and not a notebook's changes to them."""
    from data.configs.rl_configs import load_PPO_config
    from rl.rl_module import RLModule, RlConfig

    if rl_config is None:
        from data.configs.rl_configs import rl_config
    return RLModule(RlConfig(**narrowed_for_task(task, rl_config)),
                    load_PPO_config()).solve(task)


def default_rl_start_fn(task: Any) -> RLJobHandle:
    """Starts RL training (rl.rl_module.RLModule, i.e. rl.training.train_on_task)
    for `task` as a background subprocess, using rl_config as it stands in
    this process when the job starts - edits made to it in a notebook
    included - and the default PPO config. Pass a different rl_start_fn to
    solve_task() to use another config or a previously-trained policy.

    The PPO config is not carried across: it holds a learning-rate schedule
    that is a closure, and a closure does not pickle. The child loads its
    own."""
    import copy
    from functools import partial

    from data.configs.rl_configs import rl_config

    return RLJobHandle(task, partial(_rl_training_worker,
                                     rl_config=copy.deepcopy(rl_config)))
