"""Run a search on one task and say what it found, in words.

The offline path writes these blocks into a file ahead of time; this is the
same thing computed when the task arrives, for a run that wants the hint
without a scan behind it. Both share the renderer below, because the text
in the prompt is the one thing that must not differ between them.

It costs what a search costs - seconds to a few minutes per task, against
milliseconds for the rest of prompt building - so a caller decides when to
pay it. That is why nothing here is wired into PromptBuilder: the search
belongs to the rl layer, the prompt to the subsymbolic one, and the caller
joins them by putting the returned text into the build context.

What the block can say:

- the sequence that solved the task, when the search solved it;
- otherwise the furthest attempt it made, and how far that got;
- the single moves that recovered cells at some point in the search.

Everything is named in the grid's own terms - colours as digits, objects by
colour, size and bounding box - because a prompt that says "blue" beside a
grid of digits asks the model to resolve a reference nothing defines.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace

import numpy as np

import rl.mcts as mcts
from data.configs.env_configs import (ACTION_TYPES, AGENT2ACTIONS, ALL_DIRECTIONS,
                                      COLOR_DEPENDENT_ACTIONS, COLORS_MAPPING,
                                      DIRECTION_DEPENDENT_ACTIONS,
                                      DOUBLE_COLOR_DEPENDENT_ACTIONS,
                                      UNIMPLEMENTED_ACTIONS)
from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.utils import define_feasible_actions

#: colour word -> the digit that word means in a grid. The vocabulary spells
#: colours as names because that is how the transforms are declared; a grid
#: holds digits, and a prompt that mixes the two asks the model to resolve a
#: reference it has no way to resolve.
_DIGIT = {name: digit for digit, name in COLORS_MAPPING.items()}
_DIRECTIONS = set(ALL_DIRECTIONS)

#: One cell fixed moves maximal_intersection by two - it counts
#: 2 * matches - valid - so a gain of 34 is 17 cells.
POINTS_PER_CELL = 2


@dataclass(frozen=True)
class SearchSettings:
    """What the search is given, and what the block is allowed to say.

    The defaults are the ones the 262-task scan ran with - weighted
    playout, 5 rollouts, 50 iterations - because that is the configuration
    measured to reach what the scan's own file knows. On the same twenty
    tasks: one search at these settings carried a hint on 17, where the
    file pooling three searches carried one on 16, and the per-task counts
    of effective actions agree within a few (34 against 34, 19 against 18).
    `repeats` stays 1: three searches per task is a measurement budget, not
    a per-prompt one, and one is evidently enough.
    """
    rollouts: int = 5
    iterations: int = 50
    rounds: int = 1
    keep: float = 0.5
    top_k: int = 5
    episode_len: int = 25
    c: float = 1.414
    repeats: int = 1
    #: None derives the colours from the task's own output grid, which is
    #: the only palette worth painting in. A fixed pair reaches every branch
    #: of every transform - what the scans were built for - but on a task
    #: whose answer needs colour 3 no colour-dependent action can ever help:
    #: 81 of 260 scanned tasks needed a colour outside {1, 2}, and not one
    #: of them was solved.
    colours: tuple | None = None
    directions: tuple = ("N", "E")
    moves: int = 6
    min_gain: int = 5
    skip_solved: bool = False
    partials: int = 3
    #: Which action a playout tries next. 'weighted' draws from the pool
    #: the tree expands over by measured effect; 'default' samples the raw
    #: padded action space, where ~91% of draws name no object at all. The
    #: 262-task scan ran weighted, this ran default until it was measured,
    #: and that alone accounted for most of the gap between what a scan
    #: found and what an online search found on the same task.
    playout: str = "weighted"
    #: Searches to run at once. Repeats share nothing, so in principle a
    #: 60s budget on four cores buys four minutes of search - but measured,
    #: this is the weakest way to spend a budget: two workers took 146s over
    #: six tasks where the same work in line took a fifth of that, and each
    #: worker reached 2 GB during a search (544 MB of that is the imports).
    #: Four searches and four times the iterations reached the same
    #: coverage on 20 tasks, at 37.7s against 16.8s. Spend the budget on
    #: `iterations` first; this is here for a machine with cores and memory
    #: to spare, and defaults to off.
    workers: int = 1
    #: Seconds the whole task may take, 0 for no cap. Bounds repeats
    #: together rather than each on its own.
    budget: int = 0
    #: Seconds one search may take before it is cut short, 0 for no cap.
    #: Measured over 20 tasks, the median search is 2.4s and the slowest is
    #: 165s - so without a cap one task in twenty stalls a run for minutes.
    #: What the search found up to the cut is kept; only the rollouts it
    #: had not returned yet are lost.
    timeout: int = 120


def output_colours(*grids):
    """The colour words these grids contain, in digit order.

    The outputs' palette, not the inputs': painting is only ever useful in a
    colour the answer contains, so this is the exact set worth generating
    colour-dependent actions for - not a heuristic.

    Several grids because the unit that needs a vocabulary is sometimes the
    task rather than one pair. A colour absent from every example's output
    will not be in the test's either, but one that appears in only the
    second example is still the task's - and passing one grid drops it.
    A search over a single pair passes that pair; training over a task
    passes all of its outputs.
    """
    digits = sorted({int(value) for grid in grids
                     for value in np.asarray(grid).ravel()})
    return tuple(COLORS_MAPPING[d] for d in digits if d in COLORS_MAPPING)


def build_vocabulary(colours, directions):
    """The action names an env is configured with, generated rather than
    written out: a hand-written name that misses its branch returns the grid
    untouched, which is indistinguishable from a transform that had nothing
    to do."""
    bases = ({a for roster in AGENT2ACTIONS.values() for a in roster}
             | {a for group in ACTION_TYPES.values() for a in group}) \
        - UNIMPLEMENTED_ACTIONS
    names = []
    for base in sorted(bases):
        if base == "submit":
            continue
        generated = define_feasible_actions(
            [base], list(colours), list(directions), COLOR_DEPENDENT_ACTIONS,
            DOUBLE_COLOR_DEPENDENT_ACTIONS, DIRECTION_DEPENDENT_ACTIONS)
        names.extend(n for n in generated.values() if n != "submit")
    return {0: "submit", **{i + 1: n for i, n in enumerate(names)}}


def make_env(task, actions, episode_len):
    """A fresh env holding one (task_id, input, output) triple."""
    task_id, inp, out = task
    env = ARCGridWorld(max_episode_len=episode_len, feasible_actions=actions,
                       reward_approach=2, repr_level=1, input_pattern="start",
                       observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask(f"{task_id}_0", inp, out))
    return env


def replays(task, sequence, actions, episode_len=25):
    """Does this sequence reach the target in an env of its own."""
    return reached(task, sequence, actions, episode_len) == \
        int(make_env(task, actions, episode_len).target_int)


def reached(task, sequence, actions, episode_len=25):
    """The intersection this sequence ends on, or None if it raised."""
    env = make_env(task, actions, episode_len)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for step in sequence:
                env.step(np.asarray(step))
    except Exception:
        return None
    return int(env.max_int)


def minimise(task, sequence, actions, episode_len=25, goal=None):
    """The sequence with every removable step removed.

    Greedy and left to right: one pass reaches a sequence no single removal
    shortens, which is the property that matters.

    `goal` is what a shortened sequence must still reach. Left out, that is
    the target. A path that never solved has no target to hold to and needs
    the intersection it did reach named instead - otherwise every step is
    removable and the whole path disappears.
    """
    body = list(sequence)
    index = 0
    while index < len(body):
        trial = body[:index] + body[index + 1:]
        if goal is None:
            good = replays(task, trial, actions, episode_len)
        else:
            landed = reached(task, trial, actions, episode_len)
            good = landed is not None and landed >= goal
        if good:
            body = trial
        else:
            index += 1
    return tuple(body)


def distinct(sequences):
    """The recorded sequences as hashable tuples, shortest first."""
    return sorted({tuple(tuple(step) for step in seq) for seq in sequences},
                  key=len)


def readable(name):
    """`blue_emission_with_red_object_recolor_E` as something a reader can
    hold: the verb, then the arguments it was given.

    Arguments are listed, not woven into a sentence. Which colour plays
    which role differs per transform, and a rendering that guesses reads
    fluently while saying the wrong thing.
    """
    verb, colours, directions = [], [], []
    for token in name.split("_"):
        if token in _DIGIT:
            colours.append(str(_DIGIT[token]))
        elif token in _DIRECTIONS:
            directions.append(token)
        else:
            verb.append(token)
    arguments = []
    if colours:
        arguments.append(f"colour{'s' if len(colours) > 1 else ''} "
                         + ", ".join(colours))
    if directions:
        arguments.append("direction " + ", ".join(directions))
    text = " ".join(verb)
    return f"{text} ({'; '.join(arguments)})" if arguments else text


def describe_object(obj):
    """Which object a step was applied to, in the grid's own terms."""
    colours = "/".join(str(int(c)) for c in obj.color_numbers)
    return (f"colour {colours}, {obj.size} cells, rows {obj.min_i}-{obj.max_i}, "
            f"cols {obj.min_j}-{obj.max_j}")


def render_steps(task, sequence, names, actions, episode_len=25):
    """A sequence as lines, each naming the object its step touched.

    The heads after the transform are object indices, and objects are
    recomputed after every step - index 2 at step three is not the object
    index 2 named at step one - so the only way to say what a step was
    applied to is to walk the sequence and look.
    """
    env = make_env(task, actions, episode_len)
    lines = []
    with contextlib.redirect_stdout(io.StringIO()):
        for step in sequence:
            objects = env.objects
            index = int(step[1])
            target = (describe_object(objects[index]) if index < len(objects)
                      else "an empty object slot")
            lines.append(f"{readable(names[str(step[0])])} on {target}")
            env.step(np.asarray(step))
    return lines


def names_for(found, task_id):
    """The action table this task's indices are written in.

    Per task where the search derived a vocabulary from the task's own
    palette, otherwise the one table a whole scan shares. Reading a trace
    against the wrong table renames every action in it, and the result looks
    perfectly plausible.
    """
    return (found.get("names_by_task") or {}).get(task_id) or found["names"]


def render_block(task, found, actions, moves=6, min_gain=5, episode_len=25,
                 skip_solved=False):
    """The hint block for one task, or None when the search found nothing.

    `found` is what one search (or a pooled scan) reported for this task:
    names, solutions, partials, effective.

    Nothing is rendered rather than "the search found nothing" on purpose:
    a block that is sometimes empty teaches the reader to expect one, and an
    absent block is the honest form of having nothing to say.

    `skip_solved` drops the verified solving sequence, keeping the moves
    list. On a task the search solved, that sequence is the answer, and an
    arm measuring whether a hint helps would be measuring whether the model
    can follow a recipe on those tasks and something else on the rest.
    """
    task_id = task[0]
    names = names_for(found, task_id)
    lines = []
    solved = found["solutions"].get(task_id) or []
    partial = found["partials"].get(task_id) or []
    effective = found["effective"].get(task_id) or {}
    if solved and not skip_solved:
        best = min(distinct(solved), key=len)
        body = [step for step in best if names[str(step[0])] != "submit"]
        body = minimise(task, body, actions, episode_len)
        lines.append("An automated search over the first training pair "
                     "reproduced the output exactly with:")
        lines += [f"  {i + 1}. {line}" for i, line in
                  enumerate(render_steps(task, body, names, actions, episode_len))]
    elif partial:
        progress, sequence = partial[0][0], partial[0][1]
        body = [step for step in sequence if names[str(step[0])] != "submit"]
        peak = reached(task, body, actions, episode_len)
        if peak is not None:
            body = minimise(task, body, actions, episode_len, goal=peak)
        lines.append(f"An automated search over the first training pair did not "
                     f"reproduce the output. Its best attempt reached "
                     f"{progress:.0%} of the target cells with:")
        lines += [f"  {i + 1}. {line}" for i, line in
                  enumerate(render_steps(task, body, names, actions, episode_len))]
    # A gain is measured against whatever state the search was standing in,
    # not against the input grid, so it says "this move fixed cells somewhere
    # in the search" and not "this move gets you N cells closer than doing
    # nothing". Small gains are noise at that reading - on the 262-task scan
    # a floor of one cell admitted 96% of tasks and a median of 38 moves
    # each - so the block carries only moves worth naming.
    ranked = [(name, gain) for name, gain in
              sorted(effective.items(), key=lambda kv: (-kv[1], kv[0]))
              if gain >= min_gain * POINTS_PER_CELL][:moves]
    if ranked:
        opening = ("Single moves that recovered cells at some point in that "
                   "search:" if lines else
                   "An automated search over the first training pair did not "
                   "reproduce the output. Single moves that recovered cells at "
                   "some point in it:")
        lines.append(opening)
        lines += [f"  {readable(name)} (up to {gain // POINTS_PER_CELL} cells)"
                  for name, gain in ranked]
    return "\n".join(lines) if lines else None


def keep_partial(kept, closed, trace, limit):
    """Hold the `limit` furthest non-solving paths, best first; ties to the
    shorter one, since two paths that reached the same place are not equally
    worth showing anybody."""
    if any(trace == held for _, held in kept):
        return kept
    kept.append([closed, trace])
    kept.sort(key=lambda pair: (-pair[0], len(pair[1])))
    del kept[limit:]
    return kept


class SearchTimedOut(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Cut the block below short after `seconds`, where that is possible.

    SIGALRM only arrives on the main thread, so a search running in a
    worker thread is left uncapped rather than silently unprotected in a
    way that looks capped.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def ring(*_):
        raise SearchTimedOut()

    previous = signal.signal(signal.SIGALRM, ring)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _rescued(simulator, task, actions, settings):
    """The solutions a cut search had found but not yet verified.

    A fresh env, because the one the search ran in is wherever the timeout
    left it, and replay_solution steps what it is given. A candidate that
    does not replay is dropped, which is the same standard the search's own
    verification holds them to.
    """
    if simulator is None or not getattr(simulator, "solutions", None):
        return []
    env = make_env(task, actions, settings.episode_len)
    rescued = []
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        for candidate in simulator.solutions:
            try:
                replayed = mcts.replay_solution(env, candidate)
            except Exception:
                continue
            if replayed is not None:
                rescued.append(replayed)
    return rescued


def search_once(task, actions, settings):
    """One search, reported as the material a block is rendered from.

    The peak and the per-action gains come from watching every simulated
    step, not only the ones a rollout kept: the tree explores far more than
    it returns, and on the full scan most of what was found was found inside
    a playout.
    """
    env = make_env(task, actions, settings.episode_len)
    base, target = int(env.max_int), int(env.target_int)
    state = {"peak": base, "effective": {}}
    original = mcts.EnvironmentSimulator.simulate_step
    original_init = mcts.EnvironmentSimulator.__init__

    def watching(self, simulated, action):
        # The simulator is reachable only from here. It holds the solutions
        # the playouts found, which rollout_preparation verifies in a loop
        # after its own is done - so a search cut by the timeout never
        # reaches them, and they are most of what a search finds.
        state["simulator"] = self
        result = original(self, simulated, action)
        landed = result[0]["max_int"]
        if landed > state["peak"]:
            state["peak"] = landed
        gain = landed - simulated["max_int"]
        if gain > 0:
            name = actions.get(int(np.asarray(action).reshape(-1)[0]))
            if name is not None:
                state["effective"][name] = max(state["effective"].get(name, 0),
                                               int(gain))
        return result

    def weighted_init(self, env, actions=None, policy=None):
        original_init(self, env, actions=actions, policy=policy)
        self.policy = mcts.PlayoutPolicy(self.all_actions, temperature=0.2,
                                         floor=0.02)

    mcts.EnvironmentSimulator.simulate_step = watching
    if settings.playout == "weighted":
        mcts.EnvironmentSimulator.__init__ = weighted_init
    rollouts = []
    try:
        # stderr as well as stdout: the search draws tqdm bars, and a run
        # that builds one prompt per task would fill a notebook with them.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with time_limit(settings.timeout):
                rollouts = mcts.rollout_preparation(
                    env, method="mcts", n_initial_rollouts=settings.rollouts,
                    mcts_iterations=settings.iterations, top_k=settings.top_k,
                    n_rounds=settings.rounds, keep_fraction=settings.keep,
                    min_pool=4, c=settings.c)
    except SearchTimedOut:
        # The peak and the per-action gains were recorded as the search ran,
        # so a cut search still has something to say. Its solutions were not
        # so lucky: they live in the simulator until rollout_preparation
        # verifies them in a loop after its own, which a cut search never
        # reaches - so a task the search actually solved came back solved
        # with no sequence to show for it, or not solved at all.
        #
        # Measured in the budget sweep: at 640 iterations the median task
        # ran the full 600s timeout, and tasks that were solved at 40 and
        # 160 came back peak 1.0 and solved false. That column was counting
        # timeouts, not search.
        #
        # Verified here rather than trusted, exactly as the loop that was
        # missed would have: a sequence found in a playout is a claim about
        # the simulator, and the replay is what makes it a solution.
        rollouts = _rescued(state.get("simulator"), task, actions, settings)
    finally:
        mcts.EnvironmentSimulator.simulate_step = original
        mcts.EnvironmentSimulator.__init__ = original_init

    solutions, partials = [], []
    for rollout in rollouts:
        trace = [[int(x) for x in np.asarray(a).reshape(-1)]
                 for a in rollout["actions"]]
        if not trace:
            continue
        span = rollout["target_int"] - rollout["base_int"]
        closed = ((rollout["max_int"] - rollout["base_int"]) / span
                  if span else 0.0)
        if rollout["solved"]:
            if trace not in solutions:
                solutions.append(trace)
        elif closed > 0:
            keep_partial(partials, closed, trace, settings.partials)
    solutions.sort(key=len)
    span = target - base
    return {"peak": (state["peak"] - base) / span if span else 0.0,
            "effective": state["effective"],
            "solutions": solutions, "partials": partials}


def as_triple(task):
    """(task_id, input, output) from an ARCTask, an ARCSubtask or a triple.

    The search runs on the first training pair, which is what the scan used
    and what the block says it looked at.
    """
    if isinstance(task, (tuple, list)) and len(task) == 3:
        return tuple(task)
    subtasks = getattr(task, "subtasks", None)
    if subtasks:
        first = subtasks[0]
        return (str(getattr(task, "label", "task")), first.train_inp, first.train_out)
    return (str(getattr(task, "label", "task")), task.train_inp, task.train_out)


#: One pool for the process, not one per task. A spawned worker re-imports
#: torch, which costs a second or two - paid once here, paid per task if the
#: pool were built inside hints_for, where the searches themselves take
#: about as long.
_POOL = {"workers": 0, "pool": None}


def _pool(workers):
    """The shared process pool, or None when this process cannot have one.

    A daemonic worker cannot start children of its own, so a run that is
    already inside a pool falls back to searching in line rather than
    failing.
    """
    if workers <= 1:
        return None
    if _POOL["pool"] is None or _POOL["workers"] != workers:
        shutdown_pool()
        try:
            _POOL["pool"] = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            _POOL["workers"] = workers
        except (AssertionError, ValueError, OSError):
            return None
    return _POOL["pool"]


def shutdown_pool():
    """Let go of the workers. Worth calling at the end of a run; each one
    holds an interpreter with torch imported."""
    if _POOL["pool"] is not None:
        _POOL["pool"].shutdown(wait=False, cancel_futures=True)
    _POOL["pool"], _POOL["workers"] = None, 0


def _search_in_worker(payload):
    """One search, addressed by value so it can cross a process boundary."""
    triple, colours, directions, settings = payload
    return search_once(triple, build_vocabulary(colours, directions), settings)


def merge_found(merged, found, partials_limit):
    """Fold one search into the running answer for this task."""
    merged["peak"] = max(merged["peak"], found["peak"])
    for name, gain in found["effective"].items():
        merged["effective"][name] = max(merged["effective"].get(name, 0), gain)
    for trace in found["solutions"]:
        if trace not in merged["solutions"]:
            merged["solutions"].append(trace)
    for closed, trace in found["partials"]:
        keep_partial(merged["partials"], closed, trace, partials_limit)
    return merged


def search_task(task, settings=None):
    """Everything the searches found for one task, merged.

    Split out of hints_for because what a search found and how it is worded
    are different questions, and a measurement wants the first without the
    second: peak, solutions, partial paths, effective actions - plus the
    vocabulary they are named in.

    Repeats are merged the way the scan merged them: the best peak, the
    largest gain per action, every distinct solution, the furthest partial
    paths. Independent searches over one task are root parallelisation done
    serially, and this is its merge.

    Three things bound the work, because a per-prompt search cannot be given
    a measurement budget:

    - `budget` caps the whole task, not each search. Repeats stop being
      started once it is spent, and the last one gets whatever is left, so
      `repeats=3, budget=60` is at most a minute rather than at most three.
    - a solved task stops the loop. Another search cannot improve on a
      sequence that already reaches the target.
    - `timeout` still caps a single search, and a cut one keeps what it
      found.
    """
    settings = settings or SearchSettings()
    triple = as_triple(task)
    colours = settings.colours or output_colours(triple[2])
    actions = build_vocabulary(colours, settings.directions)
    merged = {"effective": {}, "solutions": [], "partials": [], "peak": 0.0}
    started = time.perf_counter()
    repeats = max(1, settings.repeats)
    pool = _pool(settings.workers) if repeats > 1 else None
    if pool is not None:
        # All the repeats at once: they share nothing, so a budget of 60s on
        # four cores buys four minutes of search rather than one. Whatever
        # has not finished when the budget runs out is dropped - the
        # searches that did finish are a smaller sample, not a wrong one.
        payload = (triple, tuple(colours), tuple(settings.directions), settings)
        try:
            futures = [pool.submit(_search_in_worker, payload)
                       for _ in range(repeats)]
            done, pending = wait(futures, timeout=settings.budget or None)
            for future in pending:
                future.cancel()
            for future in done:
                try:
                    merge_found(merged, future.result(), settings.partials)
                except Exception:
                    continue
        except BrokenProcessPool:
            # Two causes, both seen here. A caller running as a script
            # without an `if __name__ == "__main__"` guard: spawn re-imports
            # the main module in the child, which re-runs it, which spawns
            # again. Or memory - a worker reached 2 GB during a search
            # against 544 MB of imports, so four of them beside a loaded
            # model is 8 GB of search alone. Either way the pool is
            # unusable from here, so drop it and search in this process: a
            # hint is worth less than the run it would otherwise kill
            # thirteen hours in.
            print("search hints: the worker pool broke, searching in line "
                  "(a script calling this needs an "
                  "`if __name__ == \"__main__\"` guard)")
            shutdown_pool()
            pool = None
    if pool is None:
        for attempt in range(repeats):
            if attempt and settings.budget:
                left = settings.budget - (time.perf_counter() - started)
                if left <= 1:
                    break
                settings = replace(settings,
                                   timeout=int(min(settings.timeout or left, left)))
            merge_found(merged, search_once(triple, actions, settings),
                        settings.partials)
            if merged["solutions"]:
                break
    merged["solutions"].sort(key=len)
    merged["actions"] = actions
    merged["triple"] = triple
    merged["seconds"] = time.perf_counter() - started
    return merged


def hints_for(task, settings=None):
    """The hint block for one task, computed now, or None."""
    settings = settings or SearchSettings()
    merged = search_task(task, settings)
    triple, actions = merged["triple"], merged["actions"]
    task_id = triple[0]
    shaped = {"names": {str(i): n for i, n in actions.items()},
              "solutions": {task_id: merged["solutions"]},
              "partials": {task_id: merged["partials"]},
              "effective": {task_id: merged["effective"]}}
    return render_block(triple, shaped, actions, settings.moves,
                        settings.min_gain, settings.episode_len,
                        settings.skip_solved)


@dataclass
class HintCache:
    """`hints_for` with the answer remembered per task.

    A run builds the prompt for one task more than once - a retry, a second
    arm, a resumed checkpoint - and each rebuild would otherwise pay for a
    fresh search whose result is random anyway, so the same task would carry
    a different hint each time.
    """
    settings: SearchSettings = field(default_factory=SearchSettings)
    cache: dict = field(default_factory=dict)

    def __call__(self, task):
        key = str(as_triple(task)[0])
        if key not in self.cache:
            self.cache[key] = hints_for(task, self.settings)
        return self.cache[key]
