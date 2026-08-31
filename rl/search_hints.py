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
from dataclasses import dataclass, field

import numpy as np

import rl.mcts as mcts
from data.configs.env_configs import (ACTION_TYPES, AGENT2ACTIONS, ALL_DIRECTIONS,
                                      COLOR_DEPENDENT_ACTIONS, COLORS_MAPPING,
                                      DIRECTION_DEPENDENT_ACTIONS,
                                      DOUBLE_COLOR_DEPENDENT_ACTIONS)
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

    The defaults are the ones the 262-task scan ran with, so an online hint
    and a harvested one describe searches of the same strength. `repeats` is
    1 rather than the scan's 3: three searches per task is a measurement
    budget, not a per-prompt one.
    """
    rollouts: int = 4
    iterations: int = 40
    rounds: int = 1
    keep: float = 0.5
    top_k: int = 5
    episode_len: int = 25
    c: float = 1.414
    repeats: int = 1
    colours: tuple = ("red", "blue")
    directions: tuple = ("N", "E")
    moves: int = 6
    min_gain: int = 5
    skip_solved: bool = False
    partials: int = 3


def build_vocabulary(colours, directions):
    """The action names an env is configured with, generated rather than
    written out: a hand-written name that misses its branch returns the grid
    untouched, which is indistinguishable from a transform that had nothing
    to do."""
    bases = ({a for roster in AGENT2ACTIONS.values() for a in roster}
             | {a for group in ACTION_TYPES.values() for a in group})
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
    names = found["names"]
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

    def watching(self, simulated, action):
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

    mcts.EnvironmentSimulator.simulate_step = watching
    try:
        # stderr as well as stdout: the search draws tqdm bars, and a run
        # that builds one prompt per task would fill a notebook with them.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rollouts = mcts.rollout_preparation(
                env, method="mcts", n_initial_rollouts=settings.rollouts,
                mcts_iterations=settings.iterations, top_k=settings.top_k,
                n_rounds=settings.rounds, keep_fraction=settings.keep,
                min_pool=4, c=settings.c)
    finally:
        mcts.EnvironmentSimulator.simulate_step = original

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


def hints_for(task, settings=None):
    """The hint block for one task, computed now, or None.

    Repeats are merged the way the scan merged them: the best peak, the
    largest gain per action, every distinct solution, the furthest partial
    paths. Independent searches over one task are root parallelisation done
    serially, and this is its merge.
    """
    settings = settings or SearchSettings()
    triple = as_triple(task)
    actions = build_vocabulary(settings.colours, settings.directions)
    merged = {"effective": {}, "solutions": [], "partials": [], "peak": 0.0}
    for _ in range(max(1, settings.repeats)):
        found = search_once(triple, actions, settings)
        merged["peak"] = max(merged["peak"], found["peak"])
        for name, gain in found["effective"].items():
            merged["effective"][name] = max(merged["effective"].get(name, 0), gain)
        for trace in found["solutions"]:
            if trace not in merged["solutions"]:
                merged["solutions"].append(trace)
        for closed, trace in found["partials"]:
            keep_partial(merged["partials"], closed, trace, settings.partials)
    merged["solutions"].sort(key=len)
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
