#!/usr/bin/env python3
"""Turn the shard files of a search scan into traces worth training on.

The scan records, per task, every action sequence that reached the target.
That record is a claim made inside the search: the tree carries state, the
sequence was cut at its peak, and half of its steps can be scaffolding the
search walked through on the way. Two things have to happen before any of
it is a dataset.

Verified. Each sequence is replayed in a fresh env built the same way the
scan built it. A sequence that does not land on the target outside the
search that produced it is dropped, and the count of dropped ones is
printed - silently keeping them would put wrong labels in front of a
learner.

Minimal. Steps are removed one at a time, keeping the removal whenever the
shortened sequence still solves. "It changed nothing" cannot be read off
the intersection - a step can rebuild the grid elsewhere and leave the
count where it was - so removal and replay is the only honest test. On the
262-task scan this halved the step count, 735 to 374, and collapsed 219
recorded sequences into 69 distinct minimal ones with a median length of 2.

Pooling shards is only sound when they were scanned with the same
vocabulary: action 47 is a name, not a number, and two shards built with
different --colours disagree about which. The action_names tables are
compared and a mismatch stops the run.

--agents asks the other question the harvest was gathered for: whether the
actions that turned out to work on a task point at the agent that owns
them. On the full scan they do not - the best scoring lands at or below
"always answer with the commonest agent", and the effective sets of two
tasks with the same agent resemble each other exactly as much as two
tasks picked at random. The flag is here so that stays measurable when the
rosters change.

Nothing here is checked in. The traces of one scan are 69 sequences over 24
tasks, most of them one or two actions on a task simple enough that a scan
finds it again; the shard files are the thing worth keeping, and this runs
over them in seconds whenever the traces are wanted.

Usage:
    # what produces the shards, ~15 hours over five machines, one span each
    python scripts/compare_reward_approaches.py --approaches 2 --tasks 0-53 \
        --repeats 3 --rounds 1 --out shard_0.json
    # this, seconds
    python scripts/harvest_traces.py shard_*.json --out traces.json
    python scripts/harvest_traces.py shard_*.json --agents

Note the vocabulary a trace is written in. `build_actions(["red", "blue"],
["N", "E"])` is what the scan uses - two colours and two directions reach
every branch of every transform, where all ten and all eight give 2926
names - so an index re-read against a differently built vocabulary means
something else entirely. The table travels with the traces in `--out`.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.arc_env import ARCGridWorld  # noqa: E402
from rl.arc_task import ARCSubtask  # noqa: E402
from data.configs.env_configs import (AGENT2ACTIONS, ALL_DIRECTIONS,  # noqa: E402
                                      COLOR_DEPENDENT_ACTIONS, COLORS_MAPPING,
                                      DIRECTION_DEPENDENT_ACTIONS,
                                      DOUBLE_COLOR_DEPENDENT_ACTIONS)
from rl.utils import define_feasible_actions  # noqa: E402
from scripts.compare_reward_approaches import build_actions, load_tasks  # noqa: E402


def pool(paths, approach="2"):
    """One view of several shard files, or an error naming the disagreement.

    Shards are scanned separately precisely so they can be pooled, and the
    thing that silently breaks the pooling is a vocabulary difference: the
    tables map the same index to different names, every action label after
    it is wrong, and nothing about the merged file looks unusual.
    """
    per_task, effective, solutions, partials = {}, {}, {}, {}
    names, source, span = None, None, None
    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        section = data["approaches"][approach]
        covered = data.get("span")
        if covered:
            span = covered if span is None else [min(span[0], covered[0]),
                                                 max(span[1], covered[1])]
        table = section["action_names"]
        if names is None:
            names, source = table, path
        elif table != names:
            differing = sorted(k for k in set(table) | set(names)
                               if table.get(k) != names.get(k))
            raise SystemExit(
                f"{path} and {source} were scanned with different action "
                f"vocabularies ({len(differing)} indices differ, first "
                f"{differing[:3]}) - they cannot be pooled")
        per_task.update(section["per_task"])
        effective.update(section["effective_actions"])
        solutions.update(section["solutions"])
        # Shards scanned before partial paths were kept simply have none.
        partials.update(section.get("partial_paths") or {})
    return {"per_task": per_task, "effective": effective,
            "solutions": solutions, "partials": partials,
            "names": names or {}, "span": tuple(span or (0, 0))}


def distinct(sequences):
    """The recorded sequences as hashable tuples, shortest first."""
    return sorted({tuple(tuple(step) for step in seq) for seq in sequences},
                  key=len)


def make_env(task, actions, episode_len):
    task_id, inp, out = task
    env = ARCGridWorld(max_episode_len=episode_len, feasible_actions=actions,
                       reward_approach=2, repr_level=1, input_pattern="start",
                       observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask(f"{task_id}_0", inp, out))
    return env


def replays(task, sequence, actions, episode_len=25):
    """Does this sequence still reach the target in an env of its own."""
    env = make_env(task, actions, episode_len)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for step in sequence:
                env.step(np.asarray(step))
    except Exception:
        return False
    return int(env.max_int) == int(env.target_int)


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

    Greedy and left to right: one pass is enough to reach a sequence no
    single removal shortens, which is the property that matters. It is not
    the globally shortest solution and does not claim to be.

    `goal` is what a shortened sequence has to still reach. Left out, that
    is the target: a solving trace stays a solving trace. A path that never
    solved has no target to hold to and needs the intersection it did reach
    named instead - otherwise every step is removable and the whole path
    disappears. Twenty-five wandering steps come back as the two or three
    that did the work either way.
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


def harvest(solutions, names, tasks, actions, episode_len=25):
    """Verified, minimal traces per task, plus what happened on the way."""
    by_id = {task[0]: task for task in tasks}
    kept, stats = {}, collections.Counter()
    submit = {index for index, name in names.items() if name == "submit"}
    for task_id, recorded in sorted(solutions.items()):
        if not recorded:
            continue
        if task_id not in by_id:
            stats["outside the span"] += 1
            continue
        task = by_id[task_id]
        minimal = set()
        for sequence in distinct(recorded):
            stats["recorded"] += 1
            if not replays(task, sequence, actions, episode_len):
                stats["did not replay"] += 1
                continue
            # submit ends the episode without touching the grid, so it is
            # not part of what the trace teaches.
            body = [step for step in sequence if str(step[0]) not in submit]
            stats["steps before"] += len(body)
            trimmed = minimise(task, body, actions, episode_len)
            stats["steps after"] += len(trimmed)
            minimal.add(trimmed)
        if minimal:
            kept[task_id] = sorted(minimal, key=len)
    stats["tasks"] = len(kept)
    stats["minimal"] = sum(len(v) for v in kept.values())
    return kept, stats


#: colour word -> the digit that word means in a grid. The vocabulary spells
#: colours as names because that is how the transforms are declared; a grid
#: holds digits, and a prompt that mixes the two asks the model to resolve a
#: reference it has no way to resolve.
_DIGIT = {name: digit for digit, name in COLORS_MAPPING.items()}
_DIRECTIONS = set(ALL_DIRECTIONS)

#: One cell fixed moves maximal_intersection by two - it counts
#: 2 * matches - valid - so a gain of 34 is 17 cells.
POINTS_PER_CELL = 2


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


def render_block(task, pooled, actions, moves=6, min_gain=5, episode_len=25,
                 skip_solved=False):
    """The hint block for one task, or None when the search found nothing.

    Nothing is rendered rather than "the search found nothing" on purpose:
    a block that is sometimes empty teaches the reader to expect one, and
    an absent block is the honest form of having nothing to say.

    `skip_solved` drops the verified solving sequence, keeping the moves
    list. On a task the search solved, that sequence is the answer, and an
    arm measuring whether a hint helps would be measuring whether the model
    can follow a recipe on those tasks and something else on the rest -
    two experiments averaged into one number.
    """
    task_id = task[0]
    names = pooled["names"]
    lines = []
    solved = pooled["solutions"].get(task_id) or []
    partial = pooled["partials"].get(task_id) or []
    effective = pooled["effective"].get(task_id) or {}
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
    # not against the input grid, so it says "this move fixed cells
    # somewhere in the search" and not "this move gets you N cells closer
    # than doing nothing". Small gains are noise at that reading - on the
    # 262-task scan a floor of one cell admitted 96% of tasks and a median
    # of 38 moves each - so the block carries only moves worth naming.
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


def agent_names(colours, directions):
    """Each agent's roster as the generated names a search reports."""
    out = {}
    for agent, bases in AGENT2ACTIONS.items():
        owned = set()
        for base in bases:
            if base == "submit":
                continue
            generated = define_feasible_actions(
                [base], colours, directions, COLOR_DEPENDENT_ACTIONS,
                DOUBLE_COLOR_DEPENDENT_ACTIONS, DIRECTION_DEPENDENT_ACTIONS)
            owned |= {n for n in generated.values() if n != "submit"}
        out[agent] = owned
    return out


def labelled_tasks(effective, candidates):
    """Scanned tasks whose labelled agent owns actions at all.

    Five of the nine labelled agents are subsymbolic and own nothing to
    intersect with, so this can only speak about the other four - and a
    rank out of four is not a rank out of nine.
    """
    with open(REPO_ROOT / "data/datasets/ARC/idx2agent.pkl", "rb") as handle:
        idx2agent = pickle.load(handle)
    with open(REPO_ROOT / "data/datasets/ARC/training_challenges.json") as handle:
        train = json.load(handle)
    with open(REPO_ROOT / "data/datasets/ARC/evaluation_challenges.json") as handle:
        evaluation = json.load(handle)
    order = {key: index for index, key in enumerate(list(train) + list(evaluation))}
    return {task: idx2agent[order[task]] for task in effective
            if order.get(task) in idx2agent and idx2agent[order[task]] in candidates}


def agent_ranks(effective, truth, rosters, candidates, threshold, score):
    """Where the true agent lands when agents are ranked by the effective set."""
    ranks, first = [], collections.Counter()
    for task, agent in truth.items():
        found = {name: gain for name, gain in effective[task].items()
                 if gain >= threshold}
        scores = {a: score(found, rosters[a]) for a in candidates}
        if max(scores.values(), default=0) <= 0:
            continue
        order = sorted(candidates, key=lambda a: (-scores[a], a))
        first[order[0]] += 1
        ranks.append(order.index(agent) + 1)
    return np.array(ranks), first


def report_agents(effective, colours, directions):
    rosters = agent_names(colours, directions)
    candidates = [a for a in rosters if a != "connector_extended"]
    truth = labelled_tasks(effective, candidates)
    if not truth:
        print("no labelled task in these shards owns actions - nothing to rank")
        return
    spread = collections.Counter(truth.values())
    print("\n=== do the effective actions name the agent ===")
    print(f"{len(truth)} labelled tasks whose agent owns actions, "
          f"ranked over {len(candidates)}: {spread.most_common()}")
    scores = {
        "count": lambda found, own: sum(1 for n in own if n in found),
        "sum of gains": lambda found, own: sum(g for n, g in found.items() if n in own),
        "coverage": lambda found, own: sum(1 for n in own if n in found) / len(own),
    }
    print(f"{'score':14s} {'threshold':>12s} {'n':>4s} {'top1':>6s} {'top2':>6s} "
          f"{'mean rank':>10s}  first pick")
    for label, score in scores.items():
        for threshold in (1, 10, 25):
            ranks, first = agent_ranks(effective, truth, rosters, candidates,
                                       threshold, score)
            if not len(ranks):
                continue
            common = ", ".join(f"{a} {n}" for a, n in first.most_common(2))
            print(f"{label:14s} {threshold:9d} c {len(ranks):4d} "
                  f"{100 * (ranks <= 1).mean():5.1f}% {100 * (ranks <= 2).mean():5.1f}% "
                  f"{ranks.mean():10.2f}  {common}")
    top = spread.most_common(1)[0]
    print(f"  to beat: random is 25.0% top1 over four, and answering "
          f"{top[0]} every time is {100 * top[1] / len(truth):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path,
                        help="JSON files written by compare_reward_approaches --out")
    parser.add_argument("--approach", default="2",
                        help="which reward approach's section to read")
    parser.add_argument("--dataset", default="ARC")
    parser.add_argument("--tasks", default=None,
                        help="span the shards cover, A-B; taken from the files if omitted")
    parser.add_argument("--colours", nargs="+", default=["red", "blue"])
    parser.add_argument("--directions", nargs="+", default=["N", "E"])
    parser.add_argument("--episode-len", type=int, default=25)
    parser.add_argument("--out", type=Path, help="write the minimal traces as JSON")
    parser.add_argument("--prompt-block", type=Path,
                        help="write task id -> hint text, ready for a prompt resolver")
    parser.add_argument("--moves", type=int, default=6,
                        help="how many single moves a hint block lists")
    parser.add_argument("--min-gain", type=int, default=5,
                        help="cells a single move must have recovered to be listed")
    parser.add_argument("--skip-solved", action="store_true",
                        help="leave the verified solving sequence out of the "
                             "blocks, keeping the moves list - on a solved task "
                             "that sequence is the answer")
    parser.add_argument("--agents", action="store_true",
                        help="also score the effective actions against the agent rosters")
    args = parser.parse_args()

    pooled = pool(args.shards, args.approach)
    per_task, effective = pooled["per_task"], pooled["effective"]
    solutions, names, span = pooled["solutions"], pooled["names"], pooled["span"]
    print(f"{len(args.shards)} shards, {len(per_task)} tasks scored")
    solved = sorted(t for t, v in per_task.items() if v >= 1.0)
    moved = [t for t, v in per_task.items() if v > 0]
    values = np.array(list(per_task.values()))
    print(f"  moved at all    {len(moved):4d}  ({100 * len(moved) / len(per_task):.0f}%)")
    print(f"  solved outright {len(solved):4d}  "
          f"({100 * len(solved) / len(per_task):.1f}%)")
    print(f"  peak progress   mean {values.mean():.3f}  median {np.median(values):.3f}")

    if args.tasks:
        first, _, last = args.tasks.partition("-")
        span = (int(first), int(last))
    tasks, _ = load_tasks(args.dataset, span)
    print(f"  span {span[0]}-{span[1]}, {len(tasks)} tasks in it")
    actions = build_actions(args.colours, args.directions)
    if set(names.values()) - set(actions.values()):
        raise SystemExit("the shards name actions this vocabulary does not "
                         "contain - rerun with the --colours and --directions "
                         "the scan used")

    kept, stats = harvest(solutions, names, tasks, actions, args.episode_len)
    print("\n=== traces ===")
    print(f"  {stats['recorded']} recorded, {stats['did not replay']} did not "
          f"replay in a fresh env")
    print(f"  {stats['steps before']} steps before minimising, "
          f"{stats['steps after']} after")
    print(f"  {stats['minimal']} distinct minimal traces over {stats['tasks']} tasks")
    lengths = collections.Counter(len(seq) for v in kept.values() for seq in v)
    print(f"  lengths {dict(sorted(lengths.items()))}")
    for task_id, sequences in sorted(kept.items()):
        shortest = ", ".join(names.get(str(step[0]), str(step[0]))
                             for step in sequences[0])
        print(f"    {task_id} {len(sequences):2d} distinct, shortest: {shortest}")

    missing = sorted(set(solved) - set(kept))
    if missing:
        print(f"  solved in a playout but no trace returned: {missing}")

    if args.out:
        payload = {"action_names": names,
                   "traces": {t: [[list(step) for step in seq] for seq in v]
                              for t, v in kept.items()}}
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")

    if args.prompt_block:
        blocks = {}
        for task in tasks:
            text = render_block(task, pooled, actions, args.moves,
                                args.min_gain, args.episode_len,
                                args.skip_solved)
            if text:
                blocks[task[0]] = text
        args.prompt_block.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
        covered = 100 * len(blocks) / max(len(tasks), 1)
        print("\n=== prompt blocks ===")
        giving = sum(1 for text in blocks.values()
                     if "reproduced the output exactly" in text)
        print(f"  {len(blocks)} of {len(tasks)} tasks carry one ({covered:.0f}%) - "
              f"the rest get no block at all rather than an empty one")
        print(f"  {giving} of them hand over a verified solution; read those "
              f"apart from the rest, or rebuild with --skip-solved")
        # What the floor buys, since it is the one knob that decides whether
        # a task is spoken about at all.
        for floor in (1, 5, 10, 20):
            with_block = sum(1 for task in tasks
                             if render_block(task, pooled, actions, args.moves,
                                             floor, args.episode_len,
                                             args.skip_solved))
            print(f"    at >= {floor:2d} cells: {with_block} tasks")
        sizes = [len(text) for text in blocks.values()]
        if sizes:
            print(f"  {int(np.median(sizes))} characters on the median task, "
                  f"{max(sizes)} at most")
            sample = sorted(blocks)[0]
            print(f"\n  {sample}:\n" + "\n".join(f"    {line}" for line
                                                 in blocks[sample].splitlines()))
        print(f"\nwrote {args.prompt_block}")

    if args.agents:
        report_agents(effective, args.colours, args.directions)


if __name__ == "__main__":
    main()
