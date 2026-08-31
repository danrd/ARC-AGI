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
from data.configs.env_configs import (AGENT2ACTIONS,  # noqa: E402
                                      COLOR_DEPENDENT_ACTIONS,
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
    per_task, effective, solutions = {}, {}, {}
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
    return per_task, effective, solutions, (names or {}), tuple(span or (0, 0))


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


def minimise(task, sequence, actions, episode_len=25):
    """The sequence with every removable step removed.

    Greedy and left to right: one pass is enough to reach a sequence no
    single removal shortens, which is the property that matters. It is not
    the globally shortest solution and does not claim to be.
    """
    body = list(sequence)
    index = 0
    while index < len(body):
        trial = body[:index] + body[index + 1:]
        if replays(task, trial, actions, episode_len):
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
    parser.add_argument("--agents", action="store_true",
                        help="also score the effective actions against the agent rosters")
    args = parser.parse_args()

    per_task, effective, solutions, names, span = pool(args.shards, args.approach)
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

    if args.agents:
        report_agents(effective, args.colours, args.directions)


if __name__ == "__main__":
    main()
