#!/usr/bin/env python3
"""Compare reward approaches for MCTS on something they can both be measured in.

Reward cannot do it. total_reward is denominated in whichever
reward_approach the env was built with, and the approaches do not share a
scale - approach 1 runs -4..+4 across the milestones where approach 2 runs
-4..+10 - so a search that got further can score lower. What is comparable
is the intersection with the target, the same count of cells in every
approach, reported as the fraction of the distance closed:

    (best max_int - base_int) / (target_int - base_int)

0.0 is the grid as it started, 1.0 is solved.

Scored by the best intersection reached at ANY point in the search, not by
where rollouts ended. A search looks for a path and keeps the best prefix
of one, so a rollout that touched the target at step seven and wandered off
by step twenty-five is not a failure - and the endpoint metric records it
as one. The endpoint is still reported, second, because the gap between the
two says how much the search finds and then loses. Measured over 12 tasks
the gap is large: peak +0.186 against endpoint +0.033 for the default
playout.

Usage:
    python scripts/compare_reward_approaches.py                 # 36 tasks, ~30 min
    python scripts/compare_reward_approaches.py --tasks 200 --repeats 3
    python scripts/compare_reward_approaches.py --approaches 1 2 3
    python scripts/compare_reward_approaches.py --out results.json

Two things worth knowing before reading a run of this.

Sample size. On 12 tasks approach 2 led approach 1 (0.039 against 0.011)
and was ahead on every task it differed on; on 36 the ranking reversed
(0.063 against 0.054, ahead on 2 against 1, tied on 33). Most tasks score
0.000 in every approach, so the discriminating sample is the handful that
move at all - 7 or 8 of 36 - and a difference over anything less than a
few hundred tasks is not a difference. Hence --tasks.

Stopping behaviour is the axis that separated them, and it needs the
endings split three ways. An episode ends early both when something
submits and when the intersection reaches the target, so "shorter than the
cap" counts solving as stopping; this script reads the last action instead.
Over 24 tasks every voluntary submit under approach 1 came at 0.000
progress - it gives up, it does not hand in partial work - against a mean
of 0.141 under approach 2.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rl.mcts as mcts  # noqa: E402
from rl.arc_env import ARCGridWorld  # noqa: E402
from rl.arc_task import ARCSubtask  # noqa: E402
from data.configs.env_configs import (ACTION_TYPES, AGENT2ACTIONS,  # noqa: E402
                                      COLOR_DEPENDENT_ACTIONS,
                                      DIRECTION_DEPENDENT_ACTIONS,
                                      DOUBLE_COLOR_DEPENDENT_ACTIONS)
from rl.utils import define_feasible_actions  # noqa: E402


class TimedOut(Exception):
    pass


def build_actions(colours, directions):
    """The vocabulary an env is actually configured with.

    Generated rather than written out: a hand-written name that misses its
    branch returns the grid untouched, which is indistinguishable from a
    transform that had nothing to do, and a whole measurement can run
    against an action space the env never gets. Two colours and two
    directions reach every branch - the decorations pick arguments, not
    branches - where all ten and all eight give 2926 names.
    """
    bases = ({a for roster in AGENT2ACTIONS.values() for a in roster}
             | {a for group in ACTION_TYPES.values() for a in group})
    names = []
    for base in sorted(bases):
        if base == "submit":
            continue
        generated = define_feasible_actions(
            [base], colours, directions, COLOR_DEPENDENT_ACTIONS,
            DOUBLE_COLOR_DEPENDENT_ACTIONS, DIRECTION_DEPENDENT_ACTIONS)
        names.extend(n for n in generated.values() if n != "submit")
    return {0: "submit", **{i + 1: n for i, n in enumerate(names)}}


def load_tasks(dataset, limit):
    """Shape-preserving training pairs. The env's intersection metric
    compares grids cell by cell, so a pair whose output is a different size
    has no meaningful progress fraction."""
    path = REPO_ROOT / "data" / "datasets" / dataset / "training_challenges.json"
    with open(path) as f:
        challenges = json.load(f)
    tasks = []
    for task_id in sorted(challenges):
        pair = challenges[task_id]["train"][0]
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        if inp.shape == out.shape:
            tasks.append((task_id, inp, out))
        if len(tasks) == limit:
            break
    return tasks


_PEAK = {"value": None}
_ORIGINAL_STEP = mcts.EnvironmentSimulator.simulate_step
_ORIGINAL_INIT = mcts.EnvironmentSimulator.__init__


def _watch_peak(self, state, action):
    """Every simulated step, not only the ones a rollout kept - the tree
    explores far more than it returns, and the best state it touched is
    what the search actually found."""
    result = _ORIGINAL_STEP(self, state, action)
    reached = result[0]["max_int"]
    if _PEAK["value"] is None or reached > _PEAK["value"]:
        _PEAK["value"] = reached
    return result


def install_playout(mode):
    """Which policy every simulator built during a search is given.

    'default' samples the raw action space, padded slots and all - which is
    what the playout has always done, and means ~91% of its draws name no
    object. 'weighted' draws from the pool the tree expands over, weighted
    by measured effect. The two behave very differently and cost very
    differently, so a run says which it used.
    """
    def patched(self, env, actions=None, policy=None):
        _ORIGINAL_INIT(self, env, actions=actions, policy=policy)
        if mode == "weighted":
            self.policy = mcts.PlayoutPolicy(self.all_actions,
                                             temperature=0.2, floor=0.02)
    mcts.EnvironmentSimulator.__init__ = patched
    mcts.EnvironmentSimulator.simulate_step = _watch_peak


def run_one(task, actions, approach, args):
    """One search over one task, as (rollouts, peak) - or (None, None) if it
    did not finish. A task that times out or raises is dropped rather than
    scored zero: a zero is a search that found nothing, which is a different
    statement."""
    task_id, inp, out = task
    env = ARCGridWorld(max_episode_len=args.episode_len, feasible_actions=actions,
                       reward_approach=approach, repr_level=1,
                       input_pattern="start",
                       observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask(f"{task_id}_0", inp, out))
    base, target = env.max_int, env.target_int
    _PEAK["value"] = base
    signal.alarm(args.timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rollouts = mcts.rollout_preparation(
                env, method="mcts", n_initial_rollouts=args.rollouts,
                mcts_iterations=args.iterations, top_k=args.top_k,
                n_rounds=args.rounds, keep_fraction=args.keep, min_pool=4)
    except TimedOut:
        return None, None
    except Exception:
        return None, None
    finally:
        signal.alarm(0)
    span = target - base
    return rollouts, ((_PEAK["value"] - base) / span if span else 0.0)


def evaluate(approach, tasks, actions, args):
    per_task = {}
    endpoints = {}
    endings = collections.Counter()
    submit_progress = []
    lengths = []
    dropped = 0
    for task in tasks:
        best = best_peak = None
        for _ in range(args.repeats):
            rollouts, peak = run_one(task, actions, approach, args)
            if rollouts is None:
                dropped += 1
                continue
            best_peak = peak if best_peak is None else max(best_peak, peak)
            for rollout in rollouts:
                span = rollout["target_int"] - rollout["base_int"]
                closed = ((rollout["max_int"] - rollout["base_int"]) / span
                          if span else 0.0)
                best = closed if best is None else max(best, closed)
                lengths.append(rollout["length"])

                last = rollout["actions"][-1] if rollout["actions"] else None
                on_submit = (last is not None
                             and int(np.asarray(last).ravel()[0]) == 0)
                if rollout["solved"]:
                    endings["solved"] += 1
                elif on_submit:
                    endings["submitted, unsolved"] += 1
                    submit_progress.append(closed)
                elif rollout["length"] < args.episode_len:
                    endings["ended early, neither"] += 1
                else:
                    endings["ran to the cap"] += 1
        if best_peak is not None:
            per_task[task[0]] = best_peak
        if best is not None:
            endpoints[task[0]] = best
    return {
        "per_task": per_task,
        "endpoints": endpoints,
        "endings": dict(endings),
        "submit_progress": submit_progress,
        "lengths": lengths,
        "dropped": dropped,
    }


def report(approach, result, elapsed):
    progress = list(result["per_task"].values())
    if not progress:
        print(f"\nreward_approach={approach}: no task finished")
        return
    moved = [p for p in progress if p > 0]
    total = sum(result["endings"].values())
    print(f"\nreward_approach={approach}  "
          f"({len(progress)} tasks scored, {total} rollouts, "
          f"{result['dropped']} runs dropped, {elapsed:.0f}s)")
    print("  PEAK progress closed toward the target, best per task:")
    print(f"    mean {statistics.mean(progress):.3f}  "
          f"median {statistics.median(progress):.3f}  max {max(progress):.3f}")
    print(f"    tasks that moved at all: {len(moved)}/{len(progress)}")
    ends = list(result.get("endpoints", {}).values())
    if ends:
        print(f"  where rollouts ended, for contrast: mean "
              f"{statistics.mean(ends):+.3f} - the gap to the peak above is "
              f"what the search found and did not keep")
    print("  how episodes ended:")
    for kind in ("solved", "submitted, unsolved", "ended early, neither",
                 "ran to the cap"):
        n = result["endings"].get(kind, 0)
        print(f"    {kind:22s} {n:5d}  ({100 * n / total:.1f}%)")
    if result["submit_progress"]:
        print(f"    progress when it submitted unsolved: "
              f"mean {np.mean(result['submit_progress']):.3f} "
              f"max {max(result['submit_progress']):.3f}")
    print(f"  mean length {statistics.mean(result['lengths']):.1f}")


def compare(first, second, results):
    """Head to head on the tasks both approaches scored. The means hide
    this: most tasks are 0.000 either way, so a difference in means can
    come from one task, and the win/loss/tie split says whether it did."""
    shared = set(results[first]["per_task"]) & set(results[second]["per_task"])
    ahead_first = ahead_second = tied = 0
    rows = []
    for task_id in sorted(shared):
        a = results[first]["per_task"][task_id]
        b = results[second]["per_task"][task_id]
        if abs(a - b) < 1e-9:
            tied += 1
        elif a > b:
            ahead_first += 1
            rows.append((task_id, a, b))
        else:
            ahead_second += 1
            rows.append((task_id, a, b))
    print(f"\nhead to head over {len(shared)} tasks "
          f"(approach {first} vs {second}), tasks where they differ:")
    for task_id, a, b in rows:
        print(f"  {task_id}  {a:.3f}  {b:.3f}  "
              f"{'<-' + str(first) if a > b else str(second) + '->'}")
    if not rows:
        print("  none")
    print(f"\napproach {first} ahead on {ahead_first}, "
          f"approach {second} ahead on {ahead_second}, tied on {tied}")
    if ahead_first + ahead_second < 10:
        print("  Fewer than 10 discriminating tasks: this ranking has "
              "reversed before at this sample size. Raise --tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approaches", type=int, nargs="+", default=[1, 2],
                        help="reward_approach values to compare (default: 1 2)")
    parser.add_argument("--tasks", type=int, default=52,
                        help="training tasks to scan; shape-preserving ones are kept")
    parser.add_argument("--repeats", type=int, default=2,
                        help="runs per task - the search is stochastic enough "
                             "that one run of each reports noise")
    parser.add_argument("--rollouts", type=int, default=4, help="rollouts per round")
    parser.add_argument("--rounds", type=int, default=3, help="pruning rounds")
    parser.add_argument("--keep", type=float, default=0.5,
                        help="fraction of the action pool kept between rounds")
    parser.add_argument("--iterations", type=int, default=40, help="MCTS iterations")
    parser.add_argument("--episode-len", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-run cap in seconds; a run that exceeds it is dropped")
    parser.add_argument("--colours", nargs="+", default=["red", "blue"])
    parser.add_argument("--directions", nargs="+", default=["N", "E"])
    parser.add_argument("--playout", default="default", choices=["default", "weighted"],
                        help="how playouts pick actions; 'weighted' draws from the "
                             "tree's pool by measured effect, 'default' from the raw "
                             "padded action space")
    parser.add_argument("--dataset", default="ARC", help="ARC or ARC2")
    parser.add_argument("--out", type=Path, help="write the raw per-task results as JSON")
    args = parser.parse_args()

    signal.signal(signal.SIGALRM,
                  lambda *a: (_ for _ in ()).throw(TimedOut()))

    install_playout(args.playout)
    actions = build_actions(args.colours, args.directions)
    tasks = load_tasks(args.dataset, args.tasks)
    print(f"{len(tasks)} shape-preserving tasks, {len(actions)} actions, "
          f"{args.repeats} repeats, {args.rounds} rounds x {args.rollouts} "
          f"rollouts at keep={args.keep}, {args.playout} playout")
    if args.playout == "weighted" and args.rounds > 1:
        print("  note: measured over 6 tasks the two work against each other - "
              "weighted scored +0.324 at --rounds 1 and +0.243 at --rounds 3, "
              "while the default playout went the other way (+0.207 -> +0.269)")

    results = {}
    for approach in args.approaches:
        start = time.perf_counter()
        results[approach] = evaluate(approach, tasks, actions, args)
        report(approach, results[approach], time.perf_counter() - start)

    for i, first in enumerate(args.approaches):
        for second in args.approaches[i + 1:]:
            compare(first, second, results)

    if args.out:
        args.out.write_text(json.dumps(
            {str(k): {"per_task": v["per_task"], "endings": v["endings"],
                      "submit_progress": v["submit_progress"],
                      "dropped": v["dropped"]}
             for k, v in results.items()}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
