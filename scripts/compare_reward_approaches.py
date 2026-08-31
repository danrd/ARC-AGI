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

Scored by the best intersection reached at ANY point in the search - every
simulated step, including the playouts a rollout never walks. A search
looks for a path and keeps the best prefix of one, so a rollout that
touched the target at step seven and wandered off by step twenty-five is
not a failure, and scoring by where it ended records it as one.

The second figure used to be that endpoint. It no longer is: rollouts are
now cut at their own peak, so a returned trace ends at its best state by
construction. What is reported instead is the best a *returned* rollout
reached, and the gap to the figure above is what the tree touched inside a
playout and never committed to - 0.264 against 0.080 on a 20-task pilot,
so most of what the search finds is found in playouts. That gap is what
record_solution exists to recover.

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
import multiprocessing
import os
import signal
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


def parse_span(text: str):
    """"N" for the first N tasks, "A-B" for the half-open range [A, B).

    A range is what makes the scan divisible. One search over the whole set
    takes hours, and the work splits perfectly - tasks share nothing - so
    the useful unit is "cover 100-200 on this machine while another covers
    200-300", not "always start from the beginning and stop earlier".
    """
    text = str(text).strip()
    try:
        if "-" in text:
            first, _, last = text.partition("-")
            start, stop = int(first), int(last)
        else:
            start, stop = 0, int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--tasks {text}: expected a count or a range A-B with B > A >= 0")
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError(
            f"--tasks {text}: expected a count or a range A-B with B > A >= 0")
    return start, stop


def load_tasks(dataset, span):
    """Shape-preserving training pairs in `span`, and how many exist in all.

    The env's intersection metric compares grids cell by cell, so a pair
    whose output is a different size has no meaningful progress fraction.

    Positions index the shape-preserving list, not the raw file, and the
    file is walked in sorted key order - so a given span names the same
    tasks on every machine and on every run, which is what lets separately
    scanned ranges be pooled afterwards.
    """
    path = REPO_ROOT / "data" / "datasets" / dataset / "training_challenges.json"
    with open(path) as f:
        challenges = json.load(f)
    start, stop = span
    tasks, total = [], 0
    for task_id in sorted(challenges):
        pair = challenges[task_id]["train"][0]
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        if inp.shape != out.shape:
            continue
        if start <= total < stop:
            tasks.append((task_id, inp, out))
        total += 1
    return tasks, total


_PEAK = {"value": None}
#: index -> transform name, so a recorded action reads as something an agent
#: roster can be intersected with rather than as a number. Filled by main.
_ACTION_NAMES = {}
#: action name -> the largest single-step gain in intersection it produced on
#: the task being searched. Reset per task by run_one.
_EFFECTIVE = {}
_ORIGINAL_STEP = mcts.EnvironmentSimulator.simulate_step
_ORIGINAL_INIT = mcts.EnvironmentSimulator.__init__


def _watch_peak(self, state, action):
    """Every simulated step, not only the ones a rollout kept - the tree
    explores far more than it returns, and the best state it touched is
    what the search actually found.

    Which action produced each gain is recorded too. It costs one dict
    write on the steps that improve anything and it is the whole of what
    the analyst wants out of a search: the set of transforms that moved
    this task, to intersect against the set an agent owns. The first
    full scan measured the search and kept none of it - 16 CPU-hours for
    one number per task.
    """
    result = _ORIGINAL_STEP(self, state, action)
    reached = result[0]["max_int"]
    if _PEAK["value"] is None or reached > _PEAK["value"]:
        _PEAK["value"] = reached
    gain = reached - state["max_int"]
    if gain > 0:
        name = _ACTION_NAMES.get(int(np.asarray(action).reshape(-1)[0]))
        if name is not None:
            _EFFECTIVE[name] = max(_EFFECTIVE.get(name, 0), int(gain))
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
    """One search over one task, as (rollouts, peak, why) - or
    (None, None, reason) if it did not finish. A task that times out or
    raises is dropped rather than scored zero: a zero is a search that
    found nothing, which is a different statement.

    `why` names which of the two it was, because they call for opposite
    fixes and one run counted them together. Reading a scan of 300 runs
    that dropped 124 of them in 9233s as a budget problem is arithmetically
    impossible - 124 timeouts at --timeout 240 is 29760s on their own - so
    those drops were exceptions, and raising the budget would have changed
    nothing.
    """
    task_id, inp, out = task
    env = ARCGridWorld(max_episode_len=args.episode_len, feasible_actions=actions,
                       reward_approach=approach, repr_level=1,
                       input_pattern="start",
                       observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask(f"{task_id}_0", inp, out))
    base, target = env.max_int, env.target_int
    _PEAK["value"] = base
    _EFFECTIVE.clear()
    signal.alarm(args.timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rollouts = mcts.rollout_preparation(
                env, method="mcts", n_initial_rollouts=args.rollouts,
                mcts_iterations=args.iterations, top_k=args.top_k,
                n_rounds=args.rounds, keep_fraction=args.keep, min_pool=4,
                c=args.c)
    except TimedOut:
        return None, None, "timed out"
    except Exception as exc:
        return None, None, type(exc).__name__
    finally:
        signal.alarm(0)
    span = target - base
    return rollouts, ((_PEAK["value"] - base) / span if span else 0.0), None


def keep_partial(kept, closed, trace, limit):
    """Hold the `limit` furthest non-solving paths for one task, best first.

    Ties go to the shorter path. A search that wandered for twenty-five
    steps and one that arrived in three reached the same place, and the
    short one is the one worth showing anybody.
    """
    if any(trace == held for _, held in kept):
        return kept
    kept.append([closed, trace])
    kept.sort(key=lambda pair: (-pair[0], len(pair[1])))
    del kept[limit:]
    return kept


def summarise_run(task_id, rollouts, peak, effective, args):
    """One finished search reduced to the facts the scan keeps.

    Separated from the merging so that it can run in another process: a
    search returns rollouts holding grids and object graphs, and shipping
    those back would cost more than the search saved. What crosses the
    process boundary is this - counters, a peak, a handful of action
    sequences.
    """
    out = {"task": task_id, "peak": peak, "endpoint": None,
           "effective": dict(effective), "solutions": [], "partials": [],
           "endings": collections.Counter(), "lengths": [],
           "submit_progress": [], "why": None}
    for rollout in rollouts:
        span = rollout["target_int"] - rollout["base_int"]
        closed = ((rollout["max_int"] - rollout["base_int"]) / span
                  if span else 0.0)
        out["endpoint"] = (closed if out["endpoint"] is None
                           else max(out["endpoint"], closed))
        out["lengths"].append(rollout["length"])

        last = rollout["actions"][-1] if rollout["actions"] else None
        on_submit = (last is not None
                     and int(np.asarray(last).ravel()[0]) == 0)
        trace = [[int(x) for x in np.asarray(a).reshape(-1)]
                 for a in rollout["actions"]]
        if not rollout["solved"] and closed > 0 and trace:
            keep_partial(out["partials"], closed, trace, args.partials)
        if rollout["solved"]:
            out["endings"]["solved"] += 1
            # The trace, not just the tally. These are what the first
            # full scan produced and discarded - 26 solved tasks whose
            # sequences existed in memory and went nowhere.
            if trace and trace not in out["solutions"]:
                out["solutions"].append(trace)
        elif on_submit:
            out["endings"]["submitted, unsolved"] += 1
            out["submit_progress"].append(closed)
        elif rollout.get("truncated_at_peak"):
            # A short rollout no longer means the episode stopped early:
            # collect_mcts_rollouts cuts the trace at the best state it
            # passed through and throws the walk back downhill away.
            # Reading that as "ended early" counted the cut as a behaviour
            # of the search.
            out["endings"]["cut back to its peak"] += 1
        elif rollout["length"] < args.episode_len:
            out["endings"]["ended early, neither"] += 1
        else:
            out["endings"]["ran to the cap"] += 1
    return out


#: What a worker process needs and cannot be handed per call: the built
#: vocabulary and the settings. Filled by _init_worker, or by evaluate when
#: the run is sequential.
_WORKER = {}


def _init_worker(colours, directions, playout, approach, args):
    """Rebuild in this process what main() set up in the parent.

    A spawned worker starts from a fresh import: the playout patch is not
    installed, and _ACTION_NAMES - which is how a recorded action becomes a
    name rather than a number - is empty. A worker that skipped this would
    run a different search and report nameless actions, both silently.
    """
    install_playout(playout)
    actions = build_actions(colours, directions)
    _ACTION_NAMES.clear()
    _ACTION_NAMES.update(actions)
    _WORKER.update(actions=actions, approach=approach, args=args)


def _search_one(task):
    rollouts, peak, why = run_one(task, _WORKER["actions"], _WORKER["approach"],
                                  _WORKER["args"])
    if rollouts is None:
        return {"task": task[0], "why": why}
    return summarise_run(task[0], rollouts, peak, _EFFECTIVE, _WORKER["args"])


def merge(summary, into, partials_limit=3):
    """Fold one search's summary into the totals."""
    task_id = summary["task"]
    if summary["why"] is not None:
        into["dropped"] += 1
        into["drop_reasons"][summary["why"]] += 1
        return into
    if summary["peak"] is not None:
        into["per_task"][task_id] = max(into["per_task"].get(task_id, 0.0),
                                        summary["peak"])
    if summary["endpoint"] is not None:
        into["endpoints"][task_id] = max(into["endpoints"].get(task_id, 0.0),
                                         summary["endpoint"])
    found = into["effective_actions"].setdefault(task_id, {})
    for name, gain in summary["effective"].items():
        found[name] = max(found.get(name, 0), gain)
    if not found:
        del into["effective_actions"][task_id]
    kept = into["solutions"].setdefault(task_id, [])
    for trace in summary["solutions"]:
        if trace not in kept:
            kept.append(trace)
    if not kept:
        del into["solutions"][task_id]
    for closed, trace in summary["partials"]:
        keep_partial(into["partial_paths"].setdefault(task_id, []), closed,
                     trace, partials_limit)
    if not into["partial_paths"].get(task_id, None):
        into["partial_paths"].pop(task_id, None)
    into["endings"].update(summary["endings"])
    into["lengths"] += summary["lengths"]
    into["submit_progress"] += summary["submit_progress"]
    return into


def evaluate(approach, tasks, actions, args):
    """Every task, `--repeats` times each, merged into one result.

    The searches share nothing - not the task, not the tree, not a random
    seed - so `--workers` is the whole of the parallelism available here,
    and it is worth taking: a search is pure Python holding the GIL, so
    threads would buy nothing and processes buy a core each. Repeats are
    jobs of their own rather than a loop inside one, since a task that
    takes four minutes should not hold a core for twelve while others
    idle.
    """
    totals = {"per_task": {}, "endpoints": {}, "effective_actions": {},
              "solutions": {}, "partial_paths": {},
              "endings": collections.Counter(), "lengths": [],
              "submit_progress": [], "dropped": 0,
              "drop_reasons": collections.Counter()}
    jobs = [task for task in tasks for _ in range(args.repeats)]
    _init_worker(args.colours, args.directions, args.playout, approach, args)
    if args.workers > 1:
        # spawn, not fork: rl.mcts imports torch, and forking a process
        # that has torch loaded is a documented way to get a deadlock that
        # only shows up on some machines.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context,
                initializer=_init_worker,
                initargs=(args.colours, args.directions, args.playout,
                          approach, args)) as pool:
            for summary in pool.map(_search_one, jobs):
                merge(summary, totals, args.partials)
    else:
        for task in jobs:
            merge(_search_one(task), totals, args.partials)
    for task_id in totals["solutions"]:
        totals["solutions"][task_id].sort(key=len)
    totals["endings"] = dict(totals["endings"])
    totals["drop_reasons"] = dict(totals["drop_reasons"])
    return totals


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
    reasons = result.get("drop_reasons") or {}
    if reasons:
        # Broken out because the two call for opposite fixes and reading
        # them together sent one investigation after the timeout when 41%
        # of runs were raising.
        print("  why runs were dropped: " + ", ".join(
            f"{why} {n}" for why, n in sorted(reasons.items(), key=lambda kv: -kv[1])))
    print("  PEAK progress closed toward the target, best per task:")
    print(f"    mean {statistics.mean(progress):.3f}  "
          f"median {statistics.median(progress):.3f}  max {max(progress):.3f}")
    print(f"    tasks that moved at all: {len(moved)}/{len(progress)}")
    ends = list(result.get("endpoints", {}).values())
    if ends:
        # Not "where rollouts ended" any more: the trace is cut at its peak,
        # so a returned rollout ends at its own best state by construction.
        # What this contrasts is the two peaks - the best any simulated step
        # touched, above, against the best a returned rollout reached. The
        # gap is what the tree found inside a playout and never walked to
        # for real, which is the whole reason record_solution exists.
        print(f"  best a returned rollout reached: mean "
              f"{statistics.mean(ends):+.3f} - the gap to the peak above is "
              f"what the search touched in a playout and did not commit to")
    print("  how episodes ended:")
    for kind in ("solved", "submitted, unsolved", "cut back to its peak",
                 "ended early, neither", "ran to the cap"):
        n = result["endings"].get(kind, 0)
        print(f"    {kind:22s} {n:5d}  ({100 * n / total:.1f}%)")
    if result["submit_progress"]:
        print(f"    progress when it submitted unsolved: "
              f"mean {np.mean(result['submit_progress']):.3f} "
              f"max {max(result['submit_progress']):.3f}")
    print(f"  mean length {statistics.mean(result['lengths']):.1f}")

    effective = result.get("effective_actions") or {}
    if effective:
        pooled = collections.Counter()
        for found in effective.values():
            pooled.update(found.keys())
        moved = [t for t, found in effective.items() if found]
        per_task = [len(f) for f in effective.values()]
        print(f"  actions that moved something: {len(pooled)} distinct over "
              f"{len(moved)} tasks, median {statistics.median(per_task):.0f} per task")
        # Any gain at all is a low bar - a single cell counts - and on a
        # 14-task pilot it admitted 29 of 89 actions on the median task,
        # which is too broad to intersect an agent's roster against. The
        # gain is recorded per action so a consumer can raise the bar; this
        # says what raising it buys before anyone reads the flat list.
        #
        # Gains are in the env's own units, and maximal_intersection counts
        # 2 * matches - valid: fixing one cell moves it by two. Thresholds
        # are named in cells and doubled here, since a run of this is read
        # against a grid whose cells someone can count.
        for threshold in (2, 5, 12):
            narrowed = [sum(1 for g in f.values() if g >= 2 * threshold)
                        for f in effective.values()]
            print(f"    at >= {threshold:2d} cells gained: "
                  f"median {statistics.median(narrowed):.0f} per task, "
                  f"{sum(1 for n in narrowed if n)} tasks keep any")
        top = ", ".join(f"{name} ({n})" for name, n in pooled.most_common(6))
        print(f"    most often effective: {top}")
    solutions = result.get("solutions") or {}
    if solutions:
        lengths = [len(t[0]) for t in solutions.values() if t]
        print(f"  solving traces kept: {sum(len(t) for t in solutions.values())} "
              f"over {len(solutions)} tasks, shortest {min(lengths)} actions, "
              f"median {statistics.median(lengths):.0f}")
    partials = result.get("partial_paths") or {}
    if partials:
        best = [held[0][0] for held in partials.values() if held]
        print(f"  partial paths kept: {sum(len(t) for t in partials.values())} "
              f"over {len(partials)} unsolved tasks, best one closing "
              f"{statistics.median(best):.0%} of the distance on the median task")


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
    parser.add_argument("--tasks", type=parse_span, default="52",
                        help="how much of the shape-preserving task list to scan: "
                             "a count (\"100\" = the first 100) or a half-open "
                             "range (\"100-200\"). Positions are stable across "
                             "machines and runs, so ranges scanned separately can "
                             "be pooled")
    parser.add_argument("--repeats", type=int, default=2,
                        help="runs per task - the search is stochastic enough "
                             "that one run of each reports noise")
    parser.add_argument("--rollouts", type=int, default=4, help="rollouts per round")
    parser.add_argument("--rounds", type=int, default=3, help="pruning rounds")
    parser.add_argument("--keep", type=float, default=0.5,
                        help="fraction of the action pool kept between rounds")
    parser.add_argument("--iterations", type=int, default=40, help="MCTS iterations")
    parser.add_argument("--c", type=float, default=1.414,
                        help="UCB1 exploration constant. The honest knob for how "
                             "widely the tree looks - reward_approach used to move "
                             "it as a side effect, since it divides every step "
                             "reward by a different max_reward (5M under approach "
                             "1, 11M under 2) against this same fixed constant")
    parser.add_argument("--episode-len", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1,
                        help="searches to run at once; tasks and repeats share "
                             "nothing, so this is a core each")
    parser.add_argument("--partials", type=int, default=3,
                        help="how many of the furthest non-solving rollouts to "
                             "keep per task, as material for a prompt hint")
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

    if args.workers > 1:
        # Set before any worker is spawned, since a child reads these when
        # it imports numpy. Without it every worker starts a thread pool of
        # its own and the machine spends its time context-switching -
        # the arrays here are 30x30 at most and gain nothing from threads.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(variable, "1")

    install_playout(args.playout)
    actions = build_actions(args.colours, args.directions)
    _ACTION_NAMES.clear()
    _ACTION_NAMES.update(actions)
    tasks, total = load_tasks(args.dataset, args.tasks)
    if not tasks:
        raise SystemExit(f"--tasks {args.tasks[0]}-{args.tasks[1]} selects nothing; "
                         f"{total} shape-preserving tasks exist in {args.dataset}")
    print(f"tasks {args.tasks[0]}-{args.tasks[0] + len(tasks)} of {total} "
          f"shape-preserving in {args.dataset}")
    print(f"{len(tasks)} shape-preserving tasks, {len(actions)} actions, "
          f"{args.repeats} repeats, {args.rounds} rounds x {args.rollouts} "
          f"rollouts at keep={args.keep}, {args.playout} playout, "
          f"{args.workers} worker{'s' if args.workers > 1 else ''}")
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
        # The span travels with the results: pooling separately scanned
        # ranges needs to know which each file covered, and per_task alone
        # cannot say - a task missing from it was dropped, not unscanned.
        args.out.write_text(json.dumps(
            {"span": [args.tasks[0], args.tasks[0] + len(tasks)],
             "total_available": total,
             "approaches": {
                 str(k): {"per_task": v["per_task"],
                          "effective_actions": v["effective_actions"],
                          "solutions": v["solutions"],
                          "partial_paths": v["partial_paths"],
                          "action_names": {str(i): n for i, n in _ACTION_NAMES.items()},
                          "endings": v["endings"],
                          "submit_progress": v["submit_progress"],
                          "dropped": v["dropped"],
                          "drop_reasons": v.get("drop_reasons", {})}
                 for k, v in results.items()}}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
