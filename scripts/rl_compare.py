#!/usr/bin/env python3
"""Compare ways of building an agent's observation and features on the same
tasks, the same narrowed vocabulary and the same seeds.

An arm is what changes: which observations the agent gets and any PPO setting
that goes with them. Held-out accuracy - the share of the distance closed on
the test pair, 1.0 when solved - is the score; a single run says little (the
same agent on the same task ranges from -0.4 to 1.0 across seeds), so every
task is run under three seeds and arms are compared over tasks.

    python scripts/rl_compare.py select --peaks agent_searches.jsonl --symbolic sym.jsonl
    python scripts/rl_compare.py narrow --tasks $(python scripts/rl_compare.py select ...) --out configs.json
    python scripts/rl_compare.py train --configs configs.json --arms default gd_ch gd_wide \\
        --steps 250000 --shard 0 --shards 4 --out runs.jsonl
    python scripts/rl_compare.py summary --out runs.jsonl

`select` picks the tasks: see below. `narrow` decides, once per task, what rl_config leaves open (the vocabulary
the search narrows to, the object slots, the observation shape) so that every
arm trains over the same one. `train` runs one shard of the (task, arm, seed)
grid and appends a line per finished run, skipping what is already in the
file, so a run cut short - the machine restarts, a shard is killed - is
resumed by starting it again. `summary` reads the file.

Which tasks. An arm is told apart from another only on a task some agent can
make progress on: where every arm scores zero (a48eeaf7 and dc433765 in the
short series) there is nothing to compare. `select` keeps the tasks the
object search got somewhere on - a peak above zero, from a census
(search_census.py) or a roster search (agent_searches: the best over rosters)
- and drops the ones the symbolic solvers already solve, which the agents are
never asked. Those that the search solves outright stay: they say whether an
arm finds what is findable, and the rest say how far it gets.

Spreading it over machines. The grid is 1071 runs for 51 tasks, seven arms and
three seeds, about 460 s each at 250k steps, and is meant to be shared between
notebooks. `--shard k --shards K` gives a notebook the k-th of K shares of the
grid (by a hash of the run, so each holds every arm); `--workers W` runs W
processes in it, one thread each; `--hours H` stops it starting a run that
would end after H hours, so a notebook with a 12 hour limit ends by itself
with its file written. What the notebooks leave undone - a notebook that died,
a limit that came first - is what `--skip` is for: a later wave is given the
earlier outputs and does what is not in them, split however the new
notebooks are. `--tasks` and `--arms` narrow a wave the same way.

    # on Kaggle, a notebook of the first of three
    !git clone https://github.com/danrd/ARC-AGI && pip install -e "ARC-AGI[rl]"
    !cd ARC-AGI && python scripts/rl_compare.py train \\
        --configs data/experiments/rl_compare_configs.json \\
        --shard 0 --shards 3 --workers 4 --hours 11 --out /kaggle/working/runs_0.jsonl
    # after: the files of the notebooks, in a dataset, for a second wave or the table
    !python ARC-AGI/scripts/rl_compare.py summary --out /kaggle/input/runs/*.jsonl

The narrowed configs are data/experiments/rl_compare_configs.json, made once by
`narrow` and kept: the search that narrows a task's vocabulary is random, and
every arm, seed and notebook has to train over the same one.

The arms, and what each was for:

    objonly     the grid and the objects, nothing else added
    objrel      plus the relations between objects: default without the deltas
    default     objects, relations, both deltas: rl_config's default
    g           the grid alone, object actions - the grid encoder on its own
    gd          plus the deltas, each through a DeltaReadout of its own
    gd_ch       the deltas as planes of the grid encoder's convolutions
    gd_wide     the same, with the two convolutions 32 wide instead of 8 and 16

The second series, how the objects are processed (OBJECT_ARMS), each one change
from `default` and run only on the tasks where the object arms respond, with
`default`'s runs of the first series as the reference:

    o_nopos     the object branch without absolute position: a rule is
                mostly translation invariant, and position is a field a policy
                can fit without transferring
    o_noself    without self-attention among the objects
    o_nocross   without the branch's cross-attention
    o_drop3     dropout 0.3 in the object branch instead of 0.1
    o_spatial   the grid map read over each object's box (SpatialBackbone, 32
                channels)
    o_factored  per-slot scoring heads: the slot an action names is scored from
                its own embedding rather than from the pooled vector
    o_pointer64 per-object rows for the pointer heads 64 wide instead of 32

`gd_wide` used to run as gd_ch: its widths were a PPO setting nothing read,
so its runs were a second sample of gd_ch. The width is now the encoder
factory it has to be, and a test holds it to that.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import io
import json
import multiprocessing
import pickle
import sys
import time
import zlib
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "datasets" / "ARC"
DELTAS = ["delta_input", "delta_target"]
DEFAULT_ELEMENTS = ["objects_emb", "relations_emb", *DELTAS]
#: Steps of one run: the full length, not the shortened series' 100k.
STEPS = 250_000
#: What a worker assumes a run takes before it has finished one (seconds).
FIRST_RUN_SECONDS = 900

#: The fifteen training tasks the observation experiments have run on: one
#: to a few objects, shape-preserving, from the object agents' rosters.
DEFAULT_TASKS = ("05f2a901", "25ff71a9", "25d487eb", "68b16354", "d4a91cb9", "63613498", "a2fd1cf0",
                 "a5313dff", "dc433765", "4093f84a", "6cdd2623", "91714a58", "a48eeaf7", "d2abd087",
                 "d687bc17")
#: Three that respond differently: one or two steps, a few objects, several
#: objects and several steps. What the shortened series ran on.
SHORT_TASKS = ("05f2a901", "dc433765", "a48eeaf7")

#: arm -> (observation elements, PPO settings). The settings are values, or a
#: function of nothing that builds one where it must be built afresh.
ARMS = {
    "objonly": (["objects_emb"], {}),
    "objrel": (["objects_emb", "relations_emb"], {}),
    "default": (DEFAULT_ELEMENTS, {}),
    "g": ([], {}),
    "gd": (DELTAS, {}),
    "gd_ch": (DELTAS, {"delta_in_grid": True}),
    "gd_wide": (DELTAS, {"delta_in_grid": True, "extr_arch": "wide"}),
    # The second series: how the objects are processed, each arm one change
    # from `default` (same observation). The settings are PPO config keys.
    "o_nopos": (DEFAULT_ELEMENTS, {"object_arch": {"use_position": False}}),
    "o_noself": (DEFAULT_ELEMENTS, {"object_arch": {"self_attention": False}}),
    "o_nocross": (DEFAULT_ELEMENTS, {"object_arch": {"cross_attention": False}}),
    "o_drop3": (DEFAULT_ELEMENTS, {"object_arch": {"dropout": 0.3}}),
    "o_spatial": (DEFAULT_ELEMENTS, {"spatial_channels": 32}),
    "o_factored": (DEFAULT_ELEMENTS, {"object_heads": "factored"}),
    "o_pointer64": (DEFAULT_ELEMENTS, {"pointer_dim": 64}),
}
#: The arms of the first series, what `train` runs unless told which.
OBSERVATION_ARMS = ("objonly", "objrel", "default", "g", "gd", "gd_ch", "gd_wide")
#: The second series, run against `default`'s runs already made.
OBJECT_ARMS = tuple(arm for arm in ARMS if arm.startswith("o_"))


def arm_settings(arm):
    """The PPO settings of an arm, with the named encoder built."""
    elements, settings = ARMS[arm]
    settings = copy.deepcopy(settings)
    if settings.get("extr_arch") == "wide":
        from data.configs.rl_configs import lin
        settings["extr_arch"] = partial(lin, widths=(32, 32))
    return list(elements), settings


# ---------------------------------------------------------------------------
# tasks and their narrowed configs
# ---------------------------------------------------------------------------

def load_task(task_id, split="training"):
    """An ARCTask with its agent label read from idx2agent.pkl."""
    from rl.arc_task import ARCSubtask, ARCTask

    with open(DATA / f"{split}_challenges.json") as handle:
        challenges = json.load(handle)
    with open(DATA / f"{split}_solutions.json") as handle:
        solutions = json.load(handle)
    challenge = challenges[task_id]
    task = ARCTask(label=task_id,
                   subtasks=[ARCSubtask(f"{task_id}_{i}", np.array(p["input"]), np.array(p["output"]))
                             for i, p in enumerate(challenge["train"])],
                   test_inp=np.array(challenge["test"][0]["input"]), test_out=np.array(solutions[task_id][0]))
    if split == "training":
        with open(DATA / "idx2agent.pkl", "rb") as handle:
            labels = pickle.load(handle)
        task.agent = labels.get(list(challenges).index(task_id))
    return task


def narrow(tasks, out):
    """Write each task's narrowed rl_config to `out`, one task at a time: a
    task already in the file is kept and not searched again, so a run cut
    short resumes. The search is random, which is why the result is kept and
    every arm, seed and machine trains over this one."""
    from data.configs.rl_configs import rl_config
    from rl.rl_job import narrowed_for_task

    configs = {}
    if Path(out).exists():
        with open(out) as handle:
            configs = json.load(handle)
    for task_id in tasks:
        if task_id in configs:
            continue
        task = load_task(task_id)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            config = narrowed_for_task(task, rl_config)
        config["feasible_actions"] = {int(k): v for k, v in config["feasible_actions"].items()}
        configs[task_id] = config
        print(task_id, getattr(task, "agent", None), config["addressing"], len(config["feasible_actions"]),
              "actions", flush=True)
        with open(out, "w") as handle:
            json.dump(configs, handle, default=list)
    return configs


def best_peak(record):
    """How far the search got on a task's first pair, from either kind of
    record: a census line has `peak`, a roster search the best over `agents`."""
    if "peak" in record:
        return float(record["peak"])
    return max((float(found["peak"]) for found in record.get("agents", {}).values()), default=0.0)


def select_tasks(records, excluded=()):
    """The tasks of `records` whose search peak is above zero and that are not
    in `excluded`, sorted. A census record that was skipped (the grids change
    size) has no peak, reads as zero, and is not a candidate."""
    return sorted(record["task"] for record in records
                  if best_peak(record) > 0 and record["task"] not in set(excluded))


def read_lines(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def run_grid(configs, arms, seeds=(0, 1, 2)):
    """Every (task, arm, seed), tasks first: a partial file covers every arm
    on the tasks it reached."""
    return [(task_id, arm, seed) for task_id in sorted(configs) for seed in seeds for arm in arms]


def share(grid, index, total):
    """The runs of `grid` that worker `index` of `total` does.

    By a hash of the run and not by position: the grid runs through the arms
    in turn, so a stride that divides by the number of arms would give a
    worker one arm and lose that arm whole if its notebook dies. Stable
    across machines (crc32, not hash()), and a worker's runs keep the
    grid's tasks-first order."""
    return [run for run in grid if zlib.crc32("|".join(map(str, run)).encode()) % total == index]


def read_runs(path):
    """The runs already in `path`, by (task, arm, seed); a torn line is ignored."""
    runs = {}
    if not Path(path).exists():
        return runs
    with open(path) as handle:
        for line in handle:
            try:
                run = json.loads(line)
            except ValueError:
                continue
            runs[(run["task"], run["arm"], run["seed"])] = run
    return runs


def one_run(config, arm, seed, steps, task_id):
    """Train one agent and return the record of the run."""
    from data.configs.rl_configs import load_PPO_config
    from rl.training import train_on_task

    elements, settings = arm_settings(arm)
    config = dict(config, seed=seed, total_steps=steps, log_path=f"/tmp/rl_compare/{task_id}_{arm}_{seed}/")
    config["feasible_actions"] = {int(k): v for k, v in config["feasible_actions"].items()}
    for key in ("observation_grid_shape", "coordinate_shape"):
        if config.get(key) is not None:
            config[key] = tuple(config[key])
    config["observation_space_elements"] = elements
    ppo = load_PPO_config()
    ppo["verbose"] = 0
    ppo["object_heads"] = "flat"
    ppo["direction_keys"] = "both"
    ppo["spatial_channels"] = 0
    ppo.update(settings)
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        accuracies, lengths, _agent, metrics = train_on_task(load_task(task_id), config, ppo, show_plots=False)
    return {"task": task_id, "arm": arm, "seed": seed, "held_out": round(float(metrics["test_acc"]), 4),
            "train": [round(float(a), 4) for a in accuracies.values()],
            "steps_to_solve": [round(float(v), 2) for v in lengths.values()],
            "seconds": round(time.time() - started, 1)}


def _work(configs, arms, steps, index, total, out, seeds, deadline, skip=()):
    """One worker's share of the grid, appending each run as it ends. Stops
    before a run that would not be done by `deadline` (a time, or None): the
    longest run this worker has seen, or FIRST_RUN_SECONDS before it has seen
    one, is what it assumes a run takes."""
    import torch

    torch.set_num_threads(1)   # one thread a worker: several workers share the cores
    done = read_runs(out)
    for path in skip:
        done.update(read_runs(path))
    longest = None
    with open(out, "a", buffering=1) as handle:
        for task_id, arm, seed in share(run_grid(configs, arms, seeds), index, total):
            if (task_id, arm, seed) in done:
                continue
            if deadline is not None and time.time() + (longest or FIRST_RUN_SECONDS) > deadline:
                print(f"worker {index}: out of time, stopping before {task_id} {arm} {seed}", flush=True)
                return
            record = one_run(configs[task_id], arm, seed, steps, task_id)
            longest = max(longest or 0, record["seconds"])
            handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


def train(configs, arms, steps, shard, shards, out, seeds=(0, 1, 2), workers=1, hours=None, skip=()):
    """This shard's share of the grid - shard `shard` of `shards`, one per
    machine or notebook - run by `workers` processes that split it between
    them and append to `out`; `hours` is how long to keep starting runs.
    Runs in the `skip` files count as done: the outputs of earlier notebooks,
    so a later wave picks up what is left however the first was split."""
    deadline = None if hours is None else time.time() + hours * 3600
    total = shards * workers
    if workers == 1:
        return _work(configs, arms, steps, shard, total, out, seeds, deadline, skip)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_work, args=(configs, arms, steps, shard * workers + w, total, out,
                                                      seeds, deadline, skip)) for w in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


# ---------------------------------------------------------------------------
# reading them
# ---------------------------------------------------------------------------

def summarise(runs):
    """{arm: {tasks, mean held-out, solved runs, runs, gate precision}}.

    The gate is "every training pair closed" - what the orchestrator's RL
    step asks before it answers - and its precision is how often such a run
    also closed the held-out pair. A mean over runs treats every run alike;
    the per-task means are what the arms are compared on."""
    by_arm = collections.defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run)
    out = {}
    for arm, rows in by_arm.items():
        per_task = collections.defaultdict(list)
        for run in rows:
            per_task[run["task"]].append(run["held_out"])
        gated = [run for run in rows if run["train"] and all(a == 1.0 for a in run["train"])]
        out[arm] = {"tasks": len(per_task), "runs": len(rows),
                    "mean_over_tasks": round(float(np.mean([np.mean(v) for v in per_task.values()])), 3),
                    "solved_runs": sum(run["held_out"] == 1.0 for run in rows),
                    "gated": len(gated),
                    "gate_precision": (round(sum(run["held_out"] == 1.0 for run in gated) / len(gated), 3)
                                       if gated else None)}
    return out


def render(summary):
    lines = [f"{'arm':<10}{'tasks':>6}{'runs':>6}{'mean':>8}{'solved':>8}{'gated':>7}{'precision':>11}"]
    for arm, row in sorted(summary.items()):
        precision = "-" if row["gate_precision"] is None else f"{row['gate_precision']:.2f}"
        lines.append(f"{arm:<10}{row['tasks']:>6}{row['runs']:>6}{row['mean_over_tasks']:>8.3f}"
                     f"{row['solved_runs']:>8}{row['gated']:>7}{precision:>11}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    selecting = sub.add_parser("select", help="print the ids of the tasks worth comparing arms on")
    selecting.add_argument("--peaks", required=True, help="a census or roster-search jsonl")
    selecting.add_argument("--symbolic", help="symbolic_coverage jsonl: its solved training tasks are left out")
    narrowing = sub.add_parser("narrow", help="write each task's narrowed rl_config")
    narrowing.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    narrowing.add_argument("--out", default="rl_compare_configs.json")
    training = sub.add_parser("train", help="run one shard of the (task, arm, seed) grid")
    training.add_argument("--configs", default="rl_compare_configs.json")
    training.add_argument("--arms", nargs="+", default=list(OBSERVATION_ARMS), choices=sorted(ARMS))
    training.add_argument("--tasks", nargs="*", help="only these tasks of the configs file")
    training.add_argument("--steps", type=int, default=STEPS)
    training.add_argument("--shard", type=int, default=0, help="this machine's number, of --shards")
    training.add_argument("--shards", type=int, default=1, help="how many machines or notebooks share the grid")
    training.add_argument("--workers", type=int, default=1, help="processes on this machine")
    training.add_argument("--skip", nargs="*", default=[], help="runs files of other notebooks: what is in them is done")
    training.add_argument("--hours", type=float, help="stop starting runs that would end after this long")
    training.add_argument("--out", default="rl_compare_runs.jsonl")
    reading = sub.add_parser("summary", help="summarise runs files (one per notebook)")
    reading.add_argument("--out", nargs="+", default=["rl_compare_runs.jsonl"])
    args = parser.parse_args()

    if args.command == "select":
        from scripts.symbolic_coverage import solved_ids
        solved = set()
        if args.symbolic:
            for ids in solved_ids(read_lines(args.symbolic), "training").values():
                solved.update(ids)
        records = read_lines(args.peaks)
        chosen = select_tasks(records, solved)
        print(f"{len(records)} tasks searched, {len(solved)} solved symbolically, {len(chosen)} chosen",
              file=sys.stderr)
        print(" ".join(chosen))
    elif args.command == "narrow":
        narrow(args.tasks, args.out)
    elif args.command == "train":
        with open(args.configs) as handle:
            configs = json.load(handle)
        if args.tasks:
            configs = {task: configs[task] for task in args.tasks}
        train(configs, args.arms, args.steps, args.shard, args.shards, args.out,
              workers=args.workers, hours=args.hours, skip=args.skip)
    else:
        runs = {}
        for path in args.out:
            runs.update(read_runs(path))
        print(render(summarise(runs.values())))


if __name__ == "__main__":
    main()
