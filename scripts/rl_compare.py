#!/usr/bin/env python3
"""Compare ways of building an agent's observation and features on the same
tasks, the same narrowed vocabulary and the same seeds.

An arm is what changes: which observations the agent gets and any PPO setting
that goes with them. Held-out accuracy - the share of the distance closed on
the test pair, 1.0 when solved - is the score; a single run says little (the
same agent on the same task ranges from -0.4 to 1.0 across seeds), so every
task is run under three seeds and arms are compared over tasks.

    python scripts/rl_compare.py narrow --out configs.json
    python scripts/rl_compare.py train --configs configs.json --arms g gd gd_ch gd_wide \\
        --steps 100000 --shard 0 --shards 4 --out runs.jsonl
    python scripts/rl_compare.py summary --out runs.jsonl

`narrow` decides, once per task, what rl_config leaves open (the vocabulary
the search narrows to, the object slots, the observation shape) so that every
arm trains over the same one. `train` runs one shard of the (task, arm, seed)
grid and appends a line per finished run, skipping what is already in the
file, so a run cut short - the machine restarts, a shard is killed - is
resumed by starting it again. `summary` reads the file.

The arms, and what each was for:

    objonly     objects only: what the agent sees if nothing else is given
    default     objects, relations, both deltas: rl_config's default
    g           the grid alone, object actions - the grid encoder on its own
    gd          plus the deltas, each through a DeltaReadout of its own
    gd_ch       the deltas as planes of the grid encoder's convolutions
    gd_wide     the same, with the two convolutions 32 wide instead of 8 and 16

`gd_wide` used to run as gd_ch: its widths were a PPO setting nothing read,
so its runs were a second sample of gd_ch. The width is now the encoder
factory it has to be, and a test holds it to that.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "datasets" / "ARC"
DELTAS = ["delta_input", "delta_target"]

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
    "default": (["objects_emb", "relations_emb", *DELTAS], {}),
    "g": ([], {}),
    "gd": (DELTAS, {}),
    "gd_ch": (DELTAS, {"delta_in_grid": True}),
    "gd_wide": (DELTAS, {"delta_in_grid": True, "extr_arch": "wide"}),
}


def arm_settings(arm):
    """The PPO settings of an arm, with the named encoder built."""
    elements, settings = ARMS[arm]
    settings = dict(settings)
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
    """Write each task's narrowed rl_config to `out`."""
    from data.configs.rl_configs import rl_config
    from rl.rl_job import narrowed_for_task

    configs = {}
    for task_id in tasks:
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


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def run_grid(configs, arms, seeds=(0, 1, 2)):
    """Every (task, arm, seed), tasks first: a partial file covers every arm
    on the tasks it reached."""
    return [(task_id, arm, seed) for task_id in sorted(configs) for seed in seeds for arm in arms]


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


def train(configs, arms, steps, shard, shards, out, seeds=(0, 1, 2)):
    """Run this shard's share of the grid, appending each run as it ends."""
    done = read_runs(out)
    with open(out, "a", buffering=1) as handle:
        for task_id, arm, seed in run_grid(configs, arms, seeds)[shard::shards]:
            if (task_id, arm, seed) in done:
                continue
            record = one_run(configs[task_id], arm, seed, steps, task_id)
            handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


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
    narrowing = sub.add_parser("narrow", help="write each task's narrowed rl_config")
    narrowing.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    narrowing.add_argument("--out", default="rl_compare_configs.json")
    training = sub.add_parser("train", help="run one shard of the (task, arm, seed) grid")
    training.add_argument("--configs", default="rl_compare_configs.json")
    training.add_argument("--arms", nargs="+", default=["g", "gd", "gd_ch", "gd_wide"], choices=sorted(ARMS))
    training.add_argument("--tasks", nargs="*", help="only these tasks of the configs file")
    training.add_argument("--steps", type=int, default=100_000)
    training.add_argument("--shard", type=int, default=0)
    training.add_argument("--shards", type=int, default=1)
    training.add_argument("--out", default="rl_compare_runs.jsonl")
    reading = sub.add_parser("summary", help="summarise a runs file")
    reading.add_argument("--out", default="rl_compare_runs.jsonl")
    args = parser.parse_args()

    if args.command == "narrow":
        narrow(args.tasks, args.out)
    elif args.command == "train":
        with open(args.configs) as handle:
            configs = json.load(handle)
        if args.tasks:
            configs = {task: configs[task] for task in args.tasks}
        train(configs, args.arms, args.steps, args.shard, args.shards, args.out)
    else:
        print(render(summarise(read_runs(args.out).values())))


if __name__ == "__main__":
    main()
