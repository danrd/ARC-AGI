#!/usr/bin/env python3
"""Object RL on the evaluation split: each notebook narrows its own tasks, then trains on them.

RL was only ever run on training tasks, so it has no place in the table of what each source solves on evaluation. The
narrowing (rl_compare.py narrow) is a search per task that takes minutes, and a notebook only needs the configs of the
tasks it trains, so this narrows this shard's tasks in `--workers` processes and then trains them, one seed each. The
shards cut the task list, not the (task, arm, seed) grid, so no notebook waits for another's configs.

    python scripts/rl_evaluation.py --tasks-file tasks.json --shard 0 --shards 5 --workers 4 --hours 10 --out runs.jsonl
"""
import argparse
import json
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.rl_compare import ARMS, STEPS, narrow, train  # noqa: E402 - after the path is set

SPLIT = "evaluation"


def my_tasks(tasks, shard, shards):
    return tasks[shard::shards]


def narrowed(tasks, workers, directory):
    """The configs of `tasks`, narrowed by `workers` processes that each take every workers-th task."""
    chunks = [tasks[i::workers] for i in range(workers) if tasks[i::workers]]
    paths = [str(Path(directory) / f"narrow_{i}.json") for i in range(len(chunks))]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=narrow, args=(chunk, path, SPLIT)) for chunk, path in zip(chunks, paths)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [p.exitcode for p in processes if p.exitcode]
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(processes)} narrowing processes died (exit codes {failed})")
    configs = {}
    for path in paths:
        configs.update(json.loads(Path(path).read_text()))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", required=True, help="JSON list of evaluation task ids")
    parser.add_argument("--arms", nargs="+", default=["o_spatial"], choices=sorted(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, help="stop starting runs after this long, narrowing included")
    parser.add_argument("--out", default="rl_evaluation_runs.jsonl")
    args = parser.parse_args()
    started = time.time()
    tasks = my_tasks(json.loads(Path(args.tasks_file).read_text()), args.shard, args.shards)
    with tempfile.TemporaryDirectory() as directory:
        configs = narrowed(tasks, args.workers, directory)
    print(f"{len(configs)} tasks narrowed in {time.time() - started:.0f} s", flush=True)
    left = None if args.hours is None else max(args.hours - (time.time() - started) / 3600, 0.0)
    train(configs, args.arms, args.steps, 0, 1, args.out, seeds=tuple(args.seeds), workers=args.workers, hours=left)


if __name__ == "__main__":
    main()
