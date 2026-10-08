#!/usr/bin/env python3
"""A queue of jobs for the Kaggle notebooks, so that nothing waits for a person to start it.

Kaggle gives a fixed number of notebooks at once - five on the CPU, one on the GPU - and a week's hours on
each. A notebook that has finished and nothing started after it is time lost, and what is lost is lost for
good. The queue is a file (data/experiments/queue.json) of jobs, each a name from data/experiments/jobs/, the
resource it runs on and how many notebooks it takes. `tick` is the whole of the management and is meant to
be run often, by anyone: it

    1. looks at the notebooks of the jobs marked running, and for a job whose notebooks have all ended
       downloads what they wrote and marks the job done (or failed, when a notebook ended in error)
    2. counts the free notebooks of each resource
    3. starts, in the order of the file, each queued job that fits

A job that does not fit waits, and jobs behind it that do fit go first: a queue that stops at the first job too
big for the free slots leaves them idle. The state is the file, not a process, so a tick can be run from any
session or by hand, and runs the same.

    python scripts/kaggle_queue.py status
    python scripts/kaggle_queue.py tick            # then commit data/experiments/
    python scripts/kaggle_queue.py add NAME --resource cpu --shards 2
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUEUE = ROOT / "data" / "experiments" / "queue.json"
CAPACITY = {"cpu": 5, "gpu": 1}
BUSY = {"RUNNING", "QUEUED", "NEW", "STARTING"}
ENDED = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


@dataclass
class Job:
    name: str
    resource: str = "cpu"
    shards: int = 1
    status: str = "queued"           # queued | running | done | failed
    setup: List[str] = field(default_factory=list)
    extras: str = "rl"
    note: str = ""
    result: List[str] = field(default_factory=list)       # the files collected


def load(path=QUEUE) -> List[Job]:
    path = Path(path)
    if not path.exists():
        return []
    return [Job(**item) for item in json.loads(path.read_text())["jobs"]]


def save(jobs: List[Job], path=QUEUE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"capacity": CAPACITY, "jobs": [asdict(j) for j in jobs]}, indent=1) + "\n")


def slots_used(jobs: List[Job], statuses: Dict[str, List[str]]) -> Dict[str, int]:
    """The notebooks in use now, by resource: those of a running job that have not ended."""
    used = {resource: 0 for resource in CAPACITY}
    for job in jobs:
        if job.status == "running":
            used[job.resource] += sum(1 for s in statuses.get(job.name, []) if s not in ENDED)
    return used


def tick(jobs: List[Job], status_fn: Callable[[Job], List[str]], start_fn: Callable[[Job], None],
         collect_fn: Callable[[Job], List[str]], capacity: Optional[Dict[str, int]] = None) -> List[str]:
    """One pass over the queue; changes `jobs` and returns what it did, a line each."""
    capacity = capacity or CAPACITY
    events: List[str] = []
    statuses: Dict[str, List[str]] = {}

    for job in jobs:
        if job.status != "running":
            continue
        statuses[job.name] = status_fn(job)
        seen = statuses[job.name]
        if seen and all(s in ENDED for s in seen):
            job.result = collect_fn(job)
            job.status = "done" if all(s == "COMPLETE" for s in seen) else "failed"
            events.append(f"{job.name}: {job.status}, {len(job.result)} files collected")
            statuses[job.name] = seen

    used = slots_used(jobs, statuses)
    for job in jobs:
        if job.status != "queued":
            continue
        if used[job.resource] + job.shards > capacity[job.resource]:
            events.append(f"{job.name}: waits, {capacity[job.resource] - used[job.resource]} free {job.resource} "
                          f"of the {job.shards} it takes")
            continue
        start_fn(job)
        job.status = "running"
        used[job.resource] += job.shards
        events.append(f"{job.name}: started on {job.shards} {job.resource} notebook(s)")

    free = {r: capacity[r] - used[r] for r in capacity}
    idle = [f"{n} {r}" for r, n in free.items() if n > 0 and not any(j.status == "queued" and j.resource == r for j in jobs)]
    if idle:
        events.append("idle with nothing queued: " + ", ".join(idle))
    return events


def real_status(job: Job) -> List[str]:
    import kaggle_launch as launch
    return [s if s != "?" else "ERROR" for s in launch.status(job.name, job.shards).values()]


def real_start(job: Job) -> None:
    import kaggle_launch as launch
    launch.start(job.name, job.shards, gpu=job.resource == "gpu", extras=job.extras, setup=job.setup)


def real_collect(job: Job) -> List[str]:
    import kaggle_launch as launch
    return [str(p.relative_to(ROOT)) for p in launch.collect(job.name, job.shards)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "tick", "add"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--resource", choices=sorted(CAPACITY), default="cpu")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--extras", default="rl")
    parser.add_argument("--setup", action="append", default=[])
    parser.add_argument("--note", default="")
    parser.add_argument("--running", action="store_true", help="add it as already started (a job begun by hand)")
    args = parser.parse_args()
    jobs = load()
    if args.command == "add":
        if not args.name or any(j.name == args.name for j in jobs):
            sys.exit("a new job needs a name not in the queue")
        jobs.append(Job(args.name, args.resource, args.shards, "running" if args.running else "queued",
                        args.setup, args.extras, args.note))
        save(jobs)
    elif args.command == "tick":
        for line in tick(jobs, real_status, real_start, real_collect):
            print(line)
        save(jobs)
    for job in jobs:
        print(f"{job.name:<14}{job.resource:<5}{job.shards:>2}  {job.status:<8}{job.note}")


if __name__ == "__main__":
    main()
