#!/usr/bin/env python3
"""A Kaggle notebook that takes its work from the repo and gives its runs back to it.

The notebook is the same for every job and every machine: it clones the repo
and runs

    !python ARC-AGI/scripts/kaggle_worker.py JOB --shard K --shards N

JOB is data/experiments/jobs/JOB.json, a list of steps, each a name and the
arguments of `rl_compare.py train` without --shard, --shards and --out:

    {"steps": [{"name": "panel", "args": ["--configs", "...", "--arms", "default"]}]}

The worker runs the steps in turn, each into its own file under
data/experiments/runs/kaggle/JOB_STEP_K.jsonl, and every `--push-every` seconds
and at the end commits the files and pushes them to the branch kaggle/JOB-K
(this notebook's alone, so nothing collides). A notebook that restarts takes
its file back from that branch first, and rl_compare then does only what is not
in it. Fetching kaggle/* and copying the files into runs/ is the other end.

The push needs a token in the Kaggle secret Github: a fine-grained token for
this repo alone, with write access to its contents. It goes to git as a header
of the one push command, never into the remote or a file.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("data/experiments/runs/kaggle")


def load_job(name, root=ROOT):
    return json.loads((Path(root) / "data/experiments/jobs" / f"{name}.json").read_text())["steps"]


def out_path(job, step, shard):
    return RUNS / f"{job}_{step['name']}_{shard}.jsonl"


def train_command(job, step, shard, shards):
    return [sys.executable, "scripts/rl_compare.py", "train", *step["args"],
            "--shard", str(shard), "--shards", str(shards), "--out", str(out_path(job, step, shard))]


def branch_of(job, shard):
    return f"kaggle/{job}-{shard}"


def _git(root, *args, token=None, check=True):
    prefix = []
    if token:
        header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        prefix = ["-c", f"http.extraHeader=Authorization: Basic {header}"]
    return subprocess.run(["git", "-C", str(root), *prefix, *args], check=check, capture_output=True, text=True)


def restore(root, job, shard):
    """Take back what an earlier life of this notebook pushed."""
    branch = branch_of(job, shard)
    if _git(root, "fetch", "origin", branch, check=False).returncode == 0:
        _git(root, "checkout", "FETCH_HEAD", "--", str(RUNS), check=False)


def publish(root, job, shard, token=None, remote="origin"):
    """Commit the notebook's files and push them to its branch; False when there is nothing."""
    _git(root, "add", "-f", str(RUNS))
    if _git(root, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(root, "-c", "user.name=kaggle-worker", "-c", "user.email=kaggle-worker@users.noreply.github.com",
         "commit", "-m", f"Runs of {job}, shard {shard}")
    _git(root, "push", "--force", remote, f"HEAD:refs/heads/{branch_of(job, shard)}", token=token)
    return True


def run(job, shard, shards, push_every=300.0, token=None, root=ROOT):
    root = Path(root)
    os.chdir(root)
    (root / RUNS).mkdir(parents=True, exist_ok=True)
    restore(root, job, shard)
    for step in load_job(job, root):
        process = subprocess.Popen(train_command(job, step, shard, shards))
        while process.poll() is None:
            try:
                process.wait(timeout=push_every)
            except subprocess.TimeoutExpired:
                pass
            publish(root, job, shard, token)
        if process.returncode:
            print(f"step {step['name']} ended with {process.returncode}", file=sys.stderr)
    publish(root, job, shard, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--push-every", type=float, default=300.0)
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("Github")
        except ImportError:
            pass
    if not token:
        sys.exit("no token: add the Kaggle secret Github (or set GH_TOKEN), or its runs cannot leave the notebook")
    run(args.job, args.shard, args.shards, args.push_every, token)


if __name__ == "__main__":
    main()
