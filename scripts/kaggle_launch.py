#!/usr/bin/env python3
"""Run a job of kaggle_worker.py on Kaggle notebooks and bring its runs back.

    python scripts/kaggle_launch.py start JOB --shards 5      # a notebook per shard
    python scripts/kaggle_launch.py status JOB --shards 5
    python scripts/kaggle_launch.py collect JOB --shards 5    # runs into data/experiments/runs/kaggle/

Each shard is a private script notebook, arc-worker-JOB-K, that clones this repo,
installs its requirements and runs `kaggle_worker.py JOB --shard K --shards N --copy-to
/kaggle/working`. A notebook run through the API gets no Kaggle secrets, so the
files do not go to GitHub: `collect` downloads each notebook's output and puts the
files where `rl_compare.py summary` and `--skip` expect them. `--gpu` asks for a GPU.
The account is read from KAGGLE_USERNAME or, failing that, from the API; the token
from KAGGLE_API_TOKEN or KAGGLE_API_KEY.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data/experiments/runs/kaggle"
REPO = "https://github.com/danrd/ARC-AGI"


def slug(job, shard):
    return f"arc-worker-{job}-{shard}".replace("_", "-")


def script(job, shard, shards, extras="rl", setup=()):
    """The code of one notebook. The package is not installed (pip cannot build its flat
    layout): its requirements are, and the worker runs from the clone with it on the path. The
    clone is in /tmp, not in the working folder: what is there is the notebook's output, and
    the repository is not to be downloaded with every result."""
    install = ("import tomllib, subprocess, sys; "
               "project = tomllib.load(open('/tmp/ARC-AGI/pyproject.toml', 'rb'))['project']; "
               f"extras = {extras!r}.split(','); "
               "needs = project['dependencies'] + [r for e in extras for r in project['optional-dependencies'][e]]; "
               "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *needs], check=True)")
    return (
        "import subprocess\n"
        "run = lambda command: subprocess.run(command, shell=True, check=True)\n"
        f"run('git clone {REPO} /tmp/ARC-AGI')\n"
        f"run({('python -c ' + repr(install))!r})\n"
        + "".join(f"run({command!r})\n" for command in setup) +
        f"run('cd /tmp/ARC-AGI && PYTHONPATH=. python scripts/kaggle_worker.py {job} --shard {shard} --shards {shards} "
        f"--copy-to /kaggle/working')\n")


def metadata(username, job, shard, gpu=False):
    return {"id": f"{username}/{slug(job, shard)}", "title": slug(job, shard).replace("-", " "),
            "code_file": "worker.py", "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": str(gpu).lower(), "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}


def kaggle(*args):
    env = dict(os.environ)
    env.setdefault("KAGGLE_API_TOKEN", env.get("KAGGLE_API_KEY", ""))
    command = [str(Path(sys.executable).parent / "kaggle"), *args]
    if not Path(command[0]).exists():
        command[0] = "kaggle"
    return subprocess.run(command, capture_output=True, text=True, env=env)


def username():
    name = os.environ.get("KAGGLE_USERNAME")
    if name:
        return name
    out = kaggle("kernels", "list", "--mine", "--page-size", "1", "--csv").stdout.splitlines()
    return out[1].split(",")[0].split("/")[0] if len(out) > 1 else sys.exit("no Kaggle account: set KAGGLE_USERNAME")


def start(job, shards, gpu=False, extras="rl", setup=()):
    user = username()
    for shard in range(shards):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "worker.py").write_text(script(job, shard, shards, extras, setup))
            (Path(folder) / "kernel-metadata.json").write_text(json.dumps(metadata(user, job, shard, gpu)))
            result = kaggle("kernels", "push", "-p", folder)
        print(slug(job, shard), (result.stdout + result.stderr).strip().splitlines()[-1])


def status(job, shards):
    user = username()
    return {shard: (kaggle("kernels", "status", f"{user}/{slug(job, shard)}").stdout.strip().split('"')[-2:-1] or ["?"])[0]
            .replace("KernelWorkerStatus.", "") for shard in range(shards)}


def collect(job, shards):
    """Download every notebook's output; returns the files put in RUNS (the output's own folders kept)."""
    user = username()
    RUNS.mkdir(parents=True, exist_ok=True)
    got = []
    for shard in range(shards):
        with tempfile.TemporaryDirectory() as folder:
            kaggle("kernels", "output", f"{user}/{slug(job, shard)}", "-p", folder)
            for path in sorted(Path(folder).rglob("*")):
                if path.is_dir() or path.suffix == ".log" or path.name == "__results__.html":
                    continue
                target = RUNS / path.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
                got.append(target)
    return got


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["start", "status", "collect"])
    parser.add_argument("job")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--setup", action="append", default=[], help="a shell command to run before the worker; repeatable")
    parser.add_argument("--extras", default="rl", help="the extras of the package to install")
    args = parser.parse_args()
    if args.command == "start":
        start(args.job, args.shards, args.gpu, args.extras, args.setup)
    elif args.command == "status":
        print(status(args.job, args.shards))
    else:
        for path in collect(args.job, args.shards):
            print(path)


if __name__ == "__main__":
    main()
