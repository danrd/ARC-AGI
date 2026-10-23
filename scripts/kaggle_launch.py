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
import re
import shutil
import subprocess
import sys
import tempfile
import time
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
               f"extras = [e for e in {extras!r}.split(',') if e]; "
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


def model_sources(job):
    """Kaggle models a job's file asks to have attached ("model_sources": ["owner/model/framework/variation/version"]).

    A notebook started through the API cannot attach a model while it runs (the hub answers "New Models cannot be
    attached in non-interactive sessions"), so what a job loads from the model hub has to be named here, at push."""
    path = ROOT / "data" / "experiments" / "jobs" / f"{job}.json"
    return list(json.loads(path.read_text()).get("model_sources", [])) if path.exists() else []


def metadata(username, job, shard, gpu=False, tpu=False, models=()):
    return {"id": f"{username}/{slug(job, shard)}", "title": slug(job, shard).replace("-", " "),
            "code_file": "worker.py", "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": str(gpu).lower(), "enable_tpu": str(tpu).lower(), "enable_internet": "true",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": list(models)}


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
    # The CLI prints a warning before the table when it is out of date (not on every call): the first row is after
    # the header, which starts "ref,", wherever that is. Read from the first line, the account came out as "ref".
    header = next((i for i, line in enumerate(out) if line.startswith("ref,")), None)
    if header is None or header + 1 >= len(out):
        sys.exit("no Kaggle account: set KAGGLE_USERNAME")
    return out[header + 1].split(",")[0].split("/")[0]


class PushRefused(Exception):
    """Kaggle took the push and said it would not run the notebook."""


def pushed(result) -> bool:
    """Whether a `kernels push` made a notebook. The command exits 0 either way: a refusal - "Maximum weekly GPU quota
    of 30.00 hours reached." - is only in its text, as "Kernel push error: ...", and a success says "successfully
    pushed"."""
    return "successfully pushed" in (result.stdout + result.stderr).lower()


def start(job, shards, gpu=False, extras="rl", setup=(), tpu=False):
    """Push a notebook per shard. Raises PushRefused when the first is refused, before anything has been started, so
    that the caller can try again later; a refusal after the first is printed and the rest are still pushed."""
    user = username()
    for shard in range(shards):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "worker.py").write_text(script(job, shard, shards, extras, setup))
            (Path(folder) / "kernel-metadata.json").write_text(json.dumps(metadata(user, job, shard, gpu, tpu, model_sources(job))))
            result = kaggle("kernels", "push", "-p", folder)
        text = (result.stdout + result.stderr).strip()
        print(slug(job, shard), text.splitlines()[-1] if text else "no answer")
        if shard == 0 and not pushed(result):
            raise PushRefused(text.splitlines()[-1] if text else "no answer from kaggle")


#: What a failed status query says when the fault is the service's and not the notebook's: too many requests, a server
#: error, no connection.
TRANSIENT = re.compile(r"\b(429|50[0-4])\b|too many requests|timed out|connection", re.IGNORECASE)


def shard_status(result) -> str:
    """The status a `kernels status` answered, without its prefix; "UNKNOWN" when the query itself failed for a reason
    that says nothing about the notebook (so that it is asked again and not taken for the end of the job), and "?" when
    it failed for any other (no such notebook)."""
    named = result.stdout.strip().split('"')[-2:-1]
    if named:
        return named[0].replace("KernelWorkerStatus.", "")
    return "UNKNOWN" if TRANSIENT.search(result.stdout + result.stderr) else "?"


def status(job, shards, attempts=3, wait=10, sleep=time.sleep):
    """{shard: status}. A query that fails ("?" or "UNKNOWN") is asked again `attempts` times, `wait` seconds apart: one
    that failed for the service's own reasons must not read as a notebook that is gone, or a running job is taken for
    an ended one, collected with nothing and marked failed."""
    user = username()
    out = {}
    for shard in range(shards):
        for attempt in range(attempts):
            out[shard] = shard_status(kaggle("kernels", "status", f"{user}/{slug(job, shard)}"))
            if out[shard] not in ("?", "UNKNOWN"):
                break
            if attempt + 1 < attempts:
                sleep(wait)
    return out


def delete(job, shards):
    """Delete the job's notebooks from Kaggle. True when every one is gone (one that was already gone counts)."""
    user = username()
    gone = True
    for shard in range(shards):
        result = kaggle("kernels", "delete", "-y", f"{user}/{slug(job, shard)}")
        text = (result.stdout + result.stderr).lower()
        if result.returncode != 0 and "deleted successfully" not in text and "not found" not in text \
                and "permission" not in text:
            print(slug(job, shard), "not deleted:", text.strip().splitlines()[-1] if text.strip() else "?")
            gone = False
    return gone


class CollectFailed(Exception):
    """A notebook's output could not be downloaded: Kaggle answered with nothing at all, not even the log."""


#: Where a notebook's log is kept, a file a notebook: the log is what says why a notebook ended in error.
LOGS = "logs"


def collect(job, shards, attempts=3, wait=20, sleep=time.sleep):
    """Download every notebook's output; returns the files put in RUNS (the output's own folders kept).

    The notebook's log goes to RUNS/logs/ whatever the notebook did, so that a notebook that ended in error is
    explained the moment it is collected. A download that gives nothing - not even a log - is the service's fault
    and not an empty notebook: it is asked again `attempts` times, `wait` seconds apart, and then raises
    CollectFailed rather than let the job end with nothing collected."""
    user = username()
    RUNS.mkdir(parents=True, exist_ok=True)
    got = []
    for shard in range(shards):
        for attempt in range(attempts):
            with tempfile.TemporaryDirectory() as folder:
                kaggle("kernels", "output", f"{user}/{slug(job, shard)}", "-p", folder)
                found = sorted(path for path in Path(folder).rglob("*") if path.is_file())
                if found:
                    for path in found:
                        relative = path.relative_to(folder)
                        if path.suffix == ".log":
                            (RUNS / LOGS).mkdir(exist_ok=True)
                            shutil.copy(path, RUNS / LOGS / f"{slug(job, shard)}.log")
                        elif path.name != "__results__.html" and relative.parts[0] != "ARC-AGI":
                            target = RUNS / relative             # not a notebook that cloned into its working folder
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(path, target)
                            got.append(target)
                    break
            if attempt + 1 < attempts:
                sleep(wait)
        else:
            raise CollectFailed(f"{slug(job, shard)}: nothing came back in {attempts} tries")
    return got


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["start", "status", "collect", "delete"])
    parser.add_argument("job")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tpu", action="store_true")
    parser.add_argument("--setup", action="append", default=[], help="a shell command to run before the worker; repeatable")
    parser.add_argument("--extras", default="rl", help="the extras of the package to install")
    args = parser.parse_args()
    if args.command == "start":
        start(args.job, args.shards, args.gpu, args.extras, args.setup, args.tpu)
    elif args.command == "status":
        print(status(args.job, args.shards))
    elif args.command == "delete":
        print(delete(args.job, args.shards))
    else:
        for path in collect(args.job, args.shards):
            print(path)


if __name__ == "__main__":
    main()
