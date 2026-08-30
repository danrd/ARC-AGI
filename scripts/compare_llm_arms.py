#!/usr/bin/env python3
"""Compare two prompt arms of an LLM run across every shard at once.

Runs are found by filter rather than by id: llm_run puts run_description
into the config and the shard name into the run's display name, so an arm
is "config.run_description == ...", a shard is "display_name == ...", and
the pairing falls out of matching one against the other.

Pooling is the point of doing all the shards together. Tested shard by
shard, the summary-vs-baseline comparison saw 1, 4, 1, 1 and 0 discordant
pairs - a test on one pair says nothing, so five separate tests produce
five shrugs. The same data pooled is a single honest number (5 gained, 2
lost, exact p = 0.45), which at least states plainly that the sample cannot
settle it.

Not scored on the score. Score counts how much of the produced grid matches
the target, and on ARC that is not progress: a model that copies the input
scores well whenever the output resembles the input, and one that attempts
a real transformation and misses scores badly. 99% of the cells right is a
task not solved. `solved` is the outcome; score appears as a diagnostic
with that caveat attached.

Cost is read from the prompt and generation columns, not from time. Time
per task in one measured pair swung 14x between shards of the SAME arm,
which no prompt can do - the backend had changed under the run (see
llm_setup's fallback chain, which catches a server that failed to come up
and quietly continues on the next tier). The same comparison looked like a
6x slowdown from adding a summary and was 1.17x on prompt and 1.00x on
generation.

    python scripts/compare_llm_arms.py danrd/llm_run \\
        "without knowledge injection" "with summary"
    python scripts/compare_llm_arms.py ... --detail --detail-dir flips/

--detail is the part worth running. Seven flipped tasks out of 370 will
never reach significance, but seven tasks can be read: what the symbolic
summary actually said about each, how the two answers differed, and
whether the gains share anything. That is the only conclusion a sample this
size can support, and it is not a number.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def find_runs(api, path: str, description: str, model: str | None = None) -> dict:
    """Every run of one arm, keyed by the shard it covered.

    `display_name` is the run's name; `name` in a filter means the run id
    instead, which is the usual way this query comes back empty.
    """
    filters = {"config.run_description": description}
    if model:
        filters["config.model"] = {"$regex": model}
    found = {}
    for run in api.runs(path, filters=filters):
        # First wins: api.runs returns newest first, and a re-run of one
        # shard should not be silently mixed with the run it replaced.
        found.setdefault(run.name, run)
    return found


def load_checkpoint(api, path: str, run) -> dict | None:
    """The run's checkpoint artifact, which carries solved_tasks
    (authoritative, not inferred from a score threshold) and prompts_data,
    with each task's prompt, generation and time."""
    try:
        artifact = api.artifact(f"{path}/checkpoint-{run.id}:latest")
        with open(Path(artifact.download()) / "checkpoint.json") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  ! {run.name} ({run.id}): no checkpoint - {type(exc).__name__}")
        return None


def generation_text(row: dict) -> str:
    """The answer, whether the module returned it bare or wrapped.
    SubsymbolicModule.solve returns {"solution": ..., "module_results": ...};
    older records hold the string itself."""
    generation = row.get("generation_result")
    if isinstance(generation, dict):
        return generation.get("solution") or ""
    return generation or ""


def index_tasks(data: dict) -> dict:
    """task_id -> what that task cost, scored and answered."""
    out = {}
    for row in data.get("prompts_data", []):
        out[str(row["task_id"])] = {
            "score": row.get("primary_score") or 0.0,
            "prompt_text": row.get("prompt_text") or "",
            "prompt": row.get("prompt_length") or 0,
            "generated": generation_text(row),
            "minutes": row.get("processing_time_min") or 0.0,
        }
    return out


def two_sided_binomial(gained: int, lost: int) -> float:
    """Exact, not chi-square. With a handful of discordant pairs the
    chi-square approximation reads optimistic, and a handful is what a
    binary outcome at a 7% base rate produces however many tasks are run."""
    n = gained + lost
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(gained, lost), n + 1))
    return min(1.0, 2 * tail / 2 ** n)


def prompt_difference(before: str, after: str, context: int = 0) -> list:
    """The lines one prompt has and the other does not.

    Expected to be the summary block and nothing else. Worth looking at
    rather than assuming: if the arms differ anywhere further, the
    comparison is not measuring what it says it measures, and a diff is how
    that shows up.
    """
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                lineterm="", n=context)
    return [line for line in diff
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]


def save_task_image(task_id: str, directory: Path):
    """The task's train pairs and answer as a PNG. Saved rather than shown:
    these scripts are run with !python from a notebook, where a figure has
    nowhere to appear."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.datasets.ARC.arc_dataset import ARCDataset
    from utils.plotting import plot_task

    if not hasattr(save_task_image, "_dataset"):
        save_task_image._dataset = ARCDataset()
    figure = plot_task(task_id, save_task_image._dataset)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_id}.png"
    figure.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(figure)
    return path


def report_flip(task_id: str, verdict: str, a: dict, b: dict,
                arm_a: str, arm_b: str, detail_dir: Path | None):
    print(f"\n--- {task_id}  ({verdict}) ---")
    if detail_dir is not None:
        try:
            print(f"  grids: {save_task_image(task_id, detail_dir)}")
        except Exception as exc:
            print(f"  grids: could not plot - {type(exc).__name__}: {exc}")

    added = prompt_difference(a["prompt_text"], b["prompt_text"])
    print(f"  prompt: {a['prompt']} chars in A, {b['prompt']} in B; "
          f"{len(added)} lines differ")
    for line in added[:40]:
        print(f"    {line}")
    if len(added) > 40:
        print(f"    ... {len(added) - 40} more")

    for label, side in ((arm_a, a), (arm_b, b)):
        answer = side["generated"].strip()
        print(f"  [{label}] scored {side['score']:.3f}, {len(answer)} chars:")
        for line in answer.splitlines()[:12]:
            print(f"    {line}")
        if len(answer.splitlines()) > 12:
            print(f"    ... {len(answer.splitlines()) - 12} more lines")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="entity/project")
    parser.add_argument("arm_a", help="config.run_description of the baseline arm")
    parser.add_argument("arm_b", help="config.run_description of the arm under test")
    parser.add_argument("--model", help="substring of config.model, when several were run")
    parser.add_argument("--detail", action="store_true",
                        help="print each flipped task: what the prompts differ by "
                             "and what each arm answered")
    parser.add_argument("--detail-dir", type=Path,
                        help="also write one PNG per flipped task here")
    args = parser.parse_args()

    import wandb
    api = wandb.Api()
    a_runs = find_runs(api, args.path, args.arm_a, args.model)
    b_runs = find_runs(api, args.path, args.arm_b, args.model)
    shards = sorted(set(a_runs) & set(b_runs))
    print(f"A ({args.arm_a}): {sorted(a_runs)}")
    print(f"B ({args.arm_b}): {sorted(b_runs)}")
    print(f"paired shards: {shards}\n")
    if not shards:
        raise SystemExit(
            "nothing paired. Check the descriptions against what is stored:\n"
            "  api.runs(path)[0].config['run_description']\n"
            "and remember that display_name is the run's name - `name` in a "
            "filter means its id.")

    gained_all, lost_all, flips = [], [], []
    totals = collections.Counter()
    per_shard, cost = [], []
    for shard in shards:
        a_data = load_checkpoint(api, args.path, a_runs[shard])
        b_data = load_checkpoint(api, args.path, b_runs[shard])
        if a_data is None or b_data is None:
            continue
        a, b = index_tasks(a_data), index_tasks(b_data)
        shared = sorted(set(a) & set(b))
        solved_a = {str(t) for t in a_data.get("solved_tasks", [])} & set(shared)
        solved_b = {str(t) for t in b_data.get("solved_tasks", [])} & set(shared)
        gained, lost = sorted(solved_b - solved_a), sorted(solved_a - solved_b)
        gained_all += gained
        lost_all += lost
        flips += [(t, "gained by B", a[t], b[t]) for t in gained]
        flips += [(t, "lost by B", a[t], b[t]) for t in lost]

        score = np.array([[a[t]["score"], b[t]["score"]] for t in shared])
        delta = score[:, 1] - score[:, 0]
        totals["tasks"] += len(shared)
        totals["solved_a"] += len(solved_a)
        totals["solved_b"] += len(solved_b)
        totals["score_up"] += int((delta > 0).sum())
        totals["score_down"] += int((delta < 0).sum())
        per_shard.append((shard, len(shared), len(solved_a), len(solved_b),
                          len(gained), len(lost),
                          int((delta > 0).sum()), int((delta < 0).sum())))
        cost.append((shard,
                     np.mean([a[t]["prompt"] for t in shared]),
                     np.mean([b[t]["prompt"] for t in shared]),
                     np.mean([len(a[t]["generated"]) for t in shared]),
                     np.mean([len(b[t]["generated"]) for t in shared]),
                     np.mean([a[t]["minutes"] for t in shared]),
                     np.mean([b[t]["minutes"] for t in shared])))

    print(f"{'shard':12s} {'n':>4s} {'A':>3s} {'B':>3s} {'+':>3s} {'-':>3s} "
          f"{'score up':>9s} {'down':>5s}")
    for row in per_shard:
        print(f"{row[0]:12s} {row[1]:4d} {row[2]:3d} {row[3]:3d} "
              f"{row[4]:3d} {row[5]:3d} {row[6]:9d} {row[7]:5d}")

    print("\n=== solved, pooled ===")
    print(f"  A {totals['solved_a']}   B {totals['solved_b']}   of {totals['tasks']}")
    print(f"  gained {len(gained_all)}: {gained_all}")
    print(f"  lost   {len(lost_all)}: {lost_all}")
    p = two_sided_binomial(len(gained_all), len(lost_all))
    print(f"  exact two-sided p = {p:.3f} on "
          f"{len(gained_all) + len(lost_all)} discordant pairs - "
          f"{'significant' if p < 0.05 else 'NOT significant'}")
    print("  the flipped tasks are the usable output; --detail reads them")

    up, down = totals["score_up"], totals["score_down"]
    print("\n=== score, diagnostic only ===")
    print(f"  better in B on {up} tasks, in A on {down}, of {up + down} that moved")
    if up + down:
        print(f"  z = {(up - (up + down) / 2) / math.sqrt((up + down) * 0.25):+.2f}")
    print("  movement, not progress: copying the input scores well whenever "
          "the output resembles the input")

    print("\n=== cost ===")
    print(f"{'shard':12s} {'prompt A':>9s} {'B':>8s} {'gen A':>7s} {'B':>7s} "
          f"{'min A':>7s} {'B':>7s} {'slower':>7s}")
    for shard, pa, pb, ga, gb, ma, mb in cost:
        print(f"{shard:12s} {pa:9.0f} {pb:8.0f} {ga:7.0f} {gb:7.0f} "
              f"{ma:7.2f} {mb:7.2f} {mb / (ma or 1e-9):6.2f}x")
    print("  a per-task time that swings between shards of the SAME arm is the "
          "backend changing under the run, not the prompt - read the prompt "
          "and generation columns before reading anything into time")

    if args.detail and flips:
        print(f"\n=== the {len(flips)} tasks that flipped ===")
        for task_id, verdict, a, b in flips:
            report_flip(task_id, verdict, a, b, args.arm_a, args.arm_b,
                        args.detail_dir)


if __name__ == "__main__":
    main()
