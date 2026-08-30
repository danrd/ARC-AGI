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

    python scripts/compare_llm_arms.py danrd/llm_run --list
    python scripts/compare_llm_arms.py danrd/llm_run \\
        "without knowledge injection" "with summary" --report arms.md

--report is the part worth running. Seven flipped tasks out of 370 will
never reach significance, but seven tasks can be read: what the symbolic
summary actually said about each, how the two answers differed, and
whether the gains share anything. That is the only conclusion a sample this
size can support, and it is not a number. The grids have to be written
somewhere at all: a figure drawn under !python has nowhere to appear.

Which format follows the file's suffix. .html is one self-contained file
with the images inlined, for sending to someone; .md writes them into
<name>_images/ beside itself, because an editor previews Markdown and
shows HTML as source, and a relative link renders in that preview and on
GitHub where an inlined data URI renders in neither.
"""
from __future__ import annotations

import argparse
import base64
import collections
import difflib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def arm_filters(description: str, model: str | None = None) -> dict:
    """What identifies one arm in wandb.

    The description is matched as a substring, not for equality. Stored
    descriptions are free text typed per run and they drift - "with
    summary" against "with summary, without knowledge injection" - so an
    exact match is a query that returns nothing and says nothing about why.

    `display_name` is the run's name; `name` in a filter means the run id
    instead, which is the usual way one of these comes back empty.
    """
    filters = {"config.run_description": {"$regex": description}}
    if model:
        filters["config.model"] = {"$regex": model}
    return filters


def find_runs(api, path: str, description: str, model: str | None = None) -> dict:
    """Every run of one arm, keyed by the shard it covered.

    Ambiguity is refused rather than resolved. One description can cover
    several models, and quietly keeping whichever run the iterator happened
    to yield first would compare one model's baseline against another
    model's summary arm - a result that looks ordinary and means nothing.
    A shard run twice by the same model is different: that is a re-run, and
    the newer one wins.
    """
    candidates = collections.defaultdict(list)
    for run in api.runs(path, filters=arm_filters(description, model)):
        candidates[run.name].append(run)

    found = {}
    for shard, runs in candidates.items():
        models = {str(r.config.get("model", "")) for r in runs}
        if len(models) > 1:
            raise SystemExit(
                f"'{description}' matches {len(runs)} runs of shard '{shard}' "
                f"across {len(models)} models:\n  " + "\n  ".join(sorted(models)) +
                "\nNarrow it with --model, or run --list to see what is stored.")
        # api.runs returns newest first, so a re-run of one shard reports
        # the newer result rather than whichever came last.
        found[shard] = runs[0]
    return found


def describe_project(api, path: str, model: str | None = None) -> None:
    """What is actually stored, so an arm can be named rather than guessed.

    The two strings this script takes are typed by hand into a run months
    apart, and a filter that matches nothing is indistinguishable from a
    project that holds nothing.
    """
    filters = {"config.model": {"$regex": model}} if model else None
    grouped = collections.defaultdict(list)
    for run in api.runs(path, filters=filters):
        grouped[(str(run.config.get("model", "?")),
                 str(run.config.get("run_description", "?")))].append(run)

    if not grouped:
        raise SystemExit(f"{path} holds no runs matching that")
    print(f"{'model':46s} {'run_description':44s} {'n':>3s}  shards")
    for (model_name, description), runs in sorted(grouped.items()):
        shards = ", ".join(sorted(r.name for r in runs))
        print(f"{model_name[-46:]:46s} {description[:44]:44s} {len(runs):3d}  {shards}")


def load_checkpoint(api, path: str, run, downloaded: list | None = None) -> dict | None:
    """The run's checkpoint artifact, which carries solved_tasks
    (authoritative, not inferred from a score threshold) and prompts_data,
    with each task's prompt, generation and time.

    `downloaded` collects the directories wandb unpacked into, so they can
    be removed afterwards. They are not small - prompts_data holds every
    prompt and every generation - and comparing five shards leaves ten of
    them behind under ./artifacts/.
    """
    try:
        artifact = api.artifact(f"{path}/checkpoint-{run.id}:latest")
        directory = Path(artifact.download())
        if downloaded is not None:
            downloaded.append(directory)
        with open(directory / "checkpoint.json") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  ! {run.name} ({run.id}): no checkpoint - {type(exc).__name__}")
        return None


def remove_downloads(directories) -> None:
    """Delete the artifact directories this run unpacked, and nothing else.

    Each path came back from artifact.download(), so it names one
    checkpoint's own directory - but it is still a recursive delete driven
    by a value from outside, so the name is checked before the tree goes.
    A directory that does not look like a checkpoint is left alone and
    said so, rather than removed on the assumption that it must be ours.
    """
    import shutil

    removed = 0
    for directory in dict.fromkeys(directories):
        if not directory.is_dir():
            continue
        if "checkpoint-" not in directory.name:
            print(f"  left {directory} alone: not a checkpoint directory")
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed += not directory.exists()
        parent = directory.parent
        # wandb makes ./artifacts/ for these; take it too once it is empty,
        # but only if nothing else was using it.
        if parent.name == "artifacts" and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    if removed:
        print(f"removed {removed} downloaded checkpoint "
              f"{'directory' if removed == 1 else 'directories'}")


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


def task_image(task_id: str) -> bytes:
    """The task's train pairs and answer as PNG bytes.

    Rendered to memory rather than to a window: these scripts run under
    !python from a notebook, where a figure has nowhere to appear. The
    bytes go into the HTML report as a data URI so the report is one file
    that can be sent to someone.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.datasets.ARC.arc_dataset import ARCDataset
    from utils.plotting import plot_task

    if not hasattr(task_image, "_dataset"):
        task_image._dataset = ARCDataset()
    figure = plot_task(task_id, task_image._dataset)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


def save_task_image(task_id: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_id}.png"
    path.write_bytes(task_image(task_id))
    return path


REPORT_STYLE = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 2rem;
       max-width: 1100px; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2.5rem; }
h3 { font-size: 1rem; margin: 2rem 0 .5rem; }
table { border-collapse: collapse; margin: .5rem 0; }
th, td { padding: .25rem .75rem; text-align: right; border-bottom: 1px solid #8884; }
th:first-child, td:first-child { text-align: left; }
pre { overflow-x: auto; padding: .6rem .8rem; background: #8881; border-radius: 4px;
      font-size: 12px; margin: .4rem 0; }
.add { color: #1a7f37; } .del { color: #cf222e; }
img { max-width: 100%; border: 1px solid #8884; border-radius: 4px; }
.note { color: #8889; font-size: 13px; }
.flip { border-left: 3px solid #8884; padding-left: 1rem; margin: 2rem 0; }
"""


def _table(headers, rows) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
                   + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _diff_html(lines) -> str:
    if not lines:
        return "<p class=note>the prompts are identical</p>"
    out = []
    for line in lines:
        css = "add" if line.startswith("+") else "del"
        out.append(f"<span class={css}>{html.escape(line)}</span>")
    return "<pre>" + "\n".join(out) + "</pre>"


def _markdown_report(arm_a: str, arm_b: str, per_shard, cost, totals,
                     gained, lost, p_value, flips, image_dir: Path | None) -> str:
    """The same report as Markdown, for reading in an editor.

    VS Code opens .html as source - it has no built-in HTML preview the way
    it has one for Markdown (ctrl+shift+V) - so a report meant to be looked
    at where the code is has to be .md.

    Images are written beside it and linked, not inlined. A data URI put
    the whole PNG on one 74,715-character line, and it did not appear in
    the editor's preview - which runs in a webview under a content policy
    that can refuse data: sources. Rather than establish which of those it
    was, this drops the dependency on both: a relative link to a file
    renders in that preview, on GitHub (which strips data URIs outright),
    and in anything else. HTML keeps the data URIs, where being one file to
    send is worth more and a browser always shows them.
    """
    def table(headers, rows):
        out = ["| " + " | ".join(str(h) for h in headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        return "\n".join(out)

    lines = [f"# Prompt arms compared\n",
             f"- **A**: {arm_a}", f"- **B**: {arm_b}\n",
             "## By shard\n",
             table(["shard", "tasks", "solved A", "solved B", "gained", "lost",
                    "score up", "score down"], per_shard),
             "\n## Solved, pooled\n",
             f"A {totals['solved_a']}, B {totals['solved_b']} of {totals['tasks']}. "
             f"Gained {len(gained)}, lost {len(lost)}. Exact two-sided "
             f"p = {p_value:.3f} on {len(gained) + len(lost)} discordant pairs — "
             f"**{'significant' if p_value < 0.05 else 'not significant'}**.\n",
             "> A binary outcome at this base rate yields a handful of discordant "
             "pairs however many tasks are run, so the flipped tasks below are the "
             "usable output, not the p-value.\n",
             "## Cost\n",
             table(["shard", "prompt A", "prompt B", "generated A", "generated B",
                    "min/task A", "min/task B"],
                   [(s, f"{pa:.0f}", f"{pb:.0f}", f"{ga:.0f}", f"{gb:.0f}",
                     f"{ma:.2f}", f"{mb:.2f}") for s, pa, pb, ga, gb, ma, mb in cost]),
             "\n> Read the prompt and generation columns before reading anything "
             "into time: a per-task time that swings between shards of the same arm "
             "is the backend changing under the run, not the prompt.\n",
             f"## The {len(flips)} tasks that flipped\n"]

    for task_id, verdict, a, b in flips:
        lines.append(f"### {task_id} — {verdict}\n")
        if image_dir is not None:
            try:
                saved = save_task_image(task_id, image_dir)
                lines.append(f"![{task_id}]({image_dir.name}/{saved.name})\n")
            except Exception as exc:
                lines.append(f"*could not plot: {type(exc).__name__}*\n")
        lines.append(f"prompt {a['prompt']} chars in A, {b['prompt']} in B\n")
        difference = prompt_difference(a["prompt_text"], b["prompt_text"])
        lines.append("```diff\n" + ("\n".join(difference) if difference
                                    else "  (the prompts are identical)") + "\n```\n")
        for label, side in ((arm_a, a), (arm_b, b)):
            lines.append(f"**[{label}]** scored {side['score']:.3f}\n")
            lines.append("```\n" + side["generated"].strip() + "\n```\n")
    return "\n".join(lines)


def write_report(path: Path, arm_a: str, arm_b: str, per_shard, cost,
                 totals, gained, lost, p_value, flips, with_images: bool) -> None:
    """One self-contained file: tables, per-task grids and prompt diffs.

    Images are inlined as data URIs rather than written alongside, so the
    report is a single thing to open or send. It is also the only way the
    grids get seen at all - a figure drawn under !python has nowhere to go.

    Markdown when the path says .md, HTML otherwise. Which one is wanted
    depends only on where it will be read: an editor previews Markdown and
    shows HTML as source, a browser does the reverse. The Markdown form
    writes its images into <name>_images/ beside itself rather than
    inlining them - see _markdown_report.
    """
    if path.suffix.lower() in (".md", ".markdown"):
        path.parent.mkdir(parents=True, exist_ok=True)
        image_dir = path.parent / f"{path.stem}_images" if with_images else None
        path.write_text(_markdown_report(arm_a, arm_b, per_shard, cost, totals,
                                          gained, lost, p_value, flips, image_dir),
                        encoding="utf-8")
        extra = f" and {image_dir}/" if image_dir and image_dir.exists() else ""
        print(f"\nwrote {path}{extra}")
        return
    parts = [f"<style>{REPORT_STYLE}</style>",
             "<h1>Prompt arms compared</h1>",
             f"<p class=note>A: {html.escape(arm_a)}<br>B: {html.escape(arm_b)}</p>",
             "<h2>By shard</h2>",
             _table(["shard", "tasks", "solved A", "solved B", "gained", "lost",
                     "score up", "score down"], per_shard),
             "<h2>Solved, pooled</h2>",
             f"<p>A {totals['solved_a']}, B {totals['solved_b']} of {totals['tasks']}. "
             f"Gained {len(gained)}, lost {len(lost)}. Exact two-sided "
             f"p = {p_value:.3f} on {len(gained) + len(lost)} discordant pairs "
             f"&mdash; {'significant' if p_value < 0.05 else 'not significant'}.</p>",
             "<p class=note>A binary outcome at this base rate yields a handful of "
             "discordant pairs however many tasks are run, so the flipped tasks "
             "below are the usable output, not the p-value.</p>",
             "<h2>Cost</h2>",
             _table(["shard", "prompt A", "prompt B", "generated A", "generated B",
                     "min/task A", "min/task B"],
                    [(s, f"{pa:.0f}", f"{pb:.0f}", f"{ga:.0f}", f"{gb:.0f}",
                      f"{ma:.2f}", f"{mb:.2f}") for s, pa, pb, ga, gb, ma, mb in cost]),
             "<p class=note>Read the prompt and generation columns before reading "
             "anything into time: a per-task time that swings between shards of the "
             "same arm is the backend changing under the run, not the prompt.</p>"]

    parts.append(f"<h2>The {len(flips)} tasks that flipped</h2>")
    for task_id, verdict, a, b in flips:
        parts.append(f"<div class=flip><h3>{html.escape(task_id)} &mdash; "
                     f"{html.escape(verdict)}</h3>")
        if with_images:
            try:
                encoded = base64.b64encode(task_image(task_id)).decode()
                parts.append(f'<img src="data:image/png;base64,{encoded}" '
                             f'alt="{html.escape(task_id)}">')
            except Exception as exc:
                parts.append(f"<p class=note>could not plot: "
                             f"{html.escape(type(exc).__name__)}</p>")
        parts.append(f"<p class=note>prompt {a['prompt']} chars in A, "
                     f"{b['prompt']} in B</p>")
        parts.append(_diff_html(prompt_difference(a["prompt_text"], b["prompt_text"])))
        for label, side in ((arm_a, a), (arm_b, b)):
            parts.append(f"<p class=note>[{html.escape(label)}] scored "
                         f"{side['score']:.3f}</p>"
                         f"<pre>{html.escape(side['generated'].strip())}</pre>")
        parts.append("</div>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><meta charset=utf-8>"
                    "<title>Prompt arms compared</title>" + "".join(parts),
                    encoding="utf-8")
    print(f"\nwrote {path}")


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
    parser.add_argument("arm_a", nargs="?",
                        help="substring of the baseline arm's config.run_description")
    parser.add_argument("arm_b", nargs="?",
                        help="substring of the run_description of the arm under test")
    parser.add_argument("--model", help="substring of config.model, when several were run")
    parser.add_argument("--keep-downloads", action="store_true",
                        help="keep the checkpoint artifacts wandb unpacks into "
                             "./artifacts/. They are deleted when the script "
                             "finishes otherwise - comparing five shards downloads "
                             "ten of them, each holding every prompt and generation "
                             "of its run")
    parser.add_argument("--list", action="store_true",
                        help="print the model/run_description combinations the "
                             "project holds and exit - descriptions are free text "
                             "and drift, so name them from this rather than memory")
    parser.add_argument("--detail", action="store_true",
                        help="print each flipped task: what the prompts differ by "
                             "and what each arm answered")
    parser.add_argument("--report", type=Path, metavar="FILE",
                        help="write one self-contained report with the tables, "
                             "each flipped task's grids, the prompt difference and "
                             "both answers. Images are inlined, so the file is the "
                             "whole report and can be sent as it is - which is also "
                             "the only way the grids get seen, since a figure drawn "
                             "under !python has nowhere to appear. Markdown if "
                             "the name ends .md, HTML otherwise - an editor "
                             "previews Markdown and shows HTML as source, a "
                             "browser does the reverse")
    parser.add_argument("--detail-dir", type=Path, metavar="DIR",
                        help="directory to write one PNG per flipped task into, "
                             "showing that task's train pairs and answer; created "
                             "if missing. Saved rather than shown because these "
                             "scripts run under !python, where a figure has "
                             "nowhere to appear")
    args = parser.parse_args()

    import wandb
    api = wandb.Api()
    if args.list:
        describe_project(api, args.path, args.model)
        return
    if not args.arm_a or not args.arm_b:
        parser.error("two arms are required unless --list is given")
    a_runs = find_runs(api, args.path, args.arm_a, args.model)
    b_runs = find_runs(api, args.path, args.arm_b, args.model)
    shards = sorted(set(a_runs) & set(b_runs))
    print(f"A ({args.arm_a}): {sorted(a_runs)}")
    print(f"B ({args.arm_b}): {sorted(b_runs)}")
    print(f"paired shards: {shards}\n")
    if not shards:
        raise SystemExit("nothing paired - run with --list to see the "
                         "model/run_description combinations that exist")

    downloaded = []
    try:
        run_comparison(api, args, shards, a_runs, b_runs, downloaded)
    finally:
        # In a finally: a run that failed halfway still downloaded whatever
        # it got to, and leaving those behind is the case that accumulates.
        if not args.keep_downloads:
            remove_downloads(downloaded)


def run_comparison(api, args, shards, a_runs, b_runs, downloaded) -> None:
    gained_all, lost_all, flips = [], [], []
    totals = collections.Counter()
    per_shard, cost = [], []
    for shard in shards:
        a_data = load_checkpoint(api, args.path, a_runs[shard], downloaded)
        b_data = load_checkpoint(api, args.path, b_runs[shard], downloaded)
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

    if args.report:
        write_report(args.report, args.arm_a, args.arm_b, per_shard, cost,
                     totals, gained_all, lost_all, p, flips, with_images=True)


if __name__ == "__main__":
    main()
