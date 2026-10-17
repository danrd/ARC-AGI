"""Grade the kept replies of orchestration.llm_only runs again, with the grade the module has now.

Until read_grid, a reply was read by parse_llm_output, which takes only "1 x_1x_2..." - and the prompt asks for
"1 x_1 ... x_m". A model with no grammar that wrote what it was asked was graded unreadable. The replies are in
the run files, so the runs need no GPU to be graded again:

    python -m orchestration.regrade_llm_only data/experiments/runs/kaggle/llm_only_*.jsonl data/experiments/runs/kaggle/think_*.jsonl \\
        --out data/experiments/regrade_llm_only.json

Per file: solved as recorded and as graded now, replies read, and how many of those the old parser read. The
run files are left as they are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestration.llm_only import grade
from orchestration.run import load_task


def regrade(rows, loader=load_task):
    """{'runs', 'solved_before', 'solved_now', 'read_now', 'read_by_old_parser', 'gained'} over `rows`."""
    out = {"runs": 0, "solved_before": 0, "solved_now": 0, "read_now": 0, "read_by_old_parser": 0, "gained": []}
    for row in rows:
        now = grade(loader(row.get("split", "evaluation"), row["task"]), row.get("reply") or "")
        out["runs"] += 1
        out["solved_before"] += bool(row.get("solved"))
        out["solved_now"] += now["solved"]
        out["read_now"] += now["parsed"]
        out["read_by_old_parser"] += now["format_ok"]
        if now["solved"] and not row.get("solved"):
            out["gained"].append(row["task"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = {}
    for path in args.files:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        report[Path(path).name] = summary = regrade(rows)
        print(f"{Path(path).name:45s} solved {summary['solved_before']:3d} -> {summary['solved_now']:3d} of "
              f"{summary['runs']:3d}; read {summary['read_now']:3d}, by the old parser {summary['read_by_old_parser']:3d}",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
