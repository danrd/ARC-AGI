#!/usr/bin/env python3
"""Sync subsymbolic/{llm_setup,llm_runtime,prompt_builder,llm_run,logging}.py
from toolkit's llm_kit (the canonical source - see
https://github.com/danrd/toolkit/blob/main/llm_kit/README.md).

Run manually after a change lands in toolkit/llm_kit, when this repo's
copy should catch up. Not automatic: arc-agi is still fast-moving enough
that having the whole system readable in one clone - rather than living
behind a dependency, as lector/maestro now do - is worth more than
avoiding this by-hand step.

Usage:
    python scripts/sync_llm_kit.py [--ref main]

Clones toolkit at --ref (default: main) into a temp dir, copies the five
files over, and rewrites every `llm_kit` reference (imports, prose) to
`subsymbolic` - the two are otherwise meant to be identical. Always
review `git diff` before committing: subsymbolic/llm_run.py in particular
carries a real, deliberate naming difference (subsymbolic_module vs
llm_kit's more generic llm_module) that this substitution won't
reproduce - that file's diff needs a manual look, not a blind commit.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

TOOLKIT_REPO = "https://github.com/danrd/toolkit.git"
SYNCED_FILES = ["llm_setup.py", "llm_runtime.py", "prompt_builder.py", "llm_run.py", "logging.py"]
DEST_DIR = Path(__file__).resolve().parent.parent / "subsymbolic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main", help="toolkit branch/tag/commit to sync from")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "toolkit"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", args.ref, TOOLKIT_REPO, str(clone_dir)],
            check=True,
        )
        source_dir = clone_dir / "llm_kit"

        changed = []
        for name in SYNCED_FILES:
            text = (source_dir / name).read_text().replace("llm_kit", "subsymbolic")
            dest = DEST_DIR / name
            if not dest.exists() or dest.read_text() != text:
                dest.write_text(text)
                changed.append(name)

    if not changed:
        print("Already up to date with toolkit's llm_kit.")
        return

    print(f"Synced: {', '.join(changed)}")
    print(
        "Review `git diff` before committing - in particular "
        "subsymbolic/llm_run.py carries a deliberate naming difference "
        "(subsymbolic_module vs llm_kit's llm_module) this substitution "
        "doesn't reproduce; check that file's diff by hand. Run the test "
        "suite too (uv run --extra test pytest) before committing."
    )


if __name__ == "__main__":
    main()
