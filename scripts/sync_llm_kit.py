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
    python scripts/sync_llm_kit.py [--ref main]    # clone toolkit at --ref and copy
    python scripts/sync_llm_kit.py --source PATH   # a checkout already on disk
    python scripts/sync_llm_kit.py --check         # report the differences, change nothing

Copies the five files over, rewrites every `llm_kit` reference (imports,
prose) to `subsymbolic`, and then applies LOCAL_EDITS: the few places where
arc-agi deliberately differs from toolkit (the model cache path, and
llm_run's `subsymbolic_module` where toolkit says `llm_module`). Each of
those edits must find the line it changes, so a toolkit change that moves
one stops the sync instead of quietly dropping the difference. Work done
here first belongs in toolkit before this is run: a sync overwrites.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

TOOLKIT_REPO = "https://github.com/danrd/toolkit.git"
SYNCED_FILES = ["llm_setup.py", "llm_runtime.py", "prompt_builder.py", "llm_run.py", "logging.py"]
DEST_DIR = Path(__file__).resolve().parent.parent / "subsymbolic"


#: file -> [(toolkit text, arc-agi text)]: where this repo deliberately differs. Applied to the substituted
#: toolkit text, and each must match once, so that a change in toolkit that moves the line is noticed.
LOCAL_EDITS = {
    "llm_setup.py": [
        ("    pretrained_models_dir: str = '/data/pretrained_models'\n",
         "    pretrained_models_dir: str = 'data/pretrained_models'\n"),
        ('getattr(config.llm, "pretrained_models_dir", "/data/pretrained_models")',
         'getattr(config.llm, "pretrained_models_dir", LlmConfig.model_fields["pretrained_models_dir"].default)'),
    ],
    "llm_run.py": [
        ("llm_module", "subsymbolic_module"),
    ],
}


def local_text(name: str, toolkit_text: str) -> str:
    """toolkit's file as arc-agi keeps it: `llm_kit` named `subsymbolic`, and the local edits made."""
    text = toolkit_text.replace("llm_kit", "subsymbolic")
    for old, new in LOCAL_EDITS.get(name, []):
        if old not in text:
            raise SystemExit(f"{name}: the line arc-agi edits locally is gone from toolkit: {old.strip()[:70]!r}. "
                             "Update LOCAL_EDITS before syncing.")
        text = text.replace(old, new)
    return text


def differences(source_dir: Path) -> list:
    """The synced files whose arc-agi copy is not what local_text makes of toolkit's."""
    return [name for name in SYNCED_FILES
            if not (DEST_DIR / name).exists()
            or (DEST_DIR / name).read_text() != local_text(name, (source_dir / name).read_text())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="toolkit branch/tag/commit to sync from")
    parser.add_argument("--source", help="a toolkit checkout on disk, instead of cloning")
    parser.add_argument("--check", action="store_true", help="say which files differ and change nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.source:
            source_dir = Path(args.source) / "llm_kit"
        else:
            clone_dir = Path(tmp) / "toolkit"
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", args.ref, TOOLKIT_REPO, str(clone_dir)],
                check=True,
            )
            source_dir = clone_dir / "llm_kit"

        changed = differences(source_dir)
        if args.check:
            print("In sync with toolkit's llm_kit." if not changed
                  else f"Differs from toolkit's llm_kit: {', '.join(changed)}")
            raise SystemExit(1 if changed else 0)
        for name in changed:
            (DEST_DIR / name).write_text(local_text(name, (source_dir / name).read_text()))

    if not changed:
        print("Already up to date with toolkit's llm_kit.")
        return

    print(f"Synced: {', '.join(changed)}")
    print("Review `git diff` before committing, and run the test suite (uv run --extra test pytest).")


if __name__ == "__main__":
    main()
