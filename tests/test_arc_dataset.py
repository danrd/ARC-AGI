"""Tests for ARCDataset's loading, in particular the ver2 (ARC-AGI-2) path.

That path was dead: it read from `data/dataset/ARC2/` while the files sit in
`data/datasets/ARC2/`, so `ARCDataset(ver2=True)` raised FileNotFoundError
before doing anything. Nothing here exercised it, so the typo survived. The
first test below is the one that would have caught it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import data.datasets.ARC.arc_dataset as arc_dataset
from data.datasets.ARC.arc_dataset import ARCDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
ARC2_DIR = REPO_ROOT / "data" / "datasets" / "ARC2"


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    """ARCDataset reads its files through paths relative to the repo root,
    so these tests have to run from there regardless of where pytest was
    invoked."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def arc1() -> ARCDataset:
    return ARCDataset()


@pytest.fixture
def arc2() -> ARCDataset:
    return ARCDataset(ver2=True)


def _arc2_keys() -> set:
    keys = set()
    for half in ("training", "evaluation"):
        with open(ARC2_DIR / f"arc-agi_{half}_challenges.json") as f:
            keys |= set(json.load(f))
    return keys


def test_the_plain_dataset_loads_arc1(arc1):
    """800 tasks plus the extra test cases some of them carry."""
    assert len(arc1.tasks) > 800
    assert set(arc1.subsets) == {"arc1_train_add", "arc1_eval_add"}


def test_ver2_loads_arc2_from_disk(arc1, arc2):
    """The regression test for the path: this raised FileNotFoundError."""
    assert len(arc2.tasks) > len(arc1.tasks)
    assert {"arc2_train_add", "arc2_eval_add"} <= set(arc2.subsets)


def test_ver2_adds_every_arc2_task(arc1, arc2):
    """Nothing is dropped between the two files and the task list, and every
    ARC-AGI-2 task lands on top of the ARC-AGI-1 list rather than replacing
    anything in it."""
    arc2_keys = _arc2_keys()
    labels = set(task.label for task in arc2.tasks)

    assert arc2_keys <= labels
    assert len(arc2.tasks) == (len(arc1.tasks) + len(arc2_keys)
                               + arc2.subsets["arc2_train_add"]
                               + arc2.subsets["arc2_eval_add"])


def test_arc2_repeats_the_arc1_tasks_it_repackages(arc2):
    """Not a bug being asserted as correct - a known cost of ver2 being
    additive, pinned so it can't change unnoticed. ARC-AGI-2 re-packages most
    of ARC-AGI-1's training set, so those tasks sit in the list twice under
    one label, and idx2label maps only one index back to each.
    """
    repeated = [label for label, n in Counter(t.label for t in arc2.tasks).items() if n > 1]

    assert len(repeated) > 700
    assert len(set(arc2.idx2label.values())) < len(arc2.idx2label)


# -- the train/eval boundary ---------------------------------------------------

def _stub_task(color: int, test_cases: int):
    return {
        "train": [{"input": [[color, 0], [0, color]], "output": [[color]]}],
        "test": [{"input": [[color, color]]} for _ in range(test_cases)],
    }


def _stub_half(prefix: str, count: int, test_cases: int):
    challenges = {f"{prefix}{i:03d}": _stub_task(i % 9 + 1, test_cases) for i in range(count)}
    solutions = {key: [[[1]] for _ in range(test_cases)] for key in challenges}
    return challenges, solutions


def test_the_train_eval_boundary_follows_the_files_not_a_fixed_index(monkeypatch, arc1):
    """The split used to be `idx == 999` / `idx == 1119`, right only for
    exactly 1000 + 120 tasks. On anything shorter the second one never fired,
    so the eval half's extra test cases were built and then dropped without a
    word. Three plus two tasks here - both subsets still have to be recorded,
    and every extra test case has to survive.
    """
    train_ch, train_sol = _stub_half("tr", 3, test_cases=2)
    eval_ch, eval_sol = _stub_half("ev", 2, test_cases=3)
    stubs = {
        "data/datasets/ARC2/arc-agi_training_challenges.json": train_ch,
        "data/datasets/ARC2/arc-agi_training_solutions.json": train_sol,
        "data/datasets/ARC2/arc-agi_evaluation_challenges.json": eval_ch,
        "data/datasets/ARC2/arc-agi_evaluation_solutions.json": eval_sol,
    }
    # Patched only after arc1 is built, and only for the ARC2 paths - the
    # ARC-AGI-1 files this dataset already loaded are real.
    monkeypatch.setattr(arc_dataset, "load_json", lambda path: stubs[path])

    before = len(arc1.additional_tasks)
    arc1.load_ARC2()

    # One extra test case per train task, two per eval task.
    assert arc1.subsets["arc2_train_add"] == 3
    assert arc1.subsets["arc2_eval_add"] == 4
    assert len(arc1.additional_tasks) == before + 7


def test_a_missing_arc2_file_says_which_one(monkeypatch, arc1):
    """What the typo produced. Kept so the failure stays a plain, readable
    FileNotFoundError naming the path, rather than something swallowed into a
    silently short dataset."""
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(arc_dataset, "load_json", missing)

    with pytest.raises(FileNotFoundError, match="ARC2"):
        arc1.load_ARC2()
