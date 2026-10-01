"""Tests for data/datasets/ARC/evaluation_difficulty.json - the hand-graded
difficulty of the 400 evaluation tasks.

task2difficulty.json next to it says only 'easy' for every training task and
'hard' for every evaluation one, which is the split and not a grade. This
file is the grade, kept in a file because it was being carried in a message.
It was written from indices (0..799, training then evaluation, the numbering
ARCDataset stamps on its tasks), so what is pinned here is that the ids it
holds are the ones those indices name. The 27 graded `symbolic` are the ones
the symbolic modules solve, which is why they were never shown to an LLM.
Three that were first put there and that no symbolic module solves
(25094a63, dd2401ed, f9d67f8b) went to the LLM; two are graded `medium`, and
f9d67f8b `oversized`: four 30 x 30 pairs and a 30 x 30 input are about 10k tokens of prompt and
another thousand of answer, more context than the rest of the split asks for,
so it is left out of the LLM runs by being in a level nobody selects.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from data.datasets.ARC.arc_dataset import ARCDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
ARC = REPO_ROOT / "data" / "datasets" / "ARC"

LEVELS = ("easy", "medium", "hard", "very_hard", "impossible", "symbolic", "oversized")
#: How many tasks the grading put in each level.
COUNTS = {"easy": 95, "medium": 55, "hard": 67, "very_hard": 87, "impossible": 68, "symbolic": 27,
          "oversized": 1}


@pytest.fixture(scope="module")
def difficulty():
    return json.loads((ARC / "evaluation_difficulty.json").read_text())


def test_every_evaluation_task_has_a_level_and_nothing_else_does(difficulty):
    evaluation = json.loads((ARC / "evaluation_challenges.json").read_text())
    assert set(difficulty) == set(evaluation)


def test_the_levels_are_the_known_ones_in_the_known_numbers(difficulty):
    assert set(difficulty.values()) <= set(LEVELS)
    assert dict(Counter(difficulty.values())) == COUNTS


def test_two_of_the_three_the_symbolic_modules_do_not_solve_are_medium_and_left_to_the_llm(difficulty):
    assert {difficulty[task] for task in ("25094a63", "dd2401ed")} == {"medium"}


def test_the_task_whose_prompt_is_far_past_the_rest_is_in_a_level_of_its_own(difficulty):
    assert difficulty["f9d67f8b"] == "oversized"


def test_the_indices_the_grading_was_written_in_name_these_ids(difficulty, monkeypatch):
    """Index 442 was graded easy, 400 very hard, 409 symbolic: the dataset's
    own numbering has to put those on the ids the file holds."""
    monkeypatch.chdir(REPO_ROOT)
    labels = ARCDataset().idx2label
    assert (difficulty[labels[442]], difficulty[labels[400]], difficulty[labels[409]]) == \
        ("easy", "very_hard", "symbolic")
    assert difficulty[labels[799]] == "very_hard" and difficulty[labels[403]] == "impossible"
