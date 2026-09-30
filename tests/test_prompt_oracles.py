"""Tests for scripts/prompt_oracles.py - ten evaluation tasks whose rule is
known well enough to state and to run.

The sentence a prompt carries is only worth testing a model with if it is
true, so each solver is held to every training pair and to the test pair of
its task, and the wrong rule is held to being a different sentence.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prompt_oracles", ROOT / "scripts" / "prompt_oracles.py")
oracles = importlib.util.module_from_spec(SPEC)
sys.modules["prompt_oracles"] = oracles  # a dataclass looks its module up by name
SPEC.loader.exec_module(oracles)

DATA = ROOT / "data" / "datasets" / "ARC"
CHALLENGES = json.loads((DATA / "evaluation_challenges.json").read_text())
SOLUTIONS = json.loads((DATA / "evaluation_solutions.json").read_text())


def pairs(task):
    """Every training pair and the test pair, as arrays."""
    challenge = CHALLENGES[task]
    found = [(np.array(p["input"]), np.array(p["output"])) for p in challenge["train"]]
    found.append((np.array(challenge["test"][0]["input"]), np.array(SOLUTIONS[task][0])))
    return found


#: Rules that name no colour at all: objects move, whatever they are made of.
COLOURLESS = {"64a7c07e"}


def digits(colour):
    return str(colour)


@pytest.mark.parametrize("task", sorted(oracles.ORACLES))
class TestEachOracle:
    def test_the_solver_reproduces_every_training_pair_and_the_test_pair(self, task):
        for index, (grid, answer) in enumerate(pairs(task)):
            got = oracles.ORACLES[task].solve(grid)
            assert np.array_equal(got, answer), f"{task} pair {index}"

    def test_the_rule_and_the_wrong_rule_are_two_sentences(self, task):
        oracle = oracles.ORACLES[task]
        assert oracle.rule(digits) != oracle.wrong(digits)
        assert oracle.rule(digits).strip() and oracle.wrong(digits).strip()

    def test_a_permuted_palette_is_named_in_the_permuted_colours(self, task):
        """The rule is written against a colour function, so a shifted
        palette changes the digits it names and leaves the sentence alone."""
        oracle = oracles.ORACLES[task]
        shifted = lambda colour: str(colour + 1) if colour else "0"  # noqa: E731
        if task in COLOURLESS:
            assert oracle.rule(shifted) == oracle.rule(digits)
        else:
            assert oracle.rule(shifted) != oracle.rule(digits)


def test_there_are_ten_and_they_are_evaluation_tasks():
    assert len(oracles.ORACLES) == 10
    assert set(oracles.ORACLES) <= set(CHALLENGES)


def test_a_solver_that_is_wrong_is_caught_by_the_check_above():
    """The check has teeth: the identity map is not the rule of any of them."""
    for task in oracles.ORACLES:
        assert any(not np.array_equal(grid, answer) for grid, answer in pairs(task))
