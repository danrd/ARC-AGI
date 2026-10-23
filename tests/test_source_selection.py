"""Choosing one answer out of several sources'."""
import numpy as np

from orchestration import source_selection as sel

A, B = np.array([[1]]), np.array([[2]])


def test_the_cascade_takes_the_first_source_that_answers_in_the_order_modules_rules_llm():
    assert sel.cascade({"llm": (B, None), "rules": (A, 1.0)})[0] == "rules"
    assert sel.cascade({"llm": (B, None), "modules": (A, None), "rules": (A, 1.0)})[0] == "modules"
    assert sel.cascade({}) is None


def test_a_rule_longer_than_the_limit_does_not_answer_and_the_next_source_does():
    answers = {"rules": (A, 5.0), "llm": (B, None)}
    assert sel.cascade(answers, limit=3.0)[0] == "llm"
    assert sel.cascade(answers, limit=5.0)[0] == "rules"
    assert sel.cascade({"rules": (A, 5.0)}, limit=3.0) is None


def test_two_sources_that_agree_win_over_the_order():
    name, grid = sel.agree({"modules": (A, None), "rules": (B, 1.0), "llm": (B, None)})
    assert name == "agree:rules" and np.array_equal(grid, B)


def test_without_agreement_the_cascade_chooses():
    assert sel.agree({"modules": (A, None), "rules": (B, 1.0)})[0] == "modules"
    assert sel.agree({"rules": (B, 9.0), "llm": (A, None)}, limit=3.0)[0] == "llm"


def test_grids_of_another_shape_do_not_agree():
    assert sel.agree({"rules": (np.array([[1]]), 1.0), "llm": (np.array([[1, 1]]), None)})[0] == "rules"


def test_three_sources_that_agree_beat_a_pair_elsewhere():
    answers = {"modules": (A, None), "rules": (A, 1.0), "llm": (A, None)}
    assert sel.agree(answers)[0] == "agree:modules"


def test_the_score_counts_right_and_answered_apart_and_by_the_source_that_answered():
    per_task = {"a": {"modules": (A, None)}, "b": {"rules": (B, 1.0)}, "c": {}}
    solutions = {"a": A, "b": A, "c": A}
    result = sel.score(per_task, solutions, "cascade")
    assert result == {"right": 1, "answered": 2, "by_source": {"modules": [1, 1], "rules": [0, 1]}}
