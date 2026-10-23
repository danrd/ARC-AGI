"""The classes a task falls in when the cell rules fail."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import failure_probe as fp  # noqa: E402
from symbolic import transformed_rules as tr  # noqa: E402

GRIDS = [np.array([[1, 0, 0], [0, 2, 0]]), np.array([[0, 3, 0], [4, 0, 0]]), np.array([[0, 0, 5], [0, 6, 0]])]


def challenge(function, tests=(np.array([[0, 7, 0], [9, 0, 0]]),)):
    return {"train": [{"input": g.tolist(), "output": function(g).tolist()} for g in GRIDS],
            "test": [{"input": t.tolist()} for t in tests]}


def recolour(g):
    out = g.copy()
    out[out == 0] = 8
    return out


def test_a_task_a_rule_solves_is_rule_right():
    task = challenge(recolour)
    solution = [recolour(np.array([[0, 7, 0], [9, 0, 0]])).tolist()]
    record = fp.diagnose("t", task, solution, ("core",), 60)
    assert record["class"] == "rule_right" and record["transform"] == "identity" and record["conflict"] == 0.0


def test_a_task_whose_outputs_no_transformation_reaches_is_no_transformation():
    task = challenge(lambda g: np.zeros((7, 11), int))
    assert fp.diagnose("t", task, [np.zeros((7, 11), int).tolist()], ("core",), 60)["class"] == "no_transformation"


def test_cells_with_the_same_features_and_different_outputs_are_ambiguous(monkeypatch):
    task = challenge(recolour)
    monkeypatch.setattr(fp.transformed_rules, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(fp, "least_conflict", lambda reach: (0.3, "identity"))
    record = fp.diagnose("t", task, [[[0]]], ("core",), 60)
    assert record["class"] == "ambiguous" and record["conflict"] == 0.3


def test_no_rule_with_no_conflict_is_the_grammar(monkeypatch):
    task = challenge(recolour)
    monkeypatch.setattr(fp.transformed_rules, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(fp, "least_conflict", lambda reach: (0.0, "identity"))
    assert fp.diagnose("t", task, [[[0]]], ("core",), 60)["class"] == "no_rule"


class FakeRule:
    cost = 2.0

    def __init__(self, result):
        self.result = result

    def apply(self, grid):
        return self.result

    def describe(self):
        return "fake"


def test_a_wrong_rule_is_told_by_what_it_got_wrong(monkeypatch):
    test_input = np.array([[0, 0, 0, 0]])
    task = challenge(recolour, tests=(np.array([[3, 0, 6, 5]]),))      # the grid before the transformation is not the start
    fake = tr.Candidate("rot180", "core", [FakeRule(np.array([[3, 0, 6, 5]]))], test_input)
    monkeypatch.setattr(fp.transformed_rules, "candidates", lambda *a, **k: [fake])
    record = fp.diagnose("t", task, [[[0, 5, 5, 5]]], ("core",), 60)
    assert record["class"] == "rule_wrong:mixed" and record["errors"] == {"false_change": 1, "missed": 1, "wrong_colour": 1}
    only = tr.Candidate("identity", "core", [FakeRule(np.array([[0, 5, 5, 6]]))], test_input)
    monkeypatch.setattr(fp.transformed_rules, "candidates", lambda *a, **k: [only])
    assert fp.diagnose("t", task, [[[0, 5, 5, 5]]], ("core",), 60)["class"] == "rule_wrong:wrong_colour"


def test_the_least_conflict_picks_the_transformation_that_separates_the_cells_best(monkeypatch):
    def table(grids, feature_set):
        n = sum(g.size for g in grids)
        x = np.zeros((n, 1), int) if grids[0].shape[0] == 1 else np.arange(n).reshape(n, 1)
        return x, ["f"], []

    monkeypatch.setattr(fp.rule_search, "table", table)
    flat = [{"input": np.zeros((1, 4), int), "output": np.array([[1, 2, 3, 4]])}]
    spread = [{"input": np.zeros((2, 2), int), "output": np.array([[1, 2], [3, 4]])}]
    assert fp.least_conflict([("flat", flat, None), ("spread", spread, None)]) == (0.0, "spread")
    assert fp.least_conflict([("spread", spread, None), ("flat", flat, None)]) == (0.0, "spread")
    assert fp.least_conflict([("flat", flat, None)]) == (1.0, "flat")
    assert fp.least_conflict([]) == (None, None)
