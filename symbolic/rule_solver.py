"""The rule search as one of the symbolic module's solvers.

`solve(task)` has the contract symbolic_module.SymbolicModule's solvers have - a SolveResult, a grid or the reason there
is none - so that `checked_solve` holds its claim to the training pairs (each held out in turn, the rule searched again
on the rest and asked for the held-out output) and scripts/symbolic_coverage.py scores it beside the others. It is not
part of SymbolicModule, which imports a good deal more than the search needs.
"""
from typing import Sequence

import numpy as np

from symbolic.rule_search import search
from symbolic.symbolic_module import SolveResult


class RuleSearchSolver:
    """The cheapest cell rule that reproduces every training pair, applied to the test input."""

    def __init__(self, feature_sets: Sequence[str] = ("core", "rich")):
        self.feature_sets = tuple(feature_sets)

    def solve(self, task) -> SolveResult:
        pairs = [{"input": np.asarray(s.train_inp), "output": np.asarray(s.train_out)} for s in task.subtasks]
        test_input = np.asarray(task.test_subtask.train_inp)
        if any(p["input"].shape != p["output"].shape for p in pairs):
            return SolveResult.fail("an output has another shape than its input: the rules change cells in place")
        for feature_set in self.feature_sets:
            rules = search(pairs, feature_set)
            if rules:
                return SolveResult.ok(rules[0].apply(test_input), f"{feature_set}: {rules[0].describe()}")
        return SolveResult.fail("no rule over the cell features reproduces every training pair")
