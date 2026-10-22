"""The rule search as a symbolic solver: a SolveResult, held to the training pairs by checked_solve."""
import numpy as np

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.rule_solver import RuleSearchSolver
from symbolic.symbolic_module import checked_solve


def between(width, row, left, right):
    grid = np.zeros((5, width), dtype=int)
    grid[row, left] = grid[row, right] = 2
    out = grid.copy()
    out[row, left + 1:right] = 4
    return grid, out


def task_of(pairs, test):
    return ARCTask(label="t", subtasks=[ARCSubtask(f"t_{i}", i_, o) for i, (i_, o) in enumerate(pairs)],
                   test_inp=test[0], test_out=test[1])


def test_a_rule_that_fits_the_pairs_is_applied_to_the_test_and_explained():
    task = task_of([between(9, 1, 1, 5), between(9, 3, 2, 7), between(10, 0, 4, 8)], between(10, 4, 0, 3))
    result = RuleSearchSolver().solve(task)
    assert result.success and np.array_equal(result.grid, task.test_subtask.train_out)
    assert "between_left_right" in result.debug


def test_a_task_that_changes_shape_or_has_no_rule_is_declined_with_a_reason():
    grown = task_of([(np.zeros((2, 2), dtype=int), np.zeros((3, 3), dtype=int))] * 2,
                    (np.zeros((2, 2), dtype=int), np.zeros((3, 3), dtype=int)))
    assert "shape" in RuleSearchSolver().solve(grown).debug and not RuleSearchSolver().solve(grown).success
    same_in = np.arange(16).reshape(4, 4) % 3
    contradiction = [(same_in, same_in), (same_in, (same_in + 1) % 3)]          # one input, two outputs
    result = RuleSearchSolver().solve(task_of(contradiction, contradiction[0]))
    assert not result.success and "no rule" in result.debug


def test_the_claim_survives_checked_solve_when_each_pair_follows_from_the_others():
    task = task_of([between(9, 1, 1, 5), between(9, 3, 2, 7), between(10, 0, 4, 8)], between(10, 4, 0, 3))
    kept = checked_solve(RuleSearchSolver(), task)
    assert kept.success and np.array_equal(kept.grid, task.test_subtask.train_out)


def test_a_claim_that_hangs_on_a_pair_is_dropped_by_checked_solve():
    odd = (np.zeros((5, 9), dtype=int), np.zeros((5, 9), dtype=int))
    odd[1][2, 3] = 7                                                    # a change nothing else explains
    task = task_of([between(9, 1, 1, 5), between(9, 3, 2, 7), odd], between(10, 4, 0, 3))
    assert not checked_solve(RuleSearchSolver(), task).success


def test_the_cheapest_rule_answers_and_a_set_of_features_with_none_hands_over_to_the_next(monkeypatch):
    import symbolic.rule_solver as module

    class Fixed:
        def __init__(self, value):
            self.value = value

        def apply(self, grid):
            return np.full((1, 1), self.value)

        def describe(self):
            return f"gives {self.value}"

    asked = []

    def fake(pairs, feature_set):
        asked.append(feature_set)
        return [] if feature_set == "core" else [Fixed(1), Fixed(2)]

    monkeypatch.setattr(module, "search", fake)
    task = task_of([between(9, 1, 1, 5), between(9, 3, 2, 7)], between(10, 4, 0, 3))
    result = RuleSearchSolver().solve(task)
    assert result.success and result.grid[0, 0] == 1 and asked == ["core", "rich"] and "rich: gives 1" in result.debug
