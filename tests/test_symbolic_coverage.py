"""Tests for scripts/symbolic_coverage.py - what the symbolic solvers solve,
counted claim by claim and again after checked_solve holds each claim to the
task's own examples.

The solvers are stand-ins here: what is pinned is the bookkeeping - a wrong
claim is a claim and not a solve, a claim that survives the check is "kept",
a solver that hangs or raises costs its task nothing else - and that a run
cut short resumes.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.symbolic_module import SolveResult

SPEC = importlib.util.spec_from_file_location(
    "symbolic_coverage", Path(__file__).resolve().parent.parent / "scripts" / "symbolic_coverage.py")
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


def task_with(pairs, test):
    """A task whose pairs are (input, output) lists; `test` likewise."""
    subtasks = [ARCSubtask(f"t_{i}", np.array(inp), np.array(out)) for i, (inp, out) in enumerate(pairs)]
    return ARCTask(label="t", subtasks=subtasks, test_inp=np.array(test[0]), test_out=np.array(test[1]))


class Copies:
    """Returns the test input: right exactly where the output is the input."""
    def solve(self, task):
        return SolveResult.ok(np.array(task.test_subtask.train_inp))


class AlwaysZeros:
    """Claims an all-zero grid of the test input's shape on every task."""
    def solve(self, task):
        return SolveResult.ok(np.zeros_like(task.test_subtask.train_inp))


class Declines:
    def solve(self, task):
        return SolveResult.fail("no")


class Hangs:
    def solve(self, task):
        time.sleep(5)


class Breaks:
    def solve(self, task):
        raise RuntimeError("boom")


SAME = task_with([([[1, 2]], [[1, 2]]), ([[3, 4]], [[3, 4]]), ([[5, 6]], [[5, 6]])], ([[7, 8]], [[7, 8]]))
DIFFERENT = task_with([([[1, 2]], [[2, 1]]), ([[3, 4]], [[4, 3]]), ([[5, 6]], [[6, 5]])], ([[7, 8]], [[8, 7]]))


class TestOneSolverPerFontColour:
    def test_a_list_claims_if_any_claims_and_is_kept_by_the_first_that_survives(self):
        flags = coverage.score_task(SAME, {"copies": [Declines(), Copies(), Copies()]})["copies"]
        assert flags["claimed"] and flags["kept_correct"] and flags["instance"] == 1

    def test_a_list_of_solvers_that_all_decline_claims_nothing(self):
        flags = coverage.score_task(SAME, {"none": [Declines(), Declines()]})["none"]
        assert not flags["claimed"] and flags["instance"] is None

    def test_a_wrong_claim_in_the_list_does_not_hide_a_right_one_after_it(self):
        flags = coverage.score_task(DIFFERENT, {"mixed": [Copies(), Copies()]})["mixed"]
        assert flags["claimed"] and not flags["kept"]


class TestScoring:
    def test_a_solver_that_is_right_is_claimed_correct_and_kept(self):
        flags = coverage.score_task(SAME, {"copies": Copies()})["copies"]
        assert (flags["claimed"], flags["correct"], flags["kept"], flags["kept_correct"]) == \
            (True, True, True, True)

    def test_a_wrong_claim_is_a_claim_and_the_check_drops_it(self):
        """The solver hands back the input where the answer is the input
        flipped: claimed, not correct, and it cannot reproduce a held-out
        pair either, so it is not kept."""
        flags = coverage.score_task(DIFFERENT, {"copies": Copies()})["copies"]
        assert (flags["claimed"], flags["correct"], flags["kept"], flags["kept_correct"]) == \
            (True, False, False, False)

    def test_a_declining_solver_claims_nothing(self):
        flags = coverage.score_task(SAME, {"declines": Declines()})["declines"]
        assert not any((flags["claimed"], flags["correct"], flags["kept"], flags["kept_correct"]))

    def test_a_solver_that_hangs_is_cut_off_and_says_so(self):
        started = time.perf_counter()
        flags = coverage.score_task(SAME, {"hangs": Hangs()}, timeout=1)["hangs"]
        assert flags["timed_out"] and not flags["claimed"]
        assert time.perf_counter() - started < 4

    def test_a_solver_that_raises_costs_its_task_nothing_else(self):
        scored = coverage.score_task(SAME, {"breaks": Breaks(), "copies": Copies()})
        assert scored["breaks"]["error"] == "RuntimeError('boom')" and not scored["breaks"]["claimed"]
        assert scored["copies"]["kept_correct"]


def record(split, task, **solvers):
    empty = {"claimed": False, "correct": False, "kept": False, "kept_correct": False,
             "timed_out": False, "error": None}
    return {"split": split, "task": task,
            "solvers": {name: {**empty, **flags} for name, flags in solvers.items()}}


class TestSummary:
    RECORDS = [
        record("evaluation", "a", mixer={"claimed": True, "correct": True, "kept": True, "kept_correct": True},
               color_restore={"claimed": True}),
        record("evaluation", "b", mixer={"claimed": True, "kept": True},
               color_restore={"claimed": True, "correct": True, "kept": True, "kept_correct": True}),
        record("evaluation", "c", mixer={"timed_out": True}, color_restore={"error": "x"}),
        record("training", "d", mixer={"claimed": True, "correct": True}),
    ]

    def test_counts_are_per_split_and_per_solver(self):
        summary = coverage.summarise(self.RECORDS, solvers=("mixer", "color_restore"))
        evaluation = summary["evaluation"]
        assert evaluation["tasks"] == 3
        assert evaluation["solvers"]["mixer"]["claimed"] == 2
        assert evaluation["solvers"]["mixer"]["kept"] == 2
        assert evaluation["solvers"]["mixer"]["kept_correct"] == 1
        assert evaluation["solvers"]["mixer"]["timed_out"] == 1
        assert evaluation["solvers"]["color_restore"]["error"] == 1
        assert summary["training"]["solvers"]["mixer"]["correct"] == 1

    def test_a_task_two_solvers_solve_counts_once(self):
        both = [record("evaluation", "a",
                       mixer={"kept_correct": True}, color_restore={"kept_correct": True})]
        assert coverage.summarise(both, solvers=("mixer", "color_restore"))[
            "evaluation"]["solved_by_any"] == 1

    def test_which_tasks_each_solver_solved(self):
        assert coverage.solved_ids(self.RECORDS, "evaluation") == {"mixer": ["a"], "color_restore": ["b"]}

    def test_the_table_says_precision_after_the_check(self):
        text = coverage.render(coverage.summarise(self.RECORDS, solvers=("mixer",)))
        assert "evaluation: 3 tasks" in text and "50.0%" in text


class TestResuming:
    def test_records_are_read_back_and_a_torn_last_line_is_ignored(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps(record("evaluation", "a", mixer={})) + "\n" + '{"split": "eva')
        assert set(coverage.read_records(path)) == {("evaluation", "a")}

    def test_a_missing_file_is_no_records(self, tmp_path):
        assert coverage.read_records(tmp_path / "nothing.jsonl") == {}

    def test_a_task_already_in_the_file_is_not_run_again(self, tmp_path, monkeypatch):
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps(record("evaluation", "a", mixer={})) + "\n")
        monkeypatch.setattr(coverage, "load_task_ids", lambda split: ["a"])
        printed = []
        monkeypatch.setattr("builtins.print", lambda *args, **kw: printed.append(args[0]))
        coverage.run(("evaluation",), path, workers=1, timeout=0)
        assert "1 tasks already" in printed[0] and "0 to go" in printed[0]
        assert len(path.read_text().splitlines()) == 1


class TestTheRealSolvers:
    def test_the_three_the_module_has_are_named(self):
        assert set(coverage.default_solvers()) == set(coverage.SOLVERS)

    def test_a_real_task_is_scored_end_to_end_by_id(self):
        """One task from the dataset through the same call a worker makes."""
        task_id = coverage.load_task_ids("training")[0]
        scored = coverage.score_by_id("training", task_id, timeout=60)
        assert scored["task"] == task_id
        assert set(scored["solvers"]) == set(coverage.SOLVERS)
        assert all("kept_correct" in flags for flags in scored["solvers"].values())
