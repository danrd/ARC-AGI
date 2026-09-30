"""Tests for scripts/search_census.py - what the object search solves and what
it leaves behind where it does not.

The search itself is faked where it would take a minute (test_search_hints
covers it); pinned here is the description of a leftover, which is what the
census is for, the summary over records, and the file that lets a run resume.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("search_census", ROOT / "scripts" / "search_census.py")
census = importlib.util.module_from_spec(SPEC)
sys.modules["search_census"] = census
SPEC.loader.exec_module(census)


def grid(rows):
    return np.array(rows)


class TestWhatIsLeft:
    START = grid([[0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])

    def test_nothing_wrong_is_nothing_left_and_all_closed(self):
        target = self.START.copy()
        target[3, 4] = 2
        left = census.leftover(target, target, self.START)
        assert left["wrong"] == 0 and left["closed"] == 1.0 and left["regions"] == 0

    def test_cells_the_search_did_not_paint_are_paint_and_the_share_closed_is_of_the_changes(self):
        target = self.START.copy()
        target[0, 0:3] = 2                      # a line of three to paint
        target[3, 4] = 2                        # and a cell
        reached = self.START.copy()
        reached[3, 4] = 2                       # only the cell was done
        left = census.leftover(reached, target, self.START)
        assert left["wrong"] == 3 and left["to_change"] == 4 and left["closed"] == 0.25
        assert left["kinds"] == {"paint": 3} and left["regions"] == 1 and left["shapes"] == {"line": 1}

    def test_a_cell_that_should_have_been_erased_is_erase(self):
        target = self.START.copy()
        target[1, 1] = 0
        left = census.leftover(self.START, target, self.START)
        assert left["kinds"] == {"erase": 1} and left["shapes"] == {"cell": 1}

    def test_a_cell_in_the_wrong_colour_is_recolour_and_one_transition_says_it_is_one_rule(self):
        target = self.START.copy()
        target[1, 1:3] = 5
        reached = self.START.copy()
        reached[1, 1:3] = 3
        left = census.leftover(reached, target, self.START)
        assert left["kinds"] == {"recolour": 2} and left["transitions"] == 1

    def test_a_filled_block_is_a_rectangle_and_a_ragged_one_is_other(self):
        target = self.START.copy()
        target[2:4, 0:2] = 4
        assert census.leftover(self.START, target, self.START)["shapes"] == {"rectangle": 1}
        target = self.START.copy()
        target[2, 0:2] = 4
        target[3, 0] = 4
        assert census.leftover(self.START, target, self.START)["shapes"] == {"other": 1}

    def test_regions_that_touch_only_at_a_corner_are_one_region(self):
        target = self.START.copy()
        target[2, 2] = 4
        target[3, 3] = 4
        assert census.leftover(self.START, target, self.START)["regions"] == 1

    def test_a_task_with_nothing_to_change_is_closed(self):
        left = census.leftover(self.START, self.START, self.START)
        assert left["closed"] == 1.0 and left["to_change"] == 0


class TestOneTask:
    def pairs(self, same=True):
        inp = np.zeros((3, 3), dtype=int)
        out = inp + 1 if same else np.ones((2, 2), dtype=int)
        return [(f"t_{i}", inp, out) for i in range(2)]

    def test_a_task_whose_grids_change_size_is_skipped_without_a_search(self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "search_branches", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
        assert census.census_task("t", self.pairs(same=False)) == {
            "task": "t", "skipped": "the grids change size"}

    def test_an_unsolved_task_carries_what_its_furthest_path_left(self, monkeypatch):
        import rl.search_hints as hints
        found = {"solutions": [], "peak": 0.6, "partials": [[0.5, [[1, 0, 0]]]], "actions": {0: "submit"}}
        monkeypatch.setattr(hints, "search_branches", lambda *a, **k: {"connector": found, "modifier": found})
        monkeypatch.setattr(hints, "each_branch", lambda *a, **k: {"connector": found})
        monkeypatch.setattr(hints, "_first_verified", lambda *a, **k: None)
        monkeypatch.setattr(census, "final_grid", lambda *a, **k: np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
        record = census.census_task("t", self.pairs())
        assert record["verified"] is False and record["first"] is False and record["peak"] == 0.6
        assert record["left"]["wrong"] == 8 and record["left"]["path_len"] == 1

    def test_a_verified_task_has_no_leftover_to_describe(self, monkeypatch):
        import rl.search_hints as hints
        found = {"solutions": [[[1, 0, 0]]], "peak": 1.0, "partials": [], "actions": {0: "submit"}}
        monkeypatch.setattr(hints, "search_branches", lambda *a, **k: {"base": found})
        monkeypatch.setattr(hints, "_first_verified", lambda *a, **k: "the block")
        record = census.census_task("t", self.pairs())
        assert record["verified"] and record["first"] and "left" not in record

    def test_a_first_pair_solved_but_not_the_others_is_recorded_as_such(self, monkeypatch):
        import rl.search_hints as hints
        found = {"solutions": [[[1, 0, 0]]], "peak": 1.0, "partials": [[0.5, [[1, 0, 0]]]],
                 "actions": {0: "submit"}}
        monkeypatch.setattr(hints, "search_branches", lambda *a, **k: {"connector": found})
        monkeypatch.setattr(hints, "_first_verified", lambda *a, **k: None)
        monkeypatch.setattr(census, "final_grid", lambda *a, **k: np.zeros((3, 3), dtype=int))
        record = census.census_task("t", self.pairs())
        assert record["first"] and not record["verified"]
        assert "left" not in record  # solved on the first pair: nothing was left there


def record(task, verified=False, first=False, peak=0.0, **left):
    out = {"split": "evaluation", "task": task, "verified": verified, "first": first, "peak": peak}
    if left:
        out["left"] = left
    return out


class TestSummary:
    RECORDS = [
        record("a", verified=True, first=True, peak=1.0),
        record("b", first=True, peak=1.0),
        record("c", peak=0.9, regions=1, transitions=1, kinds={"paint": 5}, shapes={"rectangle": 1}),
        record("d", peak=0.3, regions=7, transitions=4, kinds={"recolour": 3, "paint": 1}, shapes={"cell": 6, "line": 1}),
        record("e", peak=0.0),
        {"split": "evaluation", "task": "f", "skipped": "the grids change size"},
    ]

    def test_counts(self):
        summary = census.summarise(self.RECORDS)
        assert (summary["tasks"], summary["skipped"], summary["verified"], summary["first_only"],
                summary["unsolved"]) == (6, 1, 1, 1, 3)
        assert summary["peak_bands"] == {">=0.8": 1, "<0.5": 1, "0": 1}

    def test_the_leftover_is_read_off_the_unsolved_that_have_one(self):
        summary = census.summarise(self.RECORDS)
        assert summary["with_leftover"] == 2 and summary["few_regions"] == 1 and summary["one_transition"] == 1
        assert summary["kinds"] == {"paint": 1, "recolour": 1}
        assert summary["dominant_shape"] == {"rectangle": 1, "cell": 1}

    def test_the_report_names_the_counts(self):
        text = census.render(census.summarise(self.RECORDS))
        assert "verified by the search: 1" in text and "at most three regions wrong: 1" in text


class TestTheFile:
    def test_a_torn_last_line_is_ignored_and_a_missing_file_is_empty(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(json.dumps(record("a")) + '\n{"split": "eva')
        assert set(census.read_records(path)) == {("evaluation", "a")}
        assert census.read_records(tmp_path / "none.jsonl") == {}

    def test_a_task_already_in_the_file_is_not_run_again(self, tmp_path, monkeypatch):
        path = tmp_path / "c.jsonl"
        path.write_text(json.dumps(record("a")) + "\n")
        monkeypatch.setattr(census, "load_tasks", lambda split: {"a": []})
        printed = []
        monkeypatch.setattr("builtins.print", lambda *args, **kw: printed.append(args[0]))
        census.run("evaluation", path, workers=1, timeout=1)
        assert "1 tasks already" in printed[0] and "0 to go" in printed[0]
