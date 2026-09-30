"""Tests for scripts/solved_by_source.py - what each way of solving solves,
read from the files the measuring scripts leave, and what they add to one
another.

Pinned: each file format reads as it should, a search that only explained the
training pairs is never counted as solved, "only this" and the overlaps are
what their names say, and tasks outside the split are ignored.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("solved_by_source", ROOT / "scripts" / "solved_by_source.py")
agg = importlib.util.module_from_spec(SPEC)
sys.modules["solved_by_source"] = agg
SPEC.loader.exec_module(agg)


def flags(**given):
    base = {"claimed": False, "correct": False, "kept": False, "kept_correct": False,
            "timed_out": False, "error": None}
    return {**base, **given}


def write_lines(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


class TestReading:
    def test_each_symbolic_solver_is_a_source_and_only_a_kept_correct_claim_is_solved(self, tmp_path):
        path = write_lines(tmp_path / "s.jsonl", [
            {"split": "evaluation", "task": "a", "solvers": {
                "mixer": flags(claimed=True, correct=True, kept=True, kept_correct=True),
                "color_restore": flags(claimed=True)}},
            {"split": "evaluation", "task": "b", "solvers": {
                "mixer": flags(), "color_restore": flags(claimed=True, correct=True)}}])
        mixer, restore = agg.read_symbolic(path)
        assert (mixer.name, mixer.solved, mixer.attempted) == ("mixer", {"a"}, {"a", "b"})
        assert restore.solved == set() and restore.claimed == {"a", "b"}

    def test_a_torn_last_line_is_ignored(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(json.dumps({"split": "e", "task": "a", "solvers": {"mixer": flags()}}) + '\n{"split": "e')
        assert agg.read_symbolic(path)[0].attempted == {"a"}

    def test_the_llm_export_is_solved_when_any_run_matched_exactly(self, tmp_path):
        path = tmp_path / "e.csv"
        path.write_text("task_id,primary_score,prompt_len\n"
                        "a,0.4,100\na,1,100\nb,0.99,100\nb,0.5,120\nc,1.0,90\n")
        (llm,) = agg.read_llm(path)
        assert llm.solved == {"a", "c"} and llm.attempted == {"a", "b", "c"}

    def test_the_rl_file_is_solved_when_a_run_closed_the_held_out_pair(self, tmp_path):
        path = write_lines(tmp_path / "r.jsonl", [
            {"task": "a", "held_out": 0.5}, {"task": "a", "held_out": 1.0}, {"task": "b", "held_out": -0.2},
            {"task": "c", "held_out": 0.5}])
        (rl,) = agg.read_rl(path)
        assert rl.solved == {"a"} and rl.attempted == {"a", "b", "c"}

    def test_a_search_that_explained_the_pairs_is_claimed_and_never_solved(self, tmp_path):
        path = tmp_path / "h.json"
        path.write_text(json.dumps({"a": "the steps", "b": None, "c": ""}))
        (search,) = agg.read_search(path)
        assert search.solved == set() and search.claimed == {"a"} and search.attempted == {"a", "b", "c"}


def sources():
    return [agg.Source("x", attempted={"a", "b", "c"}, solved={"a", "b"}),
            agg.Source("y", attempted={"b", "c", "d"}, solved={"b", "c", "z"}),
            agg.Source("w", attempted={"a"}, claimed={"a"})]


class TestCombining:
    def test_the_union_the_unique_and_the_shared(self):
        report = agg.combine(sources(), ["a", "b", "c", "d"])
        assert report["union"] == 3 and report["solved_by_any"] == ["a", "b", "c"]
        assert report["only"] == {"x": ["a"], "y": ["c"], "w": []}
        assert report["overlap"][("x", "y")] == 1
        assert report["unsolved"] == ["d"]

    def test_a_task_outside_the_split_counts_for_nothing(self):
        report = agg.combine(sources(), ["a", "b", "c", "d"])
        assert "z" not in report["solved_by_any"] and report["solved"]["y"] == 2

    def test_what_was_attempted_is_counted_inside_the_split_only(self):
        report = agg.combine(sources(), ["a", "b"])
        assert report["attempted"] == {"x": 2, "y": 1, "w": 1}

    def test_a_split_with_no_source_solving_anything(self):
        report = agg.combine([agg.Source("x", attempted={"a"})], ["a", "b"])
        assert report["union"] == 0 and report["unsolved"] == ["a", "b"]


class TestGrades:
    def test_solved_and_left_per_grade(self):
        report = agg.combine(sources(), ["a", "b", "c", "d"])
        graded = agg.by_grade(report, {"a": "easy", "b": "easy", "c": "hard", "d": "hard"})
        assert graded == {"easy": (2, 2), "hard": (2, 1)}

    def test_no_grading_no_table(self):
        assert agg.by_grade(agg.combine(sources(), ["a"]), {}) == {}

    def test_a_task_the_grading_does_not_know_is_kept_under_a_question_mark(self):
        graded = agg.by_grade(agg.combine(sources(), ["a", "d"]), {"a": "easy"})
        assert graded == {"easy": (1, 1), "?": (1, 0)}


class TestTheReport:
    def test_it_names_every_source_and_says_the_union_is_a_ceiling(self):
        report = agg.combine(sources(), ["a", "b", "c", "d"])
        text = agg.render(report, sources(), {}, "evaluation")
        for name in ("x", "y", "w"):
            assert f"\n{name} " in text
        assert "a ceiling" in text and "x & y: 1" in text

    def test_the_command_line_reads_files_and_writes_the_numbers(self, tmp_path, monkeypatch, capsys):
        hints = tmp_path / "h.json"
        hints.write_text(json.dumps({"a": "steps"}))
        out = tmp_path / "out.json"
        monkeypatch.setattr(agg, "universe", lambda split: ["a", "b"])
        monkeypatch.setattr(agg, "grades", lambda split: {})
        monkeypatch.setattr(sys, "argv", ["solved_by_source", "--search", str(hints), "--json", str(out)])
        assert agg.main() == 0
        assert "search" in capsys.readouterr().out
        assert json.loads(out.read_text())["claimed"] == {"search": 1}
