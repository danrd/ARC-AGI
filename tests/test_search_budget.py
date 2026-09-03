"""Tests for scripts/search_budget.py.

The sweep answers one question - what a solve-time budget buys - so what
is pinned is that its numbers mean what the table says: a task that
failed is not a task that scored zero, coverage counts blocks rather than
searches, and a setting reaches the search unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "search_budget", REPO_ROOT / "scripts" / "search_budget.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def _triple():
    return ("aaa", np.zeros((3, 3), dtype=int), np.array([[0, 1, 2]] * 3))


class TestOneTask:
    def test_a_task_that_raised_is_reported_not_scored(self, monkeypatch):
        """A zero is a search that found nothing, which is a different
        statement from a search that fell over."""
        def boom(task, settings):
            raise RuntimeError("no")

        monkeypatch.setattr(script, "search_task", boom)

        row = script.one_task((_triple(), script.SearchSettings()))

        assert row["why"] == "RuntimeError"
        assert "peak" not in row

    def test_carried_means_a_block_was_rendered(self, monkeypatch):
        monkeypatch.setattr(script, "search_task",
                            lambda task, settings: {
                                "peak": 0.4, "effective": {}, "solutions": [],
                                "partials": [], "actions": {0: "submit"}})
        monkeypatch.setattr(script, "render_block", lambda *a, **k: None)

        row = script.one_task((_triple(), script.SearchSettings()))

        assert row["carried"] is False
        assert row["peak"] == 0.4

    def test_a_solution_is_read_from_what_the_search_returned(self, monkeypatch):
        monkeypatch.setattr(script, "search_task",
                            lambda task, settings: {
                                "peak": 1.0, "effective": {}, "solutions": [[[1, 0, 0]]],
                                "partials": [], "actions": {0: "submit"}})
        monkeypatch.setattr(script, "render_block", lambda *a, **k: "text")

        row = script.one_task((_triple(), script.SearchSettings()))

        assert row["solved"] is True and row["carried"] is True


class TestTheSweep:
    def test_the_setting_reaches_the_search(self, monkeypatch):
        seen = []
        monkeypatch.setattr(script, "search_task",
                            lambda task, settings: seen.append(settings.iterations)
                            or {"peak": 0.0, "effective": {}, "solutions": [],
                                "partials": [], "actions": {0: "submit"}})
        monkeypatch.setattr(script, "render_block", lambda *a, **k: None)

        script.sweep([_triple(), _triple()],
                     script.SearchSettings(iterations=640), workers=1)

        assert seen == [640, 640]


class TestTheReport:
    def test_failures_are_left_out_of_the_averages(self):
        rows = [{"task": "a", "why": "TimedOut"},
                {"task": "b", "why": None, "seconds": 2.0, "peak": 0.5,
                 "solved": False, "moves": 3, "carried": True}]

        summary = script.report("x", rows)

        assert summary["tasks"] == 1 and summary["failed"] == 1
        assert summary["mean_peak"] == 0.5

    def test_a_setting_where_everything_failed_says_so(self, capsys):
        summary = script.report("x", [{"task": "a", "why": "TimedOut"}])

        assert summary == {}
        assert "every task failed" in capsys.readouterr().out


class TestTheCommandLine:
    def test_the_span_is_parsed_the_way_the_scan_parses_it(self):
        assert script.parse_span("0-50") == (0, 50)

    def test_a_bad_span_is_a_usage_error(self):
        import pytest
        with pytest.raises(argparse.ArgumentTypeError):
            script.parse_span("50-0")
