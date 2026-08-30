"""Tests for the task-selection half of scripts/compare_reward_approaches.py.

A scan of the whole set takes hours and the work splits perfectly - tasks
share nothing - so the script takes a range and separately scanned ranges
are pooled afterwards. That only works if a span names the same tasks
everywhere, which is what these pin: adjacent spans must abut exactly,
with no task scanned twice and none skipped between them.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "compare_reward_approaches", REPO_ROOT / "scripts" / "compare_reward_approaches.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


class TestParsingASpan:
    def test_a_bare_count_starts_from_the_beginning(self):
        assert script.parse_span("100") == (0, 100)

    def test_a_range_is_half_open(self):
        assert script.parse_span("100-200") == (100, 200)

    @pytest.mark.parametrize("text", ["0", "200-100", "100-100", "-5", "abc", "1-"])
    def test_what_cannot_name_a_span_is_refused(self, text):
        """argparse turns these into a usage error rather than a traceback,
        and refusing them here is what makes a typo in a shard's range fail
        at the command line instead of silently scanning the wrong tasks."""
        with pytest.raises(argparse.ArgumentTypeError):
            script.parse_span(text)


class TestSpansTile:
    """The property pooling depends on. Loading is deterministic - sorted
    keys, positions counted over the shape-preserving list only - so two
    machines given 0-10 and 10-20 between them cover exactly 0-20."""

    def test_adjacent_spans_abut_exactly(self):
        first, total = script.load_tasks("ARC", (0, 10))
        second, _ = script.load_tasks("ARC", (10, 20))
        both, _ = script.load_tasks("ARC", (0, 20))

        assert [t[0] for t in first] + [t[0] for t in second] == [t[0] for t in both]
        assert total > 0

    def test_a_span_is_the_same_tasks_every_time(self):
        once, _ = script.load_tasks("ARC", (30, 40))
        twice, _ = script.load_tasks("ARC", (30, 40))

        assert [t[0] for t in once] == [t[0] for t in twice]

    def test_only_shape_preserving_pairs_are_counted(self):
        """Positions index the kept list, not the raw file - otherwise a
        span would name different tasks depending on how many unusable
        pairs happened to fall inside it."""
        tasks, _ = script.load_tasks("ARC", (0, 25))

        assert len(tasks) == 25
        assert all(inp.shape == out.shape for _id, inp, out in tasks)

    def test_a_span_past_the_end_is_clipped_rather_than_an_error(self):
        _all_tasks, total = script.load_tasks("ARC", (0, 1))
        tail, _ = script.load_tasks("ARC", (total - 3, total + 500))

        assert len(tail) == 3

    def test_the_total_reported_does_not_depend_on_the_span(self):
        """It sizes the shards, so it has to describe the dataset rather
        than the slice being loaded."""
        _first, from_start = script.load_tasks("ARC", (0, 5))
        _middle, from_middle = script.load_tasks("ARC", (50, 55))

        assert from_start == from_middle
