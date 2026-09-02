"""Tests for the task-selection half of scripts/compare_reward_approaches.py.

A scan of the whole set takes hours and the work splits perfectly - tasks
share nothing - so the script takes a range and separately scanned ranges
are pooled afterwards. That only works if a span names the same tasks
everywhere, which is what these pin: adjacent spans must abut exactly,
with no task scanned twice and none skipped between them.
"""
from __future__ import annotations

import argparse
import collections
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


class TestKeepingPartialPaths:
    """What a prompt hint is built from on the tasks that matter. A solved
    task needs no hint - the search has the answer - so on everything else
    the furthest non-solving path is the only sequence there is."""

    def test_the_furthest_path_comes_first(self):
        kept = []
        script.keep_partial(kept, 0.2, [[1, 0, 0]], 3)
        script.keep_partial(kept, 0.7, [[2, 0, 0]], 3)

        assert [pair[0] for pair in kept] == [0.7, 0.2]

    def test_the_shorter_of_two_equally_far_paths_wins(self):
        kept = []
        script.keep_partial(kept, 0.5, [[1, 0, 0], [2, 0, 0], [3, 0, 0]], 3)
        script.keep_partial(kept, 0.5, [[4, 0, 0]], 3)

        assert kept[0][1] == [[4, 0, 0]]

    def test_the_same_path_recorded_twice_is_held_once(self):
        kept = []
        script.keep_partial(kept, 0.5, [[1, 0, 0]], 3)
        script.keep_partial(kept, 0.5, [[1, 0, 0]], 3)

        assert len(kept) == 1

    def test_nothing_beyond_the_limit_is_held(self):
        kept = []
        for i in range(10):
            script.keep_partial(kept, i / 10, [[i, 0, 0]], 3)

        assert [pair[0] for pair in kept] == [0.9, 0.8, 0.7]


def _rollout(**kwargs):
    row = {"base_int": 0, "target_int": 10, "max_int": 0, "length": 25,
           "actions": [[1, 0, 0]], "solved": False}
    row.update(kwargs)
    return row


def _args(partials=3, episode_len=25):
    return argparse.Namespace(partials=partials, episode_len=episode_len)


class TestSummarisingOneSearch:
    """What crosses a process boundary. A search returns rollouts holding
    grids and object graphs; shipping those back would cost more than the
    parallelism saves, so the worker reduces them first - and the reduction
    is where an ending can be miscounted."""

    def test_a_solved_rollout_keeps_its_trace(self):
        out = script.summarise_run("aaa", [_rollout(max_int=10, solved=True)],
                                   1.0, {}, _args())

        assert out["endings"]["solved"] == 1
        assert out["solutions"] == [[[1, 0, 0]]]

    def test_the_same_trace_twice_is_kept_once(self):
        rollouts = [_rollout(max_int=10, solved=True) for _ in range(2)]

        out = script.summarise_run("aaa", rollouts, 1.0, {}, _args())

        assert len(out["solutions"]) == 1

    def test_a_voluntary_submit_is_not_an_early_ending(self):
        """An episode ends early both when something submits and when it
        solves, so length alone cannot tell them apart."""
        out = script.summarise_run(
            "aaa", [_rollout(length=4, actions=[[1, 0, 0], [0, 0, 0]])],
            0.0, {}, _args())

        assert out["endings"]["submitted, unsolved"] == 1
        assert "ended early, neither" not in out["endings"]

    def test_a_rollout_cut_at_its_peak_is_not_an_early_ending(self):
        out = script.summarise_run(
            "aaa", [_rollout(length=4, max_int=5, truncated_at_peak=True)],
            0.5, {}, _args())

        assert out["endings"]["cut back to its peak"] == 1

    def test_an_unsolved_rollout_that_moved_becomes_a_partial_path(self):
        out = script.summarise_run("aaa", [_rollout(max_int=5)], 0.5, {}, _args())

        assert out["partials"] == [[0.5, [[1, 0, 0]]]]

    def test_an_unsolved_rollout_that_moved_nothing_does_not(self):
        out = script.summarise_run("aaa", [_rollout(max_int=0)], 0.0, {}, _args())

        assert out["partials"] == []


class TestMergingSummaries:
    @staticmethod
    def totals():
        return {"per_task": {}, "endpoints": {}, "effective_actions": {},
                "solutions": {}, "partial_paths": {},
                "endings": collections.Counter(), "lengths": [],
                "submit_progress": [], "dropped": 0,
                "drop_reasons": collections.Counter()}

    def test_a_task_keeps_the_best_of_its_repeats(self):
        totals = self.totals()

        script.merge(script.summarise_run("aaa", [_rollout(max_int=7)], 0.7, {}, _args()),
                     totals)
        script.merge(script.summarise_run("aaa", [_rollout(max_int=3)], 0.3, {}, _args()),
                     totals)

        assert totals["per_task"] == {"aaa": 0.7}, \
            "the best repeat, not the last one"

    def test_an_effective_action_keeps_its_largest_gain(self):
        totals = self.totals()

        script.merge(script.summarise_run("aaa", [], 0.0, {"fliplr": 9}, _args()), totals)
        script.merge(script.summarise_run("aaa", [], 0.0, {"fliplr": 4}, _args()), totals)

        assert totals["effective_actions"] == {"aaa": {"fliplr": 9}}, \
            "the largest gain, not the last one"

    def test_a_task_nothing_worked_on_is_absent_rather_than_empty(self):
        totals = self.totals()

        script.merge(script.summarise_run("aaa", [], 0.0, {}, _args()), totals)

        assert totals["effective_actions"] == {}
        assert totals["solutions"] == {}

    def test_a_dropped_search_is_counted_with_its_reason(self):
        totals = self.totals()

        script.merge({"task": "aaa", "why": "timed out"}, totals)

        assert totals["dropped"] == 1
        assert totals["drop_reasons"]["timed out"] == 1

    def test_endings_add_up_across_searches(self):
        totals = self.totals()

        for _ in range(3):
            script.merge(script.summarise_run("aaa", [_rollout()], 0.0, {}, _args()),
                         totals)

        assert totals["endings"]["ran to the cap"] == 3


class TestWorkerSetup:
    def test_the_names_are_filled_before_each_search(self, monkeypatch):
        """A spawned worker starts from a fresh import with _ACTION_NAMES
        empty, and under --colours auto the table belongs to one task, not
        to the run - so it is filled per search. A search that skipped this
        would record gains against another task's labels."""
        import numpy as np
        script._ACTION_NAMES.clear()
        monkeypatch.setattr(script, "run_one",
                            lambda *a, **k: (None, None, "stubbed"))
        args = argparse.Namespace(colours=["auto"], directions=["N", "E"],
                                  partials=3, episode_len=25)
        script._WORKER.update(approach=2, args=args)

        script._search_one(("aaa", np.zeros((3, 3), dtype=int),
                            np.full((3, 3), 3)))

        assert script._ACTION_NAMES[0] == "submit"
        assert any(n.startswith("green_") for n in script._ACTION_NAMES.values())


class TestTheVocabularyATaskIsSearchedWith:
    """--colours auto is the difference between 81 of 260 tasks having a
    chance and not: they need a colour outside {1, 2} in the answer, and a
    fixed red/blue vocabulary has no action that can paint one."""

    @staticmethod
    def _task():
        import numpy as np
        return ("aaa", np.zeros((3, 3), dtype=int), np.full((3, 3), 3))

    def test_auto_takes_the_palette_from_the_task(self):
        args = argparse.Namespace(colours=["auto"], directions=["N", "E"])

        names = set(script.actions_for(self._task(), args).values())

        assert any(n.startswith("green_") for n in names)
        assert not any(n.startswith("blue_") for n in names)

    def test_a_named_pair_is_used_as_given(self):
        args = argparse.Namespace(colours=["red", "blue"], directions=["N", "E"])

        names = set(script.actions_for(self._task(), args).values())

        assert any(n.startswith("red_") for n in names)
        assert not any(n.startswith("green_") for n in names)


class TestSplits:
    """A span numbers positions within one split, so the same range names a
    different set of tasks in each - which is why the split travels in the
    shard and pooling refuses to mix them."""

    def test_the_two_splits_are_different_tasks(self):
        train, train_total = script.load_tasks("ARC", (0, 5), "training")
        evaluation, eval_total = script.load_tasks("ARC", (0, 5), "evaluation")

        assert [t[0] for t in train] != [t[0] for t in evaluation]
        assert (train_total, eval_total) == (262, 270)

    def test_training_is_the_default(self):
        assert [t[0] for t in script.load_tasks("ARC", (0, 5))[0]] == \
               [t[0] for t in script.load_tasks("ARC", (0, 5), "training")[0]]

    def test_a_split_that_does_not_exist_is_refused(self):
        with pytest.raises(SystemExit):
            script.load_tasks("ARC", (0, 5), "test")

    def test_spans_tile_within_the_evaluation_split_too(self):
        first, _ = script.load_tasks("ARC", (0, 10), "evaluation")
        second, _ = script.load_tasks("ARC", (10, 20), "evaluation")
        both, _ = script.load_tasks("ARC", (0, 20), "evaluation")

        assert [t[0] for t in first] + [t[0] for t in second] == \
               [t[0] for t in both]


class TestTheDefaultPlayout:
    def test_a_scan_runs_weighted_unless_told_otherwise(self):
        """Every shard and every reference figure came from a weighted run.
        A scan that quietly used the other playout would pool with them and
        look fine - measured on one task, the default playout found 1
        effective action where weighted found 47."""
        assert script.build_parser().parse_args([]).playout == "weighted"

    def test_the_other_playout_is_still_reachable(self):
        assert script.build_parser().parse_args(
            ["--playout", "default"]).playout == "default"
