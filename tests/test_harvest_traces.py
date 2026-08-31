"""Tests for scripts/harvest_traces.py.

Two hazards, both of which produce a plausible-looking dataset rather than
an error. Pooling shards scanned with different vocabularies relabels every
action after the first difference, and keeping a sequence the search
claimed without replaying it puts a wrong label in front of a learner. The
third thing pinned here is the shape of what minimising is allowed to do:
drop steps the sequence solves without, and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import rl.search_hints as hints

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "harvest_traces", REPO_ROOT / "scripts" / "harvest_traces.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

NAMES = {"0": "submit", "1": "fliplr", "2": "flipud"}


@pytest.fixture(scope="module")
def setup():
    """The real vocabulary, and a task one recorded action solves."""
    actions = script.build_actions(["red", "blue"], ["N", "E"])
    index = {name: i for i, name in actions.items()}
    tasks, _ = script.load_tasks("ARC", (0, 262))
    return actions, index, {t[0]: t for t in tasks}["6d75e8bb"]


def shard(tmp_path, name, span, per_task, solutions, names=NAMES, effective=None,
          partials=None):
    path = tmp_path / name
    section = {"per_task": per_task,
               "effective_actions": effective or {},
               "solutions": solutions,
               "action_names": names}
    if partials is not None:
        section["partial_paths"] = partials
    path.write_text(json.dumps({"span": list(span),
                                "approaches": {"2": section}}))
    return path


class TestPoolingShards:
    def test_tasks_from_every_shard_are_kept(self, tmp_path):
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {"aaa": [[[1, 0, 0]]]})
        second = shard(tmp_path, "b.json", (2, 4), {"bbb": 0.5}, {"bbb": []})

        pooled = script.pool([first, second])

        assert pooled["per_task"] == {"aaa": 1.0, "bbb": 0.5}
        assert set(pooled["solutions"]) == {"aaa", "bbb"}
        assert pooled["names"] == NAMES

    def test_the_span_covers_every_shard(self, tmp_path):
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {})
        second = shard(tmp_path, "b.json", (5, 9), {"bbb": 0.5}, {})

        assert script.pool([first, second])["span"] == (0, 9)

    def test_shards_of_different_vocabularies_are_refused(self, tmp_path):
        """The failure this exists for is silent otherwise: action 2 means
        flipud in one file and rotate90 in the other, every label after it
        is wrong, and the merged file looks perfectly ordinary."""
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {})
        second = shard(tmp_path, "b.json", (2, 4), {"bbb": 0.5}, {},
                       names={"0": "submit", "1": "fliplr", "2": "rotate90"})

        with pytest.raises(SystemExit) as excinfo:
            script.pool([first, second])

        assert "vocabular" in str(excinfo.value)

    def test_the_same_vocabulary_pools_without_complaint(self, tmp_path):
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {})
        second = shard(tmp_path, "b.json", (2, 4), {"bbb": 0.5}, {})

        assert script.pool([first, second])["names"] == NAMES


class TestHarvest:
    """`harvest` calls the shared replay and minimiser from rl.search_hints,
    so a stub has to stand in both places: the name the script imported and
    the one `minimise` reaches for inside that module."""

    @staticmethod
    def _stub(monkeypatch, predicate):
        for module in (script, hints):
            monkeypatch.setattr(module, "replays",
                                lambda task, seq, actions, episode_len=25:
                                predicate(list(seq)))

    def test_a_sequence_that_does_not_replay_is_dropped_and_counted(self, monkeypatch):
        self._stub(monkeypatch, lambda seq: (1, 0, 0) in seq)
        solutions = {"aaa": [[[1, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]]}

        kept, stats = script.harvest(solutions, NAMES, [("aaa", None, None)], {})

        assert stats["recorded"] == 2
        assert stats["did not replay"] == 1
        assert kept == {"aaa": [((1, 0, 0),)]}

    def test_submit_is_not_part_of_what_a_trace_teaches(self, monkeypatch):
        self._stub(monkeypatch, lambda seq: True)
        solutions = {"aaa": [[[1, 0, 0], [0, 0, 0]]]}

        kept, _ = script.harvest(solutions, NAMES, [("aaa", None, None)], {})

        assert all((0, 0, 0) not in seq for seq in kept["aaa"])

    def test_a_task_outside_the_span_is_reported_not_guessed(self, monkeypatch):
        self._stub(monkeypatch, lambda seq: True)

        kept, stats = script.harvest({"zzz": [[[1, 0, 0]]]}, NAMES,
                                     [("aaa", None, None)], {})

        assert kept == {}
        assert stats["outside the span"] == 1


