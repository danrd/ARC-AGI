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


def shard(tmp_path, name, span, per_task, solutions, names=NAMES, effective=None):
    path = tmp_path / name
    path.write_text(json.dumps({
        "span": list(span),
        "approaches": {"2": {"per_task": per_task,
                             "effective_actions": effective or {},
                             "solutions": solutions,
                             "action_names": names}}}))
    return path


class TestPoolingShards:
    def test_tasks_from_every_shard_are_kept(self, tmp_path):
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {"aaa": [[[1, 0, 0]]]})
        second = shard(tmp_path, "b.json", (2, 4), {"bbb": 0.5}, {"bbb": []})

        per_task, _, solutions, names, span = script.pool([first, second])

        assert per_task == {"aaa": 1.0, "bbb": 0.5}
        assert set(solutions) == {"aaa", "bbb"}
        assert names == NAMES

    def test_the_span_covers_every_shard(self, tmp_path):
        first = shard(tmp_path, "a.json", (0, 2), {"aaa": 1.0}, {})
        second = shard(tmp_path, "b.json", (5, 9), {"bbb": 0.5}, {})

        assert script.pool([first, second])[4] == (0, 9)

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

        assert script.pool([first, second])[3] == NAMES


class TestDistinct:
    def test_the_same_sequence_recorded_twice_counts_once(self):
        assert script.distinct([[[1, 0, 0]], [[1, 0, 0]]]) == [((1, 0, 0),)]

    def test_shortest_first(self):
        out = script.distinct([[[1, 0, 0], [2, 0, 0]], [[2, 0, 0]]])

        assert [len(seq) for seq in out] == [1, 2]


class TestMinimising:
    """Algorithm only - `replays` is stubbed so the property is visible
    without an env in the way."""

    def test_steps_the_sequence_solves_without_are_dropped(self, monkeypatch):
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25:
                            (9, 0, 0) in list(seq))

        out = script.minimise(None, [(1, 0, 0), (9, 0, 0), (2, 0, 0)], None)

        assert out == ((9, 0, 0),)

    def test_a_sequence_that_needs_all_of_it_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25: len(seq) == 3)
        sequence = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]

        assert script.minimise(None, sequence, None) == tuple(sequence)

    def test_a_repeated_step_is_kept_when_the_repeat_is_needed(self, monkeypatch):
        """Two applications of one action are a real solution shape -
        b230c067 needs blue_recolor twice - so deduplicating by name would
        break it."""
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25:
                            list(seq).count((1, 0, 0)) == 2)

        assert script.minimise(None, [(1, 0, 0), (1, 0, 0), (2, 0, 0)], None) == \
               ((1, 0, 0), (1, 0, 0))


class TestHarvest:
    def test_a_sequence_that_does_not_replay_is_dropped_and_counted(self, monkeypatch):
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25:
                            (1, 0, 0) in list(seq))
        solutions = {"aaa": [[[1, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]]}

        kept, stats = script.harvest(solutions, NAMES, [("aaa", None, None)], {})

        assert stats["recorded"] == 2
        assert stats["did not replay"] == 1
        assert kept == {"aaa": [((1, 0, 0),)]}

    def test_submit_is_not_part_of_what_a_trace_teaches(self, monkeypatch):
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25: True)
        solutions = {"aaa": [[[1, 0, 0], [0, 0, 0]]]}

        kept, _ = script.harvest(solutions, NAMES, [("aaa", None, None)], {})

        assert all((0, 0, 0) not in seq for seq in kept["aaa"])

    def test_a_task_outside_the_span_is_reported_not_guessed(self, monkeypatch):
        monkeypatch.setattr(script, "replays",
                            lambda task, seq, actions, episode_len=25: True)

        kept, stats = script.harvest({"zzz": [[[1, 0, 0]]]}, NAMES,
                                     [("aaa", None, None)], {})

        assert kept == {}
        assert stats["outside the span"] == 1


class TestReplayingAgainstTheRealEnv:
    """One end-to-end check that the verification means anything: the env
    the traces are replayed in has to be the env they were recorded in,
    and a stub cannot show that."""

    def test_a_recorded_solution_still_solves(self, setup):
        actions, index, task = setup

        assert script.replays(task, [(index["red_color_outer_holes"], 0, 0)], actions)

    def test_another_action_does_not(self, setup):
        actions, index, task = setup

        assert not script.replays(task, [(index["rotate90"], 0, 0)], actions)
