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


class TestReadableNames:
    """The vocabulary spells colours as words because the transforms are
    declared that way; the grid holds digits. A prompt carrying both asks
    the model to resolve a reference nothing in it defines."""

    def test_a_colour_word_becomes_the_digit_it_means(self):
        assert script.readable("red_recolor") == "recolor (colour 2)"

    def test_both_colours_of_a_two_colour_action_are_named(self):
        assert script.readable("blue_contour_connection_red") == \
               "contour connection (colours 1, 2)"

    def test_a_direction_is_an_argument_not_a_word_in_the_verb(self):
        assert script.readable("shift_object_N") == "shift object (direction N)"

    def test_a_plain_name_is_left_alone(self):
        assert script.readable("symmetric_restoration") == "symmetric restoration"

    def test_a_word_that_only_looks_like_a_direction_stays_in_the_verb(self):
        assert script.readable("red_background_shortest_path_left") == \
               "background shortest path left (colour 2)"


class TestMinimisingTowardsWhatWasReached:
    def test_a_path_that_never_solved_keeps_the_steps_that_got_it_there(
            self, monkeypatch):
        """Without a goal every step of a non-solving path is removable and
        the path vanishes; with one, 25 wandering steps come back as the
        few that did the work."""
        monkeypatch.setattr(script, "reached",
                            lambda task, seq, actions, episode_len=25:
                            10 if (7, 0, 0) in list(seq) else 0)

        out = script.minimise(None, [(1, 0, 0), (7, 0, 0), (2, 0, 0)], None, goal=10)

        assert out == ((7, 0, 0),)

    def test_the_goal_is_a_floor_not_an_equality(self, monkeypatch):
        monkeypatch.setattr(script, "reached",
                            lambda task, seq, actions, episode_len=25: 12)

        assert script.minimise(None, [(1, 0, 0)], None, goal=10) == ()


class TestRenderingABlock:
    @staticmethod
    def pooled(**kwargs):
        base = {"per_task": {}, "effective": {}, "solutions": {},
                "partials": {}, "names": NAMES, "span": (0, 1)}
        base.update(kwargs)
        return base

    @pytest.fixture(autouse=True)
    def no_env(self, monkeypatch):
        monkeypatch.setattr(script, "render_steps",
                            lambda task, seq, names, actions, episode_len=25:
                            [f"step {names[str(s[0])]}" for s in seq])
        monkeypatch.setattr(script, "minimise",
                            lambda task, seq, actions, episode_len=25, goal=None:
                            tuple(seq))
        monkeypatch.setattr(script, "reached",
                            lambda task, seq, actions, episode_len=25: 4)

    def test_a_task_the_search_found_nothing_on_gets_no_block(self):
        """Not "the search found nothing" - a block that is sometimes empty
        teaches the reader to expect one."""
        assert script.render_block(("aaa", None, None), self.pooled(), {}) is None

    def test_a_solved_task_carries_the_trace(self):
        pooled = self.pooled(solutions={"aaa": [[[1, 0, 0], [0, 0, 0]]]})

        text = script.render_block(("aaa", None, None), pooled, {})

        assert "reproduced the output exactly" in text
        assert "1. step fliplr" in text
        assert "submit" not in text

    def test_an_unsolved_task_carries_its_furthest_attempt(self):
        pooled = self.pooled(partials={"aaa": [[0.42, [[1, 0, 0], [2, 0, 0]]]]})

        text = script.render_block(("aaa", None, None), pooled, {})

        assert "reached 42% of the target cells" in text
        assert "1. step fliplr" in text and "2. step flipud" in text

    def test_gains_are_reported_in_cells_not_in_intersection_points(self):
        """maximal_intersection counts 2 * matches - valid, so one cell
        fixed moves it by two and a gain of 34 is 17 cells."""
        pooled = self.pooled(effective={"aaa": {"fliplr": 34}})

        text = script.render_block(("aaa", None, None), pooled, {}, min_gain=1)

        assert "up to 17 cells" in text

    def test_moves_below_the_floor_are_not_listed(self):
        pooled = self.pooled(effective={"aaa": {"fliplr": 4, "flipud": 40}})

        text = script.render_block(("aaa", None, None), pooled, {}, min_gain=5)

        assert "flipud" in text and "fliplr" not in text

    def test_a_task_whose_moves_are_all_below_the_floor_gets_no_block(self):
        pooled = self.pooled(effective={"aaa": {"fliplr": 4}})

        assert script.render_block(("aaa", None, None), pooled, {},
                                   min_gain=5) is None

    def test_only_the_asked_for_number_of_moves_is_listed(self):
        pooled = self.pooled(effective={"aaa": {f"a{i}": 100 - i for i in range(10)}})

        text = script.render_block(("aaa", None, None), pooled, {}, moves=3,
                                   min_gain=1)

        assert len([line for line in text.splitlines() if "up to" in line]) == 3


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


class TestLeavingTheAnswerOut:
    """On a task the search solved, the trace in the block is the answer.
    An arm that keeps it measures recipe-following on those tasks and
    hint-following on the rest, and reports one number for both."""

    @pytest.fixture(autouse=True)
    def no_env(self, monkeypatch):
        monkeypatch.setattr(script, "render_steps",
                            lambda task, seq, names, actions, episode_len=25:
                            [f"step {names[str(s[0])]}" for s in seq])
        monkeypatch.setattr(script, "minimise",
                            lambda task, seq, actions, episode_len=25, goal=None:
                            tuple(seq))

    @staticmethod
    def pooled():
        return {"per_task": {"aaa": 1.0}, "effective": {"aaa": {"fliplr": 40}},
                "solutions": {"aaa": [[[1, 0, 0], [0, 0, 0]]]}, "partials": {},
                "names": NAMES, "span": (0, 1)}

    def test_the_solving_sequence_is_dropped(self):
        text = script.render_block(("aaa", None, None), self.pooled(), {},
                                   skip_solved=True)

        assert "reproduced the output exactly" not in text

    def test_the_moves_list_survives(self):
        text = script.render_block(("aaa", None, None), self.pooled(), {},
                                   skip_solved=True)

        assert "fliplr" in text and "up to 20 cells" in text

    def test_by_default_the_sequence_is_still_there(self):
        text = script.render_block(("aaa", None, None), self.pooled(), {})

        assert "reproduced the output exactly" in text
