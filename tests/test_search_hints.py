"""Tests for rl/search_hints.py - the search-derived hint block.

Two paths produce one text: a scan writes these blocks into a file ahead of
time, and an online run computes the same thing when the task arrives. The
renderer is shared so they cannot drift, and this is where it is pinned.

Rendering is tested against stubs so the properties are visible without an
env in the way; the last class runs a real one, because a stub cannot show
that a recorded sequence still means what it meant inside the search.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rl.search_hints as hints

REPO_ROOT = Path(__file__).resolve().parent.parent

NAMES = {"0": "submit", "1": "fliplr", "2": "flipud"}


@pytest.fixture(scope="module")
def setup():
    """The real vocabulary, and a task one recorded action solves."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_reward_approaches",
        REPO_ROOT / "scripts" / "compare_reward_approaches.py")
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    actions = hints.build_vocabulary(("red", "blue"), ("N", "E"))
    index = {name: i for i, name in actions.items()}
    tasks, _ = scan.load_tasks("ARC", (0, 262))
    return actions, index, {t[0]: t for t in tasks}["6d75e8bb"]


class TestDistinct:
    def test_the_same_sequence_recorded_twice_counts_once(self):
        assert hints.distinct([[[1, 0, 0]], [[1, 0, 0]]]) == [((1, 0, 0),)]

    def test_shortest_first(self):
        out = hints.distinct([[[1, 0, 0], [2, 0, 0]], [[2, 0, 0]]])

        assert [len(seq) for seq in out] == [1, 2]


class TestMinimising:
    """Algorithm only - `replays` is stubbed so the property is visible
    without an env in the way."""

    def test_steps_the_sequence_solves_without_are_dropped(self, monkeypatch):
        monkeypatch.setattr(hints, "replays",
                            lambda task, seq, actions, episode_len=25:
                            (9, 0, 0) in list(seq))

        out = hints.minimise(None, [(1, 0, 0), (9, 0, 0), (2, 0, 0)], None)

        assert out == ((9, 0, 0),)

    def test_a_sequence_that_needs_all_of_it_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(hints, "replays",
                            lambda task, seq, actions, episode_len=25: len(seq) == 3)
        sequence = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]

        assert hints.minimise(None, sequence, None) == tuple(sequence)

    def test_a_repeated_step_is_kept_when_the_repeat_is_needed(self, monkeypatch):
        """Two applications of one action are a real solution shape -
        b230c067 needs blue_recolor twice - so deduplicating by name would
        break it."""
        monkeypatch.setattr(hints, "replays",
                            lambda task, seq, actions, episode_len=25:
                            list(seq).count((1, 0, 0)) == 2)

        assert hints.minimise(None, [(1, 0, 0), (1, 0, 0), (2, 0, 0)], None) == \
               ((1, 0, 0), (1, 0, 0))


class TestMinimisingTowardsWhatWasReached:
    def test_a_path_that_never_solved_keeps_the_steps_that_got_it_there(
            self, monkeypatch):
        """Without a goal every step of a non-solving path is removable and
        the path vanishes; with one, 25 wandering steps come back as the
        few that did the work."""
        monkeypatch.setattr(hints, "reached",
                            lambda task, seq, actions, episode_len=25:
                            10 if (7, 0, 0) in list(seq) else 0)

        out = hints.minimise(None, [(1, 0, 0), (7, 0, 0), (2, 0, 0)], None, goal=10)

        assert out == ((7, 0, 0),)

    def test_the_goal_is_a_floor_not_an_equality(self, monkeypatch):
        monkeypatch.setattr(hints, "reached",
                            lambda task, seq, actions, episode_len=25: 12)

        assert hints.minimise(None, [(1, 0, 0)], None, goal=10) == ()


class TestReadableNames:
    """The vocabulary spells colours as words because the transforms are
    declared that way; the grid holds digits. A prompt carrying both asks
    the model to resolve a reference nothing in it defines."""

    def test_a_colour_word_becomes_the_digit_it_means(self):
        assert hints.readable("red_recolor") == "recolor (colour 2)"

    def test_both_colours_of_a_two_colour_action_are_named(self):
        assert hints.readable("blue_contour_connection_red") == \
               "contour connection (colours 1, 2)"

    def test_a_direction_is_an_argument_not_a_word_in_the_verb(self):
        assert hints.readable("shift_object_N") == "shift object (direction N)"

    def test_a_plain_name_is_left_alone(self):
        assert hints.readable("symmetric_restoration") == "symmetric restoration"

    def test_a_word_that_only_looks_like_a_direction_stays_in_the_verb(self):
        assert hints.readable("red_background_shortest_path_left") == \
               "background shortest path left (colour 2)"


class TestRenderingABlock:
    @staticmethod
    def pooled(**kwargs):
        base = {"per_task": {}, "effective": {}, "solutions": {},
                "partials": {}, "names": NAMES, "span": (0, 1)}
        base.update(kwargs)
        return base

    @pytest.fixture(autouse=True)
    def no_env(self, monkeypatch):
        monkeypatch.setattr(hints, "render_steps",
                            lambda task, seq, names, actions, episode_len=25:
                            [f"step {names[str(s[0])]}" for s in seq])
        monkeypatch.setattr(hints, "minimise",
                            lambda task, seq, actions, episode_len=25, goal=None:
                            tuple(seq))
        monkeypatch.setattr(hints, "reached",
                            lambda task, seq, actions, episode_len=25: 4)

    def test_a_task_the_search_found_nothing_on_gets_no_block(self):
        """Not "the search found nothing" - a block that is sometimes empty
        teaches the reader to expect one."""
        assert hints.render_block(("aaa", None, None), self.pooled(), {}) is None

    def test_a_solved_task_carries_the_trace(self):
        pooled = self.pooled(solutions={"aaa": [[[1, 0, 0], [0, 0, 0]]]})

        text = hints.render_block(("aaa", None, None), pooled, {})

        assert "reproduced the output exactly" in text
        assert "1. step fliplr" in text
        assert "submit" not in text

    def test_an_unsolved_task_carries_its_furthest_attempt(self):
        pooled = self.pooled(partials={"aaa": [[0.42, [[1, 0, 0], [2, 0, 0]]]]})

        text = hints.render_block(("aaa", None, None), pooled, {})

        assert "reached 42% of the target cells" in text
        assert "1. step fliplr" in text and "2. step flipud" in text

    def test_gains_are_reported_in_cells_not_in_intersection_points(self):
        """maximal_intersection counts 2 * matches - valid, so one cell
        fixed moves it by two and a gain of 34 is 17 cells."""
        pooled = self.pooled(effective={"aaa": {"fliplr": 34}})

        text = hints.render_block(("aaa", None, None), pooled, {}, min_gain=1)

        assert "up to 17 cells" in text

    def test_moves_below_the_floor_are_not_listed(self):
        pooled = self.pooled(effective={"aaa": {"fliplr": 4, "flipud": 40}})

        text = hints.render_block(("aaa", None, None), pooled, {}, min_gain=5)

        assert "flipud" in text and "fliplr" not in text

    def test_a_task_whose_moves_are_all_below_the_floor_gets_no_block(self):
        pooled = self.pooled(effective={"aaa": {"fliplr": 4}})

        assert hints.render_block(("aaa", None, None), pooled, {},
                                   min_gain=5) is None

    def test_only_the_asked_for_number_of_moves_is_listed(self):
        pooled = self.pooled(effective={"aaa": {f"a{i}": 100 - i for i in range(10)}})

        text = hints.render_block(("aaa", None, None), pooled, {}, moves=3,
                                   min_gain=1)

        assert len([line for line in text.splitlines() if "up to" in line]) == 3


class TestLeavingTheAnswerOut:
    """On a task the search solved, the trace in the block is the answer.
    An arm that keeps it measures recipe-following on those tasks and
    hint-following on the rest, and reports one number for both."""

    @pytest.fixture(autouse=True)
    def no_env(self, monkeypatch):
        monkeypatch.setattr(hints, "render_steps",
                            lambda task, seq, names, actions, episode_len=25:
                            [f"step {names[str(s[0])]}" for s in seq])
        monkeypatch.setattr(hints, "minimise",
                            lambda task, seq, actions, episode_len=25, goal=None:
                            tuple(seq))

    @staticmethod
    def pooled():
        return {"per_task": {"aaa": 1.0}, "effective": {"aaa": {"fliplr": 40}},
                "solutions": {"aaa": [[[1, 0, 0], [0, 0, 0]]]}, "partials": {},
                "names": NAMES, "span": (0, 1)}

    def test_the_solving_sequence_is_dropped(self):
        text = hints.render_block(("aaa", None, None), self.pooled(), {},
                                   skip_solved=True)

        assert "reproduced the output exactly" not in text

    def test_the_moves_list_survives(self):
        text = hints.render_block(("aaa", None, None), self.pooled(), {},
                                   skip_solved=True)

        assert "fliplr" in text and "up to 20 cells" in text

    def test_by_default_the_sequence_is_still_there(self):
        text = hints.render_block(("aaa", None, None), self.pooled(), {})

        assert "reproduced the output exactly" in text


class TestReplayingAgainstTheRealEnv:
    """One end-to-end check that the verification means anything: the env
    the traces are replayed in has to be the env they were recorded in,
    and a stub cannot show that."""

    def test_a_recorded_solution_still_solves(self, setup):
        actions, index, task = setup

        assert hints.replays(task, [(index["red_color_outer_holes"], 0, 0)], actions)

    def test_another_action_does_not(self, setup):
        actions, index, task = setup

        assert not hints.replays(task, [(index["rotate90"], 0, 0)], actions)


class _Simulator:
    """What a cut search leaves behind: the candidates the playouts found,
    not yet verified."""

    def __init__(self, solutions):
        self.solutions = solutions


class TestASearchCutByTheTimeout:
    """A timeout used to throw away everything the search had found.

    peak and the per-action gains are recorded step by step through the
    patch on simulate_step, so they survived. Solutions did not: they sit
    in the simulator until rollout_preparation verifies them in a loop
    after its own, and a cut search never gets there - so a task the search
    had solved came back with peak 1.0 and no solution.

    Visible in the budget sweep, where at 640 iterations the median task
    ran the full 600s timeout and tasks solved at 40 and 160 came back
    unsolved: 0a2355a6, 1acc24af, 3a301edc, 42918530, 84db8fc4. The column
    was counting timeouts.
    """

    def test_a_solution_the_simulator_had_found_survives(self, setup):
        actions, index, task = setup
        candidate = [(index["red_color_outer_holes"], 0, 0)]

        rescued = hints._rescued(_Simulator([candidate]), task, actions,
                                 hints.SearchSettings())

        assert len(rescued) == 1
        assert rescued[0]["solved"]

    def test_a_candidate_that_does_not_replay_is_dropped(self, setup):
        """Verified, not trusted - the same standard the loop that was
        missed holds them to. rotate90 does not solve this task."""
        actions, index, task = setup

        rescued = hints._rescued(_Simulator([[(index["rotate90"], 0, 0)]]),
                                 task, actions, hints.SearchSettings())

        assert rescued == []

    def test_nothing_to_rescue_is_not_an_error(self, setup):
        actions, _, task = setup
        settings = hints.SearchSettings()

        assert hints._rescued(None, task, actions, settings) == []
        assert hints._rescued(_Simulator([]), task, actions, settings) == []

    def test_a_cut_search_reports_the_solution_it_had(self, setup, monkeypatch):
        """End to end through search_once: the simulator is reachable only
        from inside the patch on simulate_step, so this pins that it is
        captured there and used when the search is cut."""
        import numpy as np
        actions, index, task = setup
        candidate = [(index["red_color_outer_holes"], 0, 0)]

        def cut_short(env, **kwargs):
            simulator = hints.mcts.EnvironmentSimulator(env)
            simulator.simulate_step(hints.mcts.env_state_snapshot(env),
                                    np.array([index["rotate90"], 0, 0]))
            simulator.solutions = [candidate]
            raise hints.SearchTimedOut()

        monkeypatch.setattr(hints.mcts, "rollout_preparation", cut_short)

        found = hints.search_once(task, actions, hints.SearchSettings())

        # Submit appended by the replay, the same trace the uncut path
        # would have produced - see mcts.replay_solution.
        assert found["solutions"] == [[list(candidate[0]), [0, 0, 0]]]

    def test_a_cut_search_with_nothing_found_is_still_empty(self, setup, monkeypatch):
        actions, _, task = setup

        def cut_short(env, **kwargs):
            raise hints.SearchTimedOut()

        monkeypatch.setattr(hints.mcts, "rollout_preparation", cut_short)

        found = hints.search_once(task, actions, hints.SearchSettings())

        assert found["solutions"] == [] and found["partials"] == []


class TestTheOnlinePath:
    """`hints_for` with the search stubbed: what is pinned is the merging
    of repeats and the shape handed to the renderer, not the search."""

    @staticmethod
    def _triple():
        """A real output grid: the vocabulary is now derived from its
        palette, so None no longer stands in for one."""
        import numpy as np
        return ("aaa", np.zeros((3, 3), dtype=int), np.array([[0, 1, 2]] * 3))

    def test_repeats_keep_the_largest_gain_per_action(self, monkeypatch):
        results = iter([{"peak": 0.2, "effective": {"fliplr": 40}, "solutions": [],
                         "partials": []},
                        {"peak": 0.1, "effective": {"fliplr": 4}, "solutions": [],
                         "partials": []}])
        monkeypatch.setattr(hints, "search_once",
                            lambda task, actions, settings: next(results))
        monkeypatch.setattr(hints, "render_block",
                            lambda task, found, *a, **k: found)

        found = hints.hints_for(self._triple(),
                                hints.SearchSettings(repeats=2))

        assert found["effective"] == {"aaa": {"fliplr": 40}}, \
            "the largest gain, not the last search's"

    def test_repeats_keep_the_best_peak(self, monkeypatch):
        results = iter([{"peak": 0.9, "effective": {}, "solutions": [], "partials": []},
                        {"peak": 0.1, "effective": {}, "solutions": [], "partials": []}])
        seen = {}
        monkeypatch.setattr(hints, "search_once",
                            lambda task, actions, settings: next(results))
        monkeypatch.setattr(hints, "render_block",
                            lambda task, found, *a, **k: seen.update(found) or "x")

        hints.hints_for(self._triple(), hints.SearchSettings(repeats=2))

        assert seen["solutions"] == {"aaa": []}

    def test_a_solution_found_in_any_repeat_survives(self, monkeypatch):
        results = iter([{"peak": 0.0, "effective": {}, "solutions": [[[1, 0, 0]]],
                         "partials": []},
                        {"peak": 0.0, "effective": {}, "solutions": [], "partials": []}])
        monkeypatch.setattr(hints, "search_once",
                            lambda task, actions, settings: next(results))
        monkeypatch.setattr(hints, "render_block",
                            lambda task, found, *a, **k: found)

        found = hints.hints_for(self._triple(), hints.SearchSettings(repeats=2))

        assert found["solutions"] == {"aaa": [[[1, 0, 0]]]}


class TestWhatCountsAsATask:
    def test_a_triple_is_taken_as_it_is(self):
        assert hints.as_triple(("aaa", 1, 2)) == ("aaa", 1, 2)

    def test_an_arc_task_gives_up_its_first_training_pair(self):
        from types import SimpleNamespace
        task = SimpleNamespace(label="bbb", subtasks=[
            SimpleNamespace(train_inp="in", train_out="out"),
            SimpleNamespace(train_inp="other", train_out="other")])

        assert hints.as_triple(task) == ("bbb", "in", "out")


class TestCaching:
    def test_the_same_task_is_searched_once(self, monkeypatch):
        """A run builds one task's prompt more than once - a retry, a second
        arm, a resumed checkpoint - and a fresh search each time would give
        the same task a different hint, since the search is random."""
        calls = []
        monkeypatch.setattr(hints, "hints_for",
                            lambda task, settings: calls.append(task) or "text")
        cache = hints.HintCache()

        assert cache(("aaa", None, None)) == "text"
        assert cache(("aaa", None, None)) == "text"
        assert len(calls) == 1


class TestTheTimeCap:
    """Median search is seconds, the slowest measured was 165s. Online that
    is one task in twenty stalling a run, so a search is cut short - and
    what it had already found is kept, since the peak and the per-action
    gains are recorded as it goes."""

    def test_a_search_that_runs_long_still_reports_what_it_saw(self, monkeypatch):
        def slow(*args, **kwargs):
            raise hints.SearchTimedOut()

        monkeypatch.setattr(hints.mcts, "rollout_preparation", slow)
        actions = {0: "submit", 1: "fliplr"}
        task = ("aaa", __import__("numpy").zeros((3, 3), dtype=int),
                __import__("numpy").ones((3, 3), dtype=int))

        found = hints.search_once(task, actions, hints.SearchSettings(timeout=1))

        assert found["solutions"] == [] and found["partials"] == []
        assert "peak" in found and "effective" in found

    def test_no_cap_is_asked_for_when_the_timeout_is_zero(self):
        with hints.time_limit(0):
            pass  # nothing to assert but that it does not arm or raise

    def test_the_cap_fires(self):
        import time

        with pytest.raises(hints.SearchTimedOut):
            with hints.time_limit(1):
                time.sleep(3)

    def test_the_previous_handler_is_put_back(self):
        import signal

        before = signal.getsignal(signal.SIGALRM)
        with hints.time_limit(5):
            pass

        assert signal.getsignal(signal.SIGALRM) is before


class TestTheVocabularyASearchGets:
    def test_the_stub_actions_are_not_in_it(self):
        """copy, copy_input, paste and cut are dispatched and return the grid
        untouched - verified over 43 applications apiece - so a search given
        them spends draws on four actions that cannot do anything."""
        names = set(hints.build_vocabulary(("red", "blue"), ("N", "E")).values())

        assert not names & {"copy", "copy_input", "paste", "cut"}

    def test_colours_come_from_the_output_grid(self):
        import numpy as np

        assert hints.output_colours(np.array([[0, 3], [8, 3]])) == \
               ("black", "green", "sky")

    def test_a_colour_the_answer_needs_gets_actions_of_its_own(self):
        """The measured gap this closes: with a fixed red/blue vocabulary,
        81 of 260 scanned tasks needed a colour outside {1, 2} and not one
        was solved, because nothing in the action space could paint it."""
        import numpy as np

        derived = hints.build_vocabulary(
            hints.output_colours(np.array([[0, 3], [3, 3]])), ("N", "E"))

        assert any(name.startswith("green_") for name in derived.values())
        assert not any(name.startswith("red_") for name in derived.values())


class TestTheVocabularyOneTaskGets:
    def test_hints_for_derives_the_palette_from_this_task(self, monkeypatch):
        """Not just build_vocabulary in isolation - the wiring, since a
        fixed pair here is exactly the bug being closed."""
        import numpy as np
        seen = {}

        def capture(task, actions, settings):
            seen.update(actions)
            return {"peak": 0.0, "effective": {}, "solutions": [], "partials": []}

        monkeypatch.setattr(hints, "search_once", capture)
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")
        task = ("aaa", np.zeros((3, 3), dtype=int), np.full((3, 3), 3))

        hints.hints_for(task, hints.SearchSettings())

        assert any(name.startswith("green_") for name in seen.values())
        assert not any(name.startswith("blue_") for name in seen.values())


class TestTheBudget:
    @staticmethod
    def _triple():
        import numpy as np
        return ("aaa", np.zeros((3, 3), dtype=int), np.array([[0, 1, 2]] * 3))

    def test_repeats_stop_once_the_budget_is_spent(self, monkeypatch):
        import time as clock
        calls = []

        def slow(task, actions, settings):
            calls.append(settings.timeout)
            clock.sleep(1.2)
            return {"peak": 0.0, "effective": {}, "solutions": [], "partials": []}

        monkeypatch.setattr(hints, "search_once", slow)
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(),
                        hints.SearchSettings(repeats=5, budget=2, timeout=60))

        assert len(calls) < 5, "the budget bounds the task, not each search"

    def test_a_solved_task_does_not_pay_for_more_repeats(self, monkeypatch):
        calls = []

        def solving(task, actions, settings):
            calls.append(1)
            return {"peak": 1.0, "effective": {}, "solutions": [[[1, 0, 0]]],
                    "partials": []}

        monkeypatch.setattr(hints, "search_once", solving)
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(), hints.SearchSettings(repeats=4))

        assert len(calls) == 1

    def test_without_a_budget_every_repeat_runs(self, monkeypatch):
        calls = []
        monkeypatch.setattr(hints, "search_once",
                            lambda task, actions, settings: calls.append(1) or
                            {"peak": 0.0, "effective": {}, "solutions": [],
                             "partials": []})
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(), hints.SearchSettings(repeats=3))

        assert len(calls) == 3


class TestSearchingInParallel:
    """Repeats share nothing - not a tree, not a seed, not a task - so they
    are the whole of the parallelism a solve-time budget can spend. What is
    pinned here is the plumbing, not the speedup: that every repeat is
    submitted at once, that a pool is reused rather than rebuilt per task
    (a spawned worker re-imports torch), and that a process which cannot
    have a pool searches in line instead of failing."""

    @staticmethod
    def _triple():
        import numpy as np
        return ("aaa", np.zeros((3, 3), dtype=int), np.array([[0, 1, 2]] * 3))

    def test_every_repeat_is_submitted_at_once(self, monkeypatch):
        submitted = []

        class FakePool:
            def submit(self, fn, payload):
                submitted.append(payload)
                from concurrent.futures import Future
                future = Future()
                future.set_result({"peak": 0.0, "effective": {}, "solutions": [],
                                   "partials": []})
                return future

        monkeypatch.setattr(hints, "_pool", lambda workers: FakePool())
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(),
                        hints.SearchSettings(repeats=4, workers=4))

        assert len(submitted) == 4

    def test_one_worker_stays_in_this_process(self, monkeypatch):
        calls = []
        monkeypatch.setattr(hints, "search_once",
                            lambda *a, **k: calls.append(1) or
                            {"peak": 0.0, "effective": {}, "solutions": [],
                             "partials": []})
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(), hints.SearchSettings(repeats=2, workers=1))

        assert len(calls) == 2, "workers=1 searches here, without a pool"

    def test_a_process_that_cannot_have_a_pool_searches_in_line(self, monkeypatch):
        calls = []
        monkeypatch.setattr(hints, "_pool", lambda workers: None)
        monkeypatch.setattr(hints, "search_once",
                            lambda *a, **k: calls.append(1) or
                            {"peak": 0.0, "effective": {}, "solutions": [],
                             "partials": []})
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        hints.hints_for(self._triple(),
                        hints.SearchSettings(repeats=3, workers=4))

        assert len(calls) == 3

    def test_the_pool_is_kept_between_tasks(self, monkeypatch):
        built = []

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                built.append(1)

            def shutdown(self, **kwargs):
                pass

        monkeypatch.setattr(hints, "ProcessPoolExecutor", FakeExecutor)
        hints.shutdown_pool()

        hints._pool(4)
        hints._pool(4)

        assert len(built) == 1
        hints.shutdown_pool()

    def test_asking_for_a_different_width_rebuilds_it(self, monkeypatch):
        built = []

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                built.append(kwargs.get("max_workers"))

            def shutdown(self, **kwargs):
                pass

        monkeypatch.setattr(hints, "ProcessPoolExecutor", FakeExecutor)
        hints.shutdown_pool()

        hints._pool(2)
        hints._pool(4)

        assert built == [2, 4]
        hints.shutdown_pool()

    def test_a_pool_that_dies_does_not_take_the_run_with_it(self, monkeypatch):
        """Measured, not imagined: four spawned workers beside a loaded model
        exhausted memory and the pool came back BrokenProcessPool. A hint is
        worth less than the run it would otherwise kill thirteen hours in,
        so the searches fall back into this process."""
        from concurrent.futures.process import BrokenProcessPool
        calls = []

        class DeadPool:
            def submit(self, fn, payload):
                raise BrokenProcessPool("child died")

        monkeypatch.setattr(hints, "_pool", lambda workers: DeadPool())
        monkeypatch.setattr(hints, "search_once",
                            lambda *a, **k: calls.append(1) or
                            {"peak": 0.0, "effective": {}, "solutions": [],
                             "partials": []})
        monkeypatch.setattr(hints, "render_block", lambda *a, **k: "x")

        out = hints.hints_for(self._triple(),
                              hints.SearchSettings(repeats=3, workers=4))

        assert out == "x"
        assert len(calls) == 3


class TestThePlayout:
    """Which action a playout tries next is the single biggest difference
    between a search that finds something and one that does not. Measured on
    3eda0437 at identical settings: 1 effective action on the default
    playout against 47 on the weighted one, at 3.2s against 31s. The scan
    that produced every reference figure ran weighted; this ran default
    until that was found, which is most of why an online hint looked thin."""

    @staticmethod
    def _triple():
        import numpy as np
        return ("aaa", np.zeros((3, 3), dtype=int), np.array([[0, 1, 2]] * 3))

    def test_weighted_is_the_default(self):
        assert hints.SearchSettings().playout == "weighted"

    @staticmethod
    def _during_the_search(monkeypatch, playout):
        """What the simulator class looked like while the search ran."""
        seen = {}
        monkeypatch.setattr(hints, "make_env",
                            lambda task, actions, episode_len: _FakeEnv())
        monkeypatch.setattr(
            hints.mcts, "rollout_preparation",
            lambda env, **kwargs: seen.setdefault(
                "init", hints.mcts.EnvironmentSimulator.__init__) and [])
        hints.search_once(("aaa", None, None), {0: "submit"},
                          hints.SearchSettings(playout=playout))
        return seen["init"]

    def test_weighted_replaces_what_a_simulator_is_built_with(self, monkeypatch):
        original = hints.mcts.EnvironmentSimulator.__init__

        assert self._during_the_search(monkeypatch, "weighted") is not original

    def test_the_default_playout_leaves_the_simulator_alone(self, monkeypatch):
        original = hints.mcts.EnvironmentSimulator.__init__

        assert self._during_the_search(monkeypatch, "default") is original

    def test_the_class_is_left_as_it_was_found(self, monkeypatch):
        """The patch is global to the class, so a search that did not put it
        back would change every later search in the process - including one
        a caller asked for on the default playout."""
        before_init = hints.mcts.EnvironmentSimulator.__init__
        before_step = hints.mcts.EnvironmentSimulator.simulate_step
        monkeypatch.setattr(hints, "make_env",
                            lambda task, actions, episode_len: _FakeEnv())
        monkeypatch.setattr(hints.mcts, "rollout_preparation",
                            lambda env, **kwargs: [])

        hints.search_once(self._triple(), {0: "submit"},
                          hints.SearchSettings(playout="weighted"))

        assert hints.mcts.EnvironmentSimulator.__init__ is before_init
        assert hints.mcts.EnvironmentSimulator.simulate_step is before_step


class _FakeEnv:
    max_int = 0
    target_int = 10


class TestTheDefaultsAreTheMeasuredOnes:
    """One search at these settings carried a hint on 17 of 20 tasks where
    the scan's file - three searches pooled - carried one on 16. Defaults
    that differ from the configuration the reference data came from make
    every comparison against it meaningless, which is what happened."""

    def test_the_scan_settings_are_the_defaults(self):
        settings = hints.SearchSettings()

        assert (settings.rollouts, settings.iterations, settings.playout,
                settings.rounds) == (5, 50, "weighted", 1)
