"""What an evaluation shows besides its number.

An accuracy is a fraction of a distance - it says nothing about what the
policy did to get it, and a submit on step one and a twenty-step detour
that ends in the same place score alike. evaluate_ARC_policy can fill a
trace of the episode behind the number, step_label says each step in words
that can be checked against the grid, and rl.plotting draws and prints it.

The model here is scripted, not trained: what is asserted is that the
trace says what happened, so what happens has to be known in advance.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from rl.arc_env import ARCGridWorld  # noqa: E402
from rl.arc_task import ARCSubtask  # noqa: E402
from rl.evaluation import describe_slot, evaluate_ARC_policy, step_label  # noqa: E402
from rl.plotting import (collapsed_steps, describe_trace, plot_evaluation,  # noqa: E402
                         plot_overview)

COORDINATE_ACTIONS = {0: "submit", 1: "red_fill", 2: "red_line", 3: "red_triangle"}


class Scripted:
    """A model whose predict plays a fixed list of actions, then submits -
    the same action in every env of the vector."""

    def __init__(self, actions, submit):
        self.actions = [np.asarray(a) for a in actions]
        self.submit = np.asarray(submit)
        self.calls = 0

    def predict(self, observations, state=None, deterministic=True):
        action = (self.actions[self.calls] if self.calls < len(self.actions)
                  else self.submit)
        self.calls += 1
        n_envs = len(observations["grid"])
        return np.repeat(action[None, :], n_envs, axis=0), state


def coordinate_case():
    """A block of background no object can name; one fill closes it."""
    inp = np.zeros((7, 7), dtype=int)
    inp[3, 2], inp[5, 4] = 1, 1
    out = inp.copy()
    out[2:5, 0:3] = 2
    return inp, out


def coordinate_env(inp, out, max_episode_len=5):
    env = ARCGridWorld(max_episode_len=max_episode_len, feasible_actions=COORDINATE_ACTIONS,
                       reward_approach=3, repr_level=1, input_pattern="start",
                       addressing="coordinates", coordinate_shape=(7, 7),
                       observation_space_elements=["delta_input"])
    env.set_subtask(ARCSubtask("fill_case", inp, out))
    env.reset()
    return env


def object_env(whitelist=None):
    inp = np.zeros((6, 6), dtype=int)
    inp[1:3, 1:4] = 3          # a green 2x3 block
    inp[4, 5] = 4              # a yellow cell
    out = inp.copy()
    env = ARCGridWorld(max_episode_len=5, reward_approach=3, repr_level=1,
                       input_pattern="start",
                       feasible_actions={0: "submit", 1: "rotate90", 2: "red_recolor"},
                       observation_space_elements=["objects_emb"],
                       action_whitelist=whitelist)
    env.set_subtask(ARCSubtask("object_case", inp, out))
    env.reset()
    return env


class TestAStepInWords:
    def test_a_coordinate_action_names_its_colour_and_both_cells(self):
        env = coordinate_env(*coordinate_case())
        assert step_label(env, [1, 2, 0, 4, 2]) == "red fill (2,0)-(4,2)"
        assert step_label(env, [3, 6, 0, 0, 6]) == "red triangle (6,0)-(0,6)"

    def test_submit_is_submit(self):
        env = coordinate_env(*coordinate_case())
        assert step_label(env, [0, 3, 3, 1, 1]) == "submit"

    def test_an_object_action_says_which_object_by_what_it_looks_like(self):
        """A slot number is meaningless to a reader - and not even stable,
        the objects are rebuilt every step - so the label carries the
        object's colour, size and extent instead."""
        env = object_env()
        green = next(i for i, o in enumerate(env.objects)
                     if 3 in [int(c) for c in o.color_numbers])
        label = step_label(env, [2, green, green])
        assert label == "red recolor [green: 6 cells, rows 1-2, cols 1-3]"

    def test_two_objects_are_both_named(self):
        env = object_env()
        label = step_label(env, [1, 0, 1])
        assert label.count("[") == 2 and " & " in label

    def test_a_slot_past_the_objects_is_said_to_be_empty(self):
        env = object_env()
        assert describe_slot(env.objects, len(env.objects)) == "empty slot"

    def test_under_a_whitelist_the_label_is_of_the_action_applied(self):
        """The policy picks an index into the whitelist; [0, 0, 0] then
        means its first entry, not submit."""
        env = object_env(whitelist=[(2, 0, 0), (0, 0, 0)])
        assert step_label(env, [0, 0, 0]).startswith("red recolor")
        assert step_label(env, [1, 0, 0]) == "submit"


class TestTheTraceOfAnEvaluation:
    def _evaluate(self, actions, submit=(0, 0, 0, 0, 0), max_episode_len=5):
        inp, out = coordinate_case()
        vec_env = DummyVecEnv([lambda: coordinate_env(inp, out, max_episode_len)])
        traces = []
        acc, length, _grid = evaluate_ARC_policy(Scripted(actions, submit), vec_env,
                                                 n_eval_episodes=1, traces=traces)
        assert len(traces) == 1
        return traces[0], acc, length, inp, out

    def test_it_holds_every_step_with_the_grid_after_it(self):
        """Two rows of the block, then all three - the episode ends on
        the step that solves it."""
        trace, acc, length, inp, out = self._evaluate([[1, 2, 0, 3, 2], [1, 2, 0, 4, 2]])

        assert [label for label, _g, _r in trace["steps"]] == [
            "red fill (2,0)-(3,2)", "red fill (2,0)-(4,2)"]
        partial = inp.copy()
        partial[2:4, 0:3] = 2
        assert np.array_equal(trace["start"], inp)
        assert np.array_equal(trace["steps"][0][1], partial)
        assert np.array_equal(trace["steps"][1][1], out)
        assert np.array_equal(trace["target"], out)
        assert trace["accuracy"] == acc == 1.0
        assert len(trace["steps"]) == length
        assert trace["subtask"] == "fill_case"

    def test_the_last_step_is_where_the_episode_ended_not_the_reset(self):
        """The vec env resets the moment an episode ends and hands back the
        new episode's first observation; the grid the episode ended on is
        in terminal_observation. Recording the observation would draw the
        solved grid as the input it started from."""
        trace, *_rest, out = self._evaluate([[1, 2, 0, 4, 2]])
        assert np.array_equal(trace["steps"][-1][1], out)

    def test_a_submit_is_a_step_and_ends_it_where_it_was(self):
        trace, acc, length, inp, _out = self._evaluate([[1, 2, 0, 3, 2]])
        assert [label for label, _g, _r in trace["steps"]][-1] == "submit"
        assert length == 2
        assert not np.array_equal(trace["steps"][-1][1], inp)
        assert 0 < trace["accuracy"] < 1

    def test_the_rewards_are_the_ones_the_env_paid(self):
        trace, *_ = self._evaluate([[1, 2, 0, 4, 2]])
        assert trace["steps"][0][2] > 0

    def test_no_trace_asked_for_changes_nothing(self):
        inp, out = coordinate_case()
        vec_env = DummyVecEnv([lambda: coordinate_env(inp, out)])
        result = evaluate_ARC_policy(Scripted([[1, 2, 0, 4, 2]], (0, 0, 0, 0, 0)), vec_env,
                                     n_eval_episodes=1)
        assert result[0] == 1.0 and result[1] == 1


class TestOneTracePerExample:
    """Training puts one env per example in the vector; an evaluation
    traces each of them, so the training examples can be looked at, not
    only the held-out pair."""

    def test_every_env_gets_its_own_trace(self):
        inp, out = coordinate_case()
        other = out.copy()
        other[6, 6] = 3
        vec_env = DummyVecEnv([lambda: coordinate_env(inp, out),
                               lambda: coordinate_env(inp, other)])
        vec_env.envs[1].unwrapped.set_subtask(ARCSubtask("other_case", inp, other))
        traces = []
        evaluate_ARC_policy(Scripted([[1, 2, 0, 4, 2]], (0, 0, 0, 0, 0)), vec_env,
                            n_eval_episodes=1, traces=traces)
        assert [trace["subtask"] for trace in traces] == ["fill_case", "other_case"]
        assert traces[0]["accuracy"] == 1.0 and traces[0]["steps"][-1][0] != "submit"
        assert traces[1]["accuracy"] < 1.0 and traces[1]["steps"][-1][0] == "submit"
        assert np.array_equal(traces[1]["target"], other)


def trace_of(steps, start=None):
    start = np.zeros((3, 3), dtype=int) if start is None else start
    return {"start": start, "steps": steps, "target": np.ones((3, 3), dtype=int),
            "accuracy": 0.25, "subtask": "case"}


class TestFoldingRepeats:
    def test_the_same_action_leaving_the_grid_alone_is_one_run(self):
        once = np.eye(3, dtype=int)
        steps = [("a", once, -0.1)] + [("b", once, -0.01)] * 4
        runs = collapsed_steps(trace_of(steps))
        assert [(first, last, label) for first, last, label, _g, _r in runs] == [
            (1, 1, "a"), (2, 5, "b")]
        assert runs[1][4] == pytest.approx(-0.04)

    def test_a_repeat_that_changes_the_grid_is_a_step_of_its_own(self):
        grids = [np.full((3, 3), value) for value in (1, 2, 3)]
        steps = [("a", grid, 0.0) for grid in grids]
        assert len(collapsed_steps(trace_of(steps))) == 3

    def test_a_different_action_that_changes_nothing_is_not_folded_in(self):
        same = np.zeros((3, 3), dtype=int)
        steps = [("a", same, 0.0), ("b", same, 0.0)]
        assert len(collapsed_steps(trace_of(steps))) == 2


class TestReadingItBack:
    def test_the_lines_say_what_each_step_did(self):
        once = np.eye(3, dtype=int)
        text = describe_trace(trace_of([("red fill (0,0)-(2,2)", once, 0.5)]
                                       + [("submit", once, 0.0)]))
        lines = text.splitlines()
        assert lines[0] == "case: closed +0.250 of the distance in 2 steps"
        assert "red fill (0,0)-(2,2)" in lines[1] and "+0.500" in lines[1]
        assert lines[2].strip().startswith("2. submit")

    def test_an_empty_trace_says_so(self):
        assert describe_trace({}) == "no episode recorded"

    def test_the_picture_has_start_target_result_and_a_panel_per_run(self):
        once = np.eye(3, dtype=int)
        steps = [("red fill (0,0)-(2,2)", once, 0.5)] + [("red line", once, -0.01)] * 3
        fig = plot_evaluation(trace_of(steps))
        try:
            titles = [ax.get_title() for ax in fig.axes]
            assert len(fig.axes) == 3 + 2
            assert titles[0].startswith("start") and titles[1] == "target"
            assert titles[2].startswith("result")
            assert "red fill (0,0)-(2,2)" in titles[3]
            assert titles[4].startswith("2-4 (x3)")
            assert "closed +0.250" in fig._suptitle.get_text()
        finally:
            plt.close(fig)

    def test_the_overview_is_a_row_per_example(self):
        once = np.eye(3, dtype=int)
        traces = [trace_of([("red fill (0,0)-(2,2)", once, 0.5)]), None,
                  dict(trace_of([("submit", once, 0.0)]), subtask="other")]
        fig = plot_overview(traces, title="50% of training")
        try:
            assert len(fig.axes) == 2 * 4
            texts = [text.get_text() for ax in fig.axes for text in ax.texts]
            assert any("red fill (0,0)-(2,2)" in text for text in texts)
            assert any("1. submit" in text for text in texts)
            assert fig.axes[4].get_title().startswith("other")
        finally:
            plt.close(fig)

    def test_a_long_episode_is_elided_in_the_middle(self):
        steps = [(f"step{index}", np.full((3, 3), index), 0.0) for index in range(20)]
        fig = plot_overview([trace_of(steps)], max_lines=6)
        try:
            text = fig.axes[3].texts[0].get_text()
            assert "step0" in text and "step19" in text and "step10" not in text
            assert "14 more" in text
        finally:
            plt.close(fig)

    def test_a_padded_grid_is_drawn_at_its_own_size(self):
        """Coordinate observations are padded to the task's largest grid;
        the picture crops the padding off, so the match against the target
        is computed at all."""
        padded = np.full((5, 5), 10)
        padded[:3, :3] = 1
        fig = plot_evaluation(trace_of([("x", padded, 0.0)], start=padded))
        try:
            assert "100% match" in fig.axes[3].get_title()
        finally:
            plt.close(fig)


class TestWhereAFigureGoes:
    def test_not_under_agg_outside_a_notebook(self):
        from rl.plotting import showing_figures

        assert matplotlib.get_backend().lower() == "agg"
        assert not showing_figures()

    def test_in_a_notebook_whatever_the_backend(self, monkeypatch):
        import rl.plotting

        monkeypatch.setattr(rl.plotting, "_in_notebook", lambda: True)
        assert rl.plotting.showing_figures()

    def test_under_a_backend_that_draws(self, monkeypatch):
        import rl.plotting

        monkeypatch.setattr(rl.plotting.matplotlib, "get_backend",
                            lambda: "module://matplotlib_inline.backend_inline")
        assert rl.plotting.showing_figures()

    def test_a_notebook_gets_it_through_display_and_it_is_let_go(self, monkeypatch):
        """display() renders whatever the backend; plt.show() only under
        the inline one."""
        import sys
        import types

        import rl.plotting

        shown = []
        # A stand-in, so this runs where IPython is not installed - which is
        # everywhere but a notebook.
        ipython = types.ModuleType("IPython")
        ipython.display = types.ModuleType("IPython.display")
        ipython.display.display = shown.append
        # matplotlib asks it for a shell when a figure is made.
        ipython.get_ipython = lambda: None
        monkeypatch.setitem(sys.modules, "IPython", ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", ipython.display)
        monkeypatch.setattr(rl.plotting, "_in_notebook", lambda: True)
        fig = plt.figure()
        rl.plotting.display_figure(fig)
        assert shown == [fig]
        assert not plt.fignum_exists(fig.number)

    def test_a_notebook_is_a_kernel(self, monkeypatch):
        import sys
        import types

        from rl.plotting import _in_notebook

        ipython = types.ModuleType("IPython")
        monkeypatch.setitem(sys.modules, "IPython", ipython)
        ipython.get_ipython = lambda: types.SimpleNamespace(config={"IPKernelApp": {}})
        assert _in_notebook()
        # A terminal IPython is not one: nothing renders a figure inline there.
        ipython.get_ipython = lambda: types.SimpleNamespace(config={})
        assert not _in_notebook()
        ipython.get_ipython = lambda: None
        assert not _in_notebook()
