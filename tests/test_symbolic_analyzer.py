"""Tests for symbolic/analyzer.py - the layer that turns a pair of grid
summaries into transformation patterns, and those into the hypothesis /
insight text a downstream agent reads.

The central invariant here is honesty of the output: a number in an
insight has to mean a number that was actually measured across the
training examples. A pattern parameter that varied between examples must
not reach the text as a plausible-looking default, and a pattern seen in
one example must not be described as seen in all of them - either failure
reads to an agent exactly like a real finding.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rl.arc_task import ARCSubtask
from symbolic.analyzer import SubtaskAnalysis, TransformationPattern, TaskAnalysis


def _subtask(label, inp, out):
    return ARCSubtask(label=label, train_inp=np.array(inp), train_out=np.array(out))


def _pattern(ptype, **params):
    return TransformationPattern(pattern_type=ptype, description=ptype,
                                 confidence=0.9, parameters=params)


class _StubTaskAnalysis(TaskAnalysis):
    """TaskAnalysis over pre-built pattern lists, skipping the (slow) real
    grid analysis - these tests are about how per-example patterns are
    consolidated across examples, not about detecting them."""

    def __init__(self, per_example_patterns, same_shape=True):
        self.task_id = "stub"
        self.font_color = 0
        self.levels = [2]
        self.subtasks_analyses = [
            SimpleNamespace(
                transformation_patterns=patterns,
                grid_diff=SimpleNamespace(has_size_change=not same_shape),
            )
            for patterns in per_example_patterns
        ]
        self.consistent_patterns = self._infer_consistent_patterns()


# ---------------------------------------------------------------------------
# grid diff
# ---------------------------------------------------------------------------

class TestGridDiff:
    @staticmethod
    def test_change_ratio_is_reported_for_same_shape_grids():
        analysis = SubtaskAnalysis(_subtask("s", [[1, 0], [0, 0]], [[2, 0], [0, 0]]))
        diff = analysis.grid_diff

        assert diff.num_changes == 1
        assert diff.change_ratio == pytest.approx(0.25)

    @staticmethod
    def test_change_ratio_is_reported_when_output_is_a_different_size():
        """A resize used to leave change_ratio at its 0.0 default, so the
        summary printed a cell count next to '0.0%' - the one line meant to
        say how much of the grid moved claimed nothing moved."""
        analysis = SubtaskAnalysis(_subtask("s", [[1, 0], [0, 0]], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
        diff = analysis.grid_diff

        assert diff.has_size_change is True
        assert diff.num_changes > 0
        assert diff.change_ratio > 0.0


# ---------------------------------------------------------------------------
# consolidation across examples
# ---------------------------------------------------------------------------

class TestConsistentPatterns:
    @staticmethod
    def test_parameter_agreed_by_every_example_becomes_a_common_value():
        ta = _StubTaskAnalysis([
            [_pattern("uniform_translation", shift=(1, 2))],
            [_pattern("uniform_translation", shift=(1, 2))],
        ])
        pattern = next(p for p in ta.consistent_patterns if p.pattern_type == "uniform_translation")

        assert pattern.parameters["common_values"]["shift"] == (1, 2)

    @staticmethod
    def test_parameter_that_varied_between_examples_has_no_common_value():
        ta = _StubTaskAnalysis([
            [_pattern("uniform_translation", shift=(1, 2))],
            [_pattern("uniform_translation", shift=(4, 5))],
        ])
        pattern = next(p for p in ta.consistent_patterns if p.pattern_type == "uniform_translation")

        assert "shift" not in pattern.parameters["common_values"]

    @staticmethod
    def test_a_single_example_does_not_make_a_parameter_agreed():
        """One example's value trivially equals itself. Consistency has to
        be measured over the examples that exhibited the pattern, or a
        finding from a single example is presented as a rule of the task."""
        ta = _StubTaskAnalysis([
            [_pattern("color_based_deletion", color=("green",))],
            [],
            [],
        ])

        for pattern in ta.consistent_patterns:
            if pattern.pattern_type == "color_based_deletion":
                assert "color" not in pattern.parameters["common_values"]

    @staticmethod
    def test_example_count_in_the_description_counts_examples_not_occurrences():
        """Two patterns of one type inside a single example used to be
        counted as two examples, so a one-example finding was advertised
        as holding across the whole task."""
        ta = _StubTaskAnalysis([
            [_pattern("causal_shift", rule="a"), _pattern("causal_shift", rule="a")],
            [],
        ])

        for pattern in ta.consistent_patterns:
            if pattern.pattern_type == "causal_shift":
                assert "1/2" in pattern.description

    @staticmethod
    def test_examples_taking_different_branches_do_not_agree():
        """size_scaling reports `scale_factor` when every object scaled
        alike and `scale_factors`/`mean_factor` when they didn't. Different
        keys mean the examples described different situations, so neither
        key carries a value agreed across them."""
        ta = _StubTaskAnalysis([
            [_pattern("size_scaling", scale_factor=2.0)],
            [_pattern("size_scaling", scale_factors=[2.0, 8.0], mean_factor=5.0)],
        ])
        pattern = next(p for p in ta.consistent_patterns if p.pattern_type == "size_scaling")

        assert pattern.parameters["common_values"] == {}


# ---------------------------------------------------------------------------
# honesty of the rendered output
# ---------------------------------------------------------------------------

class TestOutputHonesty:
    @staticmethod
    def test_agreed_shift_is_named_in_hypothesis_and_insights():
        ta = _StubTaskAnalysis([
            [_pattern("uniform_translation", shift=(1, 2))],
            [_pattern("uniform_translation", shift=(1, 2))],
        ])

        assert "(1, 2)" in ta.get_transformation_hypothesis()
        assert any("(1, 2)" in insight for insight in ta.get_actionable_insights())

    @staticmethod
    def test_varying_shift_never_reaches_the_output_as_a_number():
        """The failure this guards against is not a missing line but a
        convincing one: an unresolved parameter rendered as (0, 0) is
        indistinguishable from a measured zero offset."""
        ta = _StubTaskAnalysis([
            [_pattern("uniform_translation", shift=(1, 2))],
            [_pattern("uniform_translation", shift=(4, 5))],
        ])

        text = ta.get_transformation_hypothesis() + "\n".join(ta.get_actionable_insights())

        assert "(0, 0)" not in text
        assert "(1, 2)" not in text
        assert "(4, 5)" not in text

    @staticmethod
    def test_unresolved_parameters_are_not_rendered_as_placeholder_words():
        ta = _StubTaskAnalysis([
            [_pattern("causal_shift", rule="shift_equals_inner_holes")],
            [_pattern("causal_shift", rule="shift_equals_size")],
        ])

        text = ta.get_transformation_hypothesis() + "\n".join(ta.get_actionable_insights())

        assert "unknown" not in text

    @staticmethod
    def test_insights_never_end_in_an_empty_value():
        """`Delete all objects with color: ` - a label with nothing after
        it - is the same failure wearing a different mask."""
        ta = _StubTaskAnalysis([
            [_pattern("color_based_deletion", color=("green",))],
            [_pattern("color_based_deletion", color=("red",))],
        ])

        for insight in ta.get_actionable_insights():
            assert not insight.rstrip().endswith(":")
