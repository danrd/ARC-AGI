"""Tests for symbolic/findings.py - the structured form the symbolic layer
hands to its consumers, and the text rendered on top of it.

Two properties matter more than the wording, and both are things the previous
prose-only output got wrong: a parameter that varied between examples must
never surface as a value (a reader cannot tell a default from a measurement),
and every claim must name the examples backing it. The budget tests exist for
a third: fitting is done by dropping whole claims, never by cutting one in
half, since a truncated claim still reads as a complete one.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from symbolic.findings import (
    Evidence,
    Finding,
    TaskFindings,
    build_task_findings,
    render_findings,
)


def _finding(subject, statement, indices, total, confidence=0.9, **params):
    return Finding(
        subject=subject,
        statement=statement,
        evidence=Evidence(tuple(indices), total),
        confidence=confidence,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    @staticmethod
    def test_support_from_every_example_is_named_as_such():
        assert Evidence((0, 1, 2), 3).holds_everywhere is True
        assert Evidence((0, 1, 2), 3).render() == "all 3 examples"

    @staticmethod
    def test_partial_support_names_the_examples_and_is_one_based():
        """Examples are counted from 1 in the text because that is how they
        are numbered everywhere the reader sees them."""
        evidence = Evidence((0, 2), 3)

        assert evidence.holds_everywhere is False
        assert evidence.render() == "examples 1, 3 of 3"

    @staticmethod
    def test_no_examples_is_not_everywhere():
        assert Evidence((), 0).holds_everywhere is False


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

class TestRanking:
    @staticmethod
    def test_claim_true_everywhere_with_parameters_outranks_a_more_confident_one():
        """Ranking is by usefulness, not by the confidence constant the
        detector happened to assign: a parameterised claim holding in every
        example is more actionable than a vague one scored higher."""
        weak_but_confident = _finding("a", "vague", (0,), 3, confidence=1.0)
        strong = _finding("b", "concrete", (0, 1, 2), 3, confidence=0.5, offset=(1, 1))

        ordered = sorted([weak_but_confident, strong], key=lambda f: f.rank_key, reverse=True)

        assert ordered[0] is strong

    @staticmethod
    def test_parameters_break_the_tie_between_equally_supported_claims():
        without = _finding("a", "no params", (0, 1), 2, confidence=0.9)
        with_params = _finding("b", "params", (0, 1), 2, confidence=0.9, factor=2)

        ordered = sorted([without, with_params], key=lambda f: f.rank_key, reverse=True)

        assert ordered[0] is with_params


# ---------------------------------------------------------------------------
# building from a TaskAnalysis
# ---------------------------------------------------------------------------

def _grid_summary(object_count, level=2):
    objects = tuple(range(object_count)) if object_count is not None else None
    return SimpleNamespace(repr_levels={level: SimpleNamespace(objects=objects)})


def _example(inp, out, size_change=False, objects_in=1, objects_out=1):
    inp, out = np.array(inp), np.array(out)
    return SimpleNamespace(
        input_grid=inp,
        output_grid=out,
        grid_diff=SimpleNamespace(has_size_change=size_change),
        input_summary=_grid_summary(objects_in),
        output_summary=_grid_summary(objects_out),
        primary_level=2,
    )


def _pattern(ptype, confidence=0.9, common=None, indices=(0, 1)):
    return SimpleNamespace(
        pattern_type=ptype,
        confidence=confidence,
        parameters={"common_values": common or {}, "example_indices": tuple(indices)},
    )


def _stub_analysis(patterns, examples, task_id="stub"):
    return SimpleNamespace(
        task_id=task_id,
        consistent_patterns=patterns,
        subtasks_analyses=examples,
    )


class TestBuildFindings:
    @staticmethod
    def test_agreed_parameter_is_named_in_the_statement():
        analysis = _stub_analysis(
            [_pattern("uniform_translation", common={"offset": (1, 2)})],
            [_example([[1]], [[1]]), _example([[1]], [[1]])],
        )

        findings = build_task_findings(analysis)

        assert "(1, 2)" in findings.transformations[0].statement
        assert findings.transformations[0].parameters == {"offset": (1, 2)}

    @staticmethod
    def test_parameter_that_varied_never_reaches_the_statement_as_a_value():
        """The failure the whole structure exists to prevent: the audit's
        showcase was 'Translate all objects by offset: (0, 0)' where the real
        shifts were (-2, 5), (3, 6) and (0, 3)."""
        analysis = _stub_analysis(
            [_pattern("uniform_translation", common={})],
            [_example([[1]], [[1]]), _example([[1]], [[1]])],
        )

        statement = build_task_findings(analysis).transformations[0].statement

        assert "differ between examples" in statement
        assert "(0, 0)" not in statement

    @staticmethod
    def test_unknown_pattern_type_is_rendered_readably_not_dropped():
        analysis = _stub_analysis(
            [_pattern("some_new_detector", common={})],
            [_example([[1]], [[1]])],
        )

        assert build_task_findings(analysis).transformations[0].statement == "some new detector"

    @staticmethod
    def test_evidence_comes_from_the_examples_that_actually_showed_the_pattern():
        analysis = _stub_analysis(
            [_pattern("object_addition", indices=(0, 2))],
            [_example([[1]], [[1]]) for _ in range(3)],
        )

        evidence = build_task_findings(analysis).transformations[0].evidence

        assert evidence.example_indices == (0, 2)
        assert evidence.example_count == 3
        assert evidence.holds_everywhere is False


class TestInvariants:
    @staticmethod
    def test_preserved_grid_size_palette_and_object_count_are_reported():
        examples = [_example([[1, 0]], [[0, 1]]), _example([[2, 0]], [[0, 2]])]

        subjects = {f.subject for f in build_task_findings(_stub_analysis([], examples)).invariants}

        assert subjects == {"grid_size", "palette", "object_count"}

    @staticmethod
    def test_size_change_in_any_example_withdraws_the_size_invariant():
        """An invariant holding in some examples is a coincidence, not an
        invariant - reporting it would be the same class of error as naming
        a parameter that varied."""
        examples = [_example([[1]], [[1]]), _example([[1]], [[1]], size_change=True)]

        subjects = {f.subject for f in build_task_findings(_stub_analysis([], examples)).invariants}

        assert "grid_size" not in subjects

    @staticmethod
    def test_changed_palette_withdraws_the_palette_invariant():
        examples = [_example([[1]], [[2]]), _example([[1]], [[1]])]

        subjects = {f.subject for f in build_task_findings(_stub_analysis([], examples)).invariants}

        assert "palette" not in subjects

    @staticmethod
    def test_changed_object_count_withdraws_the_count_invariant():
        examples = [_example([[1]], [[1]], objects_in=1, objects_out=3)]

        subjects = {f.subject for f in build_task_findings(_stub_analysis([], examples)).invariants}

        assert "object_count" not in subjects

    @staticmethod
    def test_missing_level_does_not_claim_a_count_invariant():
        """Levels are caller-selected; when the primary one wasn't parsed
        there is no count to compare, which is not the same as it matching."""
        example = _example([[1]], [[1]])
        example.input_summary = SimpleNamespace(repr_levels={})
        example.output_summary = SimpleNamespace(repr_levels={})

        subjects = {f.subject for f in build_task_findings(_stub_analysis([], [example])).invariants}

        assert "object_count" not in subjects

    @staticmethod
    def test_no_examples_yields_no_invariants():
        assert build_task_findings(_stub_analysis([], [])).invariants == ()


# ---------------------------------------------------------------------------
# rendering under a budget
# ---------------------------------------------------------------------------

def _findings(n_transformations=2, n_invariants=1):
    return TaskFindings(
        task_id="t",
        example_count=2,
        transformations=tuple(
            _finding(f"t{i}", f"transformation {i}", (0, 1), 2) for i in range(n_transformations)
        ),
        invariants=tuple(
            _finding(f"i{i}", f"invariant {i}", (0, 1), 2) for i in range(n_invariants)
        ),
    )


class TestRender:
    @staticmethod
    def test_empty_findings_render_to_nothing_rather_than_a_bare_header():
        assert render_findings(TaskFindings(task_id="t", example_count=0)) is None

    @staticmethod
    def test_both_sections_appear_when_budget_allows():
        text = render_findings(_findings())

        assert "What changes:" in text
        assert "What stays the same:" in text
        assert "transformation 0" in text
        assert "invariant 0" in text

    @staticmethod
    def test_claims_carry_their_evidence():
        assert "[all 2 examples]" in render_findings(_findings(1, 0))

    @staticmethod
    def test_tight_budget_drops_whole_claims_without_cutting_any_in_half():
        findings = _findings(n_transformations=4, n_invariants=0)
        full = render_findings(findings)

        trimmed = render_findings(findings, budget=len(full) - 20, count_tokens=len)

        assert trimmed is not None
        assert len(trimmed) < len(full)
        # every line that survived is a complete claim, not a fragment
        for line in trimmed.splitlines():
            if line.startswith("  - "):
                assert line.rstrip().endswith("]")

    @staticmethod
    def test_budget_too_small_for_even_one_claim_omits_the_block():
        assert render_findings(_findings(), budget=1, count_tokens=len) is None

    @staticmethod
    def test_a_section_with_nothing_left_does_not_print_its_header():
        findings = _findings(n_transformations=1, n_invariants=1)
        first_only = render_findings(findings, budget=0, count_tokens=lambda s: 0 if "invariant" not in s else 99)

        assert "What stays the same:" not in first_only
