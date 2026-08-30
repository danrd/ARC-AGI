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
    render_hypothesis,
    render_insights,
)


def _parameter_keys_by_pattern_type():
    """Which parameter keys each detector puts into the pattern it emits,
    read out of the source rather than by running the (slow) real analysis
    over enough tasks to hit every branch.

    Keyed by pattern type rather than pooled: a key can be perfectly real for
    one detector and meaningless for another - `offset` belongs to
    `translation`, while `uniform_translation` writes `shift` - so a pooled
    check would wave exactly that confusion through.
    """
    import ast
    import collections
    import pathlib

    source = pathlib.Path("symbolic/analyzer.py").read_text(encoding="utf-8")
    by_type = collections.defaultdict(set)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        pattern_type = keywords.get("pattern_type")
        parameters = keywords.get("parameters")
        if not isinstance(pattern_type, ast.Constant) or not isinstance(pattern_type.value, str):
            continue
        if not isinstance(parameters, ast.Dict):
            continue
        by_type[pattern_type.value].update(
            key.value for key in parameters.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    return by_type


def test_every_phrase_references_a_parameter_its_own_detector_produces():
    """Regression test: the phrase table named `offset` where
    uniform_translation writes `shift`, and `factor` where size_scaling
    writes `scale_factor`. Nothing failed - the phrase simply never matched,
    so those findings silently fell back to "parameters differ" and threw
    away a value that had been measured. A wrong key name cannot be caught by
    reading the output, only here."""
    from symbolic.findings import _PHRASES

    produced = _parameter_keys_by_pattern_type()

    wrong = {}
    for pattern_type, (_template, required) in _PHRASES.items():
        if not required or pattern_type not in produced:
            continue
        missing = set(required) - produced[pattern_type]
        if missing:
            wrong[pattern_type] = sorted(missing)

    assert not wrong, f"phrased but never produced by that detector: {wrong}"


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
        """The parameter name has to be the one the detector actually
        produces - `shift` for uniform_translation, not `offset`. Getting it
        wrong doesn't fail loudly: the phrase silently degrades to "parameters
        differ" and a measured value is thrown away."""
        analysis = _stub_analysis(
            [_pattern("uniform_translation", common={"shift": (1, 2)})],
            [_example([[1]], [[1]]), _example([[1]], [[1]])],
        )

        findings = build_task_findings(analysis)

        assert "(1, 2)" in findings.transformations[0].statement
        assert findings.transformations[0].parameters == {"shift": (1, 2)}

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
    def test_consistent_resizing_is_stated_as_a_grid_observation():
        examples = [_example([[1]], [[1]], size_change=True) for _ in range(2)]

        observations = build_task_findings(_stub_analysis([], examples)).grid_observations

        assert [f.subject for f in observations] == ["grid_resize"]
        assert "always a different size" in observations[0].statement

    @staticmethod
    def test_inconsistent_resizing_is_stated_rather_than_passed_over():
        """Neither an invariant nor a consistent change, but still worth
        saying: it tells a solver it cannot assume a fixed size relation."""
        examples = [_example([[1]], [[1]], size_change=True),
                    _example([[1]], [[1]], size_change=False)]

        findings = build_task_findings(_stub_analysis([], examples))

        assert "resized in some examples but not others" in findings.grid_observations[0].statement
        assert "grid_size" not in {f.subject for f in findings.invariants}

    @staticmethod
    def test_preserved_size_is_an_invariant_not_a_grid_observation():
        examples = [_example([[1]], [[1]]) for _ in range(2)]

        findings = build_task_findings(_stub_analysis([], examples))

        assert findings.grid_observations == ()
        assert "grid_size" in {f.subject for f in findings.invariants}

    @staticmethod
    def test_no_examples_yields_no_invariants():
        assert build_task_findings(_stub_analysis([], [])).invariants == ()


# ---------------------------------------------------------------------------
# rendering under a budget
# ---------------------------------------------------------------------------

def _findings(n_transformations=2, n_invariants=1, n_observations=0):
    return TaskFindings(
        task_id="t",
        example_count=2,
        transformations=tuple(
            _finding(f"t{i}", f"transformation {i}", (0, 1), 2) for i in range(n_transformations)
        ),
        invariants=tuple(
            _finding(f"i{i}", f"invariant {i}", (0, 1), 2) for i in range(n_invariants)
        ),
        grid_observations=tuple(
            _finding(f"g{i}", f"observation {i}", (0, 1), 2) for i in range(n_observations)
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
    def test_grid_observations_get_their_own_heading():
        """They used to be appended to the transformations and printed under
        "What changes". Several of them state that something never changes,
        so that heading made the block say the opposite of what it means."""
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("t", "objects move right", (0, 1), 2),),
            grid_observations=(_finding("background_color",
                                        "every example sits on a black background", (0, 1), 2),),
        )

        text = render_findings(findings)

        assert "What the grids show:" in text
        changes_block = text.split("What the grids show:")[0]
        assert "objects move right" in changes_block
        assert "black background" not in changes_block

    @staticmethod
    def test_an_observation_alone_still_gets_its_own_heading_not_the_changes_one():
        findings = TaskFindings(
            task_id="t", example_count=2,
            grid_observations=(_finding("grid_resize", "the output is always resized", (0, 1), 2),),
        )

        text = render_findings(findings)

        assert "What changes:" not in text
        assert "What the grids show:" in text
        assert "always resized" in text

    @staticmethod
    def test_the_three_sections_keep_their_order():
        text = render_findings(_findings(n_transformations=1, n_invariants=1, n_observations=1))

        assert (text.index("What changes:")
                < text.index("What the grids show:")
                < text.index("What stays the same:"))

    @staticmethod
    def test_a_tight_budget_drops_the_sections_from_the_tail():
        """Section order is what decides which claims survive a cut, so
        splitting the block must not have moved anything: the transformation
        outlives the observation, which outlives the invariant."""
        findings = _findings(n_transformations=1, n_invariants=1, n_observations=1)
        # Budgets that exactly fit the leading sections, taken from rendering
        # those sections alone rather than guessed as a fraction.
        first_two = len(render_findings(_findings(1, 0, 1)))
        first_only = len(render_findings(_findings(1, 0, 0)))

        assert render_findings(findings, budget=first_two, count_tokens=len) == render_findings(
            _findings(1, 0, 1))
        assert render_findings(findings, budget=first_only, count_tokens=len) == render_findings(
            _findings(1, 0, 0))

    @staticmethod
    def test_a_section_with_nothing_left_does_not_print_its_header():
        findings = _findings(n_transformations=1, n_invariants=1)
        first_only = render_findings(findings, budget=0, count_tokens=lambda s: 0 if "invariant" not in s else 99)

        assert "What stays the same:" not in first_only


# ---------------------------------------------------------------------------
# the other two views over the same findings
# ---------------------------------------------------------------------------

class TestHypothesisView:
    @staticmethod
    def test_findings_are_tiered_by_confidence():
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(
                _finding("a", "certain thing", (0, 1), 2, confidence=0.95),
                _finding("b", "likely thing", (0, 1), 2, confidence=0.6),
            ),
        )

        text = render_hypothesis(findings)

        assert text.index("certain thing") < text.index("MEDIUM CONFIDENCE")
        assert "likely thing" in text.split("MEDIUM CONFIDENCE")[1]

    @staticmethod
    def test_low_confidence_findings_are_left_out_entirely():
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("a", "barely a guess", (0,), 2, confidence=0.2),),
        )

        assert "barely a guess" not in render_hypothesis(findings)

    @staticmethod
    def test_absent_first_section_does_not_leave_a_blank_line_at_the_top():
        """Cosmetic, but it is the first thing a reader sees."""
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("a", "likely thing", (0, 1), 2, confidence=0.6),),
        )

        assert render_hypothesis(findings).startswith("MEDIUM CONFIDENCE")

    @staticmethod
    def test_nothing_found_says_so_rather_than_returning_an_empty_string():
        text = render_hypothesis(TaskFindings(task_id="t", example_count=0))

        assert "No clear transformation hypothesis" in text


class TestInsightsView:
    @staticmethod
    def test_step_is_emitted_when_its_parameter_held_across_examples():
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("color_based_deletion", "...", (0, 1), 2,
                                       confidence=0.9, color="green"),),
        )

        assert render_insights(findings) == ["Delete all objects with color: green"]

    @staticmethod
    def test_step_is_dropped_when_its_parameter_never_held():
        """Insights are meant to be executable, so an unresolved parameter
        removes the step rather than softening it into words - which is the
        one place this view is stricter than the prose one."""
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("color_based_deletion", "...", (0, 1), 2, confidence=0.9),),
        )

        assert render_insights(findings) == []

    @staticmethod
    def test_step_is_dropped_below_the_confidence_threshold():
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("color_based_deletion", "...", (0, 1), 2,
                                       confidence=0.5, color="green"),),
        )

        assert render_insights(findings) == []

    @staticmethod
    def test_causal_shift_rule_selects_the_specific_step():
        """'hor_size' contains 'size', so the more specific rule has to be
        tested first or every shift collapses to the generic one."""
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("causal_shift", "...", (0, 1), 2,
                                       confidence=0.9, rule="shift_equals_hor_size"),),
        )

        assert render_insights(findings) == ["For each object: horizontal_shift = object.hor_size"]

    @staticmethod
    def test_unknown_subject_produces_no_step():
        findings = TaskFindings(
            task_id="t", example_count=2,
            transformations=(_finding("some_new_detector", "...", (0, 1), 2, confidence=0.9),),
        )

        assert render_insights(findings) == []


# -- grid- and task-level findings ---------------------------------------------

def _bg(consistent=None, varies=False, preserved=None, per_input=(), per_output=()):
    return SimpleNamespace(
        per_example_input=per_input,
        per_example_output=per_output,
        consistent_color=consistent,
        varies_across_examples=varies,
        preserved_by_transformation=preserved,
    )


class TestGridSizeFindings:
    """"The output is a different size" was the whole grid-level story, and
    it's the weakest true thing that can be said - it names no size at all.
    Two much stronger shapes hide inside it: an output that is the same size
    regardless of the input (15.9% of real tasks), and a whole-number
    scaling (1.8%)."""

    @staticmethod
    def _observations(examples):
        return build_task_findings(_stub_analysis([], examples)).grid_observations

    def test_a_constant_output_size_is_named(self):
        examples = [
            _example([[1, 1], [1, 1]], [[2, 2, 2]], size_change=True),
            _example([[1] * 5], [[2, 2, 2]], size_change=True),
        ]

        observations = self._observations(examples)
        subjects = {f.subject for f in observations}

        assert "grid_output_size" in subjects
        finding = next(f for f in observations if f.subject == "grid_output_size")
        assert finding.parameters == {"output_rows": 1, "output_cols": 3}
        assert "1x3" in finding.statement

    def test_a_constant_output_size_is_not_claimed_when_the_inputs_match_it(self):
        """Every input already that size makes "the output is always 2x2" a
        restatement of the input dressed up as a fact about the output."""
        examples = [_example([[1, 1], [1, 1]], [[2, 2], [2, 2]], size_change=True)
                    for _ in range(2)]

        subjects = {f.subject for f in self._observations(examples)}

        assert "grid_output_size" not in subjects
        assert "grid_resize" in subjects

    def test_a_uniform_scaling_is_named(self):
        examples = [
            _example([[1]], [[2, 2], [2, 2]], size_change=True),
            _example([[1, 1]], [[2, 2, 2, 2], [2, 2, 2, 2]], size_change=True),
        ]

        finding = next(f for f in self._observations(examples) if f.subject == "grid_scale")

        assert finding.parameters == {"row_factor": 2, "col_factor": 2}

    def test_an_inconsistent_resize_still_falls_back_to_the_plain_statement(self):
        examples = [
            _example([[1]], [[2, 2]], size_change=True),
            _example([[1]], [[2, 2, 2]], size_change=True),
        ]

        size_subjects = {f.subject for f in self._observations(examples)
                         if f.subject.startswith("grid_")}

        assert size_subjects == {"grid_resize"}


class TestPaletteFindings:
    @staticmethod
    def _changes(examples):
        return build_task_findings(_stub_analysis([], examples)).grid_observations

    def test_a_colour_introduced_in_every_example_is_reported(self):
        examples = [_example([[1, 1]], [[1, 3]]) for _ in range(2)]

        finding = next(f for f in self._changes(examples) if f.subject == "palette_added")

        assert finding.parameters == {"added_colors": (3,)}
        # The digit, not "green": the grid beside this is rendered as rows of
        # digits and nothing in the prompt maps a name onto one. Measured over
        # 60 tasks, 85% of summaries carried at least one such name.
        assert "colour 3" in finding.statement
        assert "green" not in finding.statement

    def test_a_colour_dropped_in_every_example_is_reported(self):
        examples = [_example([[1, 3]], [[1, 1]]) for _ in range(2)]

        finding = next(f for f in self._changes(examples) if f.subject == "palette_removed")

        assert finding.parameters == {"removed_colors": (3,)}

    def test_colours_that_differ_between_examples_are_not_reported(self):
        """A colour appearing in one example and a different one in the next
        says nothing about the test input."""
        examples = [_example([[1, 1]], [[1, 3]]), _example([[1, 1]], [[1, 4]])]

        subjects = {f.subject for f in self._changes(examples)}

        assert "palette_added" not in subjects

    def test_an_unchanged_palette_is_left_to_the_invariant(self):
        """_invariant_findings already reports this as `palette`; saying it
        again here would put one fact in two sections."""
        examples = [_example([[1, 3]], [[3, 1]]) for _ in range(2)]

        findings = build_task_findings(_stub_analysis([], examples))
        change_subjects = {f.subject for f in findings.grid_observations}

        assert not any(s.startswith("palette_") for s in change_subjects)
        assert "palette" in {f.subject for f in findings.invariants}


class TestBackgroundFindings:
    @staticmethod
    def _observations(examples, background):
        analysis = _stub_analysis([], examples)
        analysis.background = background
        return build_task_findings(analysis).grid_observations

    def test_a_consistent_background_is_named(self):
        examples = [_example([[7]], [[7]]) for _ in range(2)]

        finding = next(f for f in self._observations(examples, _bg(consistent=7))
                       if f.subject == "background_color")

        assert finding.parameters == {"background_color": 7}
        assert "colour 7" in finding.statement
        assert "orange" not in finding.statement

    def test_disagreeing_examples_are_reported_as_disagreeing(self):
        examples = [_example([[7]], [[7]]) for _ in range(2)]

        subjects = {f.subject for f in self._observations(examples, _bg(varies=True))}

        assert "background_varies" in subjects
        assert "background_color" not in subjects

    def test_a_repainted_background_is_called_out(self):
        examples = [_example([[7]], [[1]]) for _ in range(2)]

        subjects = {f.subject for f in
                    self._observations(examples, _bg(consistent=7, preserved=False))}

        assert "background_repainted" in subjects

    def test_nothing_is_claimed_when_no_background_was_identified(self):
        examples = [_example([[1]], [[1]]) for _ in range(2)]

        subjects = {f.subject for f in self._observations(examples, _bg())}

        assert not any(s.startswith("background") for s in subjects)

    def test_an_analysis_without_a_background_attribute_is_tolerated(self):
        """analyze_subtask builds no task-level background - the findings
        builder must not require one."""
        examples = [_example([[1]], [[1]]) for _ in range(2)]

        findings = build_task_findings(_stub_analysis([], examples))

        assert not any(f.subject.startswith("background") for f in findings.grid_observations)


class TestFindingsNameColoursTheWayTheGridDoes:
    """A finding is read next to the grid it describes, and the grid is
    rendered as rows of digits (arc_grid_formatting's "concise"). A name
    like "yellow" refers to something the prompt never defines - the reader
    has to already know yellow is 4, which is a convention of the dataset's
    pictures rather than anything in the task.
    """

    #: Every name the repository maps a digit to, in any of its three copies
    #: of that mapping.
    NAMES = ("black", "blue", "red", "green", "yellow",
             "gray", "magenta", "orange", "sky", "brown")

    def test_no_finding_statement_carries_a_colour_name(self):
        import inspect
        import re

        import symbolic.findings as findings

        source = inspect.getsource(findings)
        statements = re.findall(r'statement=f?"([^"]*)"', source)

        assert statements, "no statements found - the pattern needs updating"
        for statement in statements:
            for name in self.NAMES:
                assert not re.search(rf"\b{name}\b", statement), \
                    f"{name!r} in {statement!r}"

    def test_the_module_no_longer_reaches_for_the_name_table(self):
        """The import going away is what stops a name creeping back in."""
        import symbolic.findings as findings

        assert not hasattr(findings, "COLORS_MAPPING")

    def test_several_colours_are_listed_as_digits(self):
        from symbolic.findings import _color_phrase

        assert _color_phrase([1]) == "colour 1"
        assert _color_phrase([1, 3]) == "colour 1 and colour 3"
        assert _color_phrase([1, 3, 4]) == "colour 1, colour 3 and colour 4"
