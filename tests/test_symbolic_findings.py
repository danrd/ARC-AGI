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


def _stub_analysis(patterns, examples, task_id="stub"):
    return SimpleNamespace(
        task_id=task_id,
        consistent_patterns=patterns,
        subtasks_analyses=examples,
    )


def _changes(examples, background=None):
    analysis = _stub_analysis([], examples)
    analysis.background = _bg(consistent=background)
    return build_task_findings(analysis).transformations


def _pair(inp, out):
    return _example(inp, out, size_change=np.array(inp).shape != np.array(out).shape)


class TestWhatChanges:
    """The changes section is made of checks on the grids, each stated only
    when it holds in every example. It used to come from the pattern
    detectors, which called an object that gained a cell "scaled" (64% of
    blocks) and printed "colors are remapped consistently" for mappings
    they had marked inconsistent."""

    def test_the_detectors_are_not_read(self):
        analysis = _stub_analysis([SimpleNamespace(pattern_type="size_scaling", confidence=1.0,
                                                   parameters={})],
                                  [_pair([[1, 2]], [[1, 3]]), _pair([[4, 2]], [[4, 5]])])
        analysis.background = None
        assert all(f.subject != "size_scaling"
                   for f in build_task_findings(analysis).transformations)

    def test_a_whole_grid_move_is_named(self):
        found = _changes([_pair([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
                          _pair([[5, 6, 7]], [[5, 6, 7]]), _pair([[1], [2]], [[2], [1]])])
        assert [f.statement for f in found] == ["the output is the input flipped upside down"]

    def test_one_fixed_colour_replacement(self):
        """d511f180: 5 and 8 swap wherever they are."""
        found = _changes([_pair([[5, 8, 1]], [[8, 5, 1]]), _pair([[8, 8, 2]], [[5, 5, 2]])])
        assert found[0].subject == "colour_replacement"
        assert found[0].parameters == {"mapping": {5: 8, 8: 5}}
        assert "5 becomes 8, 8 becomes 5" in found[0].statement

    def test_a_colour_that_becomes_two_colours_is_no_replacement(self):
        found = _changes([_pair([[5, 5]], [[8, 3]]), _pair([[5]], [[8]])])
        assert all(f.subject != "colour_replacement" for f in found)

    def test_a_replacement_must_agree_between_examples(self):
        found = _changes([_pair([[5]], [[8]]), _pair([[5]], [[3]])])
        assert all(f.subject != "colour_replacement" for f in found)

    def test_only_the_background_changes_and_into_one_colour(self):
        """3aa6fb7a: the notches of each L are filled with 1."""
        found = _changes([_pair([[8, 0], [8, 8]], [[8, 1], [8, 8]]),
                          _pair([[0, 8], [0, 0]], [[1, 8], [0, 0]])], background=0)
        assert [f.subject for f in found] == ["only_background_changes",
                                              "changes_become_one_colour"]
        assert "colour 1" in found[1].statement

    def test_background_claims_need_an_established_background(self):
        found = _changes([_pair([[8, 0], [8, 8]], [[8, 1], [8, 8]]),
                          _pair([[0, 8], [0, 0]], [[1, 8], [0, 0]])], background=None)
        assert "only_background_changes" not in {f.subject for f in found}

    def test_cells_that_only_disappear(self):
        found = _changes([_pair([[3, 4, 0]], [[0, 4, 0]]), _pair([[4, 3]], [[4, 0]]),
                          _pair([[4, 4, 3]], [[4, 4, 3]])], background=0)
        assert "erased_only" in {f.subject for f in found}

    def test_the_background_is_never_painted(self):
        found = _changes([_pair([[3, 4, 0]], [[4, 3, 0]]), _pair([[4, 3, 0]], [[3, 3, 0]])],
                         background=0)
        assert "background_kept" in {f.subject for f in found}

    def test_every_cell_becomes_a_block(self):
        found = _changes([_pair([[1, 2]], [[1, 1, 2, 2], [1, 1, 2, 2]]),
                          _pair([[3]], [[3, 3], [3, 3]])])
        assert [(f.subject, f.parameters) for f in found] == [
            ("upscale", {"row_factor": 2, "col_factor": 2})]

    def test_a_block_size_that_differs_between_examples_is_not_named(self):
        found = _changes([_pair([[1]], [[1, 1], [1, 1]]),
                          _pair([[2]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]])])
        assert all(f.subject != "upscale" for f in found)

    def test_the_input_repeated(self):
        found = _changes([_pair([[1, 2]], [[1, 2, 1, 2]]), _pair([[3, 4]], [[3, 4, 3, 4]])])
        assert [f.subject for f in found] == ["tile"]

    def test_a_piece_of_the_input(self):
        found = _changes([_pair([[1, 2, 3], [4, 5, 6]], [[5, 6]]),
                          _pair([[7, 8], [9, 1]], [[7], [9]])])
        assert [f.subject for f in found] == ["crop"]

    def test_a_smaller_output_that_is_not_in_the_input_is_no_piece(self):
        found = _changes([_pair([[1, 2, 3], [4, 5, 6]], [[6, 5]]),
                          _pair([[7, 8], [9, 1]], [[7], [9]])])
        assert all(f.subject != "crop" for f in found)

    def test_one_example_that_disagrees_withdraws_the_claim(self):
        found = _changes([_pair([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
                          _pair([[1, 2], [3, 4]], [[2, 1], [4, 3]])])
        assert all(f.subject != "geometric" for f in found)

    def test_every_claim_names_every_example(self):
        found = _changes([_pair([[5, 8]], [[8, 5]]), _pair([[8]], [[5]])])
        assert found and all(f.evidence.holds_everywhere for f in found)


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
        assert finding.statement == ("the output grid has 2 times the input's rows and "
                                     "2 times its columns"), \
            "the size only - whether the content is scaled is not this finding's to say"

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
