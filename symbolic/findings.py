"""Structured analysis output: the contract between the symbolic layer and
whoever consumes it (a prompt block, an agent, RL).

The analyzer's own `get_task_summary()` / `get_transformation_hypothesis()`
return glued-together prose, which leaves a programmatic consumer nothing to
read: the structure exists inside the analyzer and is thrown away on the way
out. Here the structure is what's produced, and text is rendered on top of it,
so every consumer reads the same thing.

Two rules the rest of this module exists to enforce:

- A parameter that wasn't established has no entry at all. `TaskAnalysis`
  already distinguishes "every example agreed on this value" from "this
  varied" (see `_agreed_parameters`); nothing here may reintroduce a default,
  because a reader cannot tell a filled-in default from a measurement.
- Every claim carries the examples it came from, so it can be checked or
  refuted rather than taken on faith.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Colour names, so a finding reads "a black background" rather than "0".
#: Mirrors analyzer.colors_mapping - imported from there rather than
#: restated, so the two can't drift into naming the same number differently.
from symbolic.color_names import COLORS_MAPPING


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """Which training examples back a claim, out of how many there are."""
    example_indices: Tuple[int, ...]
    example_count: int

    @property
    def holds_everywhere(self) -> bool:
        return len(self.example_indices) == self.example_count and self.example_count > 0

    def render(self) -> str:
        if self.holds_everywhere:
            return f"all {self.example_count} examples"
        shown = ", ".join(str(i + 1) for i in self.example_indices)
        return f"examples {shown} of {self.example_count}"


@dataclass(frozen=True)
class Finding:
    """One claim about the task, with its evidence and measured parameters.

    `parameters` holds only values every supporting example agreed on -
    an absent key means "not established", never "defaulted to something".
    """
    subject: str
    statement: str
    evidence: Evidence
    confidence: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_parameters(self) -> bool:
        return bool(self.parameters)

    @property
    def rank_key(self) -> Tuple[bool, bool, float]:
        """Usefulness, not confidence alone: a claim true in every example
        and carrying concrete parameters outranks a vaguer one that happens
        to have been assigned a higher constant in the detector."""
        return (self.evidence.holds_everywhere, self.has_parameters, self.confidence)

    def render(self) -> str:
        return f"{self.statement} [{self.evidence.render()}]"


@dataclass(frozen=True)
class TaskFindings:
    """Everything the symbolic layer is prepared to claim about one task."""
    task_id: str
    example_count: int
    transformations: Tuple[Finding, ...] = field(default_factory=tuple)
    invariants: Tuple[Finding, ...] = field(default_factory=tuple)
    #: Grid-level claims that are consistent but aren't preservation ("the
    #: output is always a different size"), kept apart from `transformations`
    #: because they come from the grid diff rather than from a detector, and
    #: carry no detector confidence to rank against.
    grid_observations: Tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def changes(self) -> Tuple[Finding, ...]:
        """Everything that describes the transformation rather than what it
        leaves alone."""
        return self.transformations + self.grid_observations

    @property
    def is_empty(self) -> bool:
        return not self.transformations and not self.invariants and not self.grid_observations


def _ranked(findings: Sequence[Finding]) -> Tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: f.rank_key, reverse=True))


# ---------------------------------------------------------------------------
# Building findings out of a TaskAnalysis
# ---------------------------------------------------------------------------

# Phrasings for the pattern types the analyzer actually emits. The value is a
# format string over the agreed parameters; when a referenced parameter wasn't
# established, the generic fallback is used instead of naming a value.
#
# The parameter names here are the ones the detectors actually put in
# `parameters` (`shift`, not `offset`; `scale_factor`, not `factor`) - naming
# a key that is never produced doesn't fail loudly, it just quietly degrades
# every one of those findings to the "parameters differ" fallback, throwing
# away a value that was measured.
_PHRASES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "uniform_translation": ("every object moves by {shift}", ("shift",)),
    "translation": ("objects move between input and output", ()),
    "causal_shift": ("objects are shifted according to: {rule}", ("rule",)),
    "color_mapping": ("colors are remapped consistently", ()),
    "color_based_deletion": ("objects of color {color} are removed", ("color",)),
    "shape_based_deletion": ("all {shape} objects are removed", ("shape",)),
    "position_based_deletion": ("objects at {positions} are removed", ("positions",)),
    "object_deletion": ("objects are removed from the input", ()),
    "object_addition": ("new objects appear in the output", ()),
    "aligned_addition": ("new objects are added {alignment_type} with existing ones", ("alignment_type",)),
    "shape_duplication": ("shapes from the input are duplicated", ()),
    "size_scaling": ("objects are scaled by factor {scale_factor:.2f}", ("scale_factor",)),
    "symmetry_change": ("the symmetry of the grid changes", ()),
}


def _statement_for(pattern_type: str, agreed: Mapping[str, Any]) -> str:
    """Render a pattern as a sentence, naming only established parameters."""
    template, required = _PHRASES.get(pattern_type, (None, ()))
    if template is not None and all(key in agreed for key in required):
        try:
            return template.format(**{key: agreed[key] for key in required})
        except (ValueError, TypeError):
            # A parameter of an unexpected type for its format spec (a scale
            # factor that isn't a number, say) - fall through rather than
            # crash the whole summary over one malformed value.
            pass

    readable = pattern_type.replace("_", " ")
    if template is not None:
        # The pattern held, but the parameter it hinges on differed between
        # examples - say that, rather than picking one example's value.
        return f"{readable} occurs, but its parameters differ between examples"
    return readable


def _transformation_findings(task_analysis) -> Tuple[Finding, ...]:
    example_count = len(task_analysis.subtasks_analyses)
    findings = []
    for pattern in task_analysis.consistent_patterns:
        params = pattern.parameters or {}
        agreed = params.get("common_values") or {}
        indices = tuple(params.get("example_indices", ()))
        findings.append(Finding(
            subject=pattern.pattern_type,
            statement=_statement_for(pattern.pattern_type, agreed),
            evidence=Evidence(example_indices=indices, example_count=example_count),
            confidence=float(pattern.confidence),
            parameters=dict(agreed),
        ))
    return _ranked(findings)


def _object_count(grid_summary, level: int) -> Optional[int]:
    """Objects the summary parsed at its primary level, or None when there is
    nothing to count from - the summary is absent, or that level wasn't
    parsed (levels are caller-selected and may not include it)."""
    repr_levels = getattr(grid_summary, "repr_levels", None) or {}
    level_summary = repr_levels.get(level)
    if level_summary is None or getattr(level_summary, "objects", None) is None:
        return None
    return len(level_summary.objects)


def _invariant_findings(task_analysis) -> Tuple[Finding, ...]:
    """What the transformation leaves alone.

    Often more useful than what it changes, and cheap: every ingredient is
    already computed. Only properties preserved in *every* example are
    reported - a size that holds in two examples out of three isn't an
    invariant, it's a coincidence worth staying quiet about.
    """
    analyses = task_analysis.subtasks_analyses
    example_count = len(analyses)
    if not example_count:
        return ()

    everywhere = Evidence(tuple(range(example_count)), example_count)
    findings = []

    if all(not a.grid_diff.has_size_change for a in analyses):
        findings.append(Finding(
            subject="grid_size",
            statement="the output keeps the input's grid size",
            evidence=everywhere,
            confidence=1.0,
        ))

    # Each invariant is claimed only when every ingredient it needs is
    # present in every example. A missing ingredient means "cannot tell",
    # which is not the same as "holds" - so it withdraws the claim rather
    # than being skipped over on the way to asserting one.
    palette_kept = []
    for analysis in analyses:
        input_grid = getattr(analysis, "input_grid", None)
        output_grid = getattr(analysis, "output_grid", None)
        if input_grid is None or output_grid is None:
            palette_kept.append(None)
            continue
        palette_kept.append(
            set(np.unique(input_grid).tolist()) == set(np.unique(output_grid).tolist())
        )
    if palette_kept and all(kept is True for kept in palette_kept):
        findings.append(Finding(
            subject="palette",
            statement="input and output use the same set of colors",
            evidence=everywhere,
            confidence=1.0,
        ))

    counts = []
    for analysis in analyses:
        level = getattr(analysis, "primary_level", 2)
        before = _object_count(getattr(analysis, "input_summary", None), level)
        after = _object_count(getattr(analysis, "output_summary", None), level)
        counts.append(None if before is None or after is None else before == after)
    if counts and all(kept is True for kept in counts):
        findings.append(Finding(
            subject="object_count",
            statement="the number of objects is unchanged",
            evidence=everywhere,
            confidence=1.0,
        ))

    return _ranked(findings)


def _grid_observation_findings(task_analysis) -> Tuple[Finding, ...]:
    """Grid-level claims about size that aren't preservation.

    "Always the same size" is preservation and belongs with the invariants;
    the other two cases live here. The inconsistent case is worth stating
    outright rather than passing over in silence: "sometimes resized,
    sometimes not" tells a solver it cannot assume a fixed size relation,
    which is a different and more useful thing to know than nothing at all.
    """
    analyses = task_analysis.subtasks_analyses
    example_count = len(analyses)
    if not example_count:
        return ()

    resized = [a.grid_diff.has_size_change for a in analyses]
    everywhere = Evidence(tuple(range(example_count)), example_count)

    if not any(resized):
        return ()  # always preserved - reported as an invariant instead

    if all(resized):
        # "Always a different size" is the weakest thing that can be said
        # here, and it was all this reported. Two much stronger shapes hide
        # inside it: an output that is the same size every time whatever
        # the input was (15.9% of tasks), and a whole-number scaling of the
        # input (1.8%). Both name the output's size outright; the fallback
        # only says it won't match.
        shapes = [(getattr(a, "input_grid", None), getattr(a, "output_grid", None))
                  for a in analyses]
        if any(i is None or o is None for i, o in shapes):
            return _plain_resize_finding(everywhere)
        output_shapes = {o.shape for _, o in shapes}
        input_shapes = {i.shape for i, _ in shapes}
        # "always NxM" is only worth saying when the inputs weren't all that
        # size too - otherwise it restates the input's size back at the
        # reader and dresses it up as a fact about the output.
        if len(output_shapes) == 1 and len(input_shapes) > 1:
            rows, cols = output_shapes.pop()
            return (Finding(
                subject="grid_output_size",
                statement=f"the output grid is always {rows}x{cols}, whatever size the input is",
                evidence=everywhere,
                confidence=1.0,
                parameters={"output_rows": rows, "output_cols": cols},
            ),)

        scale = _uniform_scale(analyses)
        if scale is not None:
            row_factor, col_factor = scale
            statement = (f"the output grid is the input scaled by {row_factor}x{col_factor}"
                         if (row_factor, col_factor) > (0, 0) else
                         f"the output grid is the input reduced by {-row_factor}x{-col_factor}")
            return (Finding(
                subject="grid_scale",
                statement=statement,
                evidence=everywhere,
                confidence=1.0,
                parameters={"row_factor": row_factor, "col_factor": col_factor},
            ),)

        statement = "the output grid is always a different size from the input"
    else:
        # Worth stating outright rather than passing over in silence:
        # "sometimes resized, sometimes not" tells a solver it cannot assume
        # a fixed size relation, which is more useful than nothing at all.
        statement = "the output grid is resized in some examples but not others"

    return (Finding(
        subject="grid_resize",
        statement=statement,
        evidence=everywhere,
        confidence=1.0,
    ),)


def _plain_resize_finding(evidence: Evidence) -> Tuple[Finding, ...]:
    """The weakest true statement, for when the grids needed to say
    anything sharper aren't there."""
    return (Finding(
        subject="grid_resize",
        statement="the output grid is always a different size from the input",
        evidence=evidence,
        confidence=1.0,
    ),)


def _uniform_scale(analyses) -> Optional[Tuple[int, int]]:
    """The whole-number factor relating every input to its output, or None.

    Positive factors mean the output is larger, negative that it is smaller;
    a single sign and a single pair of factors have to hold across every
    example, or nothing is claimed.
    """
    factors = set()
    for analysis in analyses:
        input_grid = getattr(analysis, "input_grid", None)
        output_grid = getattr(analysis, "output_grid", None)
        if input_grid is None or output_grid is None:
            return None
        in_rows, in_cols = input_grid.shape
        out_rows, out_cols = output_grid.shape
        if not (in_rows and in_cols and out_rows and out_cols):
            return None
        if out_rows % in_rows == 0 and out_cols % in_cols == 0:
            factors.add((out_rows // in_rows, out_cols // in_cols))
        elif in_rows % out_rows == 0 and in_cols % out_cols == 0:
            factors.add((-(in_rows // out_rows), -(in_cols // out_cols)))
        else:
            return None
    if len(factors) != 1:
        return None
    factor = factors.pop()
    return factor if factor != (1, 1) else None


def _palette_findings(task_analysis) -> Tuple[Finding, ...]:
    """What the transformation does to the set of colours in play.

    Measured over ARC-AGI-2's training set, 45.8% of tasks never introduce
    or drop a colour, 25.7% introduce the same one in every example, and
    19.0% drop the same one - all facts about the answer's palette that
    nothing here reported. Only agreement across every example counts: a
    colour appearing in one example and not the next says nothing about
    the test input.
    """
    analyses = task_analysis.subtasks_analyses
    example_count = len(analyses)
    if not example_count:
        return ()

    added, removed = [], []
    for analysis in analyses:
        # Same rule the invariants use: an example missing a grid means
        # "cannot tell", and one of those withdraws the claim entirely
        # rather than being skipped on the way to asserting it.
        input_grid = getattr(analysis, "input_grid", None)
        output_grid = getattr(analysis, "output_grid", None)
        if input_grid is None or output_grid is None:
            return ()
        in_colors = set(np.unique(input_grid).tolist())
        out_colors = set(np.unique(output_grid).tolist())
        added.append(out_colors - in_colors)
        removed.append(in_colors - out_colors)

    everywhere = Evidence(tuple(range(example_count)), example_count)
    findings = []

    if not any(added) and not any(removed):
        # _invariant_findings already reports this as the `palette`
        # invariant ("input and output use the same set of colors");
        # repeating it here would put the same fact in two sections.
        return ()

    if any(added) and all(colors == added[0] for colors in added):
        shared = sorted(added[0])
        findings.append(Finding(
            subject="palette_added",
            statement=f"every output introduces {_color_phrase(shared)}, absent from its input",
            evidence=everywhere,
            confidence=1.0,
            parameters={"added_colors": tuple(shared)},
        ))

    if any(removed) and all(colors == removed[0] for colors in removed):
        shared = sorted(removed[0])
        findings.append(Finding(
            subject="palette_removed",
            statement=f"every output drops {_color_phrase(shared)}, present in its input",
            evidence=everywhere,
            confidence=1.0,
            parameters={"removed_colors": tuple(shared)},
        ))

    return tuple(findings)


def _color_phrase(colors: Sequence[int]) -> str:
    names = [f"{COLORS_MAPPING.get(c, c)}" for c in colors]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _background_findings(task_analysis) -> Tuple[Finding, ...]:
    """What the examples showed about the background colour.

    Reports only what BackgroundSummary established, and reports
    disagreement as disagreement: a task whose examples sit on different
    backgrounds is a fact a solver needs, not noise to average away.
    Nothing is said when no example identified one - see
    symbolic.utils.infer_background for why that happens.
    """
    background = getattr(task_analysis, "background", None)
    if background is None:
        return ()

    example_count = len(task_analysis.subtasks_analyses)
    everywhere = Evidence(tuple(range(example_count)), example_count)

    if background.varies_across_examples:
        return (Finding(
            subject="background_varies",
            statement="the background colour is not the same in every example",
            evidence=everywhere,
            confidence=1.0,
        ),)

    if background.consistent_color is None:
        return ()

    color = background.consistent_color
    name = COLORS_MAPPING.get(color, color)
    findings = [Finding(
        subject="background_color",
        statement=f"every example sits on a {name} background",
        evidence=everywhere,
        confidence=1.0,
        parameters={"background_color": color},
    )]

    if background.preserved_by_transformation is False:
        findings.append(Finding(
            subject="background_repainted",
            statement="the transformation changes the background colour itself",
            evidence=everywhere,
            confidence=1.0,
        ))

    return tuple(findings)


def build_task_findings(task_analysis) -> TaskFindings:
    """Convert a TaskAnalysis into the structured form consumers read."""
    return TaskFindings(
        task_id=str(task_analysis.task_id),
        example_count=len(task_analysis.subtasks_analyses),
        transformations=_transformation_findings(task_analysis),
        invariants=_invariant_findings(task_analysis),
        grid_observations=(_grid_observation_findings(task_analysis)
                           + _palette_findings(task_analysis)
                           + _background_findings(task_analysis)),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CHANGES_HEADER = "What changes:"
_INVARIANTS_HEADER = "What stays the same:"


def render_findings(findings: TaskFindings, budget: Optional[int] = None,
                     count_tokens: Optional[Callable[[str], int]] = None) -> Optional[str]:
    """Render findings as text that fits `budget` tokens.

    Fitting happens by dropping whole findings from the tail of the ranking -
    the least useful ones - never by truncating the text partway, which would
    leave a half-written claim looking like a complete one. Returns None when
    not even the first finding fits, so the caller can omit the block instead
    of emitting a lone header.
    """
    if findings.is_empty:
        return None

    measure = count_tokens or (lambda text: len(text))
    unlimited = budget is None

    sections = (
        (_CHANGES_HEADER, findings.changes),
        (_INVARIANTS_HEADER, findings.invariants),
    )

    rendered_sections = []
    spent = 0
    for header, items in sections:
        if not items:
            continue
        header_cost = measure(header + "\n")
        lines = []
        for finding in items:
            line = f"  - {finding.render()}\n"
            cost = measure(line)
            pending_header = header_cost if not lines else 0
            if not unlimited and spent + pending_header + cost > budget:
                break
            spent += pending_header + cost
            lines.append(line)
        if lines:
            rendered_sections.append(header + "\n" + "".join(lines))

    if not rendered_sections:
        return None
    return "\n".join(rendered_sections)


# ---------------------------------------------------------------------------
# Alternative views over the same findings
#
# These exist so there is exactly one place a claim is worded. They are
# different *presentations* of the same structure - a tiered prose read for a
# human, an imperative read for something meant to execute the rule - not
# second opinions about what the analysis found.
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
#: Insights are meant to be executable, so they demand more than prose does.
INSIGHT_CONFIDENCE = 0.7

_NO_HYPOTHESIS = ("No clear transformation hypothesis could be established. "
                   "The transformation may be highly variable or complex.")


def render_hypothesis(findings: TaskFindings) -> str:
    """Tiered prose read of the findings, for a human or an LLM."""
    high = [f for f in findings.transformations if f.confidence >= HIGH_CONFIDENCE]
    medium = [f for f in findings.transformations
              if MEDIUM_CONFIDENCE <= f.confidence < HIGH_CONFIDENCE]
    grid = list(findings.grid_observations)
    grid += [f for f in findings.invariants if f.subject == "grid_size"]

    def block(header, items, with_evidence=True):
        lines = [header]
        for finding in items:
            suffix = f" [{finding.evidence.render()}]" if with_evidence else ""
            lines.append(f"  • {finding.statement}{suffix}")
        return "\n".join(lines)

    # Assembled as whole blocks joined by a blank line, rather than each
    # header carrying a leading newline of its own - otherwise a summary
    # whose first block happens to be absent opens with a stray blank line.
    blocks = []
    if high:
        blocks.append(block("HIGH CONFIDENCE RULES:", high))
    if medium:
        blocks.append(block("MEDIUM CONFIDENCE OBSERVATIONS:", medium))
    if grid:
        blocks.append(block("GRID OBSERVATIONS:", grid, with_evidence=False))

    return "\n\n".join(blocks) if blocks else _NO_HYPOTHESIS


def _insight_causal_shift(params: Mapping[str, Any]) -> Optional[str]:
    rule = params.get("rule")
    if not isinstance(rule, str):
        return None
    # Order matters: "hor_size" and "vert_size" both contain "size", so the
    # specific tests have to come first.
    if "inner_holes" in rule:
        return "For each object: shift_amount = count(inner_holes)"
    if "hor_size" in rule:
        return "For each object: horizontal_shift = object.hor_size"
    if "vert_size" in rule:
        return "For each object: vertical_shift = object.vert_size"
    if "size" in rule:
        return "For each object: shift_amount = object.size"
    return None


def _insight_aligned_addition(params: Mapping[str, Any]) -> Optional[str]:
    alignment = params.get("alignment_type")
    if not isinstance(alignment, str):
        return None
    if "x_aligned" in alignment:
        return "Create new objects x-aligned with existing objects"
    if "y_aligned" in alignment:
        return "Create new objects y-aligned with existing objects"
    return None


def _insight_size_scaling(params: Mapping[str, Any]) -> Optional[str]:
    factor = params.get("scale_factor")
    if not isinstance(factor, (int, float)):
        return None
    return f"Scale all objects by factor: {factor:.2f}"


def _keyed_insight(template: str, key: str):
    def build(params: Mapping[str, Any]) -> Optional[str]:
        if key not in params:
            return None
        return template.format(**{key: params[key]})
    return build


#: subject -> builder producing an executable step, or None when the
#: parameters it needs weren't established across the examples.
_INSIGHT_BUILDERS: Dict[str, Callable[[Mapping[str, Any]], Optional[str]]] = {
    "causal_shift": _insight_causal_shift,
    "aligned_addition": _insight_aligned_addition,
    "size_scaling": _insight_size_scaling,
    "shape_duplication": lambda params: "Duplicate shapes from input (possibly with transformations)",
    "color_mapping": lambda params: "Apply color mapping transformation to all objects",
    "color_based_deletion": _keyed_insight("Delete all objects with color: {color}", "color"),
    "shape_based_deletion": _keyed_insight("Delete all objects with shape: {shape}", "shape"),
    "uniform_translation": _keyed_insight("Translate all objects by offset: {shift}", "shift"),
}


def render_insights(findings: TaskFindings) -> List[str]:
    """Imperative read: transformation steps that could actually be executed.

    Stricter than the prose above, and deliberately so - a step whose
    parameter never held across the examples cannot be executed, so it is
    dropped rather than softened into words. The caller gets fewer steps, all
    of them backed by a value every example agreed on.
    """
    insights = []
    for finding in findings.transformations:
        if finding.confidence < INSIGHT_CONFIDENCE:
            continue
        builder = _INSIGHT_BUILDERS.get(finding.subject)
        if builder is None:
            continue
        insight = builder(finding.parameters)
        if insight:
            insights.append(insight)
    return insights
