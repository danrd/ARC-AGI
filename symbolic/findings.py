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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


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

    @property
    def is_empty(self) -> bool:
        return not self.transformations and not self.invariants


def _ranked(findings: Sequence[Finding]) -> Tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: f.rank_key, reverse=True))


# ---------------------------------------------------------------------------
# Building findings out of a TaskAnalysis
# ---------------------------------------------------------------------------

# Phrasings for the pattern types the analyzer actually emits. The value is a
# format string over the agreed parameters; when a referenced parameter wasn't
# established, the generic fallback is used instead of naming a value.
_PHRASES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "uniform_translation": ("every object moves by offset {offset}", ("offset",)),
    "translation": ("objects move between input and output", ()),
    "causal_shift": ("objects are shifted according to: {rule}", ("rule",)),
    "color_mapping": ("colors are remapped consistently", ()),
    "color_based_deletion": ("objects of color {color} are removed", ("color",)),
    "shape_based_deletion": ("objects shaped {shape} are removed", ("shape",)),
    "position_based_deletion": ("objects at {position} are removed", ("position",)),
    "object_deletion": ("objects are removed from the input", ()),
    "object_addition": ("new objects appear in the output", ()),
    "aligned_addition": ("new objects are added {alignment_type} with existing ones", ("alignment_type",)),
    "shape_duplication": ("shapes from the input are duplicated", ()),
    "size_scaling": ("objects are scaled by factor {factor}", ("factor",)),
    "symmetry_change": ("the symmetry of the grid changes", ()),
}


def _statement_for(pattern_type: str, agreed: Mapping[str, Any]) -> str:
    """Render a pattern as a sentence, naming only established parameters."""
    template, required = _PHRASES.get(pattern_type, (None, ()))
    if template is not None and all(key in agreed for key in required):
        return template.format(**{key: agreed[key] for key in required})

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
    """Objects the summary parsed at its primary level, or None when that
    level isn't present (levels are caller-selected and may not include it)."""
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

    palette_kept = []
    for analysis in analyses:
        palette_kept.append(
            set(np.unique(analysis.input_grid).tolist()) == set(np.unique(analysis.output_grid).tolist())
        )
    if all(palette_kept):
        findings.append(Finding(
            subject="palette",
            statement="input and output use the same set of colors",
            evidence=everywhere,
            confidence=1.0,
        ))

    counts = []
    for analysis in analyses:
        level = getattr(analysis, "primary_level", 2)
        before = _object_count(analysis.input_summary, level)
        after = _object_count(analysis.output_summary, level)
        counts.append(None if before is None or after is None else before == after)
    if counts and all(kept is True for kept in counts):
        findings.append(Finding(
            subject="object_count",
            statement="the number of objects is unchanged",
            evidence=everywhere,
            confidence=1.0,
        ))

    return _ranked(findings)


def build_task_findings(task_analysis) -> TaskFindings:
    """Convert a TaskAnalysis into the structured form consumers read."""
    return TaskFindings(
        task_id=str(task_analysis.task_id),
        example_count=len(task_analysis.subtasks_analyses),
        transformations=_transformation_findings(task_analysis),
        invariants=_invariant_findings(task_analysis),
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
        (_CHANGES_HEADER, findings.transformations),
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
