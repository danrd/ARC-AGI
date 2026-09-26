"""Structured analysis output: the contract between the symbolic layer and
whoever consumes it (a prompt block, an agent, RL).

The structure is what's produced here, and text is rendered on top of it. An
analyzer that answers only in glued-together prose leaves a programmatic
consumer nothing to read - the structure exists inside it and is thrown away
on the way out - and every separately-worded view of the same analysis is one
more thing to drift. So this module holds the claims, and each way of showing
them is a rendering over it: `render_findings` for a prompt block,
`TaskAnalysis.get_transformation_hypothesis()` for prose.

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
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Colour names, so a finding reads "a black background" rather than "0".
#: Mirrors analyzer.colors_mapping - imported from there rather than
#: restated, so the two can't drift into naming the same number differently.


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
    #: output is always 3x3", "every example sits on a black background"),
    #: kept apart from `transformations` because they come from the grid diff
    #: rather than from a detector, and carry no detector confidence to rank
    #: against. They are not statements about what the transformation does -
    #: several of them say a thing never changes - so they are rendered under
    #: their own heading, not folded into either of the other two.
    grid_observations: Tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.transformations and not self.invariants and not self.grid_observations


def _ranked(findings: Sequence[Finding]) -> Tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: f.rank_key, reverse=True))


# ---------------------------------------------------------------------------
# Building findings out of a TaskAnalysis
# ---------------------------------------------------------------------------

# What changes, checked cell for cell.
#
# These replaced the pattern detectors (analyzer.SubtaskAnalysis's
# transformation_patterns) as the source of this section, because a
# detector's claim was a guess about one object and the section stated it as
# the task's rule: size_scaling fired whenever any object's cell count
# changed - an object that gained one cell was "scaled" - and reached 64% of
# blocks over 100 tasks; color_mapping printed "colors are remapped
# consistently" for mappings it had itself marked inconsistent; a pattern
# seen in half the examples was reported as the task's. With the summary in
# the prompt an LLM run solved fewer tasks, not more.
#
# Each check below compares the grids themselves and a finding is made only
# when it holds in every training example, so what the section says is true
# of the examples by construction. Which of those claims carries over to the
# test pair is a separate question, measured rather than assumed.

def _grid_pairs(task_analysis):
    """(input, output) of every example, or None when any grid is missing -
    a claim about every example cannot be checked on some of them."""
    pairs = []
    for analysis in task_analysis.subtasks_analyses:
        inp = getattr(analysis, "input_grid", None)
        out = getattr(analysis, "output_grid", None)
        if inp is None or out is None:
            return None
        pairs.append((np.asarray(inp), np.asarray(out)))
    return pairs or None


#: Whole-grid moves, in the order they are tried: a grid symmetric under
#: two of them is named by the first.
_GEOMETRIC = (
    ("flipped left to right", np.fliplr),
    ("flipped upside down", np.flipud),
    ("rotated half a turn", lambda grid: np.rot90(grid, 2)),
    ("rotated a quarter turn clockwise", lambda grid: np.rot90(grid, -1)),
    ("rotated a quarter turn anticlockwise", np.rot90),
    ("transposed - its rows become the output's columns", np.transpose),
)


def _geometric(pairs):
    for phrase, move in _GEOMETRIC:
        if all(move(inp).shape == out.shape and np.array_equal(move(inp), out)
               for inp, out in pairs):
            return phrase
    return None


def _colour_map(pairs):
    """{colour: colour it becomes} when one fixed replacement, applied to
    every cell, turns each input into its output - only the colours that
    change - or None."""
    mapping = {}
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        for colour in np.unique(inp).tolist():
            became = np.unique(out[inp == colour]).tolist()
            if len(became) != 1 or mapping.setdefault(colour, became[0]) != became[0]:
                return None
    changed = {colour: became for colour, became in mapping.items() if colour != became}
    return changed or None


def _contains(grid, piece):
    rows, cols = piece.shape
    if rows > grid.shape[0] or cols > grid.shape[1]:
        return False
    windows = np.lib.stride_tricks.sliding_window_view(grid, piece.shape)
    return bool((windows == piece).all(axis=(2, 3)).any())


def _block_factor(pairs, build):
    """The one (rows, cols) factor with build(input, factor) == output in
    every example, or None."""
    factors = set()
    for inp, out in pairs:
        if out.shape[0] % inp.shape[0] or out.shape[1] % inp.shape[1]:
            return None
        factor = (out.shape[0] // inp.shape[0], out.shape[1] // inp.shape[1])
        if factor == (1, 1) or not np.array_equal(build(inp, factor), out):
            return None
        factors.add(factor)
    return factors.pop() if len(factors) == 1 else None


def _upscaled(grid, factor):
    return np.kron(grid, np.ones(factor, dtype=grid.dtype))


def _tiled(grid, factor):
    return np.tile(grid, factor)


def _cell_change_findings(pairs, background, everywhere):
    """Which cells change and into what, when every example agrees.

    Stated against the background only when one was established for the
    whole task: "only background cells change" means nothing to a reader
    who was not told which colour that is.
    """
    was, became = set(), set()
    for inp, out in pairs:
        changed = inp != out
        was |= set(inp[changed].tolist())
        became |= set(out[changed].tolist())
    if not was:
        return []
    findings = []
    if background is not None and was == {background}:
        findings.append(Finding(
            subject="only_background_changes",
            statement=f"only background cells (colour {int(background)}) change; every "
                      "other cell keeps its colour and place",
            evidence=everywhere, confidence=1.0,
            parameters={"background_color": background}))
    elif len(was) == 1:
        colour = next(iter(was))
        findings.append(Finding(
            subject="only_one_colour_changes",
            statement=f"only cells of colour {int(colour)} change",
            evidence=everywhere, confidence=1.0, parameters={"changed_color": colour}))
    elif background is not None and background not in was:
        findings.append(Finding(
            subject="background_kept",
            statement=f"no background cell (colour {int(background)}) changes; only "
                      "cells that already have another colour do",
            evidence=everywhere, confidence=1.0,
            parameters={"background_color": background}))
    if len(became) == 1:
        colour = next(iter(became))
        erased = background is not None and colour == background
        findings.append(Finding(
            subject="erased_only" if erased else "changes_become_one_colour",
            statement=(f"cells are only erased: every cell that changes becomes the "
                       f"background colour {int(colour)}" if erased else
                       f"every cell that changes becomes colour {int(colour)}"),
            evidence=everywhere, confidence=1.0, parameters={"new_color": colour}))
    return findings


def _change_findings(task_analysis) -> Tuple[Finding, ...]:
    """What the transformation does, from the checks above - each only when
    it holds in every example, and the most specific that holds first: a
    whole-grid move or a colour replacement says everything the cell-level
    claims would, so those are left out beside it."""
    pairs = _grid_pairs(task_analysis)
    if pairs is None:
        return ()
    everywhere = Evidence(tuple(range(len(pairs))), len(pairs))
    background = getattr(getattr(task_analysis, "background", None), "consistent_color", None)

    move = _geometric(pairs)
    if move is not None:
        return (Finding(subject="geometric", statement=f"the output is the input {move}",
                        evidence=everywhere, confidence=1.0, parameters={"move": move}),)

    for subject, build, phrase in (
            ("upscale", _upscaled, "every input cell becomes a {0}x{1} block of its colour"),
            ("tile", _tiled, "the output is the input repeated {0}x{1} times")):
        factor = _block_factor(pairs, build)
        if factor is not None:
            return (Finding(subject=subject, statement=phrase.format(*factor),
                            evidence=everywhere, confidence=1.0,
                            parameters={"row_factor": factor[0], "col_factor": factor[1]}),)

    if all(out.size < inp.size and _contains(inp, out) for inp, out in pairs):
        return (Finding(subject="crop",
                        statement="the output is a piece of the input, copied cell for cell",
                        evidence=everywhere, confidence=1.0),)

    if any(inp.shape != out.shape for inp, out in pairs):
        return ()

    mapping = _colour_map(pairs)
    if mapping is not None:
        listed = ", ".join(f"{int(a)} becomes {int(b)}" for a, b in sorted(mapping.items()))
        return (Finding(subject="colour_replacement",
                        statement=f"every cell of a colour changes the same way wherever it "
                                  f"is: {listed}; other colours stay",
                        evidence=everywhere, confidence=1.0,
                        parameters={"mapping": dict(sorted(mapping.items()))}),)

    return tuple(_cell_change_findings(pairs, background, everywhere))


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
            # The size only: whether the content is scaled, tiled or
            # something else is _change_findings' to establish, and "the
            # input scaled by 3x3" claimed the content too.
            statement = (f"the output grid has {row_factor} times the input's rows and "
                         f"{col_factor} times its columns"
                         if (row_factor, col_factor) > (0, 0) else
                         f"the output grid has 1/{-row_factor} of the input's rows and "
                         f"1/{-col_factor} of its columns")
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
    """Colours as the digits the grid is written in, not as names.

    The prompt renders grids with arc_grid_formatting's "concise"
    representation - rows of digits - and a finding that says "yellow" next
    to them names something the prompt never defines. The reader has to
    already know that yellow is 4, which is a convention of the dataset's
    visualisations rather than anything in the task. Measured over 60
    tasks, 85% of summaries carried at least one such name, so this was
    most of the summary block referring to something not on the page.

    Names would be right if the grid were rendered as "color_text", which
    format_grid can do; nothing configures it, and the digit stays
    unambiguous either way because it is what the task data holds.
    """
    labels = [f"colour {int(c)}" for c in colors]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


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
    findings = [Finding(
        subject="background_color",
        # The digit, for the reason in _color_phrase: the grid beside this
        # is written in digits and nothing in the prompt maps a name onto one.
        statement=f"every example sits on a background of colour {int(color)}",
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
        transformations=_change_findings(task_analysis),
        invariants=_invariant_findings(task_analysis),
        # Ranked like the other two groups, so a budget cut drops the least
        # useful from the tail here as well: these all hold everywhere and
        # carry confidence 1.0, which leaves the ones naming a measured value
        # ("the output is always 3x3") ahead of the bare ones.
        grid_observations=_ranked(_grid_observation_findings(task_analysis)
                                  + _palette_findings(task_analysis)
                                  + _background_findings(task_analysis)),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CHANGES_HEADER = "What changes:"
_OBSERVATIONS_HEADER = "What the grids show:"
_INVARIANTS_HEADER = "What stays the same:"


def render_findings(findings: TaskFindings, budget: Optional[int] = None,
                     count_tokens: Optional[Callable[[str], int]] = None) -> Optional[str]:
    """Render findings as text that fits `budget` tokens.

    Fitting happens by dropping whole findings from the tail of the ranking -
    the least useful ones - never by truncating the text partway, which would
    leave a half-written claim looking like a complete one. Returns None when
    not even the first finding fits, so the caller can omit the block instead
    of emitting a lone header.

    Three sections, because grid observations are neither of the other two.
    Several of them state that something does not vary - "every example sits
    on a black background" - so printing them beside the transformations puts
    a claim under the heading for its opposite, and they are not preservation
    claims either: they describe the grids rather than what the transformation
    left alone. Not a rare corner: over ARC-AGI-2's 1000 training tasks they
    are 1654 of the 5103 lines outside the invariants (32.4%), 750 of those
    the background statement, and 93.7% of tasks carry at least one.

    Sections are rendered in order and the budget is spent in that order, so
    this order is also the drop order: the rule first, then what the grids
    show, then what holds.
    """
    if findings.is_empty:
        return None

    measure = count_tokens or (lambda text: len(text))
    unlimited = budget is None

    sections = (
        (_CHANGES_HEADER, findings.transformations),
        (_OBSERVATIONS_HEADER, findings.grid_observations),
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
