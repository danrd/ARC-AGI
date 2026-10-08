"""What every example of a task agrees on, held against an answer.

An answer from a model cannot be run through a rule the way a solver's rule
can: there is no rule. What can be done without a second model is to ask what
the training pairs *all* do - the output has the input's shape, or one fixed
shape, or the input's scaled up; it brings in no colour the examples never
bring in; it is symmetric if every example output is; it leaves about as many
cells alone as they do - and to ask whether the answer does the same. Each of
those is a necessary condition, not a proof: an answer that meets them all may
still be wrong, and `check` says so by never returning "correct", only the list
of what the answer breaks.

Two strengths, because the conditions are learnt from two to five pairs and a
coincidence in three pairs is not a law:

    strong   the shape of the answer, the colours it introduces, a symmetry
             every example output has. An answer that breaks one is wrong
             with high probability, and `refuses` says so
    weak     how much of the grid is left unchanged, that a colour keeps its
             count. Worth telling the model, not worth refusing for

A violation carries a sentence for a person or a model to read. That is the
other use of this module: it is where the feedback of a refinement loop comes
from, so a second attempt is told *what* was off about the first.

    invariants = learn(pairs)                       # [(input, output), ...]
    violations = check(invariants, test_input, answer)
    if refuses(violations): ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

STRONG, WEAK = "strong", "weak"

#: How far the share of unchanged cells may stray from the examples' range before a weak violation.
UNCHANGED_MARGIN = 0.2

_SYMMETRIES = {
    "left-right mirror": lambda g: g[:, ::-1],
    "top-bottom mirror": lambda g: g[::-1, :],
    "half turn": lambda g: g[::-1, ::-1],
    "transpose": lambda g: g.T,
}


@dataclass(frozen=True)
class Violation:
    name: str
    severity: str
    message: str


@dataclass(frozen=True)
class Invariants:
    """What the pairs agree on. A field is None or empty where they did not."""
    shape_rule: Optional[Tuple[str, Tuple[int, ...]]] = None   # ("same", ()), ("fixed", (r, c)), ("scale", (kr, kc)), ("shrink", (kr, kc))
    introduced: frozenset = frozenset()                        # colours some output has that its input did not
    symmetries: Tuple[str, ...] = ()
    unchanged_range: Optional[Tuple[float, float]] = None      # of cells left as they were, over same-shape pairs
    kept_counts: Tuple[int, ...] = ()                          # colours whose count no example changes
    pairs: int = 0


def _shape_rule(pairs):
    if all(o.shape == i.shape for i, o in pairs):
        return ("same", ())
    out_shapes = {o.shape for _, o in pairs}
    if len(out_shapes) == 1:
        return ("fixed", tuple(next(iter(out_shapes))))
    ratios = {(o.shape[0] / i.shape[0], o.shape[1] / i.shape[1]) for i, o in pairs}
    if len(ratios) == 1:
        rr, rc = next(iter(ratios))
        if rr >= 1 and rc >= 1 and rr == int(rr) and rc == int(rc):
            return ("scale", (int(rr), int(rc)))
        inv = (1 / rr, 1 / rc)
        if inv[0] == int(inv[0]) and inv[1] == int(inv[1]):
            return ("shrink", (int(inv[0]), int(inv[1])))
    return None


def _expected_shape(rule, test_shape):
    kind, arg = rule
    if kind == "same":
        return tuple(test_shape)
    if kind == "fixed":
        return tuple(arg)
    if kind == "scale":
        return (test_shape[0] * arg[0], test_shape[1] * arg[1])
    return (test_shape[0] // arg[0], test_shape[1] // arg[1])


def _colours(grid) -> frozenset:
    return frozenset(int(v) for v in np.unique(grid))


def _symmetric(grid, name) -> bool:
    mirrored = _SYMMETRIES[name](grid)
    return mirrored.shape == grid.shape and bool((mirrored == grid).all())


def learn(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Invariants:
    pairs = [(np.asarray(i), np.asarray(o)) for i, o in pairs]
    if not pairs:
        return Invariants()
    introduced = frozenset().union(*[_colours(o) - _colours(i) for i, o in pairs])

    symmetries = ()
    if len(pairs) >= 2:
        symmetries = tuple(name for name in _SYMMETRIES
                           if all(_symmetric(o, name) for _, o in pairs)
                           and not all(len(_colours(o)) == 1 for _, o in pairs)
                           and not all(_symmetric(i, name) for i, _ in pairs))

    unchanged = None
    kept: Tuple[int, ...] = ()
    if all(o.shape == i.shape for i, o in pairs):
        shares = [float((i == o).mean()) for i, o in pairs]
        unchanged = (min(shares), max(shares))
        everywhere = set(range(10))
        for i, o in pairs:
            everywhere &= {c for c in range(10) if int((i == c).sum()) == int((o == c).sum())
                           and (int((i == c).sum()) > 0)}
        kept = tuple(sorted(everywhere))
    return Invariants(shape_rule=_shape_rule(pairs), introduced=introduced, symmetries=symmetries,
                      unchanged_range=unchanged, kept_counts=kept, pairs=len(pairs))


def check(invariants: Invariants, test_input, answer) -> List[Violation]:
    """What `answer` breaks, among the things the examples agree on. Empty: it breaks none of them."""
    test_input, answer = np.asarray(test_input), np.asarray(answer)
    found: List[Violation] = []

    if invariants.shape_rule is not None:
        expected = _expected_shape(invariants.shape_rule, test_input.shape)
        if tuple(answer.shape) != expected:
            found.append(Violation("shape", STRONG, (
                f"The answer is {answer.shape[0]}x{answer.shape[1]}; the examples give an output of "
                f"{expected[0]}x{expected[1]} for an input of {test_input.shape[0]}x{test_input.shape[1]}, "
                f"{_describe_shape(invariants.shape_rule)}.")))

    new = _colours(answer) - _colours(test_input) - invariants.introduced
    if new and invariants.pairs:
        allowed = sorted(invariants.introduced)
        found.append(Violation("colours", STRONG, (
            f"The answer uses colour(s) {sorted(new)} that are not in the input, and no example brings in a colour "
            + (f"other than {allowed}." if allowed else "that was not already in its input."))))

    for name in invariants.symmetries:
        if not _symmetric(answer, name):
            found.append(Violation("symmetry:" + name, STRONG,
                                   f"Every example output is symmetric under the {name}; the answer is not."))

    if invariants.unchanged_range is not None and answer.shape == test_input.shape:
        share = float((answer == test_input).mean())
        low, high = invariants.unchanged_range
        if share < low - UNCHANGED_MARGIN or share > high + UNCHANGED_MARGIN:
            found.append(Violation("unchanged", WEAK, (
                f"The answer leaves {share:.0%} of the cells as they were; the examples leave "
                f"{low:.0%} to {high:.0%}.")))

    if answer.shape == test_input.shape:
        for colour in invariants.kept_counts:
            before, after = int((test_input == colour).sum()), int((answer == colour).sum())
            if before != after:
                found.append(Violation(f"count:{colour}", WEAK, (
                    f"No example changes how many cells have colour {colour}; the input has {before} "
                    f"and the answer {after}.")))
    return found


def _describe_shape(rule) -> str:
    kind, arg = rule
    if kind == "same":
        return "the same size as the input in every example"
    if kind == "fixed":
        return "always the same size"
    if kind == "scale":
        return f"the input scaled up {arg[0]}x{arg[1]} in every example"
    return f"the input shrunk by {arg[0]}x{arg[1]} in every example"


def refuses(violations: Sequence[Violation]) -> bool:
    """True when the answer breaks something strong."""
    return any(v.severity == STRONG for v in violations)
