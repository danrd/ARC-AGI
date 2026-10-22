"""A whole-grid transformation, then the cell rule search: the rule explains what the rearrangement leaves.

For every transformation of symbolic.grid_transforms that gives each training input the shape of its output (and applies
to the test input), the cell rules are searched on the pairs (transformed input, output). The identity is one of them,
so a task the plain search solves is still found. A transformation is one more thing to have guessed from the examples,
so a rule behind one is ranked as `PENALTY` longer than the same rule without it.
"""
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from symbolic import rule_search
from symbolic.grid_transforms import TRANSFORMS, transformed

#: What a transformation adds to the length of the rule behind it.
PENALTY = 1.0


@dataclass
class Candidate:
    transform: str
    feature_set: str
    rules: list
    test_input: np.ndarray

    @property
    def cost(self) -> float:
        return self.rules[0].cost + (0.0 if self.transform == "identity" else PENALTY)


def candidates(train, test_input, feature_sets: Sequence[str] = ("core", "rich"), seconds: Optional[float] = None,
               clock=time.time) -> List[Candidate]:
    """Every transformation with a rule that reproduces all the pairs, cheapest first. `seconds` is how long the
    searches may take altogether: when it is spent the transformations not tried yet are left out."""
    started, found = clock(), []
    inputs = [np.asarray(p["input"]) for p in train]
    outputs = [np.asarray(p["output"]) for p in train]
    for name in TRANSFORMS:
        if seconds is not None and clock() - started > seconds:
            break
        moved = [transformed(name, g) for g in inputs]
        test = transformed(name, test_input)
        if test is None or any(m is None or m.shape != o.shape for m, o in zip(moved, outputs)):
            continue
        pairs = [{"input": m, "output": o} for m, o in zip(moved, outputs)]
        for feature_set in feature_sets:
            rules = rule_search.search(pairs, feature_set)
            if rules:
                found.append(Candidate(name, feature_set, rules, test))
                break
    return sorted(found, key=lambda c: c.cost)


def attempts(found: Sequence[Candidate], tries: int = 2) -> List[np.ndarray]:
    """The first `tries` different outputs, taking the rules of all the candidates by their length with the
    transformation counted in."""
    ranked = sorted(((c.cost - c.rules[0].cost + r.cost, i, r, c) for i, c in enumerate(found) for r in c.rules),
                    key=lambda item: item[:2])
    out = []
    for _, _, rule, candidate in ranked:
        guess = rule.apply(candidate.test_input)
        if not any(g.shape == guess.shape and np.array_equal(g, guess) for g in out):
            out.append(guess)
        if len(out) == tries:
            break
    return out
