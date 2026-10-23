"""Why a cell rule is missing or wrong: what the features cannot tell apart, and how a wrong answer differs from the right one.

Three reasons a task is out of reach of symbolic.transformed_rules, told apart by these:
  no transformation   no transformation of symbolic.grid_transforms gives the inputs their outputs' shape
  ambiguous           cells with the same features need different colours: the features are too few
  no rule             every cell follows from its features, yet no rule within the grammar says how: the grammar is
and a rule that is found and wrong is wrong by changing cells it should not, leaving cells it should change, or by
painting the right cells the wrong colour.
"""
from typing import Dict

import numpy as np


def conflicts(x: np.ndarray, targets: np.ndarray) -> float:
    """The share of cells whose feature row also occurs with another target: 0.0 when every cell follows from its
    features, which is when a rule over those features can exist at all."""
    groups: Dict[bytes, set] = {}
    keys = [row.tobytes() for row in np.ascontiguousarray(x)]
    for key, target in zip(keys, targets.tolist()):
        groups.setdefault(key, set()).add(target)
    bad = sum(1 for key in keys if len(groups[key]) > 1)
    return bad / len(keys) if keys else 0.0


def error_kinds(predicted: np.ndarray, truth: np.ndarray, start: np.ndarray) -> Dict[str, int]:
    """How a wrong answer differs from the right one, in cells, against the grid the rule started from:
    false_change  the rule changed a cell that stays
    missed        the rule left a cell that changes
    wrong_colour  both change it, to different colours
    Empty when the shapes differ (the cells cannot be compared)."""
    if predicted.shape != truth.shape or start.shape != truth.shape:
        return {}
    changed_by_rule, changes = predicted != start, truth != start
    return {"false_change": int((changed_by_rule & ~changes).sum()),
            "missed": int((~changed_by_rule & changes).sum()),
            "wrong_colour": int((changed_by_rule & changes & (predicted != truth)).sum())}


def kind_of(counts: Dict[str, int]) -> str:
    """The one word for a wrong answer: the single kind of error it has, `mixed` when it has several, `shape` when
    its cells could not be compared."""
    present = [name for name, n in counts.items() if n]
    if not counts:
        return "shape"
    return present[0] if len(present) == 1 else "mixed" if present else "equal"
