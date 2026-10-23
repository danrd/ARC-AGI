"""One answer per task out of the answers of several sources.

A source (the symbolic modules, the cell rules behind a transformation, the language model) gives at most one grid for a
task. The system hands in one, so something has to choose; this holds the choosing, separate from the sources, so that
strategies can be compared on the answers the sources gave.

    cascade   the first source that answers, in the order given; a rule answers only when its length is at most `limit`
    agree     a grid two or more sources give wins; with none, the cascade chooses

`answers` is {source: (grid, cost)}; cost is the rule's length, None for a source that has none.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

ORDER = ("modules", "rules", "llm")


def same(a, b) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def cascade(answers: Dict[str, Tuple[np.ndarray, Optional[float]]], limit: float = float("inf"),
            order: Sequence[str] = ORDER) -> Optional[Tuple[str, np.ndarray]]:
    for source in order:
        if source not in answers:
            continue
        grid, cost = answers[source]
        if source == "rules" and cost is not None and cost > limit:
            continue
        return source, grid
    return None


def agree(answers: Dict[str, Tuple[np.ndarray, Optional[float]]], limit: float = float("inf"),
          order: Sequence[str] = ORDER) -> Optional[Tuple[str, np.ndarray]]:
    """The grid most sources give, when two or more give it (the first source in `order` that gives it names it);
    otherwise the cascade's choice."""
    best = None
    for source in order:
        if source not in answers:
            continue
        grid = answers[source][0]
        votes = sum(same(grid, answers[other][0]) for other in order if other in answers)
        if votes >= 2 and (best is None or votes > best[0]):
            best = (votes, source, grid)
    if best is not None:
        return "agree:" + best[1], best[2]
    return cascade(answers, limit, order)


STRATEGIES = {"cascade": cascade, "agree": agree}


def score(per_task: Dict[str, Dict[str, Tuple[np.ndarray, Optional[float]]]], solutions: Dict[str, np.ndarray],
          strategy: str, limit: float = float("inf")) -> Dict[str, object]:
    """{right, answered, by_source: {name: [right, answered]}} of `strategy` over the tasks' answers."""
    choose = STRATEGIES[strategy]
    right, answered, by = 0, 0, {}
    for task, answers in per_task.items():
        picked = choose(answers, limit)
        if picked is None:
            continue
        name, grid = picked
        ok = same(np.asarray(grid), np.asarray(solutions[task]))
        answered += 1
        right += ok
        counts = by.setdefault(name, [0, 0])
        counts[0] += ok
        counts[1] += 1
    return {"right": right, "answered": answered, "by_source": by}
