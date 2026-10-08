"""What a model is told about its earlier answers.

A retry that asks the same question again gets, on average, the same answer. A retry
that is shown the answer it gave and what was wrong with it has something to work
from. This is where that is put together: the reasons an answer is refused (`review`),
the history of attempts rendered for the `memory` block (`render_history`), and a
wrapper that gives the graph's own retries the same thing (`with_feedback`).

The reasons come from two places and are kept apart in the text. The symbolic checks
(symbolic.invariants) say what the answer breaks of what every example does - "the
answer is 5x5, the examples give 3x3 for an input of this size" - which is a fact the
model can use. A second model's refusal says no more than that the answer was not
accepted, which is all it knows and all it is quoted as.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from orchestration.hierarchy import _read_grid
from subsymbolic.utils import parse_llm_output
from symbolic.invariants import STRONG, Invariants, Violation, check, learn

UNREADABLE = "The reply could not be read as a grid in the format asked for."
NOT_ACCEPTED = ("A second model, shown the examples and this answer, did not accept it as the output "
                "the examples imply.")


@dataclass
class Attempt:
    number: int
    text: str
    grid: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)
    strong: int = 0
    weak: int = 0

    @property
    def clean(self) -> bool:
        return not self.notes and self.grid is not None


def review(task, grid: Optional[np.ndarray], invariants: Optional[Invariants] = None,
           verify: Optional[Callable[[Any, np.ndarray], bool]] = None):
    """(notes, strong, weak): why `grid` is not accepted. No notes: nothing was found wrong.

    The second model is asked only about an answer the symbolic checks leave standing, since
    asking it about one that is already refused costs a call and adds nothing."""
    if grid is None:
        return [UNREADABLE], 1, 0
    invariants = invariants if invariants is not None else learn(
        [(s.train_inp, s.train_out) for s in task.subtasks])
    violations: List[Violation] = check(invariants, task.test_subtask.train_inp, grid)
    notes = [v.message for v in violations]
    strong = sum(v.severity == STRONG for v in violations)
    if strong == 0 and verify is not None:
        if not verify(task, grid):
            notes.append(NOT_ACCEPTED)
            strong += 1
    return notes, strong, len(violations) - sum(v.severity == STRONG for v in violations)


def render_history(attempts: Sequence[Attempt]) -> str:
    """The attempts for the `memory` block: the answer as written, and why it was not accepted."""
    parts = []
    for attempt in attempts:
        if attempt.clean:
            continue
        shown = attempt.text.strip() if isinstance(attempt.text, str) and attempt.text.strip() else "(no answer)"
        reasons = "\n".join(f"- {note}" for note in attempt.notes) or "- (no reason recorded)"
        parts.append(f"Attempt {attempt.number}. Your answer:\n{shown}\nIt was not accepted:\n{reasons}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nDo not repeat these answers; correct what they got wrong."


def memoize_verdicts(verify: Callable[[Any, np.ndarray], bool]) -> Callable[[Any, np.ndarray], bool]:
    """`verify` that answers a (task, grid) it has been asked before from memory. The decision
    and the feedback both want the second model's word on the same answer; it is asked once."""
    verdicts: Dict[Any, bool] = {}

    def verdict(task, grid) -> bool:
        grid = np.asarray(grid)
        key = (getattr(task, "label", id(task)), grid.shape, grid.tobytes())
        if key not in verdicts:
            verdicts[key] = bool(verify(task, grid))
        return verdicts[key]

    return verdict


def best_attempt(attempts: Sequence[Attempt]) -> Optional[Attempt]:
    """The best attempt: one that reads as a grid over one that does not, then the fewest strong
    complaints, then the fewest weak; the later of equals."""
    if not attempts:
        return None
    return min(reversed(list(attempts)), key=lambda a: (a.grid is None, a.strong, a.weak))


def with_feedback(dispatch_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                  verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
                  parse: Callable[[str], Any] = parse_llm_output) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """`dispatch_fn` that tells the model, on each call after the first for a task, what its
    earlier answers got wrong.

    The graph's own retry calls the model again with nothing new; this puts the history
    under `history_text` in the context, where the `memory` block reads it (the module's
    prompt has to hold that block: `install_blocks(module, ["memory"])`, or a config that
    names it). The attempts are kept per task, in this closure, and a call for any module
    but the model passes through."""
    attempts: Dict[Any, List[Attempt]] = {}
    learnt: Dict[Any, Invariants] = {}

    def dispatch(state: Dict[str, Any]) -> Dict[str, Any]:
        if state["current_module"].module_name.lower() != "subsymbolic":
            return dispatch_fn(state)
        task = state["task"]
        key = getattr(task, "label", id(task))
        history = attempts.setdefault(key, [])
        context = dict(state.get("auxiliary_info") or {})
        text = render_history(history)
        if text:
            context["history_text"] = text
        result = dispatch_fn({**state, "auxiliary_info": context})
        solution = result.get("solution", "")
        grid = _read_grid(parse, solution) if solution not in (None, "") else None
        if key not in learnt:
            learnt[key] = learn([(s.train_inp, s.train_out) for s in task.subtasks])
        notes, strong, weak = review(task, grid, learnt[key], verify)
        history.append(Attempt(len(history) + 1, solution if isinstance(solution, str) else "", grid, notes,
                               strong, weak))
        return result

    return dispatch
