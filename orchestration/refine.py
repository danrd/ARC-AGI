"""The refinement loop: ask the model, check the answer, show it what was wrong, ask again.

A model answers; the answer is read as a grid and held to what the examples agree on
(symbolic.invariants) and, when a second model is given, to its word; if something is
wrong the model is called again with its own answer and the reasons in front of it
(`memory` block, orchestration.feedback). It stops at the first answer nothing is
found wrong with, or after `rounds` calls, and returns the attempt with the fewest
complaints. This is the loop that looks like self-correction and is not quite: the
corrections come from outside the model, from checks that do not depend on it.

    refined = RefiningModule(module, rounds=3, verify=second_model)
    refined.solve(task, context)        # the same shape as SubsymbolicModule.solve

`loop` is the same thing with the calls passed in, for anything that is not a
SubsymbolicModule. Whether it helps is a measurement, not a given: an earlier try at
asking the model again with its previous answer did not work, and what is new here is
only that the reasons are concrete. `RefineResult.attempts` keeps every attempt and
its reasons, so a run can be read afterwards for whether the second answer was ever
better than the first.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from orchestration import refine_core
from orchestration.feedback import Attempt, review
from orchestration.hierarchy import _read_grid
from orchestration.refine_core import RefineResult  # noqa: F401 - the result type, as this module has always offered it
from orchestration.trace import Tracer
from subsymbolic.utils import parse_llm_output
from symbolic.invariants import Invariants, learn


def _parts(task, verify, parse, invariants):
    """The loop's two project-specific functions for an ARC task: read the reply as a grid, and review the grid
    against what the examples agree on (and the second model's word, when `verify` is given)."""
    learnt = invariants if invariants is not None else learn([(s.train_inp, s.train_out) for s in task.subtasks])
    return (lambda text: _read_grid(parse, text)), (lambda _task, grid: review(task, grid, learnt, verify))


def loop(task, ask: Callable[[Dict[str, Any]], str], rounds: int = 3,
         verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
         parse: Callable[[str], Any] = parse_llm_output, context: Optional[Dict[str, Any]] = None,
         invariants: Optional[Invariants] = None) -> RefineResult:
    """`ask(context) -> text` up to `rounds` times; each call after the first has the attempts so far
    in `context["history_text"]`. Stops at the first attempt nothing is found wrong with."""
    read, check = _parts(task, verify, parse, invariants)
    return refine_core.loop(task, ask, check, read, rounds, context, attempt_class=Attempt)


class RefiningModule(refine_core.RefiningModule):
    """A SubsymbolicModule-shaped module whose `solve` is `loop` around the wrapped one. The task is only known
    at `solve`, so the review and the reading are made there."""

    def __init__(self, module, rounds: int = 3, verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
                 parse: Callable[[str], Any] = parse_llm_output, tracer: Optional[Tracer] = None):
        super().__init__(module, review=None, parse=None, rounds=rounds, tracer=tracer, attempt_class=Attempt)
        self.verify, self.grid_parse = verify, parse

    def solve(self, task, context: Optional[dict] = None) -> Dict[str, Any]:
        self.parse, self.review = _parts(task, self.verify, self.grid_parse, None)
        return super().solve(task, context)
