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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from orchestration.blocks import install_blocks
from orchestration.feedback import Attempt, best_attempt, render_history, review
from orchestration.hierarchy import _read_grid
from orchestration.trace import Tracer
from subsymbolic.utils import parse_llm_output
from symbolic.invariants import Invariants, learn


@dataclass
class RefineResult:
    attempts: List[Attempt] = field(default_factory=list)
    best: Optional[Attempt] = None

    @property
    def accepted(self) -> bool:
        return self.best is not None and self.best.clean

    @property
    def solution(self) -> str:
        return self.best.text if self.best is not None else ""


def loop(task, ask: Callable[[Dict[str, Any]], str], rounds: int = 3,
         verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
         parse: Callable[[str], Any] = parse_llm_output, context: Optional[Dict[str, Any]] = None,
         invariants: Optional[Invariants] = None, tracer: Optional[Tracer] = None) -> RefineResult:
    """`ask(context) -> text` up to `rounds` times; each call after the first has the attempts so far
    in `context["history_text"]`. Stops at the first attempt nothing is found wrong with."""
    invariants = invariants if invariants is not None else learn([(s.train_inp, s.train_out) for s in task.subtasks])
    attempts: List[Attempt] = []
    for _ in range(max(1, rounds)):
        call = dict(context or {})
        history = render_history(attempts)
        if history:
            call["history_text"] = history
        text = ask(call)
        text = text if isinstance(text, str) else ""
        grid = _read_grid(parse, text) if text.strip() else None
        notes, strong, weak = review(task, grid, invariants, verify, tracer)
        attempts.append(Attempt(len(attempts) + 1, text, grid, notes, strong, weak))
        if attempts[-1].clean:
            break
    return RefineResult(attempts=attempts, best=best_attempt(attempts))


class RefiningModule:
    """A SubsymbolicModule-shaped module whose `solve` is `loop` around the wrapped one."""

    def __init__(self, module, rounds: int = 3, verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
                 parse: Callable[[str], Any] = parse_llm_output, tracer: Optional[Tracer] = None):
        self.module = module
        self.rounds, self.verify, self.parse, self.tracer = rounds, verify, parse, tracer
        self.last: Optional[RefineResult] = None
        install_blocks(module, ["memory"])

    @property
    def builder(self):
        return self.module.builder

    @property
    def runner(self):
        return self.module.runner

    def close(self) -> None:
        self.module.close()

    def solve(self, task, context: Optional[dict] = None) -> Dict[str, Any]:
        errors: List[str] = []

        def ask(call: Dict[str, Any]) -> str:
            result = self.module.solve(task, context=call)
            if "error" in (result.get("module_results") or {}):
                errors.append(str(result["module_results"]["error"]))
            return result.get("solution", "")

        self.last = loop(task, ask, self.rounds, self.verify, self.parse, context, tracer=self.tracer)
        if not self.last.attempts or (not self.last.solution and errors):
            return {"solution": "", "module_results": {"error": errors[-1] if errors else "no answer"}}
        return {"solution": self.last.solution,
                "module_results": {"rounds": len(self.last.attempts), "accepted": self.last.accepted,
                                   "notes": [a.notes for a in self.last.attempts]}}
