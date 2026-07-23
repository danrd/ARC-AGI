"""Aggregates the LLM-based (subsymbolic) solving path: build a prompt for
a task via PromptBuilder, then run it through whichever inference backend
build_runner resolves for the given ExperimentConfig.

Symmetric to symbolic.symbolic_module.SymbolicModule: agents call
SubsymbolicModule().solve(task) directly to attempt a solution, with the
same result shape (a dict with "solution" and "module_results") so
orchestration can treat both the same way.

The runner (a live model/server connection) is expensive to build, so
it's constructed once, lazily, on first use rather than per solve() call.
PromptingConfig / tokenizer / ExperimentConfig are expected to be
assembled once at the system level and passed in here — logging, memory,
and anything beyond "build a prompt, run it" can layer on top of this
later.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from subsymbolic.configs import ExperimentConfig, PromptingConfig
from subsymbolic.llm_runtime import build_runner
from subsymbolic.prompt_builder import PromptBuilder


class SubsymbolicModule:
    def __init__(self, prompting_config: PromptingConfig, tokenizer,
                 experiment_config: ExperimentConfig):
        self.builder = PromptBuilder(prompting_config, tokenizer)
        self.experiment_config = experiment_config
        self._runner = None

    @property
    def runner(self):
        if self._runner is None:
            self._runner = build_runner(self.experiment_config)
        return self._runner

    def solve(self, task, context: Optional[dict] = None) -> Dict[str, Any]:
        prompt = self.builder.build(task, context=context or {})
        if prompt is None:
            return {"solution": "", "module_results": {"error": "prompt didn't fit token_limit"}}
        text = self.runner.generate(prompt)
        return {"solution": text, "module_results": {}}

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None
