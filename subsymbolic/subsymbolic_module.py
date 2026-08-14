"""Aggregates the LLM-based (subsymbolic) solving path: build a prompt for
a task via PromptBuilder, then run it through whichever inference backend
build_runner resolves for the given ExperimentConfig.

Symmetric to symbolic.symbolic_module.SymbolicModule: agents call
SubsymbolicModule().solve(task) directly to attempt a solution, with the
same result shape (a dict with "solution" and "module_results") so
orchestration can treat both the same way.

The runner (a live model/server connection) is expensive to build, so by
default it's constructed once, lazily, on first use rather than per
solve() call - or pass an already-built one via `runner=` (e.g.
llm_runtime.build_openrouter_runner(experiment_config), or a fake for
testing) to skip build_runner entirely and use experiment_config only
for prompting; useful for running a local and a hosted module side by
side off the same config without mutating it.
tokenizer / ExperimentConfig (which already carries the PromptingConfig
PromptBuilder needs, as its `prompt` field) are expected to be assembled
once at the system level and passed in here — logging, memory, and
anything beyond "build a prompt, run it" can layer on top of this later.

This is also the one place this project's resolver/filter registry
(subsymbolic.registry) gets wired into PromptBuilder - PromptBuilder
itself takes no dependency on it, so it stays reusable outside this
project (see prompt_builder.py's module docstring).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from subsymbolic.llm_setup import build_runner
from subsymbolic.prompt_builder import PromptBuilder
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY


class SubsymbolicModule:
    def __init__(self, experiment_config, tokenizer, runner=None):
        self.builder = PromptBuilder(experiment_config.prompt, tokenizer,
                                      resolver_registry=RESOLVER_REGISTRY,
                                      filter_registry=FILTER_REGISTRY)
        self.experiment_config = experiment_config
        self._runner = runner  # allows injecting an already-built runner - see module docstring

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
