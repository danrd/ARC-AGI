"""Checking one model's answer with another.

A model that solved a puzzle says so no more reliably than one that did not,
and the training pairs cannot be run through its answer as they can through a
solver's rule. What can be asked of a second model is the narrow question -
given the examples, the new input and this proposed output, is the output
what the rule gives? - and it can only be asked in the direction the answers
come from: this module puts a verdict on a candidate, and takes the word of
nothing but the second model's reply.

    verify = LlmVerifier(runner, tokenizer)
    verify(task, candidate_grid)          # True only on a clear YES

A reply that is not a clear YES is a refusal: an unreadable one, one that is
cut off inside its reasoning, and a prompt that does not fit the token
limit. A wrong "accept" costs an answer, a wrong "refuse" costs a retry.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY

BLOCKS = ["answer_check_instruction", "examples", "answer_check_query"]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_VERDICT = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def answer_check_config(base: Optional[PromptingConfig] = None) -> PromptingConfig:
    """The prompting config of the check: `base` (the main prompt's, so the
    chat template and the token limit carry over) with the check's blocks."""
    base = base or PromptingConfig(token_limit=9000, min_examples=2)
    return base.model_copy(update={"blocks": list(BLOCKS), "resolvers": ["examples"],
                                   "filters": ["grid"], "assistant_prefix": None})


def parse_verdict(text: str) -> bool:
    """True when the reply's first YES/NO, outside any <think> block, is YES.

    A reply still inside an unclosed <think> has not reached its answer, and
    reads as a refusal."""
    if not isinstance(text, str):
        return False
    text = _THINK.sub("", text)
    if re.search(r"<think>", text, re.IGNORECASE):
        return False
    found = _VERDICT.search(text)
    return found is not None and found.group(1).lower() == "yes"


class LlmVerifier:
    """Asks `runner` (any object with generate(prompt) -> str) whether a
    candidate grid is the answer. Give it a runner other than the one that
    produced the candidate; nothing here enforces that."""

    def __init__(self, runner, tokenizer, prompt: Optional[PromptingConfig] = None,
                 grid_repr_type: str = "concise"):
        self.runner = runner
        self.builder = PromptBuilder(answer_check_config(prompt), tokenizer,
                                     resolver_registry=RESOLVER_REGISTRY, filter_registry=FILTER_REGISTRY)
        self.grid_repr_type = grid_repr_type

    def __call__(self, task, candidate) -> bool:
        context = {"grid_repr_type": self.grid_repr_type,
                   "test_input_grid": task.test_subtask.train_inp,
                   "candidate_grid": np.asarray(candidate)}
        prompt = self.builder.build(task, context=context)
        if prompt is None:
            return False
        return parse_verdict(self.runner.generate(prompt))
