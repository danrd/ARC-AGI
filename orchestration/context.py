"""What joins the search to the prompt: the context each task's prompt is
built with, carrying the hint the search has verified for it.

The search belongs to rl and the prompt to subsymbolic, and neither may know
of the other, so the one place they meet is here. The prompt block
(subsymbolic.arc_resolvers.search_hints_resolver) reads `context["search_hints"]`
and says nothing when it is absent; rl.search_hints.hints_for computes the
text, about a minute of search a task, and only for a task whose training
pairs a single set of action types reproduces. This puts the second into the
first for run_llm_over_tasks:

    context_builder = with_search_hints(my_context)      # my_context(task) -> dict
    run_llm_over_tasks(tasks, module, evaluator, context_builder=context_builder)

The prompt has to ask for it: "search_hints" in PromptingConfig.blocks and in
.resolvers, or the hint is computed and never shown.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional

ContextBuilder = Callable[[Any], Dict[str, Any]]


def with_search_hints(context_builder: Optional[ContextBuilder] = None, settings=None,
                      cache: Optional[dict] = None) -> ContextBuilder:
    """`context_builder` with each task's verified search hint added under
    "search_hints", when there is one.

    A task with no verified hint gets none: its context is what the caller's
    builder returned, and the prompt goes without the block. A context that
    already holds "search_hints" - a hint the caller computed, or wants
    withheld as an empty string - is left as it is and no search is run.

    The hint is remembered per task (`cache`, a dict the caller may keep and
    share between arms), so a retry or a second arm does not pay for another
    search whose result is random anyway. A search that fails is warned
    about and costs the task its hint, not the run its task.
    """
    from rl.search_hints import HintCache, SearchSettings

    base = context_builder or (lambda task: {})
    hints = HintCache(settings or SearchSettings(), {} if cache is None else cache)

    def build(task) -> Dict[str, Any]:
        context = dict(base(task))
        if "search_hints" in context:
            return context
        try:
            text = hints(task)
        except Exception as error:  # noqa: BLE001 - the LLM run must not die of a failed search
            warnings.warn(f"search hint for {getattr(task, 'label', task)!r} failed: {error!r}")
            return context
        if text:
            context["search_hints"] = text
        return context

    return build
