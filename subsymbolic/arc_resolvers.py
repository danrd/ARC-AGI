"""Example project-specific block resolvers.

A resolver is a function `(task, budget, context) -> str | None`, registered
with PromptBuilder via the `resolvers` param, and invoked instead of
rendering a `.j2` template for a block whose name matches. Use this for
whole-task, non-per-example auxiliary content — the same seam the built-in
"examples" resolver uses internally, just registered under a different name.

This is the STATIC counterpart to the dynamic path: a resolver here always
runs when its block name appears in config.blocks (a config-time decision).
The dynamic path (an agent deciding at runtime what extra info to fetch)
instead populates context["auxiliary_info"] itself and relies on the generic
`auxiliary_info/v1.j2` template to print whatever ends up there — see
mas_langgraph.py's decision node for that side.
"""
from typing import Optional


def transformation_summary_resolver(task, budget: int, context: dict) -> Optional[str]:
    """Placeholder for a whole-task transformation summary."""
    return ""
