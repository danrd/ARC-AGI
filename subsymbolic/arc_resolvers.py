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


def build_examples_resolver(task, budget: int, context: dict, builder) -> Optional[str]:
    """Default resolver for the "examples" block: loops task.subtasks,
    rendering each with examples/v1.j2 via the builder's own Jinja
    environment, stopping once the token budget is exhausted (but requiring
    at least builder.config.min_examples to fit, or the whole block fails).

    This used to be a PromptBuilder method (_build_examples) — moved out
    here because task.subtasks / .train_inp / .train_out is ARC-specific
    knowledge the framework itself shouldn't carry. It only needed `env`,
    `config`, and the tokenizer, all of which `builder` (passed in by
    PromptBuilder.build()) already exposes.
    """
    template = builder.env.get_template("examples/v1.j2")
    accumulated = ""
    accumulated_tokens = 0

    for idx, subtask in enumerate(task.subtasks):
        example_context = {
            **context,
            "idx": idx + 1,
            "input_grid": subtask.train_inp,
            "output_grid": subtask.train_out,
        }
        rendered = template.render(**example_context)
        cost = builder.count_tokens(rendered)
        if accumulated_tokens + cost > budget:
            if idx < builder.config.min_examples:
                return None  # even the minimum didn't fit
            break
        accumulated += rendered
        accumulated_tokens += cost

    return accumulated


def transformation_summary_resolver(task, budget: int, context: dict, builder) -> Optional[str]:
    """Placeholder for a whole-task transformation summary (e.g. built from
    GridSummary / summarize_training_pair across task.subtasks). This is an
    explicit no-op seam — wire in the real implementation here once it's
    ready; PromptBuilder itself never needs to know this block exists.
    """
    return ""
