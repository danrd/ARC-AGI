"""Example project-specific block resolvers.
A resolver is a function `(task, budget, context) -> str | None`, registered
with PromptBuilder via the `resolvers` param, and invoked instead of
rendering a `.j2` template for a block whose name matches.
"""
from typing import Optional


def build_examples_resolver(task, budget: int, context: dict, builder) -> Optional[str]:
    """Default resolver for the "examples" block: loops task.subtasks,
    rendering each with examples/v1.j2 via the builder's own Jinja
    environment, stopping once the token budget is exhausted but requiring
    at least builder.config.min_examples to fit, or the whole block fails.
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
    """Placeholder for a whole-task transformation summary."""
    return ""
