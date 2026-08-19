"""Example project-specific block resolvers.
A resolver is a function `(task, budget, context) -> str | None`, registered
with PromptBuilder via the `resolvers` param, and invoked instead of
rendering a `.j2` template for a block whose name matches.
"""
from collections import OrderedDict
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


_findings_cache: "OrderedDict[str, object]" = OrderedDict()
_FINDINGS_CACHE_SIZE = 64


def _task_findings(task):
    """Symbolic findings for one task, memoised.

    Analysis is by far the most expensive thing a prompt block can trigger -
    on the measured spread it is milliseconds for most tasks but tens of
    seconds for the worst - and prompt building for the same task happens
    more than once across a notebook session, a retry, or several prompt
    variants. Keyed by task label, bounded so a long run can't grow it
    without limit.
    """
    from symbolic.analyzer import SymbolicAnalyzer

    key = str(getattr(task, "label", None) or getattr(task, "id", id(task)))
    if key in _findings_cache:
        _findings_cache.move_to_end(key)
        return _findings_cache[key]

    findings = SymbolicAnalyzer().analyze_task(task).get_findings()
    _findings_cache[key] = findings
    if len(_findings_cache) > _FINDINGS_CACHE_SIZE:
        _findings_cache.popitem(last=False)
    return findings


def transformation_summary_resolver(task, budget: int, context: dict, builder) -> Optional[str]:
    """Whole-task transformation summary from the symbolic analyzer.

    Returns None - omitting the block entirely - when the analyzer found
    nothing to claim, or when not even its best finding fits the budget. An
    empty-but-present block would spend tokens on a header introducing
    nothing.
    """
    from symbolic.findings import render_findings

    return render_findings(_task_findings(task), budget, builder.count_tokens)
