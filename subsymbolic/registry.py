"""
Unified registry for methods related to subsymbolic functionality.
It's purpose is to aggregate and isolate task-specific functions from task-agnostic features.
"""

from subsymbolic.arc_resolvers import build_examples_resolver, transformation_summary_resolver
from subsymbolic.arc_grid_formatting import format_grid

RESOLVER_REGISTRY = {
                    "examples": build_examples_resolver,
                    "summary": transformation_summary_resolver
                    }


def _grid_filter(grid, type: str = "concise") -> str:
    """Adapts format_grid's `repr_type` kwarg to `type`, since that's the
    name every `.j2` template's `| grid(type=...)` call already uses."""
    return format_grid(grid, repr_type=type)


FILTER_REGISTRY = {
                   "grid": _grid_filter
                  }
