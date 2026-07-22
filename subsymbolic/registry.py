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

FILTER_REGISTRY = {
                   "grid": format_grid  
                  }
