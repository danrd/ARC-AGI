"""
Universal Jinja2-based prompt builder.

A prompt is composed of ordered "blocks" (BlockSpec). Each block resolves to
a `<name>/<version>.j2` template file under `config.blocks_dir` (default:
`data/prompts/`), rendered with the shared `context` dict. Blocks are
token-budgeted and joined according to `config.join_format`, or via
`tokenizer.apply_chat_template` when `config.chat_template` is set.

This module is domain-agnostic. ARC-specific grid formatting, role texts,
and instruction bodies live in `arc_grid_formatting.py` / `arc_blocks_data.py`
/ the `.j2` files themselves — this file only knows how to assemble blocks.

Editing while experimenting (e.g. from a notebook):
    Templates are re-read from disk automatically (Environment(auto_reload=True)
    compares each template file's mtime on every `get_template()` call), so
    editing a .j2 file under `blocks_dir` and re-running `build()` picks up
    the change without recreating the PromptBuilder. Use `render_block(...)`
    to iterate on a single block without needing a full task/tokenizer setup,
    and `list_blocks()` to see what's on disk.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict

from subsymbolic.arc_grid_formatting import format_grid
from subsymbolic.configs import BlockSpec, PromptingConfig

from subsymbolic.registry import RESOLVER_REGISTRY, FILTER_REGISTRY


class PromptBuilder:
    """Composes a prompt string (or chat message list) from configured blocks."""

    def __init__(self, config: PromptingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.env = self._make_env()
        self.resolvers: Dict[str, Callable] = {func_name: RESOLVER_REGISTRY[func_name] for func_name in self.config.resolvers}

    def _make_env(self) -> Environment:
        env = Environment(
            loader=FileSystemLoader(self.config.blocks_dir),
            undefined=StrictUndefined,   # KeyError on unknown variables
            trim_blocks=True,            # strip \n after {% block %} tags
            lstrip_blocks=True,          # strip leading whitespace before {% %}
            auto_reload=True,            # re-read .j2 files whose mtime changed
        )
        for filter_name in self.config.filters:
            env.filters[filter_name] = FILTER_REGISTRY[filter_name]
        return env

    def reload_env(self) -> None:
        """Force a brand new Environment (e.g. after changing blocks_dir).
        Not required for ordinary template edits — those are picked up
        automatically via auto_reload."""
        self.env = self._make_env()

    def list_blocks(self) -> List[str]:
        """List '<name>/<version>' pairs found on disk under blocks_dir —
        handy for discovering what's available while experimenting."""
        root = Path(self.config.blocks_dir)
        if not root.exists():
            return []
        return sorted(
            f"{p.parent.name}/{p.stem}"
            for p in root.glob("*/*.j2")
        )

    def render_block(self, name: str, version: str = "v1", **context) -> str:
        """Render a single block template directly, bypassing the block list,
        token budget, and join step. Useful for iterating on one .j2 file
        (e.g. from a notebook) without building the whole prompt."""
        template = self.env.get_template(f"{name}/{version}.j2")
        return template.render(**context)

    def build(self, task, context: Optional[dict] = None,
              overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Render and join all configured blocks.

        `context` feeds Jinja variables to every block template.
        `overrides` fully replaces a block's rendered text by name (keeps the
        old prompts_modifications behaviour, without touching templates).
        Returns None if the prompt can't fit within token_limit (even after
        trimming the examples block down to `min_examples`).
        """
        context = context or {}
        overrides = overrides or {}
        parts: "OrderedDict[str, Tuple[BlockSpec, str]]" = OrderedDict()
        used_tokens = 0

        for spec_raw in self.config.blocks:
            spec = BlockSpec.parse(spec_raw)

            if spec.name in overrides:
                rendered = overrides[spec.name]
            elif spec.name in self.resolvers:
                rendered = self.resolvers[spec.name](
                    task, self.config.token_limit - used_tokens, context, self,
                )
                if rendered is None:
                    return None  # this resolver couldn't fit even its minimum
            else:
                template = self.env.get_template(f"{spec.name}/{spec.version}.j2")
                rendered = template.render(**context)

            cost = self.count_tokens(rendered)
            if used_tokens + cost > self.config.token_limit:
                return None
            parts[spec.name] = (spec, rendered)
            used_tokens += cost

        return self._join(parts)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def _join(self, parts: "OrderedDict[str, Tuple[BlockSpec, str]]") -> str:
        if self.config.chat_template is None:
            sections = [
                self._wrap(spec.tag or spec.name.upper(), content) if spec.tag else content
                for spec, content in parts.values()
            ]
            return "\n".join(sections)

        role_buckets: Dict[str, list] = {"system": [], "user": []}
        for spec, content in parts.values():
            wrapped = self._wrap(spec.tag or spec.name.upper(), content) if spec.tag else content
            role_buckets[spec.role].append(wrapped)

        messages = []
        if role_buckets["system"]:
            messages.append({"role": "system", "content": "\n".join(role_buckets["system"])})
        messages.append({"role": "user", "content": "\n".join(role_buckets["user"])})

        if self.config.assistant_prefix:
            messages.append({"role": "assistant", "content": self.config.assistant_prefix})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _wrap(self, tag: str, content: str) -> str:
        if self.config.join_format == "xml":
            return f"<{tag}>\n{content}\n</{tag}>"
        if self.config.join_format == "md":
            return f"## {tag}\n\n{content}\n\n---"
        return content  # "plain"

    def _format_grid(self, grid, type: str = "concise") -> str:
        return format_grid(grid, repr_type=type)
