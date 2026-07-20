"""
Universal Jinja2-based prompt builder.

A prompt is composed of ordered "blocks" (BlockSpec). Each block resolves to
a `<name>/<version>.j2` template file under `config.blocks_dir` (default:
`data/prompts/`), rendered with the shared `context` dict. Blocks are
token-budgeted and joined according to `config.join_format`, or via
`tokenizer.apply_chat_template` when `config.chat_template` is set.

"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict

from arc_grid_formatting import format_grid


class BlockSpec(BaseModel):
    """Prompt block specification."""
    name: str
    version: str = "v1"
    role: Literal["system", "user"] = "user"   # role for chat template
    tag: Optional[str] = None                   # tag for non-chat wrapping

    @classmethod
    def parse(cls, spec: Union[str, tuple, "BlockSpec"]) -> "BlockSpec":
        if isinstance(spec, BlockSpec):
            return spec
        if isinstance(spec, str):
            return cls(name=spec)
        if isinstance(spec, tuple):
            return cls(name=spec[0], version=spec[1])
        raise TypeError(f"Unsupported block spec: {spec!r}")


class PromptingConfig(BaseModel):
    """Configuration for a single prompt (which blocks, in which order)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    blocks_dir: str = "data/prompts"
    blocks: List[Union[str, tuple, BlockSpec]]
    token_limit: int = 4096
    join_format: Literal["xml", "md", "plain"] = "xml"
    chat_template: Optional[Any] = None          # non-None => use tokenizer.apply_chat_template
    assistant_prefix: Optional[str] = None
    min_examples: int = 2                        # examples block must fit at least this many
    include_transformation_summary: bool = False


class PromptBuilder:
    """Composes a prompt string (or chat message list) from configured blocks."""

    def __init__(self, config: PromptingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.env = self._make_env()

    def _make_env(self) -> Environment:
        env = Environment(
            loader=FileSystemLoader(self.config.blocks_dir),
            undefined=StrictUndefined,   # KeyError on unknown variables
            trim_blocks=True,            # strip \n after {% block %} tags
            lstrip_blocks=True,          # strip leading whitespace before {% %}
            auto_reload=True,            # re-read .j2 files whose mtime changed
        )
        env.filters["grid"] = self._format_grid
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
            elif spec.name == "examples":
                rendered = self._build_examples(
                    task, budget=self.config.token_limit - used_tokens, context=context,
                )
                if rendered is None:
                    return None  # even min_examples didn't fit
            else:
                template = self.env.get_template(f"{spec.name}/{spec.version}.j2")
                rendered = template.render(**context)

            cost = self._count_tokens(rendered)
            if used_tokens + cost > self.config.token_limit:
                return None
            parts[spec.name] = (spec, rendered)
            used_tokens += cost

        return self._join(parts)

    def _build_examples(self, task, budget: int, context: dict) -> Optional[str]:
        template = self.env.get_template("examples/v1.j2")
        accumulated = ""
        accumulated_tokens = 0

        for idx, subtask in enumerate(task.subtasks):
            example_context = {
                **context,
                "idx": idx + 1,
                "input_grid": subtask.train_inp,
                "output_grid": subtask.train_out,
            }
            if self.config.include_transformation_summary:
                example_context["transformation_summary"] = self._transformation_summary(subtask)

            rendered = template.render(**example_context)
            cost = self._count_tokens(rendered)
            if accumulated_tokens + cost > budget:
                if idx < self.config.min_examples:
                    return None  # even the minimum didn't fit
                break
            accumulated += rendered
            accumulated_tokens += cost

        return accumulated

    def _transformation_summary(self, subtask) -> str:
        """Hook for project-specific example enrichment (e.g. a symbolic
        input/output diff via GridSummary). Override or monkeypatch this in
        the project layer once that logic is wired back in."""
        return ""

    def _count_tokens(self, text: str) -> int:
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
