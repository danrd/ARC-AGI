"""Tests for subsymbolic/prompt_builder.py's registry injection.

resolvers/filters used to come from a hardcoded import of
subsymbolic.registry (this project's ARC-specific resolver/filter
lookup), which secretly coupled a module whose whole point is to be
domain-agnostic to this one project - anyone trying to reuse PromptBuilder
elsewhere would have pulled in ARC-specific code they can't use.
resolver_registry/filter_registry are now constructor params instead;
subsymbolic_module.py is what wires in this project's registry.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


def _write_block(blocks_dir, name, version, content):
    block_dir = blocks_dir / name
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / f"{version}.j2").write_text(content)


def test_prompt_builder_module_has_no_project_specific_imports():
    """Regression guard for the coupling this change removed: prompt_builder.py
    must not import anything from this project's ARC-specific modules -
    that defeats the point of making the registries injectable."""
    import subsymbolic.prompt_builder as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    project_specific = {m for m in imported if m.startswith(("subsymbolic.registry", "subsymbolic.arc_"))}
    assert not project_specific, f"prompt_builder.py still imports project-specific modules: {project_specific}"


def test_builds_without_any_registry_when_config_uses_no_resolvers_or_filters(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100)

    builder = PromptBuilder(config, _FakeTokenizer())
    result = builder.build(task=None, context={"name": "world"})

    assert result == "Hello, world!"


def test_resolver_registry_is_used_when_provided(tmp_path):
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["dynamic"], token_limit=100,
                              resolvers=["dynamic"])

    def my_resolver(task, remaining_tokens, context, builder):
        return f"resolved: {task}"

    builder = PromptBuilder(config, _FakeTokenizer(), resolver_registry={"dynamic": my_resolver})
    result = builder.build(task="my-task", context={})

    assert result == "resolved: my-task"


def test_filter_registry_is_used_when_provided(tmp_path):
    _write_block(tmp_path, "shout", "v1", "{{ text | shout }}")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["shout"], token_limit=100,
                              filters=["shout"])

    builder = PromptBuilder(config, _FakeTokenizer(), filter_registry={"shout": str.upper})
    result = builder.build(task=None, context={"text": "hi"})

    assert result == "HI"


def test_unregistered_resolver_name_raises_keyerror(tmp_path):
    """No registry passed -> empty default -> a resolver name not found in
    it should fail loudly (KeyError), not silently skip or crash later
    with a confusing error deep inside build()."""
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["missing"], token_limit=100,
                              resolvers=["missing"])

    with pytest.raises(KeyError):
        PromptBuilder(config, _FakeTokenizer())
