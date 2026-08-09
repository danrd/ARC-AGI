"""Tests for orchestration/configs.py."""
from __future__ import annotations

from orchestration.configs import ExperimentConfig
from subsymbolic.llm_runtime import GenerationConfig
from subsymbolic.prompt_builder import PromptingConfig


def test_chat_template_kwargs_syncs_from_generation_to_prompt():
    config = ExperimentConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.prompt.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_syncs_from_prompt_to_generation():
    config = ExperimentConfig(prompt=PromptingConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.generation.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_untouched_when_neither_side_set():
    config = ExperimentConfig()

    assert config.prompt.chat_template_kwargs == {}
    assert config.generation.chat_template_kwargs == {}


def test_chat_template_kwargs_left_alone_when_both_sides_set_independently():
    config = ExperimentConfig(
        generation=GenerationConfig(chat_template_kwargs={"a": 1}),
        prompt=PromptingConfig(chat_template_kwargs={"b": 2}),
    )

    assert config.generation.chat_template_kwargs == {"a": 1}
    assert config.prompt.chat_template_kwargs == {"b": 2}


def test_chat_template_kwargs_sync_does_not_alias_the_dict():
    config = ExperimentConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    config.prompt.chat_template_kwargs["extra"] = True

    assert "extra" not in config.generation.chat_template_kwargs


def test_str_is_readable_nested_yaml_not_the_default_single_line_repr():
    config = ExperimentConfig()

    text = str(config)

    assert "base:" in text
    assert "generation:" in text
    assert "prompt:" in text
    assert "\n" in text  # multi-line, unlike pydantic's default repr


def test_str_reflects_field_overrides():
    config = ExperimentConfig()
    config.base.seed = 123

    assert "123" in str(config)


def test_repr_is_left_untouched():
    config = ExperimentConfig()

    assert repr(config).startswith("ExperimentConfig(")
