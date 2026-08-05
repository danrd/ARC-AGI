"""Tests for orchestration/configs.py."""
from __future__ import annotations

from orchestration.configs import ExperimentConfig


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
