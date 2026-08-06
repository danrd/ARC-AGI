"""Tests for subsymbolic/llm_setup.py's LlmConfig."""
from __future__ import annotations

from subsymbolic.llm_setup import LlmConfig


def test_tokenizer_model_defaults_to_none():
    config = LlmConfig()
    assert config.tokenizer_model is None


def test_tokenizer_model_can_be_set_separately_from_model():
    config = LlmConfig(model="unsloth/Qwen3.6-27B-GGUF", tokenizer_model="Qwen/Qwen3.6-27B")

    assert config.model == "unsloth/Qwen3.6-27B-GGUF"
    assert config.tokenizer_model == "Qwen/Qwen3.6-27B"
