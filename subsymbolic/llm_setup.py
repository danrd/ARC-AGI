"""Config describing which LLM/backend to set up: device, framework, model
identity. Generation-time parameters (temperature, max_tokens, ...) live in
subsymbolic.llm_runtime.GenerationConfig instead - this is "what to load",
not "how to sample from it".
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict


class LlmConfig(BaseModel):
    """Setup config specifing all meta parameters for system functionality."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)
    seed: int = 42
    checkpoint_interval: int = 1  # number of examples to process before printing relevant info
    device: str = 'cpu'
    framework: str = 'llama_cpp'  # llama_cpp | vllm | hf
    model: str = 'unsloth/Qwen3.6-27B-GGUF'
    quant_file: str = 'Qwen3.6-27B-Q4_K_M.gguf'
    max_context: int = 9000  # llm token limit for computational resources to control
    openrouter_models: List[str] = ["google/gemma-4-26b-a4b-it",
                                    "nvidia/nemotron-3-ultra-550b-a55b"]
