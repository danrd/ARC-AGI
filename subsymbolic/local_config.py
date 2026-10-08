"""A model loaded the way the GPU notebooks load it, without the rest of an experiment.

`build_runner` reads three parts of a config - the technical settings, which model, how it
generates - and the four `to_*` views of generation. An ExperimentConfig has those and a good
deal more (the prompt, RL, the system); a script that only wants to put a prompt to a model has
no use for the rest and, living a layer below orchestration, may not import it. This is the part
the two share: ExperimentConfig is a LocalModelConfig with more fields.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from subsymbolic.llm_runtime import GenerationConfig
from subsymbolic.llm_setup import BaseConfig, LlmConfig
from subsymbolic.utils import build_grid_grammar


class LocalModelConfig(BaseModel):
    base: BaseConfig = Field(default_factory=BaseConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    def to_llama_cpp(self) -> dict:
        return self.generation.to_llama_cpp(seed=self.base.seed)

    def to_vllm(self) -> dict:
        return self.generation.to_vllm(seed=self.base.seed)

    def to_hf(self) -> dict:
        return self.generation.to_hf(seed=self.base.seed)

    def to_chat_completions(self, grammar_backend: str = "llama_cpp") -> dict:
        return self.generation.to_chat_completions(seed=self.base.seed, grammar_backend=grammar_backend)


def gpu_notebook_params(model: str, tokenizer: Optional[str] = None, max_tokens: int = 1200) -> Dict[str, Any]:
    """The settings of the GPU notebooks for one model, as the dict ExperimentConfig.from_dict reads:
    llama.cpp with every layer on the cards and flash attention, greedy, thinking off, and the output
    held to the grid grammar. `model` is the GGUF repository and the quantisation file joined by a colon."""
    repo, quant_file = model.split(":")
    return {
        "base": {"device": "cpu", "server_ready_timeout": 1000.0, "request_timeout": 6000.0},
        "llm": {"framework": "llama_cpp", "model": repo, "quant_file": quant_file, "tokenizer_model": tokenizer,
                "max_context": 10000, "n_gpu_layers": 999, "flash_attn": True, "use_mlock": False},
        "generation": {"temperature": 0.0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "grammar": build_grid_grammar(colors_str=False)},
    }


def local_model_config(model: str, tokenizer: Optional[str] = None, max_tokens: int = 1200) -> LocalModelConfig:
    params = gpu_notebook_params(model, tokenizer, max_tokens)
    return LocalModelConfig(base=BaseConfig(**params["base"]), llm=LlmConfig(**params["llm"]),
                            generation=GenerationConfig(**params["generation"]))
