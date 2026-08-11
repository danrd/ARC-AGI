"""System-level configuration.

ExperimentConfig aggregates one config per module (llm setup, generation
params, prompting, RL) plus `system` - the orchestration-wide settings
(iteration bounds, timeouts) previously scattered as bare dataclasses in
orchestration.__main__. subsymbolic.subsymbolic_module.SubsymbolicModule
takes the whole thing (using only `prompt` + `llm`/`generation` off of
it) - ExperimentConfig just gives solve_task() a single object to build
the whole system from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict

from rl.rl_module import RlConfig
from subsymbolic.llm_run import WandbLogConfig
from subsymbolic.llm_setup import BaseConfig, LlmConfig
from subsymbolic.llm_runtime import GenerationConfig
from subsymbolic.prompt_builder import PromptingConfig


@dataclass
class AgentRunConfig:
    """Execution settings for the agent-level (module) loop."""
    max_agent_iterations: int = 3
    rl_wait_timeout: float = 30.0
    verbose: bool = False


@dataclass
class SystemRunConfig:
    """Execution settings for the system-level (agent) loop."""
    max_system_iterations: int = 5
    agent_run_config: AgentRunConfig = field(default_factory=AgentRunConfig)
    verbose: bool = True


class ExperimentConfig(BaseModel):
    """Main config for guiding system setup and processing."""
    base: BaseConfig = Field(default_factory=BaseConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptingConfig = Field(default_factory=PromptingConfig)
    rl: RlConfig = Field(default_factory=RlConfig)
    system: SystemRunConfig = Field(default_factory=SystemRunConfig)
    logging: WandbLogConfig = Field(default_factory=WandbLogConfig)
    project: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_chat_template_kwargs(self) -> "ExperimentConfig":
        """chat_template_kwargs (e.g. Qwen3's enable_thinking) has to live
        on both generation (server-backed tiers - sent as extra_body) and
        prompt (local in-process tiers - baked into the prompt string via
        apply_chat_template), since those are two structurally different
        delivery points - see PromptingConfig.chat_template_kwargs /
        GenerationConfig.chat_template_kwargs. Callers shouldn't have to
        know that split: if only one side was set, mirror it onto the
        other so setting it once is enough. Leaves both alone if either
        both or neither were set explicitly."""
        if self.generation.chat_template_kwargs and not self.prompt.chat_template_kwargs:
            self.prompt.chat_template_kwargs = dict(self.generation.chat_template_kwargs)
        elif self.prompt.chat_template_kwargs and not self.generation.chat_template_kwargs:
            self.generation.chat_template_kwargs = dict(self.prompt.chat_template_kwargs)
        return self

    def to_llama_cpp(self) -> dict:
        return self.generation.to_llama_cpp(seed=self.base.seed)

    def to_vllm(self) -> dict:
        return self.generation.to_vllm(seed=self.base.seed)

    def to_hf(self) -> dict:
        return self.generation.to_hf(seed=self.base.seed)

    def to_chat_completions(self) -> dict:
        return self.generation.to_chat_completions(seed=self.base.seed)

    def dump(self):
        with open("exp.yaml", "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    def __str__(self) -> str:
        """Human-readable, nested view of every attribute - unlike pydantic's default single-line repr, meant to be read at a glance via print(config)."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_dict(cls, exp_params: dict) -> "ExperimentConfig":
        base = BaseConfig(**exp_params.get("base", {}))
        llm = LlmConfig(**exp_params.get("llm", {}))
        generation = GenerationConfig(**exp_params.get("generation", {}))
        prompt = PromptingConfig(**exp_params.get("prompt", {}))
        rl = RlConfig(**exp_params.get("rl", {}))
        return cls(base=base, llm=llm, generation=generation, prompt=prompt, rl=rl)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
