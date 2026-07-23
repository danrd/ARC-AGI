"""System-level configuration.

ExperimentConfig aggregates one config per module (llm setup, generation
params, prompting, RL) plus `system` - the orchestration-wide settings
(iteration bounds, timeouts) previously scattered as bare dataclasses in
orchestration.__main__. Each piece can still be built and passed around on
its own (e.g. subsymbolic.subsymbolic_module.SubsymbolicModule only needs
`prompt` + `llm`/`generation`) - ExperimentConfig just gives solve_task() a
single object to build the whole system from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field
from typing import Any, Dict

from rl.rl_module import RlConfig
from subsymbolic.llm_setup import LlmConfig
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
    base: LlmConfig = Field(default_factory=LlmConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptingConfig = Field(default_factory=PromptingConfig)
    rl: RlConfig = Field(default_factory=RlConfig)
    system: SystemRunConfig = Field(default_factory=SystemRunConfig)
    project: Dict[str, Any] = Field(default_factory=dict)

    def to_llama_cpp(self) -> dict:
        return self.generation.to_llama_cpp(seed=self.base.seed)

    def to_vllm(self) -> dict:
        return self.generation.to_vllm(seed=self.base.seed)

    def to_hf(self) -> dict:
        return self.generation.to_hf(seed=self.base.seed)

    def to_open_router(self) -> dict:
        return self.generation.to_open_router(seed=self.base.seed)

    def dump(self):
        with open("exp.yaml", "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    @classmethod
    def from_dict(cls, exp_params: dict) -> "ExperimentConfig":
        base = LlmConfig(**exp_params.get("base", {}))
        generation = GenerationConfig(**exp_params.get("generation", {}))
        prompt = PromptingConfig(**exp_params.get("prompt", {}))
        rl = RlConfig(**exp_params.get("rl", {}))
        return cls(base=base, generation=generation, prompt=prompt, rl=rl)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
