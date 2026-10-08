"""System-level configuration.

ExperimentConfig aggregates one config per module (llm setup, generation
params, prompting, RL) plus `system` - the orchestration-wide settings
(iteration bounds, timeouts), which belong here rather than beside the
entry point that happens to read them first, so every caller assembling a
run finds them in the same place.
subsymbolic.subsymbolic_module.SubsymbolicModule
takes the whole thing (using only `prompt` + `llm`/`generation` off of
it) - ExperimentConfig just gives solve_task() a single object to build
the whole system from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml
from pydantic import Field, model_validator
from typing import Any, Dict, Optional

from rl.rl_module import RlConfig
from subsymbolic.llm_run import WandbLogConfig
from subsymbolic.llm_setup import BaseConfig, LlmConfig
from subsymbolic.local_config import LocalModelConfig, gpu_notebook_params
from subsymbolic.llm_runtime import GenerationConfig
from subsymbolic.prompt_builder import PromptingConfig


@dataclass
class AgentRunConfig:
    """Execution settings for the agent-level (module) loop."""
    max_agent_iterations: int = 3
    # The one wait for the background RL job after the model first answers.
    # A run at 100k steps took 85-522 s (median 183 s) over 36 runs, so a
    # shorter wait cancels RL on almost every task before it can answer. The
    # value follows the speed of the final RL step; revisit it when that moves.
    rl_wait_timeout: float = 600.0
    verbose: bool = False


@dataclass
class OrchestrationOptions:
    """How the system decides, on top of the deterministic rule that is its default.

    Every field's default is the system as it was before: the rule decides, the model is asked
    once, nothing is added to its prompt. Each option adds something and none removes the rule -
    it stays the fallback whenever the added piece cannot answer (orchestration.assemble)."""
    decision: str = "deterministic"      # "llm": a model chooses among the open actions (the rule still validates)
    coordinator: str = "deterministic"   # "llm": a model picks the next agent when one fails
    refine_rounds: int = 0               # > 1: the model is asked again with its answer and what was wrong, up to this many calls
    feedback: bool = False               # the graph's own retries carry the earlier answers and what was wrong with them
    info_tools: bool = False             # the model may reply REQUEST: summary / search_hints before it answers
    max_info_requests: int = 1


@dataclass
class SystemRunConfig:
    """Execution settings for the system-level (agent) loop."""
    max_system_iterations: int = 5
    agent_run_config: AgentRunConfig = field(default_factory=AgentRunConfig)
    orchestration: OrchestrationOptions = field(default_factory=OrchestrationOptions)
    verbose: bool = True


class ExperimentConfig(LocalModelConfig):
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

    def to_wandb_config(self) -> Dict[str, Any]:
        """LLM-relevant slice for subsymbolic.llm_run.run_llm_over_tasks'
        extra_config= - model identity/serving knobs plus sampling
        params, so a run's own wandb metadata records what was actually
        queried instead of a free-text description."""
        return {**self.llm.model_dump(), **self.generation.model_dump()}

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
        logging = WandbLogConfig(**exp_params.get("logging", {}))
        system_params = dict(exp_params.get("system", {}))
        agent_run_config = AgentRunConfig(**system_params.pop("agent_run_config", {}))
        orchestration = OrchestrationOptions(**system_params.pop("orchestration", {}))
        system = SystemRunConfig(agent_run_config=agent_run_config, orchestration=orchestration, **system_params)
        return cls(base=base, llm=llm, generation=generation, prompt=prompt, rl=rl,
                    logging=logging, system=system)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def local_experiment(model: str, tokenizer: Optional[str] = None, max_tokens: int = 1200) -> ExperimentConfig:
    """An ExperimentConfig for one model as the GPU notebooks load it (subsymbolic.local_config)."""
    return ExperimentConfig.from_dict(gpu_notebook_params(model, tokenizer, max_tokens))
