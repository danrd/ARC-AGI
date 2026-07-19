import yaml
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Literal

class BaseConfig(BaseModel):
    """Setup config specifing all meta parameters for system functionality."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)
    seed: int = 42 
    checkpoint_interval: int = 1 # number of examples to process before printing relevant info
    device: str = 'cpu' 
    model: str = 'unsloth/Qwen3.6-27B-GGUF'
    quant_file: str = 'Qwen3.6-27B-Q4_K_M.gguf'
    max_context: int = 9000 # llm token limit for computational resources to control
    openrouter_models: List[str] = ["google/gemma-4-26b-a4b-it", 
                                    "nvidia/nemotron-3-ultra-550b-a55b",]

class GenerationConfig(BaseModel):
    """Base framework-agnostic config specifying key llm generation parameters."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, 
                               description="Scales logit distribution before softmax. 0.0 = greedy (argmax). < 1.0 = sharper, > 1.0 = flatter.")
    
    max_tokens:  int   = Field(default=256,  ge=1,
                               description="Maximum number of tokens to generate.")
    
    top_p:       float = Field(default=1.0,  ge=0.0, le=1.0,
                               description="Nucleus sampling: keep smallest token set whose cumulative probability ≥ top_p. 1.0 = disabled.")
    
    top_k:       int   = Field(default=-1,   ge=-1,
                               description="Sample from top-k most probable tokens. -1 = disabled.")
    
    stop:        List[str] = Field(default_factory=list,
                                   description="Stop generation immediately when any of these strings is produced.")
    
    repetition_penalty: float = Field(default=1.0, ge=0.0,
                                      description="Multiplicative penalty on previously generated tokens. 1.0 = no penalty, > 1.0 = penalise repetition.")
    
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0,
                                     description="Additive logit penalty scaled by token frequency in context. Positive = reduce repetition.")
                                     
    use_beam_search: bool = Field(default=False,
                                   description="Use beam search instead of sampling. Requires temperature=0.0.")
    
    best_of: int = Field(default=1, ge=1,
                           description="Generate best_of candidates, return the best n. Must be ≥ n. Required for beam search.")
            
    def to_dict(self, exclude_none: bool = True,exclude_unset: bool = False,) -> Dict[str, Any]:
        """Plain dict for **kwargs unpacking."""
        return self.model_dump(exclude_none=exclude_none, exclude_unset=exclude_unset)

    def merge(self, overrides: Dict[str, Any]) -> BaseConfig:
        """Return a new config with override values applied (non-mutating)."""
        return self.model_copy(update=overrides)
                            
    def to_llama_cpp(self, seed: int) -> dict:
        """Prepare generation config for llama_cpp framework using a set of defaults parameters."""
        return {
            "temperature":        self.temperature,
            "max_tokens":         self.max_tokens,
            "top_p":              self.top_p,
            "top_k":              self.top_k if self.top_k != -1 else 0,
            "seed":               seed,
            "stop":               self.stop,
            "repeat_penalty":     self.repetition_penalty,
            # defaults
            "repeat_last_n":      64,
            "penalize_nl":        True,
            "echo":               False,
        }
    
    def to_vllm(self, seed: int):
        """Prepare generation config for vllm framework using a set of defaults parameters."""
        from vllm import SamplingParams
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=seed,
            stop=self.stop,
            repetition_penalty=self.repetition_penalty,
            # defaults
        )

class BlockSpec(BaseModel):
    """Prompt block specification."""
    name: str
    version: str = "v1"
    role: Literal["system", "user"] = "user" # role for chat template
    tag: Optional[str] = None # specify tags for wrapping cusomization

    @classmethod
    def parse(cls, spec: "str | tuple | Blockspec"):
        if isinstance(spec, str):
            return cls(name=spec)
        if isinstance(spec, tuple):
            return cls(name=spec[0], version=spec[1])
        return spec

class PromptingConfig(BaseModel):
    """Config to guide prompt construction."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)
    blocks: List[BlockSpec|str] = ["general_instruction", "examples", "output_format"] # list of element types to compose prompt
    block_overrides: Optional[Dict[str, str]] = Field(default_factory=dict) # specific blocks subsitution while experimenting
    token_limit: int = 9000 # resources management 
    filters: Optional[List[str]] = None # filters to set up custom functionality for each project
    join_format: Literal["xml", "md", "plain"] = "xml" # approach for blocks composing
    chat_template: Optional[str] = None # optionaly use specific chat template
    assistant_prefix: Optional[str] = None # string to add before assistant response

class ExperimentConfig(BaseModel):
    base: BaseConfig = Field(default_factory=BaseConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptingConfig = Field(default_factory=PromptingConfig)

    def to_llama_cpp(self) -> dict:
        return self.generation.to_llama_cpp(seed=self.base.seed)
    
    def to_vllm(self) -> dict:
        return self.generation.to_vllm(seed=self.base.seed)

    def dump(self):
        with open("exp.yaml", "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
    
    @classmethod
    def from_dict(cls, exp_params: dict) -> "ExperimentConfig":
        base = BaseConfig(**exp_params.get("base", {}))
        generation = GenerationConfig(**exp_params.get("generation", {}))
        prompt = PromptingConfig(**exp_params.get("prompt", {}))
        return cls(base=base, generation=generation, prompt=prompt)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))