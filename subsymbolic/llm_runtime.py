"""
Universal LLM inference runner.

One interface regardless of backend: `Runner.generate(prompt: str) -> str`.
`prompt` is always a plain string (whatever PromptBuilder already produces,
with or without a chat template) — the string-to-chat-messages translation
happens only at the one boundary that actually needs it (ServerRunner /
OpenRouterRunner talking to an OpenAI-compatible endpoint), not in every
caller.

`build_runner(config)` picks a backend for LOCAL inference per
config.base.device, with a fallback chain — since a server (vLLM, or even
llama.cpp) may fail to start:
    CPU:  llama.cpp server -> llama.cpp in-process
    GPU:  vLLM server -> vLLM in-process -> HF in-process (4-bit)
Every tier's error is collected; if all tiers fail, RuntimeError chains them.

Hosted/proprietary models (OpenRouter, and in principle OpenAI/Anthropic/
Gemini) are a deliberately SEPARATE, explicit path (`OpenRouterRunner`) —
not merged into build_runner. Whether to use local inference or a hosted
model is the caller's decision, not something to infer from config.

Heavy dependencies (torch, transformers, llama_cpp, vllm, openai) are
imported lazily inside whichever class/function actually needs them, so
importing this module — or building a runner for one backend — never
requires every other backend's library to be installed.
"""
from __future__ import annotations

import gc
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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

    def to_dict(self, exclude_none: bool = True, exclude_unset: bool = False) -> Dict[str, Any]:
        """Plain dict for **kwargs unpacking."""
        return self.model_dump(exclude_none=exclude_none, exclude_unset=exclude_unset)

    def merge(self, overrides: Dict[str, Any]) -> "GenerationConfig":
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
        )

    def to_hf(self, seed: int) -> dict:
        """Prepare generation config for HuggingFace Transformers."""
        from transformers import set_seed
        set_seed(seed)

        if self.use_beam_search:
            if self.temperature != 0.0:
                raise ValueError(
                    "HF beam search should be used with temperature=0.0 / do_sample=False."
                )

            params = {
                "max_new_tokens": self.max_tokens,
                "do_sample": False,
                "num_beams": self.best_of,
                "num_return_sequences": 1,
                "early_stopping": True,
                "repetition_penalty": self.repetition_penalty,
                "stop_strings": self.stop
            }
            return params

        do_sample = self.temperature > 0.0

        params = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": self.repetition_penalty,
        }

        if do_sample:
            params.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k if self.top_k != -1 else 0,
                }
            )

        if self.stop:
            params["stop_strings"] = self.stop

        return params

    def to_open_router(self, seed: int) -> dict:
        """Prepare generation config for OpenRouter Chat Completions API."""
        if self.use_beam_search:
            raise ValueError(
                "OpenRouter API does not support beam search via `use_beam_search`."
            )
        if self.best_of != 1:
            raise ValueError(
                "OpenRouter API does not support `best_of` in the same way as vLLM. "
                "Use best_of=1."
            )

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop if self.stop else None,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.top_k != -1:
            params["top_k"] = self.top_k

        if seed is not None:
            params["seed"] = seed

        return {k: v for k, v in params.items() if v is not None}


class AllModelsFailedError(Exception):
    """Raised when every model in a resilience chain (e.g. OpenRouter's
    model list) failed — never silently swallowed into a fake "sorry"
    string that could be mistaken for a real answer downstream."""


class BaseRunner:
    """Common interface for every backend. Use as a context manager to
    guarantee server processes / GPU memory are cleaned up:
        with build_runner(config) as runner:
            text = runner.generate(prompt)
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Server-backed runner (llama.cpp-server or `vllm serve`, OpenAI-compatible)
# ---------------------------------------------------------------------------

class ServerRunner(BaseRunner):
    """Wraps a local OpenAI-compatible HTTP server. This is the one place a
    plain prompt string gets wrapped into a single-turn chat message — every
    other backend just consumes the string directly."""

    def __init__(self, process: Optional[subprocess.Popen], port: int,
                 model_name: str, generation_kwargs: Dict[str, Any], client=None):
        self.process = process
        self.port = port
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs
        self._client = client  # allows injecting a fake client for testing

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="not-needed")
        return self._client

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **self.generation_kwargs,
        )
        return response.choices[0].message.content

    def close(self) -> None:
        if self.process is not None:
            _terminate_process(self.process)
            self.process = None


# ---------------------------------------------------------------------------
# In-process backends (no HTTP server)
# ---------------------------------------------------------------------------

class LlamaCppRunner(BaseRunner):
    """Wraps an in-process llama_cpp.Llama instance."""

    def __init__(self, model, generation_kwargs: Dict[str, Any]):
        self.model = model
        self.generation_kwargs = generation_kwargs

    def generate(self, prompt: str) -> str:
        return self.model(prompt, **self.generation_kwargs)["choices"][0]["text"]

    def close(self) -> None:
        self.model = None
        gc.collect()


class VLLMRunner(BaseRunner):
    """Wraps an in-process vllm.LLM instance."""

    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def close(self) -> None:
        self.llm = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFRunner(BaseRunner):
    """Wraps a plain transformers model + tokenizer (e.g. 4-bit via
    bitsandbytes) — the last-resort tier on GPU."""

    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate(self, prompt: str) -> str:
        import torch
        from transformers import GenerationConfig

        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            if isinstance(self.generation_config, GenerationConfig):
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(**inputs, **self.generation_config)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def close(self) -> None:
        self.model = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Hosted / proprietary models — separate, explicit path (not part of
# build_runner's local cpu/gpu selection; the caller opts into this).
# ---------------------------------------------------------------------------

class OpenRouterRunner(BaseRunner):
    """Tries a list of OpenRouter models in order, with backoff on rate
    limits. Raises AllModelsFailedError if every model failed — no silent
    "service unavailable" string standing in for a real answer."""

    def __init__(self, models: List[str], generation_kwargs: Dict[str, Any],
                 api_key: Optional[str] = None, max_retries: int = 2,
                 timeout: float = 30.0, client=None):
        self.models = models
        self.generation_kwargs = generation_kwargs
        self._client = client
        if self._client is None:
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                                   max_retries=max_retries, timeout=timeout)

    def generate(self, prompt: str) -> str:
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

        messages = [{"role": "user", "content": prompt}]
        last_error: Optional[BaseException] = None

        for index, model in enumerate(self.models):
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=messages, **self.generation_kwargs,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                wait_time = 2 ** (index + 1) + random.uniform(0, 1)
                time.sleep(wait_time)
                last_error = e

            except (APIError, APITimeoutError, APIConnectionError, ConnectionError) as e:
                time.sleep(1)
                last_error = e

            except Exception as e:  # noqa: BLE001 - deliberately broad: keep trying remaining models
                last_error = e

        raise AllModelsFailedError(f"All {len(self.models)} OpenRouter models failed: {last_error}")


# ---------------------------------------------------------------------------
# Server startup + health check + cleanup
# ---------------------------------------------------------------------------

def _wait_for_server_ready(process: subprocess.Popen, port: int,
                            timeout: float = 60.0, interval: float = 1.0) -> bool:
    """Poll the OpenAI-compatible /v1/models endpoint until it answers, the
    server process dies, or timeout is hit."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            return False  # process already exited — no point polling further
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(interval)
    return False


def _terminate_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log_file = getattr(process, "log_file", None)
    if log_file is not None:
        log_file.close()


def _start_llama_cpp_server(config) -> subprocess.Popen:
    # TODO: installing at runtime is convenient but slow and non-reproducible;
    # move to requirements.txt when this settles.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python[server]"])

    port = getattr(config.base, "port", 8001)
    n_ctx = str(getattr(config.base, "n_ctx", getattr(config.generation, "max_tokens", 2048)))
    log_file = open("llama_cpp.log", "w", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", "--model", config.base.model,
         "--port", str(port), "--use_mlock", "True", "--n_ctx", n_ctx],
        stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    process.log_file = log_file
    return process


def _start_vllm_server(config) -> subprocess.Popen:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "vllm"])

    port = getattr(config.base, "port", 8001)
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log_file = open("vllm_server.log", "w", encoding="utf-8")

    process = subprocess.Popen(
        ["vllm", "serve", config.base.model, "--port", str(port)],
        stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    process.log_file = log_file
    return process


# ---------------------------------------------------------------------------
# In-process backend construction
# ---------------------------------------------------------------------------

def setup_hf_model(model_id: str):
    """Initialize an HF causal LM in 4-bit (bitsandbytes) + its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, torch_dtype=compute_dtype,
        use_cache=True, device_map="auto", trust_remote_code=True,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
    return model, tokenizer


def setup_llama_cpp_model(model_path: str, config=None, tokenizer_id: Optional[str] = None):
    """In-process llama.cpp, using the same GGUF file the server tier would
    have used — the CPU fallback tier when the server fails to come up."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python") from e

    tokenizer = None
    if tokenizer_id is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    base_cfg = getattr(config, "base", config) if config is not None else object()
    gen_cfg = getattr(config, "generation", config) if config is not None else object()

    model = Llama(
        model_path=model_path,
        n_ctx=getattr(base_cfg, "n_ctx", getattr(gen_cfg, "max_tokens", 2048)),
        n_batch=getattr(base_cfg, "n_tokens_batch", 512),
        use_mlock=getattr(base_cfg, "use_mlock", True),
        n_gpu_layers=getattr(base_cfg, "n_gpu_layers", 0),
        verbose=getattr(base_cfg, "verbose", False),
    )
    return model, tokenizer


def _hf_generation_config(config):
    from transformers import GenerationConfig
    gen = config.generation
    temperature = getattr(gen, "temperature", 0.7)
    return GenerationConfig(
        max_new_tokens=getattr(gen, "max_tokens", 512),
        temperature=temperature,
        do_sample=temperature > 0,
    )


def _llama_cpp_generation_kwargs(config) -> Dict[str, Any]:
    gen = config.generation
    return {"max_tokens": getattr(gen, "max_tokens", 512), "temperature": getattr(gen, "temperature", 0.7)}


def _vllm_sampling_params(config):
    from vllm import SamplingParams
    gen = config.generation
    return SamplingParams(max_tokens=getattr(gen, "max_tokens", 512), temperature=getattr(gen, "temperature", 0.7))


def _server_generation_kwargs(config) -> Dict[str, Any]:
    gen = config.generation
    return {"max_tokens": getattr(gen, "max_tokens", 512), "temperature": getattr(gen, "temperature", 0.7)}


# ---------------------------------------------------------------------------
# Factory: local inference, with fallback chain
# ---------------------------------------------------------------------------

def build_runner(config) -> BaseRunner:
    """Build a local inference runner per config.base.device, falling back
    through progressively simpler backends if a tier fails to start:
        CPU:  llama.cpp server -> llama.cpp in-process
        GPU:  vLLM server -> vLLM in-process -> HF in-process (4-bit)
    Raises RuntimeError (chaining every tier's error) if all tiers fail.
    """
    device = config.base.device.lower()
    if device == "cpu":
        return _build_cpu_runner(config)
    if device == "gpu":
        return _build_gpu_runner(config)
    raise ValueError(f"Unsupported device: {config.base.device}")


def _build_cpu_runner(config) -> BaseRunner:
    errors = []
    port = getattr(config.base, "port", 8001)

    try:
        process = _start_llama_cpp_server(config)
        if _wait_for_server_ready(process, port):
            return ServerRunner(process, port, config.base.model, _server_generation_kwargs(config))
        _terminate_process(process)
        errors.append("llama.cpp server: failed health check")
    except Exception as e:
        errors.append(f"llama.cpp server: {type(e).__name__}: {e}")

    try:
        model, _ = setup_llama_cpp_model(config.base.model, config=config)
        return LlamaCppRunner(model, _llama_cpp_generation_kwargs(config))
    except Exception as e:
        errors.append(f"llama.cpp in-process: {type(e).__name__}: {e}")

    raise RuntimeError("All CPU backends failed:\n" + "\n".join(errors))


def _build_gpu_runner(config) -> BaseRunner:
    errors = []
    port = getattr(config.base, "port", 8001)

    try:
        process = _start_vllm_server(config)
        if _wait_for_server_ready(process, port):
            return ServerRunner(process, port, config.base.model, _server_generation_kwargs(config))
        _terminate_process(process)
        errors.append("vLLM server: failed health check")
    except Exception as e:
        errors.append(f"vLLM server: {type(e).__name__}: {e}")

    try:
        from vllm import LLM
        llm = LLM(model=config.base.model)
        return VLLMRunner(llm, _vllm_sampling_params(config))
    except Exception as e:
        errors.append(f"vLLM in-process: {type(e).__name__}: {e}")

    try:
        model, tokenizer = setup_hf_model(config.base.model)
        return HFRunner(model, tokenizer, _hf_generation_config(config))
    except Exception as e:
        errors.append(f"HF in-process: {type(e).__name__}: {e}")

    raise RuntimeError("All GPU backends failed:\n" + "\n".join(errors))
