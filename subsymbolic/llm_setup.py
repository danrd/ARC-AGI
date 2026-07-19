import os
import subprocess
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def setup_model(config):
    """
    Start local OpenAI-compatible server:
    - cpu: llama-cpp-python server
    - gpu: vLLM server

    Returns:
        subprocess.Popen: started server process
    """
    env = os.environ.copy()
    device = config.base.device.lower()
    port = str(getattr(config.base, "port", 8001))

    if device == "cpu":
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "llama-cpp-python[server]",
        ])

        n_ctx = str(
            getattr(
                config.base,
                "n_ctx",
                getattr(config.generation, "max_tokens", 2048),
            )
        )

        log_file = open("llama_cpp.log", "w", encoding="utf-8")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "llama_cpp.server",
                "--model",
                config.base.model,
                "--port",
                port,
                "--use_mlock",
                "True",
                "--n_ctx",
                n_ctx,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        process.log_file = log_file
        return process

    if device == "gpu":
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "vllm",
        ])

        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        log_file = open("vllm_server.log", "w", encoding="utf-8")

        process = subprocess.Popen(
            [
                "vllm",
                "serve",
                config.base.model,
                "--port",
                port,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        process.log_file = log_file
        return process

    raise ValueError(f"Unsupported device: {config.base.device}")


def setup_hf_model(model_id):
    """
    Initialize a Hugging Face model and tokenizer.

    Args:
        model_id (str): Hugging Face model ID.

    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        use_cache=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # gradient_checkpointing_enable() нужен для обучения,
    # для инференса он обычно не нужен и может замедлять генерацию.

    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    return model, tokenizer


def setup_llama_cpp_model(model_path, config=None, tokenizer_id=None):
    """
    Setup quantized LLM inference using llama_cpp.

    Args:
        model_path (str): local path to GGUF model.
        config: optional config object.
        tokenizer_id (str | None): HF tokenizer ID, if tokenizer is needed.

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python not installed. "
            "Install with: pip install llama-cpp-python"
        ) from e

    tokenizer = None

    if tokenizer_id is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
            padding_side="right",
        )

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