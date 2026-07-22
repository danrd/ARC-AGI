import gc
import os
import random
import time

import torch
from openai import (
    OpenAI,
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
)
from transformers import (
    GenerationConfig,
)

def process_prompt(messages, config) -> str:
    """
    Process prompt using predefined local OpenAI-compatible server.
    """
    port = getattr(config.base, "port", 8001)

    client = OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed",
    )

    response = client.chat.completions.create(
        model=config.base.model,
        messages=messages,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )

    return response.choices[0].message.content


def process_prompt_with_model(
    prompt: str,
    model,
    tokenizer,
    generation_config=None,
    device: str = "cpu",
) -> str:
    """
    Process prompt with given model.

    Args:
        prompt (str): prompt to process.
        model: model to use.
        tokenizer: tokenizer for HF mode.
        generation_config: generation parameters.
        device (str): "cpu" for llama.cpp, otherwise HF generation.

    Returns:
        str: generation result.
    """
    generation_config = generation_config or {}

    if device == "cpu":
        if not isinstance(generation_config, dict):
            raise TypeError(
                "For llama.cpp generation_config must be a dict "
                "with llama_cpp-compatible parameters."
            )

        output = model(prompt, **generation_config)["choices"][0]["text"]

    else:
        if tokenizer is None:
            raise ValueError("tokenizer is required for HF generation")

        model.eval()

        model_device = next(model.parameters()).device

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model_device)

        with torch.no_grad():
            if isinstance(generation_config, GenerationConfig):
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
            elif isinstance(generation_config, dict):
                outputs = model.generate(
                    **inputs,
                    **generation_config,
                )
            else:
                raise TypeError(
                    "generation_config must be dict or transformers.GenerationConfig"
                )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]

        output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


def process_prompt_with_resilience(messages, config):
    """
    Return response trying different models from predefined OpenRouter list.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=2,
        timeout=30.0,
    )

    last_error = None
    models = config.base.openrouter_models

    for index, model in enumerate(models):
        try:
            print(f"🔄 Пробуем модель: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=getattr(config.generation, "temperature", 0.7),
                max_tokens=getattr(config.generation, "max_tokens", 1024),
            )

            print(f"✅ Успех с моделью: {model}")
            return response.choices[0].message.content

        except RateLimitError as e:
            wait_time = 2 ** (index + 1) + random.uniform(0, 1)
            print(
                f"⚠️ Превышен лимит запросов для {model}. "
                f"Ожидание {wait_time:.2f} сек..."
            )
            time.sleep(wait_time)
            last_error = e

        except (APIError, APITimeoutError, APIConnectionError, ConnectionError) as e:
            print(f"❌ Ошибка с моделью {model}: {type(e).__name__}")
            last_error = e
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Неизвестная ошибка с моделью {model}: {e}")
            last_error = e

    print(
        "💥 Критическая ошибка: ни одна модель не ответила. "
        f"Последняя ошибка: {last_error}"
    )

    return "Извините, сервис временно недоступен."
