"""End-to-end smoke test: build a prompt for one ARC task in a handful of
representative PromptingConfig variants, and run it through the real
prompt-building -> inference pipeline (subsymbolic.prompt_builder ->
subsymbolic.llm_runtime.build_runner).

Scope, deliberately: not every block/config combination - just the ones most
likely to exercise code paths that could actually break (each join_format,
a minimal vs. a fuller block set). Not an accuracy benchmark: the CPU model
is 51M params, the GPU candidates are 1-8M - too small to solve ARC. Success
here means the pipeline ran without raising and produced *some* text, not
that the text is a correct answer.

GPU variants are opt-in only (ARC_TEST_GPU=1) - see conftest.gpu_enabled.
"""
from __future__ import annotations

import pytest

from subsymbolic.configs import BaseConfig, ExperimentConfig, GenerationConfig
from subsymbolic.llm_runtime import build_runner
from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.utils import parse_llm_output

from .conftest import GPU_MODEL_CANDIDATES, gpu_enabled

MINIMAL_BLOCKS = ["general_instruction", "examples", "output_format"]
FULLER_BLOCKS = [
    "role_instruction", "grid_description", "general_instruction",
    "examples", "task_repr", "output_format",
]

PROMPT_CONTEXT = {
    "color_mapping_text": (
        "0=black, 1=blue, 2=red, 3=green, 4=yellow, "
        "5=gray, 6=magenta, 7=orange, 8=sky, 9=brown"
    ),
    "role_text": "You are participating in the ARC-AGI benchmark.",
    "grid_repr_type": "concise",
}

# (id, blocks, join_format) - the "most obvious use cases" per block-set x
# join_format, not a full cross product. chat_template isn't covered here:
# it needs a real tokenizer with `apply_chat_template`, which ApproxTokenizer
# (conftest.py) intentionally doesn't provide - add a case once one of the
# candidate models' tokenizers is confirmed to ship a chat template.
PROMPT_CONFIG_CASES = [
    ("minimal-xml", MINIMAL_BLOCKS, "xml"),
    ("minimal-md", MINIMAL_BLOCKS, "md"),
    ("minimal-plain", MINIMAL_BLOCKS, "plain"),
    ("fuller-xml", FULLER_BLOCKS, "xml"),
]


def _make_config(blocks, join_format) -> PromptingConfig:
    return PromptingConfig(
        blocks=blocks,
        join_format=join_format,
        token_limit=4096,
        min_examples=1,
        filters=["grid"],
        resolvers=["examples"],
    )


def _make_builder(config: PromptingConfig, tokenizer) -> PromptBuilder:
    return PromptBuilder(config, tokenizer)


def _context_for(task) -> dict:
    """PROMPT_CONTEXT plus the one value that's inherently per-task
    (test_input_grid, consumed by task_repr/v1.j2's `grid` filter)."""
    return {**PROMPT_CONTEXT, "test_input_grid": task.test_subtask.train_inp}


@pytest.mark.parametrize("case_id,blocks,join_format", PROMPT_CONFIG_CASES)
def test_prompt_builds_without_a_model(case_id, blocks, join_format, arc_task, tiny_tokenizer):
    """Cheap check with no model involved: does prompt assembly itself
    survive this config (missing context keys, token-budget edge cases,
    unresolved blocks, join_format handling)?"""
    config = _make_config(blocks, join_format)
    builder = _make_builder(config, tiny_tokenizer)

    prompt = builder.build(arc_task, context=_context_for(arc_task))

    assert prompt is not None, f"[{case_id}] prompt didn't fit token_limit"
    assert isinstance(prompt, str) and prompt.strip()


@pytest.fixture(scope="session")
def cpu_runner(supra_router_gguf_path):
    base = BaseConfig(device="cpu", framework="llama_cpp", model=supra_router_gguf_path)
    generation = GenerationConfig(max_tokens=64, temperature=0.0)
    config = ExperimentConfig(base=base, generation=generation)
    with build_runner(config) as runner:
        yield runner


@pytest.mark.parametrize("case_id,blocks,join_format", PROMPT_CONFIG_CASES)
def test_cpu_llm_smoke(case_id, blocks, join_format, arc_task, tiny_tokenizer, cpu_runner):
    """Full pipeline on CPU: build the prompt, run it through the tiny GGUF
    model, and make sure the output can at least be handed to the grid
    parser without it raising (an empty/unparsed result is an acceptable
    "didn't solve it" outcome for a 51M model - not a test failure)."""
    config = _make_config(blocks, join_format)
    builder = _make_builder(config, tiny_tokenizer)
    prompt = builder.build(arc_task, context=_context_for(arc_task))
    assert prompt is not None, f"[{case_id}] prompt didn't fit token_limit"

    output = cpu_runner.generate(prompt)
    assert isinstance(output, str) and output != "", f"[{case_id}] model produced no output"

    parsed = parse_llm_output(output)  # "" on unparsed output - not an error
    assert parsed is not None


@pytest.fixture(scope="session")
def gpu_runner():
    if not gpu_enabled():
        pytest.skip("GPU tests are opt-in: set ARC_TEST_GPU=1 to run")

    errors = []
    for model_id in GPU_MODEL_CANDIDATES:
        try:
            base = BaseConfig(device="gpu", framework="hf", model=model_id)
            generation = GenerationConfig(max_tokens=64, temperature=0.0)
            config = ExperimentConfig(base=base, generation=generation)
            runner = build_runner(config)
        except Exception as e:  # try the next candidate
            errors.append(f"{model_id}: {type(e).__name__}: {e}")
            continue
        try:
            yield runner
        finally:
            runner.close()
        return

    pytest.skip("No GPU model candidate loaded:\n" + "\n".join(errors))


@pytest.mark.gpu
def test_gpu_llm_smoke(arc_task, tiny_tokenizer, gpu_runner):
    """Same pipeline as test_cpu_llm_smoke, on whichever GPU model candidate
    actually loaded (see GPU_MODEL_CANDIDATES in conftest.py). One config is
    enough here - this is about the GPU backend path itself, the config
    sweep is already covered on CPU."""
    config = _make_config(MINIMAL_BLOCKS, "xml")
    builder = _make_builder(config, tiny_tokenizer)
    prompt = builder.build(arc_task, context=_context_for(arc_task))
    assert prompt is not None

    output = gpu_runner.generate(prompt)
    assert isinstance(output, str) and output != ""
