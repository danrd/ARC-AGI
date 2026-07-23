"""Shared fixtures for the test suite (tests/test_llm_smoke.py,
tests/test_rl_*.py) - includes the fixed ARC task loader most of these
suites build on, plus the LLM smoke test's model/tokenizer fixtures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

# test_symbolic.py is its own pre-existing, non-pytest test framework
# (UnifiedTestRunner.run_test_on_all_grids(...) calls test methods directly,
# passing a `grid` positional argument) - pytest's collector mistakes that
# `grid` parameter for a fixture request and errors on most of them. Excluded
# from auto-collection here; run it via its own run_all_tests()/quick_test()
# entry points instead.
collect_ignore = ["test_symbolic.py"]

from rl.arc_task import ARCSubtask, ARCTask

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "data" / "pretrained_models"

CPU_GGUF_REPO = "SupraLabs/Supra-Router-51M-gguf"
CPU_GGUF_FILENAME = "Supra-Router-51M-Q4_K_M.gguf"

# Tried in order on the GPU path; first one that actually loads wins. The
# first entry is the same GGUF-only repo as the CPU model - transformers/vLLM
# most likely can't load it (no HF-format weights), it's kept first only
# because it's already cached locally from the CPU test and costs nothing to
# try. The other two are plain HF-format tiny models, more likely to work.
GPU_MODEL_CANDIDATES: List[str] = [
    CPU_GGUF_REPO,
    "SimpleStories/SimpleStories-1.25M",
    "BananaMind/BananaMind-2-Nano",
]


class ApproxTokenizer:
    """Whitespace-split token counter, standing in for a real tokenizer.

    PromptBuilder only needs `.tokenize(text)` to return something with a
    `len()` for token-budget accounting during prompt assembly - it doesn't
    need real subword tokenization for that. Using a real HF tokenizer here
    would mean downloading one just to count tokens approximately anyway.
    Doesn't support `apply_chat_template`, so configs with `chat_template`
    set aren't exercised by this suite (see test_llm_smoke.py's note).
    """

    def tokenize(self, text: str) -> List[str]:
        return text.split()


@pytest.fixture(scope="session")
def tiny_tokenizer() -> ApproxTokenizer:
    return ApproxTokenizer()


@pytest.fixture(scope="session")
def arc_task() -> ARCTask:
    """One fixed ARC task, picked deterministically (first key, sorted) from
    the training set so the suite doesn't depend on a hardcoded task id that
    might not exist in a given checkout."""
    challenges_path = REPO_ROOT / "data" / "datasets" / "ARC" / "training_challenges.json"
    solutions_path = REPO_ROOT / "data" / "datasets" / "ARC" / "training_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)

    task_id = sorted(challenges.keys())[0]
    task_data = challenges[task_id]

    subtasks = [
        ARCSubtask(label=f"{task_id}_{i}", train_inp=_to_array(pair["input"]),
                   train_out=_to_array(pair["output"]))
        for i, pair in enumerate(task_data["train"])
    ]
    test_inp = _to_array(task_data["test"][0]["input"])
    test_out = _to_array(solutions[task_id][0])

    return ARCTask(label=task_id, subtasks=subtasks, test_inp=test_inp, test_out=test_out)


def _to_array(grid):
    import numpy as np
    return np.array(grid)


@pytest.fixture(scope="session")
def supra_router_gguf_path() -> str:
    """Downloads (once, cached under data/pretrained_models/) the tiny GGUF
    model used for the CPU llama.cpp path."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = huggingface_hub.hf_hub_download(
        repo_id=CPU_GGUF_REPO,
        filename=CPU_GGUF_FILENAME,
        local_dir=str(MODEL_CACHE_DIR),
    )
    return path


def gpu_enabled() -> bool:
    return os.environ.get("ARC_TEST_GPU") == "1"
