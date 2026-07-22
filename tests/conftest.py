"""Shared fixtures for the LLM smoke-test suite (tests/test_llm_smoke.py).

Goal of this suite: exercise the real prompt-building -> inference pipeline
end to end on one fixed ARC task, using models small enough (single-digit to
double-digit MB) to run on a CPU in seconds. This is NOT an accuracy
benchmark - these models are far too small to solve ARC tasks. Success means
the pipeline ran without raising, not that the answer is right.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from rl.ARC_task import ARCSubtask, ARCTask

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
