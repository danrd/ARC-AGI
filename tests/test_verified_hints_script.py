"""Tests for scripts/verified_hints.py - precomputing verified hints into
the file the search_hints resolver reads."""
from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import subsymbolic.arc_resolvers as arc_resolvers
from subsymbolic.prompt_builder import OMIT

_SPEC = importlib.util.spec_from_file_location(
    "verified_hints", Path(__file__).resolve().parent.parent / "scripts" / "verified_hints.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_a_split_is_every_task_with_its_training_pairs():
    tasks = script.load_split("training")
    pairs = tasks["3aa6fb7a"]
    assert len(tasks) == 400 and len(pairs) == 2
    assert pairs[1][0] == "3aa6fb7a_1" and isinstance(pairs[1][1], np.ndarray)


def _run(monkeypatch, tmp_path, answers, existing=None):
    monkeypatch.setattr(script, "load_split", lambda split: {t: [] for t in "abcd"})
    monkeypatch.setattr(script, "ProcessPoolExecutor",
                        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=1))
    asked = []

    def hint_for(task_id, pairs, timeout):
        asked.append(task_id)
        text, error = answers[task_id]
        return task_id, text, 1.0, error

    monkeypatch.setattr(script, "hint_for", hint_for)
    out = tmp_path / "hints.json"
    if existing is not None:
        out.write_text(json.dumps(existing))
    script.main(["--out", str(out)])
    return json.loads(out.read_text()), asked


def test_checked_tasks_are_kept_nothing_verified_as_null_failures_left_out(monkeypatch, tmp_path):
    written, _ = _run(monkeypatch, tmp_path, {"a": ("TEXT", None), "b": (None, None),
                                              "c": (None, "RuntimeError()"), "d": (None, None)})
    assert written == {"a": "TEXT", "b": None, "d": None}, \
        "a failed task is not marked done, so a rerun tries it again"


def test_a_rerun_only_asks_what_is_not_in_the_file(monkeypatch, tmp_path):
    written, asked = _run(monkeypatch, tmp_path, {"c": ("C", None), "d": (None, None)},
                          existing={"a": "TEXT", "b": None})
    assert sorted(asked) == ["c", "d"]
    assert written == {"a": "TEXT", "b": None, "c": "C", "d": None}


def test_the_resolver_reads_null_as_no_block(tmp_path):
    path = tmp_path / "hints.json"
    path.write_text(json.dumps({"a": "TEXT", "b": None}))
    arc_resolvers._hints_cache.clear()
    builder = SimpleNamespace(config=SimpleNamespace(project={"search_hints": str(path)}),
                              count_tokens=lambda text: len(text.split()))
    assert arc_resolvers.search_hints_resolver(SimpleNamespace(label="a"), 100, {}, builder) == "TEXT"
    assert arc_resolvers.search_hints_resolver(SimpleNamespace(label="b"), 100, {}, builder) is OMIT
