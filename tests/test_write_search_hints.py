"""Tests for scripts/write_search_hints.py - precomputing search hints into
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
    "write_search_hints", Path(__file__).resolve().parent.parent / "scripts" / "write_search_hints.py")
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
                        lambda max_workers, max_tasks_per_child, mp_context:
                        ThreadPoolExecutor(max_workers=1))
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


class _Pools:
    """Pools whose workers die on the tasks named in `kills`, one list per
    pool built: a killed worker fails every task still in the pool."""

    def __init__(self, kills):
        self.kills, self.built = list(kills), 0

    def __call__(self, max_workers, max_tasks_per_child, mp_context):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        kills = self.kills[self.built] if self.built < len(self.kills) else ()
        self.built += 1

        class Pool:
            broken = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, task_id, pairs, timeout):
                future = Future()
                if Pool.broken or task_id in kills:
                    Pool.broken = True
                    future.set_exception(BrokenProcessPool("killed"))
                else:
                    future.set_result(fn(task_id, pairs, timeout))
                return future

        return Pool()


def _run_with(monkeypatch, tmp_path, pools):
    monkeypatch.setattr(script, "load_split", lambda split: {t: [] for t in "abcd"})
    monkeypatch.setattr(script, "ProcessPoolExecutor", pools)
    # Last submitted first: a broken future reached before tasks that had
    # already finished, which as_completed is free to do.
    monkeypatch.setattr(script, "as_completed", lambda futures: list(reversed(futures)))
    monkeypatch.setattr(script, "hint_for", lambda task_id, pairs, timeout:
                        (task_id, task_id.upper(), 1.0, None))
    out = tmp_path / "hints.json"
    script.main(["--out", str(out)])
    return json.loads(out.read_text())


def test_a_killed_worker_sends_what_was_unfinished_to_a_fresh_pool(monkeypatch, tmp_path):
    pools = _Pools([("c",)])
    written = _run_with(monkeypatch, tmp_path, pools)
    assert written == {"a": "A", "b": "B", "c": "C", "d": "D"}
    assert pools.built == 2


def test_a_task_caught_in_every_break_is_given_up_without_a_hint(monkeypatch, tmp_path):
    pools = _Pools([("c",)] * script.BREAKS_ALLOWED)
    written = _run_with(monkeypatch, tmp_path, pools)
    assert written["c"] is None and written["a"] == "A"
    assert written["d"] is None, "d sat behind c in every broken pool"
