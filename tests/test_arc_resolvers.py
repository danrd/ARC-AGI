"""Tests for subsymbolic/arc_resolvers.py's transformation_summary_resolver -
the point where symbolic analysis finally reaches the prompt.

It stayed a `return ""` stub while the whole symbolic layer had no consumer at
all, so what is pinned here is the wiring: that analysis is actually invoked,
that its cost is paid once per task rather than once per prompt build, and
that an empty analysis omits the block instead of spending tokens on a header
introducing nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import subsymbolic.arc_resolvers as arc_resolvers
from symbolic.findings import Evidence, Finding, TaskFindings


@pytest.fixture(autouse=True)
def clear_cache():
    arc_resolvers._findings_cache.clear()
    yield
    arc_resolvers._findings_cache.clear()


def _builder():
    return SimpleNamespace(count_tokens=len)


def _task(label="task-1"):
    return SimpleNamespace(label=label)


def _findings(task_id="task-1", n=2):
    return TaskFindings(
        task_id=task_id,
        example_count=2,
        transformations=tuple(
            Finding(subject=f"t{i}", statement=f"transformation {i}",
                    evidence=Evidence((0, 1), 2), confidence=0.9, parameters={"k": i})
            for i in range(n)
        ),
        invariants=(
            Finding(subject="grid_size", statement="the output keeps the input's grid size",
                    evidence=Evidence((0, 1), 2), confidence=1.0),
        ),
    )


class _FakeAnalyzer:
    """Stands in for SymbolicAnalyzer, counting how often analysis ran."""
    calls = 0
    findings = None

    def analyze_task(self, task):
        type(self).calls += 1
        return SimpleNamespace(task=task, get_findings=lambda: type(self).findings)


@pytest.fixture
def fake_analysis(monkeypatch):
    _FakeAnalyzer.calls = 0
    _FakeAnalyzer.findings = _findings()
    monkeypatch.setattr("symbolic.analyzer.SymbolicAnalyzer", _FakeAnalyzer)
    return _FakeAnalyzer


def test_resolver_returns_the_rendered_analysis(fake_analysis):
    text = arc_resolvers.transformation_summary_resolver(_task(), budget=10_000,
                                                          context={}, builder=_builder())

    assert "What changes:" in text
    assert "transformation 0" in text
    assert "What stays the same:" in text


def test_resolver_omits_the_block_when_there_is_nothing_to_claim(fake_analysis):
    fake_analysis.findings = TaskFindings(task_id="task-1", example_count=0)

    result = arc_resolvers.transformation_summary_resolver(_task(), budget=10_000,
                                                            context={}, builder=_builder())

    assert result is None


def test_resolver_omits_the_block_when_nothing_fits_the_budget(fake_analysis):
    result = arc_resolvers.transformation_summary_resolver(_task(), budget=1,
                                                            context={}, builder=_builder())

    assert result is None


def test_analysis_runs_once_per_task_across_repeated_prompt_builds(fake_analysis):
    """Analysis is the most expensive thing a prompt block can trigger; a
    retry or a second prompt variant for the same task must not pay for it
    again."""
    task = _task()
    for _ in range(3):
        arc_resolvers.transformation_summary_resolver(task, budget=10_000,
                                                       context={}, builder=_builder())

    assert fake_analysis.calls == 1


def test_different_tasks_are_analyzed_separately(fake_analysis):
    arc_resolvers.transformation_summary_resolver(_task("a"), budget=10_000,
                                                   context={}, builder=_builder())
    arc_resolvers.transformation_summary_resolver(_task("b"), budget=10_000,
                                                   context={}, builder=_builder())

    assert fake_analysis.calls == 2


def test_cache_does_not_grow_without_bound(fake_analysis, monkeypatch):
    monkeypatch.setattr(arc_resolvers, "_FINDINGS_CACHE_SIZE", 4)

    for i in range(10):
        arc_resolvers.transformation_summary_resolver(_task(f"task-{i}"), budget=10_000,
                                                       context={}, builder=_builder())

    assert len(arc_resolvers._findings_cache) <= 4


class TestSearchHints:
    """The block that carries what an automated search measured on the task.

    Written offline - a search costs hours per shard, a prompt is built in
    milliseconds - so what this pins is the reading side: the right task's
    hint, the budget honoured, and silence rather than noise for a task
    nothing is known about.
    """

    @pytest.fixture(autouse=True)
    def clear_hints(self):
        arc_resolvers._hints_cache.clear()
        yield
        arc_resolvers._hints_cache.clear()

    @staticmethod
    def _builder(path, limit=10_000):
        return SimpleNamespace(count_tokens=len,
                              config=SimpleNamespace(project={"search_hints": str(path)}))

    @staticmethod
    def _file(tmp_path, mapping):
        import json
        path = tmp_path / "hints.json"
        path.write_text(json.dumps(mapping))
        return path

    def test_the_hint_of_this_task_is_returned(self, tmp_path):
        path = self._file(tmp_path, {"task-1": "a search found X", "task-2": "Y"})

        out = arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                 self._builder(path))

        assert out == "a search found X"

    def test_a_task_the_search_says_nothing_about_omits_the_block(self, tmp_path):
        path = self._file(tmp_path, {"task-2": "Y"})

        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                  self._builder(path)) is None

    def test_a_hint_that_does_not_fit_the_budget_omits_the_block(self, tmp_path):
        path = self._file(tmp_path, {"task-1": "a" * 50})

        assert arc_resolvers.search_hints_resolver(_task("task-1"), 10, {},
                                                   self._builder(path)) is None

    def test_a_missing_file_loses_the_block_rather_than_the_run(self, tmp_path):
        """A run configured with hints on a machine that has not scanned yet
        should lose the block, not fail at the first task."""
        missing = tmp_path / "nothing.json"

        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                   self._builder(missing)) is None

    def test_the_file_is_read_once_however_many_tasks_ask(self, tmp_path):
        path = self._file(tmp_path, {"task-1": "X"})
        builder = self._builder(path)

        arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {}, builder)
        path.write_text('{"task-1": "replaced"}')

        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                   builder) == "X"

    def test_a_missing_file_is_remembered_as_missing(self, tmp_path):
        """Otherwise every task of the run re-opens a file that is not there
        and re-prints the notice saying so."""
        missing = tmp_path / "nothing.json"

        arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                            self._builder(missing))

        assert str(missing) in arc_resolvers._hints_cache
