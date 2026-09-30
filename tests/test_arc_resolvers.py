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
from subsymbolic.prompt_builder import OMIT
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

    assert result is OMIT


def test_resolver_omits_the_block_when_nothing_fits_the_budget(fake_analysis):
    result = arc_resolvers.transformation_summary_resolver(_task(), budget=1,
                                                            context={}, builder=_builder())

    assert result is OMIT


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
    """The block that carries what a search measured on the task, computed
    by the caller for this task now and handed over in the build context.

    What this pins is the reading side: the hint as given, the budget
    honoured, and silence rather than noise for a task nobody brought one
    for.
    """

    @staticmethod
    def _builder():
        return SimpleNamespace(count_tokens=len, config=SimpleNamespace(project={}))

    def test_the_hint_the_caller_computed_is_returned(self):
        out = arc_resolvers.search_hints_resolver(
            _task("task-1"), 1000, {"search_hints": "a search found X"}, self._builder())

        assert out == "a search found X"

    def test_a_task_with_no_hint_omits_the_block(self):
        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                   self._builder()) is OMIT
        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, None,
                                                   self._builder()) is OMIT

    def test_an_empty_hint_omits_the_block(self):
        assert arc_resolvers.search_hints_resolver(
            _task("task-1"), 1000, {"search_hints": ""}, self._builder()) is OMIT

    def test_a_hint_is_budgeted_like_any_other(self):
        out = arc_resolvers.search_hints_resolver(
            _task("task-1"), 10, {"search_hints": "a" * 50}, self._builder())

        assert out is OMIT

    def test_no_file_is_read_any_more(self):
        """The precomputed file the block used to fall back on is gone: a
        config still naming one changes nothing."""
        builder = SimpleNamespace(count_tokens=len,
                                  config=SimpleNamespace(project={"search_hints": "hints.json"}))

        assert arc_resolvers.search_hints_resolver(_task("task-1"), 1000, {},
                                                   builder) is OMIT
