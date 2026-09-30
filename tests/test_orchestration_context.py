"""Tests for orchestration/context.py - the context an LLM run builds each
task's prompt with, carrying the hint the search has verified.

The search is faked (a real one is a minute a task and is tested in
test_search_hints); what is pinned is the join: the hint reaches the prompt
block when there is one, is absent when there is not, is computed once per
task, never overrides what the caller supplied, and cannot take the run down.
"""
from __future__ import annotations

import numpy as np
import pytest

from orchestration.context import with_search_hints
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.llm_run import EvalResult, run_llm_over_tasks
from subsymbolic.prompt_builder import ApproxTokenizer, PromptBuilder, PromptingConfig
from subsymbolic.registry import RESOLVER_REGISTRY

HINT = "A search reproduced every training pair (2 of 2) exactly."


def task(label="t"):
    grid = np.zeros((3, 3), dtype=int)
    return ARCTask(label=label, subtasks=[ARCSubtask(f"{label}_0", grid, grid + 1),
                                          ARCSubtask(f"{label}_1", grid, grid + 1)],
                   test_inp=grid, test_out=grid + 1)


@pytest.fixture
def searched(monkeypatch):
    """rl.search_hints.hints_for, replaced: what it was asked for, and what
    it says for which task."""
    import rl.search_hints as hints

    calls, says = [], {"t": HINT}

    def fake(subject, settings=None, candidates=3):
        calls.append(subject.label)
        return says.get(subject.label)

    monkeypatch.setattr(hints, "hints_for", fake)
    return calls, says


class TestTheContext:
    def test_a_verified_hint_is_added_and_the_callers_context_is_kept(self, searched):
        build = with_search_hints(lambda t: {"role_text": "You solve puzzles."})
        assert build(task()) == {"role_text": "You solve puzzles.", "search_hints": HINT}

    def test_a_task_the_search_cannot_stand_behind_gets_no_key(self, searched):
        assert "search_hints" not in with_search_hints()(task("other"))

    def test_the_callers_own_hint_is_left_alone_and_nothing_is_searched(self, searched):
        calls, _ = searched
        build = with_search_hints(lambda t: {"search_hints": "mine"})
        assert build(task())["search_hints"] == "mine" and calls == []

    def test_an_empty_hint_from_the_caller_withholds_the_block(self, searched):
        calls, _ = searched
        build = with_search_hints(lambda t: {"search_hints": ""})
        assert build(task()) == {"search_hints": ""} and calls == []

    def test_a_task_is_searched_once_however_often_it_is_asked_for(self, searched):
        calls, _ = searched
        build = with_search_hints()
        for _ in range(3):
            build(task())
        assert calls == ["t"]

    def test_a_cache_the_caller_keeps_is_shared_between_builders(self, searched):
        calls, _ = searched
        shared = {}
        with_search_hints(cache=shared)(task())
        with_search_hints(cache=shared)(task())
        assert calls == ["t"] and shared

    def test_the_callers_context_is_not_written_into(self, searched):
        original = {"role_text": "x"}
        with_search_hints(lambda t: original)(task())
        assert original == {"role_text": "x"}

    def test_a_search_that_fails_is_warned_about_and_costs_only_the_hint(self, monkeypatch):
        import rl.search_hints as hints

        def broken(subject, settings=None, candidates=3):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(hints, "hints_for", broken)
        with pytest.warns(UserWarning, match="search hint for"):
            context = with_search_hints(lambda t: {"a": 1})(task())
        assert context == {"a": 1}


class TestReachingThePrompt:
    def builder(self, blocks=("general_instruction", "search_hints", "output_format")):
        config = PromptingConfig(blocks=list(blocks), token_limit=10 ** 6, min_examples=1,
                                 filters=["grid"], resolvers=["search_hints"])
        return PromptBuilder(config, ApproxTokenizer(), resolver_registry=RESOLVER_REGISTRY,
                             filter_registry={"grid": lambda grid, type="concise": str(grid)})

    def test_the_block_says_the_hint_when_there_is_one_and_is_absent_when_there_is_not(self, searched):
        build = with_search_hints()
        with_hint = self.builder().build(task(), context=build(task()))
        without = self.builder().build(task("other"), context=build(task("other")))
        assert HINT in with_hint and "SEARCH_HINTS" in with_hint
        assert "SEARCH_HINTS" not in without and "GENERAL_INSTRUCTION" in without

    def test_a_prompt_that_does_not_list_the_block_never_shows_the_hint(self, searched):
        prompt = self.builder(blocks=("general_instruction", "output_format")).build(
            task(), context=with_search_hints()(task()))
        assert HINT not in prompt

    def test_it_is_what_an_llm_run_hands_each_prompt(self, searched):
        class Runner:
            prompts = []

            def generate(self, prompt):
                self.prompts.append(prompt)
                return "x"

        class Module:
            builder = self.builder()
            runner = Runner()

        class Subject:
            def __init__(self, arc):
                self.arc, self.id, self.index = arc, arc.label, None
                self.label, self.subtasks = arc.label, arc.subtasks
                self.test_subtask = arc.test_subtask

        run_llm_over_tasks(tasks=[Subject(task("t")), Subject(task("other"))], subsymbolic_module=Module(),
                           evaluator=lambda t, g: EvalResult(), context_builder=with_search_hints(), debug=True)
        assert [HINT in prompt for prompt in Runner.prompts] == [True, False]
