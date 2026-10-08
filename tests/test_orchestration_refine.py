"""Showing a model what was wrong with its answer, and asking again."""
from types import SimpleNamespace

import numpy as np

from orchestration.feedback import (NOT_ACCEPTED, UNREADABLE, Attempt, best_attempt, memoize_verdicts,
                                    render_history, review, with_feedback)
from orchestration.graph import ModuleInvConfig
from orchestration.refine import RefiningModule, loop
from orchestration.trace import Tracer
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY


def make_task(label="t"):
    pairs = [(np.array([[0, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 2, 0]])),
             (np.array([[1, 0, 0], [0, 0, 0]]), np.array([[1, 0, 0], [0, 2, 0]]))]
    subtasks = [ARCSubtask(f"{label}_{i}", a, b) for i, (a, b) in enumerate(pairs)]
    return ARCTask(label=label, subtasks=subtasks, test_inp=np.array([[0, 0, 1], [0, 0, 0]]),
                   test_out=np.array([[0, 0, 1], [0, 0, 2]]))


GOOD = "2,3:\n1 001\n2 002"
WRONG_SHAPE = "1,3:\n1 001"
OFF_PALETTE = "2,3:\n1 001\n2 007"


def fmt(text):
    from subsymbolic.utils import parse_llm_output
    return parse_llm_output(text)


class TestReview:
    def test_an_answer_that_breaks_nothing_has_no_notes_and_the_second_model_is_asked_only_then(self):
        asked = []
        notes, strong, weak = review(make_task(), fmt(GOOD), verify=lambda t, g: asked.append(1) or True)
        assert (notes, strong) == ([], 0) and asked == [1]

    def test_a_refused_answer_is_not_put_to_the_second_model(self):
        asked = []
        notes, strong, _ = review(make_task(), fmt(WRONG_SHAPE), verify=lambda t, g: asked.append(1) or True)
        assert strong >= 1 and asked == [] and "1x3" in notes[0]

    def test_the_second_models_refusal_is_a_note_of_its_own(self):
        notes, strong, _ = review(make_task(), fmt(GOOD), verify=lambda t, g: False)
        assert notes == [NOT_ACCEPTED] and strong == 1

    def test_an_unreadable_answer_says_so(self):
        assert review(make_task(), None) == ([UNREADABLE], 1, 0)

    def test_the_second_model_is_timed_when_there_is_a_tracer(self):
        tracer = Tracer()
        review(make_task(), fmt(GOOD), verify=lambda t, g: True, tracer=tracer)
        assert [s.phase for s in tracer.spans] == ["verify"]


class TestHistory:
    def test_the_history_shows_each_refused_answer_and_why_and_skips_the_clean_ones(self):
        attempts = [Attempt(1, WRONG_SHAPE, None, ["too small"], 1, 0), Attempt(2, GOOD, fmt(GOOD), [], 0, 0),
                    Attempt(3, "", None, [UNREADABLE], 1, 0)]
        text = render_history(attempts)
        assert "Attempt 1. Your answer:\n1,3:" in text and "- too small" in text
        assert "Attempt 2" not in text and "(no answer)" in text and "Attempt 3" in text
        assert render_history([]) == "" and render_history([attempts[1]]) == ""

    def test_the_best_attempt_has_fewest_strong_then_weak_complaints_and_the_later_of_equals(self):
        grid = np.zeros((1, 1), int)
        a = Attempt(1, "a", grid, ["x"], 1, 0)
        b = Attempt(2, "b", grid, ["x"], 0, 2)
        c = Attempt(3, "c", grid, ["x"], 0, 2)
        d = Attempt(4, "d", grid, ["x"], 0, 1)
        assert best_attempt([a, b, c]).number == 3 and best_attempt([a, b, c, d]).number == 4
        assert best_attempt([]) is None
        unreadable = Attempt(5, "", None, [UNREADABLE], 0, 0)
        assert best_attempt([a, unreadable]).number == 1       # a wrong grid beats no grid, however few the complaints

    def test_a_verdict_is_asked_for_once_per_task_and_grid(self):
        calls = []
        verdict = memoize_verdicts(lambda t, g: calls.append(1) or len(calls) == 1)
        task = make_task()
        grid = np.array([[1, 2]])
        assert verdict(task, grid) is True and verdict(task, grid) is True and len(calls) == 1
        assert verdict(task, np.array([[1, 3]])) is False and len(calls) == 2
        assert verdict(make_task("other"), grid) is False and len(calls) == 3


class TestTheLoop:
    def test_it_stops_at_the_first_answer_nothing_is_found_wrong_with(self):
        replies = [WRONG_SHAPE, GOOD, GOOD]
        seen = []

        def ask(context):
            seen.append(context.get("history_text", ""))
            return replies[len(seen) - 1]

        result = loop(make_task(), ask, rounds=3)
        assert result.accepted and len(result.attempts) == 2 and result.solution == GOOD
        assert seen[0] == "" and "1x3" in seen[1] and WRONG_SHAPE in seen[1]

    def test_when_nothing_is_ever_right_the_least_wrong_attempt_is_returned_after_the_rounds(self):
        replies = iter([WRONG_SHAPE, OFF_PALETTE, "nonsense"])
        result = loop(make_task(), lambda context: next(replies), rounds=3)
        assert not result.accepted and len(result.attempts) == 3
        assert result.best.number in (1, 2) and result.best.strong >= 1
        assert result.attempts[2].notes == [UNREADABLE]

    def test_the_context_the_caller_gave_goes_to_every_call(self):
        contexts = []
        loop(make_task(), lambda c: contexts.append(dict(c)) or WRONG_SHAPE, rounds=2, context={"mark": 1})
        assert all(c["mark"] == 1 for c in contexts) and len(contexts) == 2

    def test_a_non_text_reply_is_an_unreadable_answer_and_not_a_crash(self):
        result = loop(make_task(), lambda c: None, rounds=1)
        assert result.attempts[0].notes == [UNREADABLE]


class TestTheModule:
    def make(self, tiny_tokenizer, replies):
        config = PromptingConfig(blocks=["general_instruction", "examples", "output_format"],
                                 resolvers=["examples"], filters=["grid"], token_limit=20000)
        builder = PromptBuilder(config, tiny_tokenizer, resolver_registry=RESOLVER_REGISTRY,
                                filter_registry=FILTER_REGISTRY)
        calls = []

        def solve(task, context=None):
            calls.append(dict(context or {}))
            return {"solution": replies[len(calls) - 1], "module_results": {}}

        return SimpleNamespace(builder=builder, runner=None, close=lambda: None, solve=solve), calls

    def test_the_wrapped_module_gets_the_memory_block_and_the_loop_runs_through_it(self, tiny_tokenizer):
        inner, calls = self.make(tiny_tokenizer, [WRONG_SHAPE, GOOD])
        refined = RefiningModule(inner, rounds=3)
        assert "memory" in [b if isinstance(b, str) else b.name for b in inner.builder.config.blocks]
        result = refined.solve(make_task(), {"mark": 1})
        assert result["solution"] == GOOD
        assert result["module_results"]["rounds"] == 2 and result["module_results"]["accepted"] is True
        assert "history_text" not in calls[0] and "1x3" in calls[1]["history_text"]

    def test_the_block_is_empty_without_a_history_and_carries_it_with_one(self, tiny_tokenizer):
        inner, _ = self.make(tiny_tokenizer, [GOOD])
        RefiningModule(inner)
        task = make_task()
        base = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
        assert "Memory" not in inner.builder.build(task, context=base)
        assert "Attempt 1" in inner.builder.build(task, context={**base, "history_text": "Attempt 1. Your answer:\nx"})

    def test_an_error_from_the_module_with_no_answer_stays_an_error(self, tiny_tokenizer):
        inner, _ = self.make(tiny_tokenizer, [])
        inner.solve = lambda task, context=None: {"solution": "", "module_results": {"error": "too long"}}
        result = RefiningModule(inner, rounds=2).solve(make_task())
        assert result["solution"] == "" and result["module_results"]["error"] == "too long"


class TestInTheGraph:
    def test_the_retry_is_told_what_the_earlier_answer_got_wrong_and_the_other_modules_pass_through(self):
        task = make_task()
        replies = iter([WRONG_SHAPE, GOOD])
        contexts = []

        def dispatch(state):
            contexts.append(dict(state["auxiliary_info"]))
            if state["current_module"].module_name == "symbolic":
                return {"solution": "", "module_results": {"error": "none"}}
            return {"solution": next(replies), "module_results": {}}

        wrapped = with_feedback(dispatch)
        llm = ModuleInvConfig(1, "subsymbolic")
        wrapped({"task": task, "current_module": ModuleInvConfig(0, "symbolic"), "auxiliary_info": {}})
        wrapped({"task": task, "current_module": llm, "auxiliary_info": {"a": 1}})
        wrapped({"task": task, "current_module": llm, "auxiliary_info": {"a": 1}})
        assert contexts[0] == {} and contexts[1] == {"a": 1}
        assert "1x3" in contexts[2]["history_text"] and contexts[2]["a"] == 1

    def test_attempts_are_kept_per_task(self):
        replies = iter([WRONG_SHAPE, WRONG_SHAPE])
        contexts = []

        def dispatch(state):
            contexts.append(dict(state["auxiliary_info"]))
            return {"solution": next(replies), "module_results": {}}

        wrapped = with_feedback(dispatch)
        llm = ModuleInvConfig(1, "subsymbolic")
        wrapped({"task": make_task("a"), "current_module": llm, "auxiliary_info": {}})
        wrapped({"task": make_task("b"), "current_module": llm, "auxiliary_info": {}})
        assert "history_text" not in contexts[1]
