"""Tests for orchestration.hierarchy - who is believed when: the symbolic
solvers as they answer, RL when it closed the training pairs, and the model
only when a second model agrees.

Nothing runs for real: the dispatch is canned, RL is a handle that resolves
when the test says so, and the second model is a function that records what
it was asked. What is pinned is the order of trust and the ways out of it.
"""
from __future__ import annotations

import numpy as np

from orchestration.configs import AgentRunConfig, SystemRunConfig
from orchestration.hierarchy import hierarchical_decision_fn, solve_with_hierarchy

GRID = np.array([[1, 2], [3, 4]])
TEXT = "2,2:\n1 12\n2 34"
RL_GRID = np.array([[5, 5], [5, 5]])


class Handle:
    """An RL job: `ready` is what poll() answers from the start, `late` what
    wait() brings, `after` how many polls pass before poll() answers with it."""

    def __init__(self, ready=None, late=None, after=None):
        self.ready, self.late, self.after = ready, late, after
        self.polls = 0
        self.cancelled = False
        self.waited = 0

    def poll(self):
        self.polls += 1
        if self.ready is not None:
            return self.ready
        if self.after is not None and self.polls > self.after:
            return self.late
        return None

    def wait(self, timeout):
        self.waited += 1
        return self.late

    def cancel(self, timeout=5.0):
        self.cancelled = True

    @property
    def process(self):
        return self

    def is_alive(self):
        return not self.cancelled


def ok(grid=RL_GRID):
    return {"status": "ok", "solution": grid}


def dispatch(symbolic, texts):
    """Symbolic gives `symbolic`; the model gives `texts` in turn, the last one again."""
    asked = []

    def run(state):
        name = state["current_module"].module_name.lower()
        if name == "symbolic":
            return symbolic
        asked.append(name)
        text = texts[min(len(asked) - 1, len(texts) - 1)]
        return text if isinstance(text, dict) else {"solution": text, "module_results": {}}

    run.asked = asked
    return run


class Verifier:
    """The second model: `verdicts` are its answers to the first model's grids in turn
    (the last one again), `rl` its answer to RL_GRID, whenever that is put to it."""

    def __init__(self, *verdicts, rl=True):
        self.verdicts, self.rl, self.calls = list(verdicts), rl, []

    def of(self, grid):
        return [call for call in self.calls if np.array_equal(call[1], grid)]

    def __call__(self, task, grid):
        self.calls.append((task, grid))
        if np.array_equal(grid, RL_GRID):
            return self.rl
        asked = len(self.calls) - len(self.of(RL_GRID))
        return self.verdicts[min(asked - 1, len(self.verdicts) - 1)]


NO_SYMBOLIC = {"solution": "", "module_results": {"error": "no rule"}}


def solve(task, verify, run, handle, max_iterations=3):
    started = []

    def start(_task):
        started.append(handle)
        return handle

    config = SystemRunConfig(agent_run_config=AgentRunConfig(max_agent_iterations=max_iterations))
    result = solve_with_hierarchy(task, verify, run, rl_start_fn=start, system_run_config=config)
    result["started"] = started
    return result


class TestTheOrderOfTrust:
    def test_a_symbolic_answer_is_taken_and_nothing_else_is_started(self, arc_task):
        verify = Verifier(True)
        result = solve(arc_task, verify, dispatch({"solution": GRID, "module_results": {}}, [TEXT]), Handle())
        assert result["accepted_source"] == "symbolic" and np.array_equal(result["solution"], GRID)
        assert result["started"] == [] and verify.calls == []

    def test_rl_that_is_ready_and_agreed_to_is_taken_over_the_model_which_is_not_asked_about(self, arc_task):
        verify, handle = Verifier(True), Handle(ready=ok())
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "rl" and np.array_equal(result["solution"], RL_GRID)
        assert len(verify.calls) == 1 and len(verify.of(RL_GRID)) == 1 and handle.waited == 0

    def test_rl_the_second_model_refuses_leaves_the_first_model_to_be_checked(self, arc_task):
        verify, handle = Verifier(True, rl=False), Handle(ready=ok())
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "llm" and result["solution"] == TEXT
        assert len(verify.of(RL_GRID)) == 1 and len(verify.of(GRID)) == 1

    def test_a_refused_rl_grid_is_not_put_to_the_second_model_again_on_a_retry(self, arc_task):
        verify, handle = Verifier(False, True, rl=False), Handle(ready=ok())
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle, max_iterations=4)
        assert result["accepted_source"] == "llm" and len(verify.of(GRID)) == 2
        assert len(verify.of(RL_GRID)) == 1

    def test_rl_that_is_not_a_grid_is_never_put_to_the_second_model(self, arc_task):
        verify = Verifier(True)
        handle = Handle(ready={"status": "ok", "solution": np.array([1, 2, 3])})
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "llm" and len(verify.calls) == 1

    def test_rl_still_training_is_waited_for_once_and_taken_when_it_arrives(self, arc_task):
        verify, handle = Verifier(True), Handle(late=ok())
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "rl" and handle.waited == 1
        assert len(verify.of(RL_GRID)) == 1 and len(verify.of(GRID)) == 0

    def test_rl_that_does_not_arrive_in_time_leaves_the_model_to_be_checked(self, arc_task):
        verify, handle = Verifier(True), Handle(late=None)
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "llm" and result["solution"] == TEXT
        assert handle.waited == 1 and handle.cancelled

    def test_rl_that_finished_without_closing_the_training_pairs_is_no_answer(self, arc_task):
        verify, handle = Verifier(True), Handle(ready={"status": "ok", "solution": None})
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "llm" and handle.waited == 0

    def test_rl_that_failed_is_no_answer(self, arc_task):
        verify, handle = Verifier(True), Handle(ready={"status": "error", "debug": "boom"})
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), handle)
        assert result["accepted_source"] == "llm"


class TestTheModelIsCheckedNotTrusted:
    def test_the_second_model_is_asked_about_the_grid_the_text_reads_as(self, arc_task):
        verify = Verifier(True)
        solve(arc_task, verify, dispatch(NO_SYMBOLIC, [TEXT]), Handle(late=None))
        (task, grid), = verify.calls
        assert task is arc_task and np.array_equal(grid, GRID)

    def test_a_refusal_sends_the_model_round_again_until_the_rounds_run_out(self, arc_task):
        verify, handle, run = Verifier(False), Handle(late=None), dispatch(NO_SYMBOLIC, [TEXT])
        result = solve(arc_task, verify, run, handle, max_iterations=3)
        # the symbolic step is the first of the three iterations
        assert run.asked == ["subsymbolic", "subsymbolic"] and len(verify.calls) == 2
        assert result["accepted_source"] is None and result["solution"] == TEXT
        assert handle.cancelled

    def test_an_agreement_on_a_later_round_is_taken(self, arc_task):
        verify, run = Verifier(False, True), dispatch(NO_SYMBOLIC, ["2,2:\n1 00\n2 00", TEXT])
        result = solve(arc_task, verify, run, Handle(late=None))
        assert result["accepted_source"] == "llm" and result["solution"] == TEXT
        assert len(verify.calls) == 2

    def test_text_that_is_not_a_grid_is_a_refusal_the_second_model_never_sees(self, arc_task):
        verify = Verifier(True)
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, ["I think the answer is blue"]),
                       Handle(late=None))
        assert verify.calls == [] and result["accepted_source"] is None

    def test_a_model_error_is_a_refusal(self, arc_task):
        verify = Verifier(True)
        error = {"solution": "", "module_results": {"error": "prompt didn't fit token_limit"}}
        result = solve(arc_task, verify, dispatch(NO_SYMBOLIC, [error, TEXT]), Handle(late=None))
        assert result["accepted_source"] == "llm" and len(verify.calls) == 1

    def test_an_answer_the_module_reported_an_error_with_is_not_read(self):
        verify = Verifier(True)
        state = {"task": object(), "last_dispatch": "llm", "solution": TEXT,
                 "module_results": {"error": "cut off"}}
        assert hierarchical_decision_fn(verify)(state)["action"] == "retry_llm" and verify.calls == []

    def test_rl_that_arrives_while_the_model_is_retried_is_taken(self, arc_task):
        """Not ready at the first answer, timed out at the one wait, ready by the third."""
        handle = Handle(late=None)
        verify, run = Verifier(False, rl=True), dispatch(NO_SYMBOLIC, [TEXT])

        original = handle.wait

        def wait(timeout):
            original(timeout)
            handle.ready = ok()      # arrives just after the wait gave up
            return None

        handle.wait = wait
        result = solve(arc_task, verify, run, handle, max_iterations=4)
        assert result["accepted_source"] == "rl" and np.array_equal(result["solution"], RL_GRID)


class TestTheDecisionFunction:
    def state(self, **kw):
        return {"task": object(), "last_dispatch": "llm", "solution": TEXT, "module_results": {},
                "rl_handle": None, "rl_status": None, "rl_wait_used": False, **kw}

    def test_a_symbolic_answer_with_an_error_is_not_taken(self):
        decide = hierarchical_decision_fn(Verifier(True))
        state = self.state(last_dispatch="symbolic", solution="", module_results={"error": "x"})
        assert decide(state)["status"] == "INVALID"

    def test_the_parser_is_the_callers(self):
        seen = []

        def parse(text):
            seen.append(text)
            return GRID

        decide = hierarchical_decision_fn(Verifier(True), parse=parse)
        assert decide(self.state(solution="whatever"))["source"] == "llm" and seen == ["whatever"]

    def test_a_parser_that_raises_is_a_refusal(self):
        def parse(text):
            raise IndexError

        assert hierarchical_decision_fn(Verifier(True), parse=parse)(self.state())["action"] == "retry_llm"

    def test_the_verdict_of_a_grid_the_model_returned_as_an_array_is_asked_as_it_is(self):
        verify = Verifier(True)
        assert hierarchical_decision_fn(verify)(self.state(solution=GRID))["source"] == "llm"
        assert np.array_equal(verify.calls[0][1], GRID)

    def test_a_verdict_is_remembered_for_that_task_and_that_grid_only(self):
        """Another grid of the same task, or the same grid of another task, is a new question."""
        asked = []

        def verify(task, grid):
            asked.append((task.label, grid.tobytes()))
            return False

        class Task:
            def __init__(self, label):
                self.label = label

        decide = hierarchical_decision_fn(verify)
        first, other = Task("a"), Task("b")
        for task, grid in ((first, GRID), (first, GRID), (first, RL_GRID), (other, GRID)):
            decide(self.state(task=task, rl_status="ok", rl_solution=grid, solution=""))
        assert len(asked) == 3
