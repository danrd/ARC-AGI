"""The orchestrator as a model: guarded, and the rule whenever the model cannot be used."""
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.configs import AgentRunConfig
from orchestration.graph import AgentInvConfig, InteractionRecord, ModuleInvConfig
from orchestration.hierarchy import hierarchical_decision_fn
from orchestration.llm_orchestrator import (COORDINATOR_BLOCKS, Choice, decision_config, delegating_coordinator_fn,
                                            describe_evidence, describe_modules, llm_coordinator_fn, llm_decision_fn,
                                            open_actions, parse_choice)
from orchestration.trace import Tracer
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY

GOOD = "2,3:\n1 001\n2 002"
WRONG_SHAPE = "1,3:\n1 001"


def make_task(label="t"):
    pairs = [(np.array([[0, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 2, 0]])),
             (np.array([[1, 0, 0], [0, 0, 0]]), np.array([[1, 0, 0], [0, 2, 0]]))]
    subtasks = [ARCSubtask(f"{label}_{i}", a, b) for i, (a, b) in enumerate(pairs)]
    return ARCTask(label=label, subtasks=subtasks, test_inp=np.array([[0, 0, 1], [0, 0, 0]]),
                   test_out=np.array([[0, 0, 1], [0, 0, 2]]))


MODULES = [{"index": 0, "name": "symbolic"}, {"index": 1, "name": "subsymbolic"}]


def state(solution=WRONG_SHAPE, **over):
    base = {"task": make_task(), "solution": solution, "module_results": {}, "last_dispatch": "llm", "iteration": 1,
            "run_config": AgentRunConfig(max_agent_iterations=3), "available_modules": MODULES,
            "history": [InteractionRecord(1, "llm", "subsymbolic", solution, "PENDING")],
            "rl_handle": None, "rl_status": None, "rl_wait_used": False,
            "llm_module": ModuleInvConfig(1, "subsymbolic")}
    base.update(over)
    return base


class Runner:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def decider(tokenizer, reply, blocks=None):
    config = decision_config(PromptingConfig(token_limit=20000), blocks) if blocks else \
        decision_config(PromptingConfig(token_limit=20000))
    builder = PromptBuilder(config, tokenizer, resolver_registry=RESOLVER_REGISTRY, filter_registry=FILTER_REGISTRY)
    return SimpleNamespace(builder=builder, runner=Runner(reply))


def rule(verify=lambda t, g: False):
    return hierarchical_decision_fn(verify)


class TestReadingTheChoice:
    def test_a_json_decision_is_read_and_anything_else_is_not(self):
        assert parse_choice('{"status": "INVALID", "action": "Retry_LLM", "delegate_to_module": 1, "reasoning": "r"}') \
            == Choice("INVALID", "retry_llm", 1, None, "r")
        assert parse_choice('```json\n{"status": "validated"}\n```') == Choice("VALIDATED")
        assert parse_choice('{"status": "INVALID", "delegate_to_agent": 2}').delegate_to_agent == 2
        for bad in (None, "", "retry please", '{"status": "MAYBE"}', "[1, 2]", '{"action": "wait_rl"}'):
            assert parse_choice(bad) is None
        assert parse_choice('{"status": "INVALID", "delegate_to_module": true}').delegate_to_module is None

    def test_the_actions_open_follow_the_graphs_own_rules(self):
        assert open_actions(state()) == ["retry_llm", "give_up"]
        assert open_actions(state(iteration=3)) == ["give_up"]
        waiting = state(rl_handle=object())
        assert open_actions(waiting) == ["retry_llm", "wait_rl", "give_up"]
        assert "wait_rl" not in open_actions(state(rl_handle=object(), rl_wait_used=True))
        assert "wait_rl" not in open_actions(state(rl_handle=object(), rl_status="ok"))


class TestTheEvidence:
    def test_each_source_has_a_line_and_the_second_models_word_is_in_it(self):
        text = describe_evidence(state(GOOD, rl_handle=object()), verify=lambda t, g: True)
        assert "symbolic solvers: no answer" in text and "RL: still training" in text
        assert "2x3 grid; it breaks nothing" in text and "The second model accepted it." in text
        refused = describe_evidence(state(GOOD), verify=lambda t, g: False)
        assert "The second model did not accept it." in refused and "RL: not running." in refused

    def test_what_the_answer_breaks_is_said(self):
        text = describe_evidence(state(WRONG_SHAPE), verify=None)
        assert "1x3 grid; it breaks: The answer is 1x3" in text and "second model" not in text

    def test_no_answer_and_an_unreadable_one_are_said(self):
        assert "language model: no answer" in describe_evidence(state("", module_results={}))
        assert "no answer" in describe_evidence(state("x", module_results={"error": "e"}))
        assert "could not be read" in describe_evidence(state("nonsense"))

    def test_rl_that_finished_is_described_by_what_it_produced(self):
        done = state(GOOD, rl_handle=object(), rl_status="ok", rl_solution=np.array([[0, 0, 1], [0, 0, 2]]))
        assert "closed every training pair. The second model accepted" in describe_evidence(done, lambda t, g: True)
        failed = state(GOOD, rl_handle=object(), rl_status="error")
        assert "RL: finished without an answer" in describe_evidence(failed)

    def test_the_modules_are_listed_by_index(self):
        assert describe_modules(MODULES).splitlines()[1].startswith("1. subsymbolic: the language model")


class TestTheDecision:
    def test_a_validated_answer_is_the_rules_and_the_model_is_not_asked(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up"}')
        decide = llm_decision_fn(d, rule(lambda t, g: True), verify=lambda t, g: True)
        assert decide(state(GOOD)) == {"status": "VALIDATED", "source": "llm"} and d.runner.prompts == []

    def test_after_the_symbolic_solvers_and_with_one_action_open_the_rule_decides(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up"}')
        decide = llm_decision_fn(d, rule())
        assert decide(state("", last_dispatch="symbolic"))["action"] == "give_up"
        assert decide(state(WRONG_SHAPE, iteration=3))["action"] == "retry_llm"      # the rule's, the graph refuses it
        assert d.runner.prompts == []

    def test_the_model_chooses_among_the_open_actions(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up", "reasoning": "hopeless"}')
        decide = llm_decision_fn(d, rule())
        assert decide(state(WRONG_SHAPE)) == {"status": "INVALID", "action": "give_up"}
        assert decide.log[-1] == {"iteration": 1, "fallback": "retry_llm", "model": "give_up", "used_model": True,
                                  "reasoning": "hopeless"}

    def test_the_prompt_carries_the_evidence_the_open_actions_and_the_history(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "retry_llm"}')
        llm_decision_fn(d, rule())(state(WRONG_SHAPE, rl_handle=object()))
        (prompt,) = d.runner.prompts
        assert "RL: still training" in prompt and "retry_llm, wait_rl, give_up" in prompt
        assert "1. llm subsymbolic: PENDING" in prompt and WRONG_SHAPE in prompt
        assert "1. subsymbolic: the language model" in prompt

    def test_an_action_that_is_closed_is_not_taken(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "wait_rl"}')
        decide = llm_decision_fn(d, rule())
        assert decide(state(WRONG_SHAPE))["action"] == "retry_llm"
        assert decide.log[-1]["model"] == "wait_rl" and decide.log[-1]["used_model"] is False

    def test_a_wish_to_validate_an_unsupported_answer_is_not_a_validation(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "VALIDATED", "reasoning": "looks right"}')
        decide = llm_decision_fn(d, rule(lambda t, g: False))
        assert decide(state(GOOD)) == {"status": "INVALID", "action": "retry_llm"}
        assert decide.log[-1]["used_model"] is False

    @pytest.mark.parametrize("reply", ["no idea", "", None, RuntimeError("down")])
    def test_a_reply_that_cannot_be_used_leaves_the_rule(self, tiny_tokenizer, reply):
        d = decider(tiny_tokenizer, reply)
        decide = llm_decision_fn(d, rule())
        assert decide(state(WRONG_SHAPE))["action"] == "retry_llm" and decide.log[-1]["used_model"] is False

    def test_a_retry_may_be_handed_to_another_module_the_model_names(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "retry_llm", "delegate_to_module": 0}')
        decision = llm_decision_fn(d, rule())(state(WRONG_SHAPE))
        assert decision["next_module"].module_name == "symbolic"
        giving_up = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up", "delegate_to_module": 0}')
        assert llm_decision_fn(giving_up, rule())(state(WRONG_SHAPE)) == {"status": "INVALID", "action": "give_up"}
        unknown = decider(tiny_tokenizer, '{"status": "INVALID", "action": "retry_llm", "delegate_to_module": 9}')
        assert "next_module" not in llm_decision_fn(unknown, rule())(state(WRONG_SHAPE))

    def test_the_decision_is_timed(self, tiny_tokenizer):
        tracer = Tracer()
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up"}')
        llm_decision_fn(d, rule(), tracer=tracer)(state(WRONG_SHAPE))
        assert [s.phase for s in tracer.spans] == ["decide"] and tracer.spans[0].tokens_in > 0

    def test_a_grammar_on_the_runner_is_taken_off_for_the_decision(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "action": "give_up"}')
        d.runner.generation_kwargs = {"extra_body": {"grammar": "g"}}
        seen = []
        original = Runner.generate

        def generate(self, prompt):
            seen.append(dict(self.generation_kwargs["extra_body"]))
            return original(self, prompt)

        Runner.generate = generate
        try:
            llm_decision_fn(d, rule())(state(WRONG_SHAPE))
        finally:
            Runner.generate = original
        assert seen == [{}] and d.runner.generation_kwargs["extra_body"] == {"grammar": "g"}


AGENTS = [{"index": 1, "name": "Constructor", "purpose": "Create"}, {"index": 2, "name": "Highlighter"},
          {"index": 3, "name": "Shifter"}]


def factory(entry):
    return AgentInvConfig(agent_index=entry["index"], agent_name=entry["name"],
                          initial_module=ModuleInvConfig(0, "symbolic"), available_modules=MODULES)


def system_state(history, task=None):
    return {"task": task or make_task(), "solution": WRONG_SHAPE, "available_agents": AGENTS, "history": history,
            "iteration": len([h for h in history if h.level == "agent"]), "auxiliary_info": {}}


def agent_record(name, status):
    return InteractionRecord(1, "agent", name, "s", status)


class TestTheCoordinator:
    def test_a_validated_agent_is_accepted_and_a_failed_one_hands_over_to_the_next_untried(self):
        coordinate = delegating_coordinator_fn(factory)
        assert coordinate(system_state([agent_record("Constructor", "VALIDATED")])) == \
            {"status": "VALIDATED", "next_agent": None}
        nxt = coordinate(system_state([agent_record("Constructor", "INVALID")]))
        assert nxt["status"] == "INVALID" and nxt["next_agent"].agent_name == "Highlighter"
        done = coordinate(system_state([agent_record(a["name"], "INVALID") for a in AGENTS]))
        assert done == {"status": "INVALID", "next_agent": None}

    def test_the_model_picks_among_the_agents_left(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "delegate_to_agent": 3}', COORDINATOR_BLOCKS)
        coordinate = llm_coordinator_fn(d, factory)
        out = coordinate(system_state([agent_record("Constructor", "INVALID")]))
        assert out["next_agent"].agent_name == "Shifter" and coordinate.log[-1]["used_model"] is True
        (prompt,) = d.runner.prompts
        assert "2. Highlighter" in prompt and "3. Shifter" in prompt and "1. Constructor" not in prompt

    def test_only_text_of_the_auxiliary_information_goes_to_the_model(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "delegate_to_agent": 3}', COORDINATOR_BLOCKS)
        crowded = system_state([agent_record("Constructor", "INVALID")])
        crowded["auxiliary_info"] = {"test_input_grid": np.zeros((2, 2), int), "Agents worth trying": "1. Shifter"}
        llm_coordinator_fn(d, factory)(crowded)
        (prompt,) = d.runner.prompts
        assert "1. Shifter" in prompt and "test_input_grid" not in prompt

    def test_an_agent_already_tried_or_unknown_is_not_taken(self, tiny_tokenizer):
        for reply in ('{"status": "INVALID", "delegate_to_agent": 1}', '{"status": "INVALID", "delegate_to_agent": 9}',
                      "gibberish"):
            d = decider(tiny_tokenizer, reply, COORDINATOR_BLOCKS)
            out = llm_coordinator_fn(d, factory)(system_state([agent_record("Constructor", "INVALID")]))
            assert out["next_agent"].agent_name == "Highlighter"

    def test_one_agent_left_or_a_validated_one_is_not_a_question(self, tiny_tokenizer):
        d = decider(tiny_tokenizer, '{"status": "INVALID", "delegate_to_agent": 2}', COORDINATOR_BLOCKS)
        coordinate = llm_coordinator_fn(d, factory)
        coordinate(system_state([agent_record("Constructor", "INVALID"), agent_record("Highlighter", "INVALID")]))
        coordinate(system_state([agent_record("Constructor", "VALIDATED")]))
        assert d.runner.prompts == []
