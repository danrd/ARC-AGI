"""Tests for orchestration.graph - the coordinator/agent/module control flow
itself (routing, iteration bounds, RL start/cancel), not the individual
solvers those modules wrap.

RL is never actually trained here: real training needs torch/stable-
baselines3 and is slow, so every test injects a fake rl_start_fn
(_FakeRLHandle) that never resolves and is safely cancellable, instead of
rl.rl_job.default_rl_start_fn. Symbolic and subsymbolic get exercised for
real - they're cheap (no training) and the whole point of these tests is to
catch it if the graph stops wiring them correctly.
"""
from __future__ import annotations

import numpy as np

from orchestration.graph import (
    AgentInvConfig,
    AgentRunConfig,
    ModuleInvConfig,
    SystemRunConfig,
    default_decision_fn,
    make_module_dispatch_fn,
    solve_task,
)


class _FakeRLHandle:
    """Stands in for rl.rl_job.RLJobHandle without spawning a real subprocess
    - poll()/wait() never resolve, cancel() just flips a flag. Keeps these
    tests from needing torch/stable-baselines3 or actually training anything."""

    def __init__(self):
        self.cancelled = False

    def poll(self):
        return None

    def wait(self, timeout):
        return None

    def cancel(self, timeout=5.0):
        self.cancelled = True

    @property
    def process(self):
        return self

    def is_alive(self):
        return not self.cancelled


def _fake_rl_start_fn(task):
    return _FakeRLHandle()


def _agent(available_modules):
    return AgentInvConfig(
        agent_index=0, agent_name="test-agent",
        initial_module=ModuleInvConfig(0, available_modules[0]["name"]),
        available_modules=available_modules,
    )


AVAILABLE_MODULES = [{"index": 0, "name": "symbolic"}, {"index": 1, "name": "subsymbolic"}]
OK = {"solution": "OK", "module_results": {}}
FAIL = {"solution": "", "module_results": {"error": "no solve"}}


def _agent_record(result):
    """The agent-level InteractionRecord from a solve_task() result. Needed
    for the "gives up" tests below because default_coordinator_fn is an
    always-accept placeholder (documented as such) - result["validated"] is
    True regardless of whether the agent itself succeeded, so the agent's
    own outcome has to be read from its history entry instead."""
    return next(r for r in result["history"] if r.level == "agent")


def _canned_dispatch_fn(symbolic_result, subsymbolic_result):
    """Deterministic module_dispatch_fn stand-in: returns a fixed result per
    module name, so tests exercise the graph's routing logic without
    depending on whether a real solver actually succeeds on some task."""
    def dispatch(state):
        module_name = state["current_module"].module_name.lower()
        if module_name == "symbolic":
            return symbolic_result
        if module_name == "subsymbolic":
            return subsymbolic_result
        raise AssertionError(f"unexpected module dispatched: {module_name!r}")
    return dispatch


def test_default_decision_fn_handles_grid_solution():
    """Regression test: default_decision_fn used to do `not state.get(
    "solution")`, which raises for a numpy grid ("truth value of an array...
    is ambiguous") - exactly what a real symbolic solve returns, so it broke
    the moment a solver actually succeeded on a real grid."""
    state = {"solution": np.zeros((3, 3), dtype=int), "module_results": {}, "last_dispatch": "symbolic"}
    assert default_decision_fn(state)["status"] == "VALIDATED"


def test_symbolic_success_short_circuits_before_subsymbolic(arc_task):
    """If the symbolic module validates, the graph should end there - the
    subsymbolic module is never dispatched."""
    calls = []

    def dispatch(state):
        module_name = state["current_module"].module_name.lower()
        calls.append(module_name)
        return OK if module_name == "symbolic" else FAIL

    agent = _agent(AVAILABLE_MODULES)
    result = solve_task(
        arc_task, initial_agent=agent, available_agents=[{"index": 0, "name": "test-agent"}],
        module_dispatch_fn=dispatch, rl_start_fn=_fake_rl_start_fn,
    )

    assert result["validated"] is True
    assert result["solution"] == "OK"
    assert calls == ["symbolic"]


def test_falls_through_to_subsymbolic_and_cancels_unused_rl(arc_task):
    """Symbolic fails -> RL starts in the background and the LLM is called;
    once the LLM's answer validates, the still-running RL job (never used)
    gets cancelled."""
    rl_handles = []

    def rl_start_fn(task):
        handle = _FakeRLHandle()
        rl_handles.append(handle)
        return handle

    agent = _agent(AVAILABLE_MODULES)
    result = solve_task(
        arc_task, initial_agent=agent, available_agents=[{"index": 0, "name": "test-agent"}],
        module_dispatch_fn=_canned_dispatch_fn(FAIL, OK), rl_start_fn=rl_start_fn,
    )

    assert result["validated"] is True
    assert result["solution"] == "OK"
    assert len(rl_handles) == 1
    assert rl_handles[0].cancelled is True


def test_gives_up_when_every_module_fails(arc_task):
    agent = _agent(AVAILABLE_MODULES)
    result = solve_task(
        arc_task, initial_agent=agent, available_agents=[{"index": 0, "name": "test-agent"}],
        module_dispatch_fn=_canned_dispatch_fn(FAIL, FAIL), rl_start_fn=_fake_rl_start_fn,
    )
    assert _agent_record(result).status == "INVALID"


def test_retry_llm_is_bounded_by_max_agent_iterations(arc_task):
    """A decision_fn that always asks to retry must still terminate, bounded
    by AgentRunConfig.max_agent_iterations rather than looping forever."""
    llm_calls = {"count": 0}

    def dispatch(state):
        if state["current_module"].module_name.lower() == "subsymbolic":
            llm_calls["count"] += 1
        return FAIL

    def always_retry_decision_fn(state):
        return {"status": "INVALID", "action": "retry_llm"}

    agent = _agent(AVAILABLE_MODULES)
    run_config = SystemRunConfig(agent_run_config=AgentRunConfig(max_agent_iterations=2))
    result = solve_task(
        arc_task, initial_agent=agent, available_agents=[{"index": 0, "name": "test-agent"}],
        module_dispatch_fn=dispatch, rl_start_fn=_fake_rl_start_fn,
        agent_decision_fn=always_retry_decision_fn, system_run_config=run_config,
    )

    assert _agent_record(result).status == "INVALID"
    assert llm_calls["count"] <= 2


def test_real_symbolic_path_solves_a_real_task(arc_task):
    """End-to-end with the actual default wiring (default_module_dispatch ->
    symbolic.symbolic_module.SymbolicModule) - the same path orchestration's
    __main__.py stub runs. Confirms the real integration, not just the fake
    routing logic exercised by the tests above."""
    agent = _agent([{"index": 0, "name": "symbolic"}])
    result = solve_task(arc_task, initial_agent=agent, available_agents=[{"index": 0, "name": "test-agent"}],
                         rl_start_fn=_fake_rl_start_fn)

    assert result["validated"] is True
    assert np.array_equal(result["solution"], arc_task.test_out)


def test_make_module_dispatch_fn_routes_symbolic_and_subsymbolic(arc_task, tiny_tokenizer):
    """Unit-level check of make_module_dispatch_fn's routing to real module
    instances (not the full graph) - symbolic needs no setup, subsymbolic
    gets a fake runner injected (same trick as test_llm_smoke.py) so this
    doesn't need a real model."""
    from orchestration.configs import ExperimentConfig
    from subsymbolic.prompt_builder import PromptingConfig
    from subsymbolic.subsymbolic_module import SubsymbolicModule
    from symbolic.symbolic_module import SymbolicModule

    pconf = PromptingConfig(blocks=["general_instruction", "examples", "output_format"],
                             token_limit=4096, min_examples=1, filters=["grid"], resolvers=["examples"])
    subsymbolic_module = SubsymbolicModule(pconf, tiny_tokenizer, ExperimentConfig())

    class _FakeRunner:
        def generate(self, prompt):
            return "FAKE_OUTPUT"

    subsymbolic_module._runner = _FakeRunner()

    dispatch = make_module_dispatch_fn(symbolic_module=SymbolicModule(), subsymbolic_module=subsymbolic_module)

    symbolic_result = dispatch({"current_module": ModuleInvConfig(0, "symbolic"), "task": arc_task})
    assert "error" not in symbolic_result["module_results"]

    context = {"color_mapping_text": "x", "role_text": "y", "grid_repr_type": "concise",
               "test_input_grid": arc_task.test_subtask.train_inp}
    subsymbolic_result = dispatch({
        "current_module": ModuleInvConfig(0, "subsymbolic"), "task": arc_task, "auxiliary_info": context,
    })
    assert subsymbolic_result == {"solution": "FAKE_OUTPUT", "module_results": {}}
