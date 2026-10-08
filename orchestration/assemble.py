"""Putting the orchestration pieces together from the options.

`graph.solve_task` takes three functions - how a module is run, how an agent decides, how
the system coordinates - and the rest of the orchestration package is what can be put in
them. This reads `OrchestrationOptions` and builds the three:

    options                  what changes
    -----------------------  -----------------------------------------------------------
    (none: the defaults)     the rule decides (hierarchical_decision_fn); the model is asked
                             once per round; the system graph accepts the agent's answer
    info_tools               the model may reply `REQUEST: summary|search_hints` and is asked again
    refine_rounds > 1        the model is asked again with its answer and what was wrong with it,
                             up to that many calls per round
    feedback                 the graph's own retries carry the history too
    decision = "llm"         a model chooses among the open actions (the rule still validates)
    coordinator = "llm"      a model picks the next agent when one fails

Nothing replaces the rule. Acceptance of an answer is `hierarchical_decision_fn`'s in every
setting, and the model's choices are made inside what it leaves open (llm_orchestrator).

A `Tracer`, when given, is told the time and the tokens of every call, so one run says where it
went: the symbolic solvers, the model, the second model, the decision, the wait for RL.

    run = assemble(options, module, verify, tracer=Tracer())
    result = solve_with_orchestration(task, run)
    run.tracer.summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from orchestration.blocks import install_blocks
from orchestration.configs import OrchestrationOptions
from orchestration.feedback import memoize_verdicts, with_feedback
from orchestration.graph import (AgentInvConfig, ModuleInvConfig, default_coordinator_fn, default_rl_start_fn,
                                 make_module_dispatch_fn, solve_task)
from orchestration.hierarchy import hierarchical_decision_fn
from orchestration.llm_orchestrator import (COORDINATOR_BLOCKS, decision_config, delegating_coordinator_fn,
                                            llm_coordinator_fn, llm_decision_fn)
from orchestration.refine import RefiningModule
from orchestration.tools import ToolUsingModule, default_tools
from orchestration.trace import Tracer, traced
from subsymbolic.prompt_builder import PromptBuilder
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY
from subsymbolic.utils import parse_llm_output


@dataclass
class Orchestration:
    """The three functions for solve_task, and what was kept to read the run by."""
    options: OrchestrationOptions
    module_dispatch_fn: Callable
    agent_decision_fn: Callable
    coordinator_fn: Callable
    rl_start_fn: Callable
    tracer: Optional[Tracer] = None
    module: Any = None
    verify: Optional[Callable] = None

    def kwargs(self) -> Dict[str, Any]:
        return {"module_dispatch_fn": self.module_dispatch_fn, "agent_decision_fn": self.agent_decision_fn,
                "coordinator_fn": self.coordinator_fn, "rl_start_fn": self.rl_start_fn}

    @property
    def decisions(self) -> List[Dict[str, Any]]:
        return list(getattr(self.agent_decision_fn, "log", []))


def make_decider(module, blocks=None):
    """`module`'s runner with a builder for the decision (or other `blocks`): the shape llm_decision_fn takes."""
    config = decision_config(module.builder.config, blocks) if blocks else decision_config(module.builder.config)
    builder = PromptBuilder(config, module.builder.tokenizer, resolver_registry=RESOLVER_REGISTRY,
                            filter_registry=FILTER_REGISTRY)
    return SimpleNamespace(builder=builder, runner=module.runner)


def timed_dispatch(dispatch_fn: Callable, tracer: Optional[Tracer]) -> Callable:
    """`dispatch_fn` with each call a span: `symbolic` for the solvers, `llm` for the model."""
    if tracer is None:
        return dispatch_fn

    def dispatch(state: Dict[str, Any]) -> Dict[str, Any]:
        name = state["current_module"].module_name.lower()
        with traced(tracer, "symbolic" if name == "symbolic" else "llm", name):
            return dispatch_fn(state)

    return dispatch


class _TimedHandle:
    """An RL job handle whose `wait` is a span: the time the graph stands waiting for RL."""

    def __init__(self, handle, tracer: Tracer):
        self._handle, self._tracer = handle, tracer

    def wait(self, timeout=None):
        with self._tracer.span("rl_wait", "wait for RL"):
            return self._handle.wait(timeout=timeout)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def timed_rl_start(rl_start_fn: Callable, tracer: Optional[Tracer]) -> Callable:
    if tracer is None:
        return rl_start_fn

    def start(task):
        handle = rl_start_fn(task)
        return _TimedHandle(handle, tracer) if handle is not None else handle

    return start


def timed_verify(verify: Callable, tracer: Optional[Tracer]) -> Callable:
    if tracer is None:
        return verify

    def timed(task, grid):
        with tracer.span("verify", "second model"):
            return verify(task, grid)

    return timed


def assemble(options: OrchestrationOptions, module, verify: Callable[[Any, np.ndarray], bool],
             decider=None, tracer: Optional[Tracer] = None, parse: Callable[[str], Any] = parse_llm_output,
             agent_factory: Optional[Callable[[Dict[str, Any]], AgentInvConfig]] = None,
             rl_start_fn: Callable = default_rl_start_fn, symbolic_module=None) -> Orchestration:
    """The functions for solve_task that `options` ask for, around `module` (a SubsymbolicModule)
    and `verify` (the second model's verdict, `LlmVerifier`). `decider` is the module that makes
    the model-made decisions, `module` itself when not given."""
    verdict = memoize_verdicts(timed_verify(verify, tracer))
    decider = decider or module

    solver = module
    if options.info_tools:
        solver = ToolUsingModule(solver, default_tools(), options.max_info_requests, tracer)
    if options.feedback:
        install_blocks(solver, ["memory"])
    if options.refine_rounds and options.refine_rounds > 1:
        solver = RefiningModule(solver, rounds=options.refine_rounds, verify=verdict, parse=parse, tracer=tracer)

    dispatch = make_module_dispatch_fn(symbolic_module=symbolic_module, subsymbolic_module=solver)
    dispatch = timed_dispatch(dispatch, tracer)
    if options.feedback:
        dispatch = with_feedback(dispatch, verdict, parse, tracer)

    rule = hierarchical_decision_fn(verdict, parse)
    if options.decision == "llm":
        decision = llm_decision_fn(make_decider(decider), rule, verdict, parse, tracer)
    else:
        def decision(state, _rule=rule):
            with traced(tracer, "decide", "rule"):
                return _rule(state)

    if options.coordinator == "llm" and agent_factory is not None:
        coordinator = llm_coordinator_fn(make_decider(decider, COORDINATOR_BLOCKS), agent_factory, tracer=tracer)
    elif agent_factory is not None:
        coordinator = delegating_coordinator_fn(agent_factory)
    else:
        coordinator = default_coordinator_fn

    return Orchestration(options=options, module_dispatch_fn=dispatch, agent_decision_fn=decision,
                         coordinator_fn=coordinator, rl_start_fn=timed_rl_start(rl_start_fn, tracer),
                         tracer=tracer, module=solver, verify=verdict)


def solve_with_orchestration(task, run: Orchestration, agents: Optional[List[Dict[str, Any]]] = None,
                             **solve_kwargs) -> Dict[str, Any]:
    """graph.solve_task with the assembled functions. With no `agents`, one agent holds the symbolic
    solvers and the model, as in `solve_with_hierarchy`."""
    modules = [{"index": 0, "name": "symbolic"}, {"index": 1, "name": "subsymbolic"}]
    agents = agents or [{"index": 0, "name": "hierarchy"}]
    first = agents[0]
    agent = AgentInvConfig(agent_index=first["index"], agent_name=first["name"],
                           initial_module=ModuleInvConfig(0, "symbolic"), available_modules=modules)
    return solve_task(task, initial_agent=agent, available_agents=agents, **run.kwargs(), **solve_kwargs)
