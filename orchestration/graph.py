"""
LangGraph-based multi-agent orchestration skeleton for ARC-AGI solving.

Two-level graph:
  - System graph (Coordinator): picks an Agent, gets its solution, validates
    it, delegates to another Agent if rejected. Bounded by
    SystemRunConfig.max_system_iterations.
  - Agent graph (Decision module): gated + parallel module execution.
        1. Symbolic runs synchronously first (cheap) — if it validates, done.
        2. Otherwise RL training starts as a background subprocess
           (RLJobHandle — non-blocking, cancellable) and, without waiting on
           it, the LLM (Subsymbolic) is called (blocking, but fast — it's a
           request to an already-running server).
        3. Each time the LLM answers, the decision module looks at what's
           available (LLM's answer, and RL's result if it happened to
           already be ready) and picks ONE of: accept, retry the LLM with
           different params (RL keeps running), or wait on RL once with a
           bounded timeout. Whichever path ends the agent's turn cancels RL
           if it's still running and wasn't the accepted source.
    Bounded by AgentRunConfig.max_agent_iterations (counts symbolic + every
    LLM attempt) and AgentRunConfig.rl_wait_timeout (the RL wait is one-shot
    and bounded, never an open-ended loop).
    Compiled as its own graph and invoked as a single node from the system
    graph (standard LangGraph pattern for hierarchical agents).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from orchestration.configs import AgentRunConfig, SystemRunConfig
from rl.rl_job import RLJobHandle, default_rl_start_fn


# ============================================================================
# IDENTITY DATACLASSES
# ============================================================================

@dataclass
class ModuleInvConfig:
    """Identifies a module invocation (which module, with what params)."""
    module_index: int
    module_name: str
    config_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInvConfig:
    """Identifies an agent invocation (which agent, its modules)."""
    agent_index: int
    agent_name: str
    initial_module: ModuleInvConfig
    available_modules: List[Dict]


@dataclass
class InteractionRecord:
    """One entry in the interaction history (agent- or system-level)."""
    iteration: int
    level: str      # 'symbolic' | 'llm' | 'rl' | 'agent'
    name: str        # module_name or agent_name
    solution: str
    status: str       # 'VALIDATED' | 'INVALID' | 'ERROR' | 'PENDING' | 'CANCELLED'


# ============================================================================
# AGENT-LEVEL GRAPH: symbolic gate -> (LLM retry loop || background RL)
# ============================================================================

class AgentState(TypedDict, total=False):
    task: Any
    task_repr: str
    auxiliary_info: Dict[str, Any]
    prompts_modifications: Dict[str, str]

    current_module: ModuleInvConfig       # whichever module module_dispatch_fn should act on right now
    symbolic_module: ModuleInvConfig
    llm_module: ModuleInvConfig
    available_modules: List[Dict]
    run_config: AgentRunConfig

    iteration: int
    solution: str                          # last dispatched candidate's text (symbolic or llm)
    module_results: Dict[str, Any]         # last dispatched candidate's raw results
    last_dispatch: str                     # 'symbolic' | 'llm' — who state["solution"] came from

    rl_handle: Optional[RLJobHandle]
    rl_solution: Optional[Any]
    rl_status: Optional[str]               # None (not resolved yet) | 'ok' | 'error'
    rl_wait_used: bool

    status: str
    validated: bool
    accepted_source: Optional[str]         # 'symbolic' | 'llm' | 'rl'
    next_action: Optional[str]             # 'retry_llm' | 'wait_rl' | 'give_up' (set by decision nodes)

    module_dispatch_fn: Callable[["AgentState"], Dict[str, Any]]
    rl_start_fn: Callable[[Any], RLJobHandle]
    decision_fn: Callable[["AgentState"], Dict[str, Any]]

    history: Annotated[List[InteractionRecord], operator.add]


def _dispatch_symbolic(task: Any, symbolic_module: Optional[Any] = None) -> Dict[str, Any]:
    """Tries each of SymbolicModule's solvers in turn, returns the first
    success. Solvers need no model/config, so a default instance is built
    here if the caller didn't supply one."""
    from symbolic.symbolic_module import SymbolicModule

    module = symbolic_module or SymbolicModule()
    errors = []
    for solver in (module.mixer, module.pattern_planting, module.upscale, module.color_restore):
        result = solver.solve(task)
        if result.success:
            return {"solution": result.grid, "module_results": {"debug": result.debug}}
        errors.append(result.debug)
    return {"solution": "", "module_results": {"error": "; ".join(errors)}}


def default_module_dispatch(state: AgentState) -> Dict[str, Any]:
    """Runs `current_module`. Symbolic solvers need no model/config, so
    they get a real default here. Subsymbolic (LLM) needs a tokenizer and
    inference backend configured at the system level — pass
    module_dispatch_fn=make_module_dispatch_fn(subsymbolic_module=...) to
    solve_task() instead of relying on this default for that case."""
    module_name = state["current_module"].module_name.lower()
    if module_name == "symbolic":
        return _dispatch_symbolic(state["task"])
    return {"solution": "", "module_results": {
        "error": f"no default dispatch for module {module_name!r} — "
                 "configure it via make_module_dispatch_fn()",
    }}


def make_module_dispatch_fn(
    symbolic_module: Optional[Any] = None,
    subsymbolic_module: Optional[Any] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Builds a module_dispatch_fn for solve_task(), routing to real module
    instances assembled once at the system level — pass a
    symbolic.symbolic_module.SymbolicModule and/or a
    subsymbolic.subsymbolic_module.SubsymbolicModule (the latter needs a
    tokenizer + inference backend, too expensive/config-dependent to build
    a working default for here)."""
    def dispatch(state: AgentState) -> Dict[str, Any]:
        module_name = state["current_module"].module_name.lower()
        task = state["task"]

        if module_name == "symbolic":
            return _dispatch_symbolic(task, symbolic_module)
        if module_name == "subsymbolic":
            if subsymbolic_module is None:
                return {"solution": "", "module_results": {"error": "no SubsymbolicModule configured"}}
            return subsymbolic_module.solve(task, context=state.get("auxiliary_info", {}))
        return {"solution": "", "module_results": {"error": f"unknown module: {module_name!r}"}}

    return dispatch


def default_decision_fn(state: AgentState) -> Dict[str, Any]:
    """Accepts whatever was most recently (successfully) dispatched, never
    retries or waits on RL. Placeholder — replace with the real validator."""
    solution = state.get("solution")
    no_solution = solution is None or (isinstance(solution, str) and solution == "")
    # not "not solution": solution may be a numpy grid, whose truth value
    # (for anything but a single cell) is ambiguous to bool().
    if "error" in state.get("module_results", {}) or no_solution:
        return {"status": "INVALID", "action": "give_up"}
    return {"status": "VALIDATED", "source": state.get("last_dispatch", "symbolic")}


def _cancel_rl_if_unused(state: AgentState, accepted_source: Optional[str]) -> None:
    handle: Optional[RLJobHandle] = state.get("rl_handle")
    if handle is not None and accepted_source != "rl" and handle.process.is_alive():
        handle.cancel()


def _max_iterations(state: AgentState) -> int:
    run_config: AgentRunConfig = state.get("run_config") or AgentRunConfig()
    return run_config.max_agent_iterations


# -- nodes -------------------------------------------------------------------

def _execute_symbolic_node(state: AgentState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    dispatch_fn = state.get("module_dispatch_fn", default_module_dispatch)
    result = dispatch_fn({**state, "current_module": state["symbolic_module"]})
    solution = result.get("solution", "")
    module_results = result.get("module_results", {})

    record = InteractionRecord(
        iteration=iteration, level="symbolic", name=state["symbolic_module"].module_name,
        solution=solution, status="ERROR" if "error" in module_results else "PENDING",
    )
    return {
        "iteration": iteration, "solution": solution, "module_results": module_results,
        "last_dispatch": "symbolic", "history": [record],
    }


def _decide_symbolic_node(state: AgentState) -> Dict[str, Any]:
    decision_fn = state.get("decision_fn", default_decision_fn)
    decision = decision_fn(state)
    if decision.get("status") == "VALIDATED":
        return {"validated": True, "accepted_source": "symbolic"}
    return {"validated": False}


def _dispatch_parallel_node(state: AgentState) -> Dict[str, Any]:
    rl_start_fn = state.get("rl_start_fn", default_rl_start_fn)
    rl_handle = rl_start_fn(state["task"])

    iteration = state.get("iteration", 0) + 1
    dispatch_fn = state.get("module_dispatch_fn", default_module_dispatch)
    result = dispatch_fn({**state, "current_module": state["llm_module"]})
    solution = result.get("solution", "")
    module_results = result.get("module_results", {})

    record = InteractionRecord(
        iteration=iteration, level="llm", name=state["llm_module"].module_name,
        solution=solution, status="ERROR" if "error" in module_results else "PENDING",
    )
    return {
        "rl_handle": rl_handle, "rl_solution": None, "rl_status": None, "rl_wait_used": False,
        "iteration": iteration, "solution": solution, "module_results": module_results,
        "last_dispatch": "llm", "history": [record],
    }


def _call_llm_again_node(state: AgentState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    dispatch_fn = state.get("module_dispatch_fn", default_module_dispatch)
    result = dispatch_fn({**state, "current_module": state["llm_module"]})
    solution = result.get("solution", "")
    module_results = result.get("module_results", {})

    record = InteractionRecord(
        iteration=iteration, level="llm", name=state["llm_module"].module_name,
        solution=solution, status="ERROR" if "error" in module_results else "PENDING",
    )
    return {
        "iteration": iteration, "solution": solution, "module_results": module_results,
        "last_dispatch": "llm", "history": [record],
    }


def _poll_rl(state: AgentState) -> Dict[str, Any]:
    handle: Optional[RLJobHandle] = state.get("rl_handle")
    if handle is None or state.get("rl_status") is not None:
        return {}
    result = handle.poll()
    if result is None:
        return {}
    return {"rl_solution": result.get("solution"), "rl_status": result.get("status")}


def _decide_after_llm_node(state: AgentState) -> Dict[str, Any]:
    update = _poll_rl(state)
    merged_state = {**state, **update}

    decision_fn = merged_state.get("decision_fn", default_decision_fn)
    decision = decision_fn(merged_state)
    status = decision.get("status", "INVALID")

    if status == "VALIDATED":
        source = decision.get("source", "llm")
        solution = merged_state["solution"] if source != "rl" else merged_state.get("rl_solution")
        _cancel_rl_if_unused(merged_state, source)
        update.update({"validated": True, "accepted_source": source, "solution": solution, "next_action": None})
        return update

    action = decision.get("action", "give_up")
    if action == "retry_llm" and merged_state.get("iteration", 0) < _max_iterations(merged_state):
        next_module = decision.get("next_module") or merged_state["llm_module"]
        update.update({"validated": False, "next_action": "retry_llm", "llm_module": next_module})
        return update

    if action == "wait_rl" and not merged_state.get("rl_wait_used") and merged_state.get("rl_handle") is not None:
        update.update({"validated": False, "next_action": "wait_rl"})
        return update

    _cancel_rl_if_unused(merged_state, accepted_source=None)
    update.update({"validated": False, "next_action": "give_up"})
    return update


def _wait_for_rl_node(state: AgentState) -> Dict[str, Any]:
    run_config: AgentRunConfig = state.get("run_config") or AgentRunConfig()
    handle: Optional[RLJobHandle] = state.get("rl_handle")
    result = handle.wait(timeout=run_config.rl_wait_timeout) if handle is not None else None

    update: Dict[str, Any] = {"rl_wait_used": True}
    if result is not None:
        update["rl_solution"] = result.get("solution")
        update["rl_status"] = result.get("status")
        record = InteractionRecord(iteration=state.get("iteration", 0), level="rl",
                                    name="rl", solution=str(result.get("solution")),
                                    status="PENDING" if result.get("status") == "ok" else "ERROR")
        update["history"] = [record]
    else:
        # timed out — give up on RL, we already spent the one wait we get
        if handle is not None:
            handle.cancel()
        record = InteractionRecord(iteration=state.get("iteration", 0), level="rl",
                                    name="rl", solution="", status="CANCELLED")
        update["history"] = [record]
    return update


def _decide_final_node(state: AgentState) -> Dict[str, Any]:
    decision_fn = state.get("decision_fn", default_decision_fn)
    decision = decision_fn(state)
    status = decision.get("status", "INVALID")

    if status == "VALIDATED":
        source = decision.get("source", "llm")
        solution = state["solution"] if source != "rl" else state.get("rl_solution")
        _cancel_rl_if_unused(state, source)
        return {"validated": True, "accepted_source": source, "solution": solution, "next_action": None}

    action = decision.get("action", "give_up")
    if action == "retry_llm" and state.get("iteration", 0) < _max_iterations(state):
        next_module = decision.get("next_module") or state["llm_module"]
        return {"validated": False, "next_action": "retry_llm", "llm_module": next_module}

    _cancel_rl_if_unused(state, accepted_source=None)
    return {"validated": False, "next_action": "give_up"}


# -- routing ------------------------------------------------------------------

def _route_after_symbolic(state: AgentState) -> str:
    return END if state.get("validated") else "dispatch_parallel"


def _route_after_llm_decision(state: AgentState) -> str:
    if state.get("validated"):
        return END
    action = state.get("next_action")
    if action == "retry_llm":
        return "call_llm_again"
    if action == "wait_rl":
        return "wait_for_rl"
    return END


def _route_after_final_decision(state: AgentState) -> str:
    if state.get("validated"):
        return END
    if state.get("next_action") == "retry_llm":
        return "call_llm_again"
    return END


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("execute_symbolic", _execute_symbolic_node)
    graph.add_node("decide_symbolic", _decide_symbolic_node)
    graph.add_node("dispatch_parallel", _dispatch_parallel_node)
    graph.add_node("call_llm_again", _call_llm_again_node)
    graph.add_node("decide_after_llm", _decide_after_llm_node)
    graph.add_node("wait_for_rl", _wait_for_rl_node)
    graph.add_node("decide_final", _decide_final_node)

    graph.add_edge(START, "execute_symbolic")
    graph.add_edge("execute_symbolic", "decide_symbolic")
    graph.add_conditional_edges("decide_symbolic", _route_after_symbolic)

    graph.add_edge("dispatch_parallel", "decide_after_llm")
    graph.add_edge("call_llm_again", "decide_after_llm")
    graph.add_conditional_edges("decide_after_llm", _route_after_llm_decision)

    graph.add_edge("wait_for_rl", "decide_final")
    graph.add_conditional_edges("decide_final", _route_after_final_decision)

    return graph.compile()


AGENT_GRAPH = build_agent_graph()


# ============================================================================
# SYSTEM-LEVEL GRAPH (coordinator <-> agents)
# ============================================================================

class SystemState(TypedDict, total=False):
    task: Any
    task_repr: str
    auxiliary_info: Dict[str, Any]
    prompts_modifications: Dict[str, str]

    current_agent: AgentInvConfig
    available_agents: List[Dict]
    run_config: SystemRunConfig

    iteration: int
    solution: str
    status: str
    validated: bool
    has_next: bool

    module_dispatch_fn: Callable[[AgentState], Dict[str, Any]]
    rl_start_fn: Callable[[Any], RLJobHandle]
    agent_decision_fn: Callable[[AgentState], Dict[str, Any]]
    coordinator_fn: Callable[["SystemState"], Dict[str, Any]]

    history: Annotated[List[InteractionRecord], operator.add]


def default_coordinator_fn(state: SystemState) -> Dict[str, Any]:
    """System-level coordinator: accept the agent's solution or delegate to
    another agent. Placeholder — replace with the real LLM-backed validator."""
    return {"status": "VALIDATED", "next_agent": None}


def _find_module_by_type(available_modules: List[Dict], type_name: str) -> Optional[ModuleInvConfig]:
    """Exact (case-insensitive) match on module name — not substring, since
    "symbolic" is itself a substring of "subsymbolic" and a substring match
    would silently pick the wrong module depending on list order."""
    for module in available_modules:
        if module.get("name", "").lower() == type_name:
            return ModuleInvConfig(module_index=module["index"], module_name=module["name"], config_params={})
    return None


def _run_agent_node(state: SystemState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    agent = state["current_agent"]
    run_config: SystemRunConfig = state.get("run_config") or SystemRunConfig()

    symbolic_module = _find_module_by_type(agent.available_modules, "symbolic") or agent.initial_module
    llm_module = _find_module_by_type(agent.available_modules, "subsymbolic") or agent.initial_module

    agent_state: AgentState = {
        "task": state["task"],
        "task_repr": state["task_repr"],
        "auxiliary_info": state["auxiliary_info"],
        "prompts_modifications": state["prompts_modifications"],
        "symbolic_module": symbolic_module,
        "llm_module": llm_module,
        "available_modules": agent.available_modules,
        "run_config": run_config.agent_run_config,
        "iteration": 0,
        "module_dispatch_fn": state.get("module_dispatch_fn", default_module_dispatch),
        "rl_start_fn": state.get("rl_start_fn", default_rl_start_fn),
        "decision_fn": state.get("agent_decision_fn", default_decision_fn),
    }
    agent_result = AGENT_GRAPH.invoke(agent_state)
    solution = agent_result.get("solution", "")

    record = InteractionRecord(
        iteration=iteration, level="agent", name=agent.agent_name,
        solution=str(solution),
        status="VALIDATED" if agent_result.get("validated") else "INVALID",
    )
    return {
        "iteration": iteration,
        "solution": solution,
        "history": [record],
    }


def _coordinator_node(state: SystemState) -> Dict[str, Any]:
    coordinator_fn = state.get("coordinator_fn", default_coordinator_fn)
    decision = coordinator_fn(state)
    status = decision.get("status", "INVALID")
    next_agent = decision.get("next_agent")

    update: Dict[str, Any] = {"status": status, "validated": status == "VALIDATED", "has_next": False}
    if status != "VALIDATED" and next_agent is not None:
        update["current_agent"] = next_agent
        update["has_next"] = True
    return update


def _system_should_continue(state: SystemState) -> str:
    if state.get("validated"):
        return END
    run_config: SystemRunConfig = state.get("run_config") or SystemRunConfig()
    if state.get("iteration", 0) >= run_config.max_system_iterations:
        return END
    if not state.get("has_next"):
        return END
    return "run_agent"


def build_system_graph():
    graph = StateGraph(SystemState)
    graph.add_node("run_agent", _run_agent_node)
    graph.add_node("coordinate", _coordinator_node)
    graph.add_edge(START, "run_agent")
    graph.add_edge("run_agent", "coordinate")
    graph.add_conditional_edges("coordinate", _system_should_continue)
    return graph.compile()


SYSTEM_GRAPH = build_system_graph()

# ============================================================================
# ENTRY POINT
# ============================================================================

def solve_task(
    task: Any,
    initial_agent: AgentInvConfig,
    available_agents: List[Dict],
    system_run_config: Optional[SystemRunConfig] = None,
    task_repr: str = "",
    auxiliary_info: Optional[Dict[str, Any]] = None,
    prompts_modifications: Optional[Dict[str, str]] = None,
    module_dispatch_fn: Callable[[AgentState], Dict[str, Any]] = default_module_dispatch,
    rl_start_fn: Callable[[Any], RLJobHandle] = default_rl_start_fn,
    agent_decision_fn: Callable[[AgentState], Dict[str, Any]] = default_decision_fn,
    coordinator_fn: Callable[[SystemState], Dict[str, Any]] = default_coordinator_fn,
) -> Dict[str, Any]:
    """Run the full coordinator -> agent -> module loop for a single task.

    Returns the final SystemState dict (solution text, solution validated
    flag, interaction history). Grid parsing and scoring against the target
    are intentionally left to the caller — hook them onto result["solution"].
    """
    initial_state: SystemState = {
        "task": task,
        "task_repr": task_repr,
        "auxiliary_info": auxiliary_info or {},
        "prompts_modifications": prompts_modifications or {},
        "current_agent": initial_agent,
        "available_agents": available_agents,
        "run_config": system_run_config or SystemRunConfig(),
        "iteration": 0,
        "module_dispatch_fn": module_dispatch_fn,
        "rl_start_fn": rl_start_fn,
        "agent_decision_fn": agent_decision_fn,
        "coordinator_fn": coordinator_fn,
    }
    return SYSTEM_GRAPH.invoke(initial_state)
