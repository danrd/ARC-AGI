from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


# ============================================================================
# CONFIG / IDENTITY DATACLASSES
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
class AgentRunConfig:
    """Execution settings for the agent-level (module) loop.

    NOTE: renamed from the original `AgentConfig` to avoid a name collision —
    the source file defined two different classes both called `AgentConfig`
    (agent identity vs. agent execution settings).
    """
    max_agent_iterations: int = 3
    verbose: bool = False


@dataclass
class SystemRunConfig:
    """Execution settings for the system-level (agent) loop."""
    max_system_iterations: int = 5
    agent_run_config: AgentRunConfig = field(default_factory=AgentRunConfig)
    verbose: bool = True


@dataclass
class InteractionRecord:
    """One entry in the interaction history (agent- or system-level)."""
    iteration: int
    level: str      # 'module' or 'agent'
    name: str        # module_name or agent_name
    solution: str
    status: str       # 'VALIDATED' | 'INVALID' | 'ERROR' | 'PENDING'


# ============================================================================
# AGENT-LEVEL GRAPH (decision module <-> modules)
# ============================================================================

class AgentState(TypedDict, total=False):
    task: Any
    task_repr: str
    auxiliary_info: Dict[str, Any]
    prompts_modifications: Dict[str, str]

    current_module: ModuleInvConfig
    available_modules: List[Dict]
    run_config: AgentRunConfig

    iteration: int
    solution: str
    module_results: Dict[str, Any]
    status: str
    validated: bool
    has_next: bool
    fallback_used: bool

    module_dispatch_fn: Callable[["AgentState"], Dict[str, Any]]
    decision_fn: Callable[["AgentState"], Dict[str, Any]]

    history: Annotated[List[InteractionRecord], operator.add]


def default_module_dispatch(state: AgentState) -> Dict[str, Any]:
    """Runs `current_module`. Placeholder — this is where the real
    symbolic / subsymbolic (LLM) / interactive execution goes (the old
    compose_prompt / process_prompt call sites)."""
    return {"solution": "", "module_results": {"error": "module dispatch not wired up yet"}}


def default_decision_fn(state: AgentState) -> Dict[str, Any]:
    """Agent-level decision module: accept or delegate to another module.
    Placeholder — replace with the real LLM-backed validator."""
    return {"status": "VALIDATED", "next_module": None}


def _execute_module_node(state: AgentState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    dispatch_fn = state.get("module_dispatch_fn", default_module_dispatch)
    result = dispatch_fn(state)
    solution = result.get("solution", "")
    module_results = result.get("module_results", {})

    record = InteractionRecord(
        iteration=iteration, level="module",
        name=state["current_module"].module_name,
        solution=solution,
        status="ERROR" if "error" in module_results else "PENDING",
    )
    return {
        "iteration": iteration,
        "solution": solution,
        "module_results": module_results,
        "history": [record],
    }


def _find_subsymbolic_fallback(state: AgentState) -> Optional[ModuleInvConfig]:
    """Look for a subsymbolic module among available_modules, other than the
    one currently in use. Returns None if there isn't one."""
    current_name = state["current_module"].module_name.lower()
    for module in state.get("available_modules", []):
        name = module.get("name", "")
        if "subsymbolic" in name.lower() and name.lower() != current_name:
            return ModuleInvConfig(module_index=module["index"], module_name=name, config_params={})
    return None


def _agent_decision_node(state: AgentState) -> Dict[str, Any]:
    if "error" in state.get("module_results", {}):
        # One-shot fallback: on module failure, try the subsymbolic module
        # once before giving up (mirrors the original _select_fallback_module).
        if not state.get("fallback_used"):
            fallback = _find_subsymbolic_fallback(state)
            if fallback is not None:
                return {
                    "status": "ERROR", "validated": False, "has_next": True,
                    "current_module": fallback, "fallback_used": True,
                }
        return {"status": "ERROR", "validated": False, "has_next": False}

    decision_fn = state.get("decision_fn", default_decision_fn)
    decision = decision_fn(state)
    status = decision.get("status", "INVALID")
    next_module = decision.get("next_module")

    update: Dict[str, Any] = {"status": status, "validated": status == "VALIDATED", "has_next": False}
    if status != "VALIDATED" and next_module is not None:
        update["current_module"] = next_module
        update["has_next"] = True
    return update


def _agent_should_continue(state: AgentState) -> str:
    if state.get("validated"):
        return END
    run_config: AgentRunConfig = state.get("run_config") or AgentRunConfig()
    if state.get("iteration", 0) >= run_config.max_agent_iterations:
        return END
    if not state.get("has_next"):
        return END
    return "execute_module"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("execute_module", _execute_module_node)
    graph.add_node("decide", _agent_decision_node)
    graph.add_edge(START, "execute_module")
    graph.add_edge("execute_module", "decide")
    graph.add_conditional_edges("decide", _agent_should_continue)
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
    agent_decision_fn: Callable[[AgentState], Dict[str, Any]]
    coordinator_fn: Callable[["SystemState"], Dict[str, Any]]

    history: Annotated[List[InteractionRecord], operator.add]


def default_coordinator_fn(state: SystemState) -> Dict[str, Any]:
    """System-level coordinator: accept the agent's solution or delegate to
    another agent. Placeholder — replace with the real LLM-backed validator."""
    return {"status": "VALIDATED", "next_agent": None}


def _run_agent_node(state: SystemState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    agent = state["current_agent"]
    run_config: SystemRunConfig = state.get("run_config") or SystemRunConfig()

    agent_state: AgentState = {
        "task": state["task"],
        "task_repr": state["task_repr"],
        "auxiliary_info": state["auxiliary_info"],
        "prompts_modifications": state["prompts_modifications"],
        "current_module": agent.initial_module,
        "available_modules": agent.available_modules,
        "run_config": run_config.agent_run_config,
        "iteration": 0,
        "module_dispatch_fn": state.get("module_dispatch_fn", default_module_dispatch),
        "decision_fn": state.get("agent_decision_fn", default_decision_fn),
    }
    agent_result = AGENT_GRAPH.invoke(agent_state)
    solution = agent_result.get("solution", "")
    agent_status = agent_result.get("status")

    record = InteractionRecord(
        iteration=iteration, level="agent", name=agent.agent_name,
        solution=solution,
        status="ERROR" if agent_status == "ERROR" else "PENDING",
    )
    return {
        "iteration": iteration,
        "solution": solution,
        "status": "ERROR" if agent_status == "ERROR" else None,
        "history": [record],
    }


def _coordinator_node(state: SystemState) -> Dict[str, Any]:
    if state.get("status") == "ERROR":
        return {"validated": False, "has_next": False}

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
    if state.get("validated") or state.get("status") == "ERROR":
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
    agent_decision_fn: Callable[[AgentState], Dict[str, Any]] = default_decision_fn,
    coordinator_fn: Callable[[SystemState], Dict[str, Any]] = default_coordinator_fn,
) -> Dict[str, Any]:
    """Run the full coordinator -> agent -> module loop for a single task.

    Returns the final SystemState dict (solution text, status, validated flag,
    interaction history). Grid parsing and scoring against the target are
    intentionally left to the caller — hook them onto result["solution"].
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
        "agent_decision_fn": agent_decision_fn,
        "coordinator_fn": coordinator_fn,
    }
    return SYSTEM_GRAPH.invoke(initial_state)