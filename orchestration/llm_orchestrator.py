"""The orchestrator as a model: it reads the evidence and chooses the next step.

The graph (orchestration.graph) is the loop and `hierarchical_decision_fn` is the
rule: symbolic first, RL and the model each only if a second model agrees, one wait
for RL, retry or give up. That rule is deterministic and it is what runs by default.
This is the other orchestrator, the one `decision_instruction` and
`coordinator_instruction` were written for: a model is shown what each source has
produced and what the system checked about it, and says what to do.

It is *guarded*. The model chooses among the actions that are open - retry the model,
wait once for RL, give up - and cannot open one that is closed; and it cannot validate
an answer the evidence does not support, because acceptance is the deterministic rule's
and the model's "VALIDATED" is read as a wish. What is left to the model is the part
that is a judgement: whether a refused answer is worth another try, whether to wait for
RL that has not reported, which agent to hand a task to next. Whatever goes wrong - the
reply is not JSON, it names an action that is closed, the runner raises - the
deterministic choice is made instead, and the log says that the model's was not used.

    decide = llm_decision_fn(decider, hierarchical_decision_fn(verify), verify=verify)
    coordinator = llm_coordinator_fn(decider, agent_factory)

`decider` is anything with SubsymbolicModule's shape (a `.builder` with the decision
blocks and a `.runner`): it may be the model that solves, or a second one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from orchestration.feedback import UNREADABLE, review
from orchestration.graph import AgentInvConfig, ModuleInvConfig
from orchestration.hierarchy import _read_grid, _rl_grid
from orchestration.tools import without_grammar
from orchestration.trace import Tracer, traced
from subsymbolic.analyst import _load_json
from subsymbolic.prompt_builder import PromptingConfig
from subsymbolic.utils import parse_llm_output
from symbolic.invariants import learn

DECISION_BLOCKS = ["decision_instruction", "modules_info", "memory", "solution", "decision_evidence"]
COORDINATOR_BLOCKS = ["coordinator_instruction", "agents_info", "auxiliary_info", "memory", "solution"]

MODULE_NOTES = {
    "symbolic": "rule solvers; an answer is checked against every training pair before it is given",
    "subsymbolic": "the language model; its answer is checked by the system and a second model",
}


def decision_config(base: Optional[PromptingConfig] = None, blocks: Sequence[str] = DECISION_BLOCKS) -> PromptingConfig:
    """The prompting config of the decision: `base` (so the chat template and token limit carry over) with the blocks."""
    base = base or PromptingConfig(token_limit=9000)
    return base.model_copy(update={"blocks": list(blocks), "resolvers": [], "filters": [], "assistant_prefix": None})


@dataclass(frozen=True)
class Choice:
    """What the model said, read: a status, an action, the module it would hand the retry to."""
    status: str
    action: Optional[str] = None
    delegate_to_module: Optional[int] = None
    delegate_to_agent: Optional[int] = None
    reasoning: str = ""


def parse_choice(text) -> Optional[Choice]:
    """The model's JSON reply as a Choice, or None when it is not one."""
    if not isinstance(text, str) or not text.strip():
        return None
    payload, why = _load_json(text)
    if why is not None or not isinstance(payload, dict):
        return None
    status = str(payload.get("status", "")).strip().upper()
    if status not in ("VALIDATED", "INVALID"):
        return None

    def index(key):
        value = payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    action = payload.get("action")
    return Choice(status=status, action=action.strip().lower() if isinstance(action, str) else None,
                  delegate_to_module=index("delegate_to_module"), delegate_to_agent=index("delegate_to_agent"),
                  reasoning=str(payload.get("reasoning", ""))[:500])


# -- what the model is shown ---------------------------------------------------------------

def open_actions(state: Mapping[str, Any]) -> List[str]:
    """The actions the graph would carry out: retry while there are iterations left, wait for a
    background RL job that has not reported and has not been waited on, and giving up."""
    run_config = state.get("run_config")
    limit = getattr(run_config, "max_agent_iterations", 3)
    actions = []
    if state.get("iteration", 0) < limit:
        actions.append("retry_llm")
    if state.get("rl_handle") is not None and state.get("rl_status") is None and not state.get("rl_wait_used"):
        actions.append("wait_rl")
    actions.append("give_up")
    return actions


def describe_evidence(state: Mapping[str, Any], verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
                      parse: Callable[[str], Any] = parse_llm_output) -> str:
    """What each source has and what the system found out about it, a line a source."""
    lines = ["- symbolic solvers: no answer that held to every training pair."]
    task = state["task"]

    if state.get("rl_handle") is None:
        lines.append("- RL: not running.")
    elif state.get("rl_status") is None:
        lines.append("- RL: still training in the background; it has not reported.")
    elif state.get("rl_status") != "ok" or _rl_grid(state.get("rl_solution")) is None:
        lines.append("- RL: finished without an answer.")
    else:
        verdict = ""
        if verify is not None:
            verdict = (" The second model accepted its answer." if verify(task, _rl_grid(state["rl_solution"]))
                       else " The second model did not accept its answer.")
        lines.append("- RL: finished; its policy closed every training pair." + verdict)

    solution = state.get("solution")
    has_answer = solution not in (None, "") and "error" not in (state.get("module_results") or {})
    grid = _read_grid(parse, solution) if has_answer else None
    if not has_answer:
        lines.append("- language model: no answer.")
    elif grid is None:
        lines.append(f"- language model: answered, and {UNREADABLE[0].lower() + UNREADABLE[1:]}")
    else:
        notes, _, _ = review(task, grid, learn([(s.train_inp, s.train_out) for s in task.subtasks]), None)
        found = "it breaks nothing the training pairs agree on" if not notes else \
            "it breaks: " + " ".join(notes)
        verdict = ""
        if verify is not None:
            verdict = (" The second model accepted it." if verify(task, grid)
                       else " The second model did not accept it.")
        lines.append(f"- language model: answered with a {grid.shape[0]}x{grid.shape[1]} grid; {found}.{verdict}")
    return "\n".join(lines)


def describe_modules(modules: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"{m['index']}. {m['name']}: {MODULE_NOTES.get(str(m['name']).lower(), '')}".rstrip(": ")
                     for m in modules)


def render_records(records) -> str:
    """The graph's own history as lines for the `memory` block."""
    return "\n".join(f"{r.iteration}. {r.level} {r.name}: {r.status}" for r in records or [])


# -- the agent-level decision --------------------------------------------------------------

def llm_decision_fn(decider, fallback: Callable[[Dict[str, Any]], Dict[str, Any]],
                    verify: Optional[Callable[[Any, np.ndarray], bool]] = None,
                    parse: Callable[[str], Any] = parse_llm_output, tracer: Optional[Tracer] = None):
    """The agent graph's decision_fn with a model choosing among the open actions.

    `fallback` is the deterministic decision (hierarchical_decision_fn). It decides when the
    evidence validates an answer, when only one action is open, and whenever the model's reply
    cannot be used. `verify` should be a memoised verifier, so that showing the second model's
    verdict to the model costs nothing the fallback has not already paid.

    The returned function has a `.log`: one entry per decision, with what the fallback chose,
    what the model chose, and whether the model's choice was the one used."""
    free = without_grammar(decider.runner)
    log: List[Dict[str, Any]] = []

    def decide(state: Dict[str, Any]) -> Dict[str, Any]:
        base = fallback(state)
        entry = {"iteration": state.get("iteration", 0), "fallback": base.get("action") or base.get("status"),
                 "model": None, "used_model": False, "reasoning": ""}
        log.append(entry)
        if base.get("status") == "VALIDATED" or state.get("last_dispatch") == "symbolic":
            return base
        actions = open_actions(state)
        if len(actions) < 2:
            return base

        context = {
            "modules_info_text": describe_modules(state.get("available_modules") or []),
            "history_text": render_records(state.get("history")),
            "current_iteration": state.get("iteration", 0),
            "current_solution": state.get("solution") if isinstance(state.get("solution"), str) else "(a grid)",
            "evidence_text": describe_evidence(state, verify, parse),
            "open_actions": ", ".join(actions),
        }
        prompt = decider.builder.build(state["task"], context=context)
        if prompt is None:
            entry["reasoning"] = "the decision prompt did not fit"
            return base
        try:
            with traced(tracer, "decide", f"iteration {entry['iteration']}",
                        tokens_in=decider.builder.count_tokens(prompt)) as span:
                reply = free.generate(prompt)
                span.tokens_out = decider.builder.count_tokens(reply) if isinstance(reply, str) else 0
        except Exception as error:  # noqa: BLE001 - an orchestrator that fails is the rule, not a crash
            entry["reasoning"] = f"the model failed: {type(error).__name__}"
            return base
        choice = parse_choice(reply)
        if choice is None:
            entry["reasoning"] = "the reply was not a decision"
            return base
        entry["model"], entry["reasoning"] = choice.action or choice.status, choice.reasoning
        if choice.status != "INVALID" or choice.action not in actions:
            return base                          # a wish to validate, or an action that is closed

        entry["used_model"] = True
        decision: Dict[str, Any] = {"status": "INVALID", "action": choice.action}
        target = _module_by_index(state.get("available_modules") or [], choice.delegate_to_module)
        if choice.action == "retry_llm" and target is not None:
            decision["next_module"] = target
        return decision

    decide.log = log
    return decide


def _module_by_index(modules: Sequence[Mapping[str, Any]], index: Optional[int]) -> Optional[ModuleInvConfig]:
    for module in modules:
        if index is not None and module.get("index") == index:
            return ModuleInvConfig(module_index=module["index"], module_name=module["name"], config_params={})
    return None


# -- the system-level coordinator ----------------------------------------------------------

def _tried(state: Mapping[str, Any]) -> List[str]:
    return [r.name for r in state.get("history") or [] if r.level == "agent"]


def _last_agent_validated(state: Mapping[str, Any]) -> bool:
    agents = [r for r in state.get("history") or [] if r.level == "agent"]
    return bool(agents) and agents[-1].status == "VALIDATED"


def _untried(state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    tried = set(_tried(state))
    return [a for a in state.get("available_agents") or [] if a.get("name") not in tried]


def delegating_coordinator_fn(agent_factory: Callable[[Mapping[str, Any]], AgentInvConfig]):
    """The deterministic coordinator: accept a validated agent, otherwise hand the task to the next
    agent of the roster that has not had it, and stop when there is none."""
    def coordinate(state: Dict[str, Any]) -> Dict[str, Any]:
        if _last_agent_validated(state):
            return {"status": "VALIDATED", "next_agent": None}
        untried = _untried(state)
        if not untried:
            return {"status": "INVALID", "next_agent": None}
        return {"status": "INVALID", "next_agent": agent_factory(untried[0])}

    return coordinate


def llm_coordinator_fn(decider, agent_factory: Callable[[Mapping[str, Any]], AgentInvConfig],
                       fallback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                       tracer: Optional[Tracer] = None):
    """A coordinator that asks a model which agent to hand the task to when the last one failed.

    Accepting is not its to do: an agent that validated is accepted as it is. With one agent left,
    or a reply that names none of the agents left, the deterministic coordinator's choice stands.
    The agents' relevance, if the analyst ranked them, is in the state's `auxiliary_info`."""
    fallback = fallback or delegating_coordinator_fn(agent_factory)
    free = without_grammar(decider.runner)
    log: List[Dict[str, Any]] = []

    def coordinate(state: Dict[str, Any]) -> Dict[str, Any]:
        base = fallback(state)
        if _last_agent_validated(state) or base.get("next_agent") is None:
            return base
        untried = _untried(state)
        entry = {"fallback": getattr(base["next_agent"], "agent_name", None), "model": None, "used_model": False}
        log.append(entry)
        if len(untried) < 2:
            return base
        roster = "\n".join(f"{a['index']}. {a['name']}" + (f": {a['purpose']}" if a.get("purpose") else "")
                           for a in untried)
        context = {
            "agents_info_text": roster,
            "auxiliary_info": state.get("auxiliary_info") or {},
            "history_text": render_records(state.get("history")),
            "current_iteration": state.get("iteration", 0),
            "current_solution": state.get("solution") if isinstance(state.get("solution"), str) else "(a grid)",
        }
        prompt = decider.builder.build(state["task"], context=context)
        if prompt is None:
            return base
        try:
            with traced(tracer, "decide", "coordinator", tokens_in=decider.builder.count_tokens(prompt)) as span:
                reply = free.generate(prompt)
                span.tokens_out = decider.builder.count_tokens(reply) if isinstance(reply, str) else 0
        except Exception:  # noqa: BLE001
            return base
        choice = parse_choice(reply)
        chosen = next((a for a in untried if choice is not None and a.get("index") == choice.delegate_to_agent), None)
        entry["model"] = choice.delegate_to_agent if choice else None
        if chosen is None:
            return base
        entry["used_model"] = True
        return {"status": "INVALID", "next_agent": agent_factory(chosen)}

    coordinate.log = log
    return coordinate
