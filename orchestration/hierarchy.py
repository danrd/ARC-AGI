"""Who is believed when: the symbolic solvers, then RL, then the model.

The agent graph (orchestration.graph) runs the three in parallel and leaves
the choice to a decision function. This is that function, and what it rests
on is what each source can show for its answer without the target:

    symbolic   the rule reproduced every training pair when that pair was
               held out (symbolic_module.checked_solve, run by the dispatch),
               so an answer that arrives without an error is taken as it is
    RL         the policy closed every training pair (rl_module.RLModule),
               and its held-out grid is put to the second model like the
               first model's answer. The gate is not a proof: on 945 runs
               over 51 tasks, 47 of the 201 that passed it closed the held-out
               pair, and on many tasks none did. A policy fits each pair by
               its own actions without finding the rule
    the model  nothing it says about itself counts. Its answer is read as a
               grid and put to a second model, which is asked whether the
               answer is what the training pairs imply (subsymbolic.answer_check);
               a refusal sends the first model round again

The second model's word is asked once for each RL grid and remembered, so a
retry of the first model does not ask it again.

The order is the order of trust, not of arrival: RL is usually still training
when the model answers, so the first time the model answers and RL has not
reported, the graph waits for it once (AgentRunConfig.rl_wait_timeout) before
turning to the model's answer. RL that finishes later is still picked up on
the next round. When the rounds run out the last answer stays as the agent's
solution and the agent is recorded as not validated.

    verify = LlmVerifier(other_runner, tokenizer)
    result = solve_with_hierarchy(task, verify, module_dispatch_fn)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from orchestration.graph import (AgentInvConfig, ModuleInvConfig, default_rl_start_fn, solve_task)
from subsymbolic.utils import parse_llm_output


def _has_answer(state) -> bool:
    solution = state.get("solution")
    if "error" in (state.get("module_results") or {}):
        return False
    return solution is not None and not (isinstance(solution, str) and solution == "")


def _read_grid(parse: Callable[[str], Any], text) -> Optional[np.ndarray]:
    """The model's text as a 2-D grid, or None when it does not read as one."""
    if not isinstance(text, str):
        return np.asarray(text) if isinstance(text, np.ndarray) and text.ndim == 2 else None
    try:
        grid = parse(text)
    except Exception:  # noqa: BLE001 - a malformed answer is a refusal, not a crash
        return None
    return grid if isinstance(grid, np.ndarray) and grid.ndim == 2 else None


def _rl_grid(solution) -> Optional[np.ndarray]:
    """The RL job's held-out grid as a 2-D array, or None."""
    if solution is None:
        return None
    grid = np.asarray(solution)
    return grid if grid.ndim == 2 and grid.size else None


def hierarchical_decision_fn(verify: Callable[[Any, np.ndarray], bool],
                             parse: Callable[[str], Any] = parse_llm_output
                             ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """The agent graph's decision_fn: symbolic, then RL and the model, each
    answer checked by `verify`.

    `verify(task, grid) -> bool` is the second model's verdict (LlmVerifier);
    `parse(text) -> grid` reads the first model's text - pass a partial of
    parse_llm_output for a prompt that pre-seeds the answer's header
    (expected_prefix) or letter-coded colors.
    """
    rl_verdicts: Dict[Any, bool] = {}

    def rl_accepted(state: Dict[str, Any]) -> bool:
        grid = _rl_grid(state.get("rl_solution"))
        if grid is None:
            return False
        task = state["task"]
        key = (getattr(task, "label", id(task)), grid.shape, grid.tobytes())
        if key not in rl_verdicts:
            rl_verdicts[key] = bool(verify(task, grid))
        return rl_verdicts[key]

    def decide(state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("last_dispatch") == "symbolic":
            if _has_answer(state):
                return {"status": "VALIDATED", "source": "symbolic"}
            return {"status": "INVALID", "action": "give_up"}

        if state.get("rl_status") == "ok" and rl_accepted(state):
            return {"status": "VALIDATED", "source": "rl"}

        if (state.get("rl_handle") is not None and state.get("rl_status") is None
                and not state.get("rl_wait_used")):
            return {"status": "INVALID", "action": "wait_rl"}

        grid = _read_grid(parse, state.get("solution")) if _has_answer(state) else None
        if grid is not None and verify(state["task"], grid):
            return {"status": "VALIDATED", "source": "llm"}
        return {"status": "INVALID", "action": "retry_llm"}

    return decide


def solve_with_hierarchy(task, verify: Callable[[Any, np.ndarray], bool],
                         module_dispatch_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                         rl_start_fn: Callable[[Any], Any] = default_rl_start_fn,
                         parse: Callable[[str], Any] = parse_llm_output, **solve_kwargs) -> Dict[str, Any]:
    """graph.solve_task with the symbolic solvers and the model as the agent's
    modules and hierarchical_decision_fn as its decision. `solve_kwargs` are
    solve_task's own (system_run_config, task_repr, auxiliary_info -
    the model's prompt context -, prompts_modifications)."""
    modules = [{"index": 0, "name": "symbolic"}, {"index": 1, "name": "subsymbolic"}]
    agent = AgentInvConfig(agent_index=0, agent_name="hierarchy",
                           initial_module=ModuleInvConfig(0, "symbolic"), available_modules=modules)
    return solve_task(task, initial_agent=agent, available_agents=[{"index": 0, "name": "hierarchy"}],
                      module_dispatch_fn=module_dispatch_fn, rl_start_fn=rl_start_fn,
                      agent_decision_fn=hierarchical_decision_fn(verify, parse), **solve_kwargs)
