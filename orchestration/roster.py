"""The agents as the registry lists them, with the modules they have.

`data.configs.agents_config.AGENTS_REGISTRY` says which modules each agent works with: the symbolic
solvers, the model, and the interactive one, which is RL. This turns an entry into an agent the graph
runs - strict, so that an agent without a module really lacks it - and does the one thing a run
without RL needs: take a module out of every agent (`without_module`), dropping those left with
nothing to do. That is the module-level switch, as opposed to removing agents one by one: an agent
that works with the model as well as RL stays, with the model alone.

    registry = without_module(AGENTS_REGISTRY, "Interactive")
    factory = agent_factory(registry)
    agents = order_by_analyst(registry, task, analyst, context)     # the analyst's shortlist first
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Sequence

from orchestration.graph import AgentInvConfig, ModuleInvConfig


def without_module(registry: Sequence[Mapping[str, Any]], module: str) -> List[Dict[str, Any]]:
    """`registry` with `module` taken from every agent; an agent left with no module is dropped."""
    wanted = module.lower()
    kept = []
    for agent in registry:
        entry = copy.deepcopy(dict(agent))
        entry["modules"] = [m for m in entry.get("modules", []) if m.lower() != wanted]
        entry["available_modules"] = [m for m in entry.get("available_modules", [])
                                      if str(m.get("name", "")).lower() != wanted]
        if entry["modules"] or entry["available_modules"]:
            kept.append(entry)
    return kept


def agent_factory(registry: Sequence[Mapping[str, Any]]):
    """`entry -> AgentInvConfig` for entries of `registry` (the graph's `available_agents` items).
    The agent starts with its symbolic module when it has one, and is strict."""
    by_index = {agent["index"]: agent for agent in registry}

    def make(entry: Mapping[str, Any]) -> AgentInvConfig:
        agent = by_index.get(entry["index"], entry)
        modules = [{"index": m["index"], "name": m["name"]} for m in agent.get("available_modules", [])]
        first = next((m for m in modules if m["name"].lower() == "symbolic"), modules[0] if modules else None)
        initial = ModuleInvConfig(first["index"], first["name"]) if first else ModuleInvConfig(0, "none")
        return AgentInvConfig(agent_index=agent["index"], agent_name=agent["name"], initial_module=initial,
                              available_modules=modules, strict=True)

    return make


def order_by_analyst(registry: Sequence[Mapping[str, Any]], shortlist) -> List[Dict[str, Any]]:
    """`registry` with the agents the analyst named first, in its order, and the rest after, in theirs.
    An empty shortlist (the model failed, or named nobody) leaves the registry as it is."""
    names = {name.lower(): rank for rank, name in enumerate(getattr(shortlist, "agents", ()) or ())}
    entries = [dict(a) for a in registry]
    return sorted(entries, key=lambda a: names.get(str(a["name"]).lower(), len(names)))
