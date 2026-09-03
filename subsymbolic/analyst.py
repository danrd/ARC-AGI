"""The analyst: reads what the symbolic analysis found and says which
specialized agents are worth trying on the task, in what order.

Its output is what `coordinator_instruction/v1.j2` calls "agent relevance
scores", and the ordering it produces is the whole of it. Deliberately not
numeric: measured over ARC-AGI-1's 800 hand-labelled tasks, a classifier on
the symbolic features ranked well (the right agent was in its top 3 for
91.6% of tasks, and 1.73 agents were tried on average before hitting it)
while its confidence said almost nothing about correctness - accepting only
predictions above 0.8 bought 4 points of accuracy for a fifth of the tasks.
An LLM's self-reported scores will be worse on that count, not better, so
what it is asked for is a ranked shortlist. Numbers in the response would
only invite the coordinator to read a difference between 0.72 and 0.68 that
is not there.

Those same measurements are the bar this has to clear: a shortlist of three
that misses the right agent more than 8.4% of the time is worse than
counting shapes and colours.

The roster is read from data.configs.agents_config at call time rather than
baked in - agents get added and dropped, and nothing here should need to
change when they do.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: How many agents to ask for. Three because that is where the measured
#: coverage curve flattens: top-2 catches 82.5% of tasks, top-3 91.6%,
#: top-4 95.4% - the third slot is worth 9 points, the fourth under 4.
DEFAULT_SHORTLIST_SIZE = 3

#: Prompt blocks the analyst is built from. `summary` is the symbolic
#: findings (subsymbolic.registry's resolver), `examples` the training
#: pairs - the analyst reads the same evidence a solver would, minus the
#: instruction to solve anything. Not `examples_intro`: that block frames
#: the pairs as something to learn a transformation from "to later predict
#: the output", which is the solver's job and contradicts the line above it
#: here. analyst_instruction introduces them instead.
ANALYST_BLOCKS = ["analyst_instruction", "examples", "summary",
                   "agents_info", "shortlist_format"]


@dataclass(frozen=True)
class AgentShortlist:
    """Agents worth trying, best first, and what the model said about them.

    `unknown` records names the model produced that no agent answers to.
    They are dropped rather than corrected - a near-miss on a name is not
    evidence about which agent it meant - but kept here, because a model
    that keeps inventing names is a prompt problem, and silently discarding
    them is how that stays invisible.
    """
    agents: Tuple[str, ...] = ()
    reasoning: str = ""
    unknown: Tuple[str, ...] = ()
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.agents

    def as_auxiliary_info(self) -> Dict[str, str]:
        """The form `auxiliary_info/v1.j2` renders: a heading per key. Empty
        when nothing was established, so the coordinator sees no relevance
        section at all rather than an empty one - its instruction already
        reads them as "if given".
        """
        if self.is_empty:
            return {}
        block = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(self.agents))
        if self.reasoning:
            block += f"\n\n{self.reasoning}"
        return {"Agents worth trying, most promising first": block}


def render_agent_roster(registry: Sequence[Mapping[str, Any]],
                         role_instructions: Optional[Mapping[str, str]] = None) -> str:
    """The roster as the analyst sees it: one entry per agent, name first.

    Both descriptions go in where they exist. AGENTS_REGISTRY's `purpose` is
    a single line (35-87 characters across the nine agents); ROLE_INSTRUCTIONS
    carries the Role/Focus/Key Actions breakdown (246-293 characters). The
    long one is what an agent gets told about itself, and it is also what
    distinguishes "establish correspondences between objects" from "transform
    object coloration" well enough to choose between them.
    """
    role_instructions = role_instructions or {}
    entries = []
    for agent in registry:
        name = agent["name"]
        lines = [f"### {name}"]
        purpose = agent.get("purpose")
        if purpose:
            lines.append(purpose)
        detail = role_instructions.get(name)
        if detail:
            lines.append(detail)
        modules = agent.get("modules")
        if modules:
            lines.append(f"Modules: {', '.join(modules)}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def _known_names(registry: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Lowercased name -> the spelling the registry uses, so a model that
    answers "constructor" is understood without its casing being taken as
    a different agent."""
    return {agent["name"].lower(): agent["name"] for agent in registry}


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _load_json(text: str) -> Tuple[Any, Optional[str]]:
    """Whatever JSON the response carries, and why there wasn't any.

    Tries the response as a whole before looking for an object inside it,
    so an answer that is valid JSON of the wrong shape (a bare list of
    names, say) is reported as the wrong shape rather than as no JSON:
    searching for a `{...}` span first would find nothing in it and give
    the same message a refusal gives, which are different problems.
    """
    stripped = _FENCE.sub("", text.strip())
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT.search(stripped)
    if match is not None:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON: {exc.msg}"

    if "{" in stripped:
        # An opening brace and nothing closing it: the usual shape of a
        # response cut off at max_tokens, which is a generation-length
        # problem rather than a prompt one.
        return None, "response looks truncated - a JSON object was started but never closed"
    return None, "no JSON object in the response"


def parse_shortlist(text: str, registry: Sequence[Mapping[str, Any]],
                     shortlist_size: int = DEFAULT_SHORTLIST_SIZE) -> AgentShortlist:
    """Read the model's answer into a shortlist.

    Every failure is reported rather than papered over: an unparseable
    answer, or one naming no real agent, gives an empty shortlist with the
    reason attached. An empty shortlist is a usable answer - the coordinator
    falls back to its own judgement, which is what it did before any of this
    existed - whereas a guessed one would send it after the wrong agent
    while looking exactly like a real result.
    """
    if not text or not text.strip():
        return AgentShortlist(error="the model returned nothing")

    payload, why = _load_json(text)
    if why is not None:
        return AgentShortlist(error=why)
    if not isinstance(payload, dict):
        return AgentShortlist(error="response JSON is not an object")

    raw = payload.get("shortlist")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return AgentShortlist(error="response has no 'shortlist' list")

    known = _known_names(registry)
    agents: List[str] = []
    unknown: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        canonical = known.get(item.strip().lower())
        if canonical is None:
            unknown.append(item)
        elif canonical not in agents:      # a repeat is one vote, not two
            agents.append(canonical)

    reasoning = payload.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""

    if not agents:
        return AgentShortlist(reasoning=reasoning, unknown=tuple(unknown),
                               error="no agent on the roster was named")
    return AgentShortlist(agents=tuple(agents[:shortlist_size]), reasoning=reasoning,
                           unknown=tuple(unknown))


def rank_agents(task, module, registry: Optional[Sequence[Mapping[str, Any]]] = None,
                 role_instructions: Optional[Mapping[str, str]] = None,
                 shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
                 context: Optional[dict] = None) -> AgentShortlist:
    """Ask `module` which agents to try on `task`.

    `module` is anything with SubsymbolicModule's shape - a `.builder` to
    render the prompt with and a `.runner` to send it to - so a fake stands
    in for the model in tests, and a second SubsymbolicModule configured for
    a cheaper model can do the ranking while an expensive one solves.

    The blocks are the caller's to set: this reads whatever
    `module.builder.config.blocks` holds, so a project that wants the
    analyst to see something else (MCTS rollouts, once the RL branch runs)
    adds a block rather than an argument here.
    """
    if registry is None or role_instructions is None:
        from data.configs.agents_config import AGENTS_REGISTRY, ROLE_INSTRUCTIONS
        registry = AGENTS_REGISTRY if registry is None else registry
        role_instructions = ROLE_INSTRUCTIONS if role_instructions is None else role_instructions

    full_context = {
        "agents_info_text": render_agent_roster(registry, role_instructions),
        "shortlist_size": shortlist_size,
        **(context or {}),
    }
    prompt = module.builder.build(task, context=full_context)
    if prompt is None:
        return AgentShortlist(error="the analyst prompt didn't fit token_limit")
    return parse_shortlist(module.runner.generate(prompt), registry, shortlist_size)
