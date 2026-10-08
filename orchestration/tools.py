"""Tools the model can ask for before it answers.

The orchestrator already treats its modules as tools - the symbolic solvers,
the background RL job, the model - and calls them itself. This is the other
direction: the model, shown what it can ask for, replies `REQUEST: <tool>`
instead of an answer, the result is put into the prompt, and it is asked
again. It is retrieval in the plain sense - the information is looked up on
demand, from what the symbolic analysis and the search found about this task,
and not stuffed into every prompt whether or not it is wanted.

    module = ToolUsingModule(subsymbolic_module, default_tools(), max_requests=1)
    module.solve(task, context)            # the same shape as SubsymbolicModule.solve

Off unless asked for: a prompt without the `tools_info` block is the prompt
it always was. Measured on the evaluation tasks, the summary and the hints
changed little when given up front (the summary 5 won, 2 lost, p = 0.45), so
this is a mechanism that has been built, not one that has been shown to pay.

Two things the model's side of this has to be given. The tools are announced
in a block (`tools_info`) and their results come back in another
(`tool_results`); both are put into the prompt here, before `output_format`,
so a config does not have to name them. And the request is a line of text,
which a grammar that holds the output to a grid forbids: the round in which
the model may ask goes to the runner with the grammar taken off
(`without_grammar`), and the answer, which wants the grammar, to the runner
as it is.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from orchestration.blocks import with_blocks
from orchestration.trace import Tracer, traced
from subsymbolic.prompt_builder import PromptBuilder
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY

_REQUEST = re.compile(r"^\s*REQUEST:\s*([A-Za-z_][\w-]*)\s*$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
GRAMMAR_KEYS = ("grammar", "guided_grammar")


@dataclass(frozen=True)
class InfoTool:
    """What the model may ask for: a name, a line saying what it is, and
    `fetch(task, builder) -> text or None` (None: nothing to say)."""
    name: str
    description: str
    fetch: Callable[[Any, PromptBuilder], Optional[str]]


def summary_tool() -> InfoTool:
    """The symbolic analysis of the task: what changes between input and output, in words."""
    def fetch(task, builder):
        from subsymbolic.arc_resolvers import transformation_summary_resolver
        from subsymbolic.prompt_builder import OMIT

        text = transformation_summary_resolver(task, builder.config.token_limit // 3, {}, builder)
        return None if text is OMIT else text

    return InfoTool("summary", "what the symbolic analysis found changes between the inputs and the outputs", fetch)


def search_hints_tool(settings=None, cache: Optional[dict] = None) -> InfoTool:
    """What a search over the grid transforms found to reproduce every example, if it found one.
    A search is minutes of CPU: it is run once per task and kept in `cache`."""
    state: Dict[str, Any] = {}

    def fetch(task, builder):
        if "hints" not in state:
            from rl.search_hints import HintCache, SearchSettings
            state["hints"] = HintCache(settings or SearchSettings(), {} if cache is None else cache)
        return state["hints"](task) or None

    return InfoTool("search_hints", "a sequence of grid operations that a search found to reproduce every example, "
                    "when it found one", fetch)


def default_tools(settings=None, cache: Optional[dict] = None) -> List[InfoTool]:
    return [summary_tool(), search_hints_tool(settings, cache)]


def without_grammar(runner):
    """`runner` with the output grammar left out of its requests, or `runner` itself when
    it has none to leave out. The copy shares the server; nothing is restarted."""
    kwargs = getattr(runner, "generation_kwargs", None)
    body = (kwargs or {}).get("extra_body") or {}
    if not any(key in body for key in GRAMMAR_KEYS):
        return runner
    free = copy.copy(runner)
    free.generation_kwargs = {**kwargs, "extra_body": {k: v for k, v in body.items() if k not in GRAMMAR_KEYS}}
    return free


def parse_request(text, tools: Sequence[InfoTool]) -> Optional[str]:
    """The tool a reply asks for - a line `REQUEST: <name>` of a tool that exists - or None.
    A request for a tool that does not exist is not a request."""
    if not isinstance(text, str):
        return None
    known = {tool.name.lower(): tool.name for tool in tools}
    for found in _REQUEST.findall(_THINK.sub("", text)):
        if found.lower() in known:
            return known[found.lower()]
    return None


def tools_description(tools: Sequence[InfoTool]) -> str:
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


TOOL_BLOCKS = ("tools_info", "tool_results")


def with_tool_blocks(config):
    """The prompting config with `tools_info` and `tool_results` added before `output_format`."""
    return with_blocks(config, TOOL_BLOCKS)


class ToolUsingModule:
    """A SubsymbolicModule that may ask for information before it answers.

    Quacks like the module it wraps (`solve`, `builder`, `runner`, `close`), so it goes
    wherever that one goes: into `make_module_dispatch_fn`, into `rank_agents`.
    """

    def __init__(self, module, tools: Sequence[InfoTool], max_requests: int = 1,
                 tracer: Optional[Tracer] = None):
        self.module = module
        self.tools = list(tools)
        self.max_requests = max_requests
        self.tracer = tracer
        self.builder = PromptBuilder(with_tool_blocks(module.builder.config), module.builder.tokenizer,
                                     resolver_registry=RESOLVER_REGISTRY, filter_registry=FILTER_REGISTRY)
        self._free = None

    @property
    def runner(self):
        return self.module.runner

    @property
    def free_runner(self):
        if self._free is None:
            self._free = without_grammar(self.module.runner)
        return self._free

    def close(self) -> None:
        self.module.close()

    def _ask(self, runner, prompt: str, name: str) -> str:
        with traced(self.tracer, "llm", name, tokens_in=self.builder.count_tokens(prompt)) as span:
            reply = runner.generate(prompt)
            span.tokens_out = self.builder.count_tokens(reply) if isinstance(reply, str) else 0
        return reply

    def _fetch(self, tool: Optional[InfoTool], task) -> Optional[str]:
        with traced(self.tracer, "tools", tool.name if tool else "unknown"):
            return tool.fetch(task, self.builder) if tool else None

    def solve(self, task, context: Optional[dict] = None) -> Dict[str, Any]:
        context = dict(context or {})
        by_name = {tool.name: tool for tool in self.tools}
        context.setdefault("tools_info_text", tools_description(self.tools))
        context.setdefault("max_requests", self.max_requests)
        results: Dict[str, str] = dict(context.get("tool_results") or {})
        calls: List[Dict[str, Any]] = []

        for number in range(self.max_requests):
            prompt = self.builder.build(task, context={**context, "tool_results": results}, assistant_prefix="")
            if prompt is None:
                return {"solution": "", "module_results": {"error": "prompt didn't fit token_limit",
                                                           "tool_calls": calls}}
            reply = self._ask(self.free_runner, prompt, f"ask {number + 1}")
            wanted = parse_request(reply, self.tools)
            if wanted is None:
                if self.free_runner is self.module.runner:
                    return {"solution": reply, "module_results": {"tool_calls": calls}}
                break
            text = self._fetch(by_name.get(wanted), task)
            calls.append({"tool": wanted, "answered": bool(text)})
            results[wanted] = text or "nothing found"
        prompt = self.builder.build(task, context={**context, "tool_results": results, "tools_info_text": ""})
        if prompt is None:
            return {"solution": "", "module_results": {"error": "prompt didn't fit token_limit", "tool_calls": calls}}
        return {"solution": self._ask(self.module.runner, prompt, "answer"), "module_results": {"tool_calls": calls}}
