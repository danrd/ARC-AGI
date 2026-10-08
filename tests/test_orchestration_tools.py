"""The model asking for information: reading a request, putting the result in the next prompt,
and the grammar that has to be off for the request to be sayable."""
from types import SimpleNamespace

from orchestration.tools import (InfoTool, ToolUsingModule, default_tools, parse_request, tools_description,
                                 with_tool_blocks, without_grammar)
from orchestration.trace import Tracer
from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY

TOOLS = [InfoTool("summary", "what changes", lambda task, builder: "THE SUMMARY"),
         InfoTool("search_hints", "a search result", lambda task, builder: None)]


class Runner:
    def __init__(self, replies, generation_kwargs=None):
        self.replies, self.prompts = list(replies), []
        self.generation_kwargs = generation_kwargs

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def module(tokenizer, runner, blocks=("general_instruction", "examples", "output_format")):
    config = PromptingConfig(blocks=list(blocks), resolvers=["examples"], filters=["grid"], token_limit=20000)
    builder = PromptBuilder(config, tokenizer, resolver_registry=RESOLVER_REGISTRY, filter_registry=FILTER_REGISTRY)
    return SimpleNamespace(builder=builder, runner=runner, close=lambda: None)


class TestReadingARequest:
    def test_a_line_naming_a_tool_that_exists_is_a_request(self):
        assert parse_request("REQUEST: summary", TOOLS) == "summary"
        assert parse_request("  request: SUMMARY  \n", TOOLS) is None   # the word REQUEST is the marker, upper case
        assert parse_request("Let me think.\nREQUEST: search_hints\n", TOOLS) == "search_hints"
        assert parse_request("<think>REQUEST: summary</think>\n3,3:\n1 000", TOOLS) is None

    def test_a_request_for_a_tool_there_is_not_is_not_a_request(self):
        assert parse_request("REQUEST: telepathy", TOOLS) is None
        assert parse_request(None, TOOLS) is None
        assert parse_request("I would REQUEST: summary if I could", TOOLS) is None

    def test_the_tools_are_described_one_to_a_line(self):
        assert tools_description(TOOLS) == "- summary: what changes\n- search_hints: a search result"
        assert [tool.name for tool in default_tools()] == ["summary", "search_hints"]


class TestTheGrammar:
    def test_a_runner_with_a_grammar_gets_a_copy_without_it_and_the_original_keeps_it(self):
        runner = Runner([], {"temperature": 0, "extra_body": {"grammar": "root ::= x", "top_k": 5}})
        free = without_grammar(runner)
        assert free is not runner and free.generation_kwargs["extra_body"] == {"top_k": 5}
        assert runner.generation_kwargs["extra_body"]["grammar"] == "root ::= x"
        assert free.generation_kwargs["temperature"] == 0

    def test_a_runner_without_one_is_left_as_it_is(self):
        runner = Runner([], {"extra_body": {"top_k": 5}})
        assert without_grammar(runner) is runner and without_grammar(Runner([])) is not None
        assert without_grammar(Runner([], {"extra_body": {"guided_grammar": "g"}})).generation_kwargs["extra_body"] == {}


class TestThePromptBlocks:
    def test_the_tool_blocks_go_before_the_output_format_once(self):
        config = PromptingConfig(blocks=["examples", "output_format"])
        blocks = with_tool_blocks(config).blocks
        assert blocks == ["examples", "tools_info", "tool_results", "output_format"]
        assert with_tool_blocks(with_tool_blocks(config)).blocks == blocks
        assert with_tool_blocks(PromptingConfig(blocks=["examples"])).blocks[-2:] == ["tools_info", "tool_results"]


class TestAskingAndAnswering:
    def test_a_model_that_asks_gets_the_result_in_the_next_prompt_and_answers_with_it(self, tiny_tokenizer, arc_task):
        runner = Runner(["REQUEST: summary", "3,3:\n1 000\n2 000\n3 000"])
        wrapped = ToolUsingModule(module(tiny_tokenizer, runner), TOOLS, max_requests=1)
        result = wrapped.solve(arc_task, {"grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert result["solution"].startswith("3,3:")
        assert result["module_results"]["tool_calls"] == [{"tool": "summary", "answered": True}]
        first, second = runner.prompts
        assert "REQUEST: <tool>" in first and "THE SUMMARY" not in first
        assert "THE SUMMARY" in second and "REQUEST: <tool>" not in second     # asked once, not invited again

    def test_a_model_that_does_not_ask_answers_in_one_call_when_nothing_forbids_the_request(self, tiny_tokenizer, arc_task):
        runner = Runner(["3,3:\n1 000\n2 000\n3 000"])
        result = ToolUsingModule(module(tiny_tokenizer, runner), TOOLS).solve(arc_task, {
            "grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert len(runner.prompts) == 1 and result["module_results"]["tool_calls"] == []

    def test_requests_are_capped(self, tiny_tokenizer, arc_task):
        runner = Runner(["REQUEST: summary", "REQUEST: search_hints", "3,3:\n1 000"])
        result = ToolUsingModule(module(tiny_tokenizer, runner), TOOLS, max_requests=2).solve(arc_task, {
            "grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert [c["tool"] for c in result["module_results"]["tool_calls"]] == ["summary", "search_hints"]
        assert result["solution"] == "3,3:\n1 000" and len(runner.prompts) == 3

    def test_a_tool_with_nothing_to_say_says_so(self, tiny_tokenizer, arc_task):
        runner = Runner(["REQUEST: search_hints", "3,3:\n1 000"])
        result = ToolUsingModule(module(tiny_tokenizer, runner), TOOLS).solve(arc_task, {
            "grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert result["module_results"]["tool_calls"] == [{"tool": "search_hints", "answered": False}]
        assert "nothing found" in runner.prompts[1]

    def test_with_a_grammar_the_ask_is_free_and_the_answer_is_held_to_it(self, tiny_tokenizer, arc_task):
        kwargs = {"extra_body": {"grammar": "root ::= grid"}}
        seen = []

        class Constrained(Runner):
            def generate(self, prompt):
                seen.append(self.generation_kwargs["extra_body"].get("grammar"))
                return super().generate(prompt)

        runner = Constrained(["REQUEST: summary", "3,3:\n1 000"], kwargs)
        wrapped = ToolUsingModule(module(tiny_tokenizer, runner), TOOLS)
        # the free copy shares the replies list, so the two calls draw from one script
        result = wrapped.solve(arc_task, {"grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert seen == [None, "root ::= grid"] and result["solution"] == "3,3:\n1 000"

    def test_the_spans_say_where_the_time_went(self, tiny_tokenizer, arc_task):
        tracer = Tracer()
        runner = Runner(["REQUEST: summary", "3,3:\n1 000"])
        ToolUsingModule(module(tiny_tokenizer, runner), TOOLS, tracer=tracer).solve(arc_task, {
            "grid_repr_type": "concise", "test_input_grid": arc_task.test_subtask.train_inp})
        assert [(s.phase, s.name) for s in tracer.spans] == [("llm", "ask 1"), ("tools", "summary"), ("llm", "answer")]
        assert tracer.spans[0].tokens_in > 0 and tracer.spans[0].tokens_out > 0
