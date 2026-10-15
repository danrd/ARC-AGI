"""scripts/elicit_trace_needs.py: asking a model what it worked out by hand in its own reasoning."""
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "elicit", Path(__file__).resolve().parents[1] / "scripts" / "elicit_trace_needs.py")
elicit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(elicit)


class TestReasoning:
    def test_gemma_marks_are_taken_off(self):
        assert elicit.reasoning_of("<|channel>thought\nRow 1: 0033\n<channel|>11,5:\n1 0033") == "Row 1: 0033"

    def test_qwen_closing_mark_alone_ends_it(self):
        """The opening <think> is in the chat template for Qwen, so a reply may hold only the closing one."""
        assert elicit.reasoning_of("Row 1: 0033\n</think>\nanswer") == "Row 1: 0033"

    def test_a_reasoning_that_never_closed_is_all_reasoning(self):
        assert elicit.reasoning_of("<|channel>thought\nRow 1: 0033 and then the limit") == "Row 1: 0033 and then the limit"

    def test_a_reply_without_marks_is_itself(self):
        assert elicit.reasoning_of("  plain  ") == "plain"


class TestShorten:
    def test_a_short_text_is_whole(self):
        assert elicit.shorten("abc", 5, 5) == "abc"

    def test_a_long_one_keeps_its_ends_and_says_where_the_middle_was(self):
        out = elicit.shorten("A" * 10 + "M" * 100 + "Z" * 10, 10, 10)
        assert out == "A" * 10 + "\n[...]\n" + "Z" * 10 and "M" not in out


class TestParse:
    def test_items_and_the_most_useful(self):
        reply = ("1. ROW_INDEX | each row with its number | Row 1: 0033\n"
                 "- DIFF_CELLS | cells changed, (r,c) old->new | Changed: rows 2-4\n"
                 "MOST: DIFF_CELLS")
        items, most = elicit.parse(reply)
        assert items == {"ROW_INDEX": {"prints": "each row with its number", "by_hand": "Row 1: 0033"},
                         "DIFF_CELLS": {"prints": "cells changed, (r,c) old->new", "by_hand": "Changed: rows 2-4"}}
        assert most == "DIFF_CELLS"

    def test_a_table_rule_and_prose_are_not_items(self):
        items, most = elicit.parse("Here is the list:\n---|---|---\n**COUNTS** | per colour | count = 5\nmost: counts")
        assert list(items) == ["COUNTS"] and most == "counts"

    def test_a_missing_third_column_is_empty(self):
        assert elicit.parse("OBJECTS | boxes")[0]["OBJECTS"]["by_hand"] == ""


def write(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")


class TestRun:
    def test_every_trace_is_asked_with_its_reasoning_in_the_prompt(self, tmp_path):
        write(tmp_path / "t.jsonl", [{"task": "a", "reply": "<think>REASON-A</think>x", "solved": True, "reply_tokens": 9},
                                     "torn line {", {"task": "b", "reply": ""}])
        prompts = []
        records = elicit.run(elicit.read_traces([tmp_path / "t.jsonl"]),
                             lambda p: prompts.append(p) or "NAME | prints | by hand\nMOST: NAME",
                             tmp_path / "out.jsonl", model="m")
        assert len(prompts) == 1 and "REASON-A" in prompts[0] and "</think>" not in prompts[0]
        assert records[0]["source"] == "t" and records[0]["task"] == "a" and records[0]["most"] == "NAME"
        assert records[0]["trace_solved"] is True and records[0]["trace_tokens"] == 9 and records[0]["model"] == "m"

    def test_what_is_answered_is_not_asked_again(self, tmp_path):
        write(tmp_path / "t.jsonl", [{"task": "a", "reply": "r1"}, {"task": "b", "reply": "r2"}])
        out, asked = tmp_path / "out.jsonl", []
        ask = lambda p: asked.append(p) or "X | y | z"  # noqa: E731
        elicit.run(elicit.read_traces([tmp_path / "t.jsonl"]), ask, out, limit=1)
        assert len(asked) == 1
        elicit.run(elicit.read_traces([tmp_path / "t.jsonl"]), ask, out)
        assert len(asked) == 2 and len(out.read_text().splitlines()) == 2
        elicit.run(elicit.read_traces([tmp_path / "t.jsonl"]), ask, out)
        assert len(asked) == 2

    def test_the_same_task_in_two_sources_is_two_questions(self, tmp_path):
        write(tmp_path / "one.jsonl", [{"task": "a", "reply": "r"}])
        write(tmp_path / "two.jsonl", [{"task": "a", "reply": "r"}])
        records = elicit.run(elicit.read_traces([tmp_path / "one.jsonl", tmp_path / "two.jsonl"]),
                             lambda p: "X | y", tmp_path / "out.jsonl")
        assert [r["source"] for r in records] == ["one", "two"]

    def test_a_long_reasoning_reaches_the_model_cut(self, tmp_path):
        write(tmp_path / "t.jsonl", [{"task": "a", "reply": "H" * 100 + "M" * 1000 + "T" * 100}])
        prompts = []
        elicit.run(elicit.read_traces([tmp_path / "t.jsonl"]), lambda p: prompts.append(p) or "", tmp_path / "o.jsonl",
                   head=100, tail=100)
        assert "[...]" in prompts[0] and "M" * 50 not in prompts[0]
