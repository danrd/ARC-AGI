"""Tests for subsymbolic.answer_check - a second model's verdict on a candidate.

The verdict is only as strict as the reading of the reply, so that is pinned
hard: a clear YES accepts and nothing else does. The prompt is pinned for what
the second model must be given to judge at all - the examples, the new input
and the candidate - and for not being handed the target.
"""
from __future__ import annotations

import numpy as np
import pytest

from subsymbolic.answer_check import BLOCKS, LlmVerifier, answer_check_config, parse_verdict
from subsymbolic.prompt_builder import PromptingConfig


class TestReadingTheReply:
    @pytest.mark.parametrize("reply", ["YES", "yes", "Yes.", "  YES\n", "The answer is: YES",
                                       "<think>it is not the same, no</think>YES"])
    def test_a_yes_is_accepted(self, reply):
        assert parse_verdict(reply) is True

    @pytest.mark.parametrize("reply", ["NO", "no", "No, the corner is wrong", "", "maybe", "Yesterday",
                                       "nothing to say", "<think>yes it is</think> NO",
                                       "<think>the rule is yes, so", None, 3])
    def test_anything_else_is_a_refusal(self, reply):
        assert parse_verdict(reply) is False

    def test_the_first_verdict_is_the_one_that_counts(self):
        assert parse_verdict("NO. Although YES would be the answer if the corner were red") is False


class Runner:
    def __init__(self, reply="YES"):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class TestTheVerifier:
    def test_it_puts_the_examples_the_input_and_the_candidate_to_the_model(self, arc_task, tiny_tokenizer):
        runner = Runner("YES")
        candidate = np.full_like(arc_task.test_subtask.train_out, 7)
        assert LlmVerifier(runner, tiny_tokenizer)(arc_task, candidate) is True
        (prompt,) = runner.prompts
        first = arc_task.subtasks[0]
        assert "Training example 1" in prompt and "Training example 2" in prompt
        assert f"grid shape: {first.train_out.shape[0]},{first.train_out.shape[1]}" in prompt
        assert "New input" in prompt and "Proposed output" in prompt
        assert "777" in prompt or "7" * candidate.shape[1] in prompt

    def test_the_target_is_never_in_the_prompt(self, arc_task, tiny_tokenizer):
        """The candidate and the test input are the only grids of the test pair it sees."""
        runner = Runner()
        target = arc_task.test_subtask.train_out
        candidate = np.zeros_like(target)
        LlmVerifier(runner, tiny_tokenizer)(arc_task, candidate)
        from subsymbolic.arc_grid_formatting import format_grid
        target_text, candidate_text = format_grid(target), format_grid(candidate)
        if target_text != candidate_text:
            assert target_text not in runner.prompts[0]

    def test_a_no_is_false(self, arc_task, tiny_tokenizer):
        assert LlmVerifier(Runner("NO"), tiny_tokenizer)(arc_task, np.zeros((3, 3), dtype=int)) is False

    def test_a_prompt_that_does_not_fit_is_a_refusal_and_the_model_is_not_called(self, arc_task, tiny_tokenizer):
        runner = Runner("YES")
        prompt = PromptingConfig(token_limit=5, min_examples=1)
        assert LlmVerifier(runner, tiny_tokenizer, prompt)(arc_task, np.zeros((3, 3), dtype=int)) is False
        assert runner.prompts == []

    def test_the_check_takes_over_the_callers_prompt_settings_but_not_its_blocks(self):
        base = PromptingConfig(blocks=["output_format"], token_limit=1234, join_format="md",
                               assistant_prefix="3,3:")
        config = answer_check_config(base)
        assert config.blocks == BLOCKS and config.token_limit == 1234 and config.join_format == "md"
        assert config.assistant_prefix is None and config.resolvers == ["examples"]
        assert base.blocks == ["output_format"]
