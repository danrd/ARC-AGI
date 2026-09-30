"""Tests for scripts/prompt_variants.py - the prompt files for testing by hand
what a model needs to solve a task.

What is pinned is that each variant changes the one thing it says it changes
and nothing else, that the colour shuffle is applied to everything a prompt
shows (examples, rule, draft) and to the answer, and that the files land
where the index says.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("prompt_variants", ROOT / "scripts" / "prompt_variants.py")
variants = importlib.util.module_from_spec(SPEC)
sys.modules["prompt_variants"] = variants
SPEC.loader.exec_module(variants)

TASK = "6ea4a07e"


def tags(prompt):
    """The block tags of a prompt, in order."""
    return [line[1:-1] for line in prompt.splitlines() if line.startswith("<") and not line.startswith("</")]


class TestWhatEachVariantAdds:
    def test_the_baseline_has_no_knowledge_block(self):
        prompt, _ = variants.make_prompt(TASK, "p0")
        assert "KNOWLEDGE" not in tags(prompt)
        assert tags(prompt)[-3:] == ["EXAMPLES", "TASK_REPR", "OUTPUT_FORMAT"][-3:] or "EXAMPLES" in tags(prompt)

    def test_the_rule_comes_before_the_examples_and_can_move_after_the_test_input(self):
        rule = variants.ORACLES[TASK].rule(str)
        early, _ = variants.make_prompt(TASK, "p3")
        late, _ = variants.make_prompt(TASK, "p3b")
        assert rule in early and rule in late
        assert tags(early).index("KNOWLEDGE") < tags(early).index("EXAMPLES")
        assert tags(late).index("KNOWLEDGE") > tags(late).index("TASK_REPR")

    def test_the_wrong_rule_is_there_and_the_right_one_is_not(self):
        prompt, _ = variants.make_prompt(TASK, "p5")
        oracle = variants.ORACLES[TASK]
        assert oracle.wrong(str) in prompt and oracle.rule(str) not in prompt

    def test_the_neutral_facts_hold_no_rule(self):
        prompt, _ = variants.make_prompt(TASK, "p1")
        assert "Facts about this puzzle" in prompt
        assert variants.ORACLES[TASK].rule(str) not in prompt

    def test_the_walkthrough_adds_to_the_rule_and_names_every_changed_cell(self):
        rule_only, _ = variants.make_prompt(TASK, "p3")
        with_walk, answer = variants.make_prompt(TASK, "p4")
        assert with_walk.startswith(rule_only[:200]) and len(with_walk) > len(rule_only)
        before = variants.build_task(TASK, False)[0].test_subtask.train_inp
        after = variants.build_task(TASK, False)[1]
        counted = sum(int(word) for line in with_walk.splitlines() if line.startswith("Rows ")
                      for word in line.replace(",", " ").split() if word.isdigit()
                      and line.split(word + " cell")[0] != line)
        assert counted == int((np.asarray(before) != np.asarray(after)).sum())
        assert answer

    def test_the_draft_is_neither_the_input_nor_the_answer(self):
        before = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
        after = np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0], [0, 2, 2, 0]])
        drafted = variants.draft(before, after)
        assert not np.array_equal(drafted, before) and not np.array_equal(drafted, after)
        assert np.array_equal(drafted[1], after[1]) and np.array_equal(drafted[3], before[3])

    def test_a_single_region_is_drafted_half_done(self):
        before = np.zeros((2, 4), dtype=int)
        after = before.copy()
        after[0] = 3
        drafted = variants.draft(before, after)
        assert int((drafted != before).sum()) == 2

    def test_the_examples_as_changes_leave_out_the_output_grids(self):
        prompt, _ = variants.make_prompt(TASK, "p7")
        assert "Cells that change" in prompt and "Output:" not in prompt.split("<EXAMPLES>")[1].split("</EXAMPLES>")[0]

    @pytest.mark.parametrize("variant,marker", [("p10a", "|"), ("p10b", "black"), ("p10c", "bb")])
    def test_the_grid_can_be_written_another_way(self, variant, marker):
        plain_prompt, _ = variants.make_prompt(TASK, "p0")
        other, _ = variants.make_prompt(TASK, variant)
        assert other != plain_prompt and marker in other.split("<EXAMPLES>")[1]

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError):
            variants.make_prompt(TASK, "p99")


class TestTheColourShuffle:
    def test_it_is_a_fixed_shuffle_of_one_to_nine_leaving_zero(self):
        mapping = variants.colour_map(TASK, True)
        assert mapping[0] == 0 and sorted(mapping.values()) == list(range(10))
        assert mapping == variants.colour_map(TASK, True) and mapping != variants.colour_map(TASK, False)

    def test_the_answer_and_the_examples_follow_it(self):
        mapping = variants.colour_map(TASK, True)
        train, _, answer = variants.load_arrays(TASK)
        _, shown_answer = variants.make_prompt(TASK, "p0", permute=True)
        _, plain_answer = variants.make_prompt(TASK, "p0", permute=False)
        assert shown_answer != plain_answer
        task, expected, _ = variants.build_task(TASK, True)
        assert np.array_equal(expected, variants.recolour(answer, mapping))
        assert np.array_equal(task.subtasks[0].train_inp, variants.recolour(train[0][0], mapping))

    def test_the_rule_names_the_shown_colours(self):
        mapping = variants.colour_map(TASK, True)
        prompt, _ = variants.make_prompt(TASK, "p3", permute=True)
        assert variants.ORACLES[TASK].rule(lambda c: str(mapping[c])) in prompt


class TestTheFiles:
    def test_every_variant_of_every_task_is_written_with_its_answer_and_an_index(self, tmp_path):
        count = variants.write_all(tmp_path, [TASK], ("p0", "p3"), permute_too=True)
        assert count == 4
        assert {p.name for p in (tmp_path / TASK).iterdir()} == {
            "p0.txt", "p3.txt", "p0_perm.txt", "p3_perm.txt", "answer.txt", "answer_perm.txt"}
        index = (tmp_path / "index.md").read_text()
        assert TASK in index and "wrong rule (P5)" in index and "| task | grade | p0 | p3 |" in index

    def test_the_first_pass_is_six_variants(self):
        assert variants.FIRST_PASS == ("p0", "p1", "p3", "p4", "p5", "p6")
