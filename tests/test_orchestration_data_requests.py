"""Asking a model what it wants to be told: the menu's answers, the question without the output format,
and the requests recorded in order."""
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration import data_requests as dr
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.prompt_builder import PromptBuilder, PromptingConfig
from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY

GRID_IN = np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]])
GRID_OUT = np.array([[0, 0, 3], [0, 4, 0], [3, 0, 0]])


@pytest.fixture
def small_task():
    shrink = np.array([[1, 1, 1, 1]])
    return ARCTask("small", [ARCSubtask("a", GRID_IN, GRID_OUT), ARCSubtask("b", GRID_IN, GRID_IN),
                             ARCSubtask("c", GRID_IN, shrink)], GRID_IN, GRID_OUT)


class Counter:
    count_tokens = staticmethod(lambda text: len(text.split()))


class Runner:
    def __init__(self, replies):
        self.replies, self.prompts, self.generation_kwargs = list(replies), [], None

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def module(tokenizer, runner, blocks=("general_instruction", "examples_intro", "examples", "task_repr", "output_format")):
    config = PromptingConfig(blocks=list(blocks), resolvers=["examples"], filters=["grid"], token_limit=20000)
    builder = PromptBuilder(config, tokenizer, resolver_registry=RESOLVER_REGISTRY, filter_registry=FILTER_REGISTRY)
    return SimpleNamespace(builder=builder, runner=runner, close=lambda: None)


class TestTheNumberedGrids:
    def test_rows_and_columns_count_from_one_as_the_answer_does(self, small_task):
        text = dr.indexed_grids(small_task)
        assert "Example 1 input (3x3), columns 1..3:" in text
        assert "     123\nr1   003\nr2   000\nr3   300" in text

    def test_a_wide_grid_gets_a_row_of_tens(self):
        task = ARCTask("w", [ARCSubtask("a", np.zeros((1, 12), int), np.zeros((1, 12), int))], np.zeros((1, 12), int),
                       np.zeros((1, 12), int))
        assert "     " + " " * 9 + "111\n     123456789012" in dr.indexed_grids(task)

    def test_the_test_output_is_never_shown(self, small_task):
        assert "Test output" not in dr.indexed_grids(small_task) and "Test input" in dr.indexed_grids(small_task)


class TestTheCellChanges:
    def test_a_change_is_where_from_what_to_what_counting_from_one(self, small_task):
        first = dr.cell_changes(small_task).splitlines()[0]
        assert first == "Example 1: 1 of 9 cells change (0->4 x1): (r2,c2) 0->4"

    def test_an_example_that_does_not_change_or_changes_size_says_so(self, small_task):
        lines = dr.cell_changes(small_task).splitlines()
        assert lines[1] == "Example 2: nothing changes"
        assert lines[2] == "Example 3: the size changes, 3x3 to 1x4"

    def test_a_long_list_is_cut_and_says_how_much_is_left(self):
        grid = np.zeros((5, 5), int)
        task = ARCTask("t", [ARCSubtask("a", grid, grid + 1)], grid, grid + 1)
        line = dr.cell_changes(task, limit=3).splitlines()[0]
        assert "25 of 25 cells change (0->1 x25)" in line and line.endswith(", and 22 more")


class TestTheCounts:
    def test_every_grid_counts_its_colours(self, small_task):
        lines = dr.colour_counts(small_task).splitlines()
        assert lines[0] == "Example 1 input: 0: 7, 3: 2"
        assert lines[1] == "Example 1 output: 0: 6, 3: 2, 4: 1"
        assert lines[-1] == "Test input: 0: 7, 3: 2"


class TestTheObjects:
    def test_pieces_of_a_colour_are_counted_apart_and_the_background_is_not_one(self, small_task):
        first = dr.objects(small_task).splitlines()[0]
        assert first.startswith("Example 1 input (background 0): 2 pieces")
        assert "colour 3, 1 cells, rows 1-1, cols 3-3" in first and "colour 3, 1 cells, rows 3-3, cols 1-1" in first

    def test_a_long_list_is_cut(self):
        grid = np.zeros((9, 9), int)
        grid[::2, ::2] = 5
        task = ARCTask("t", [ARCSubtask("a", grid, grid)], grid, grid)
        assert "25 pieces" in dr.objects(task, per_grid=2) and "; and 23 more" in dr.objects(task, per_grid=2)


class TestTheMenu:
    def test_it_offers_ten_and_the_last_three_have_nothing_behind_them(self, small_task):
        tools = dr.menu_tools()
        assert [t.name for t in tools] == ["changes", "kept", "grids_shown", "cell_changes", "colour_counts", "objects",
                                           "numbered_grids", "symmetry", "row_column_comparison", "object_relations"]
        assert [t.fetch(small_task, Counter) for t in tools[-3:]] == [None, None, None]
        assert all(t.fetch(small_task, Counter) for t in tools[3:7])

    def test_each_symbolic_entry_gives_its_own_section_and_no_other(self):
        from orchestration.run import load_task
        task = load_task("evaluation", "770cc55f")
        by_name = {t.name: t.fetch(task, Counter) for t in dr.menu_tools()}
        assert by_name["changes"].startswith("What changes:") and "What stays" not in by_name["changes"]
        assert by_name["kept"].startswith("What stays the same:") and "What changes" not in by_name["kept"]
        assert by_name["grids_shown"].startswith("What the grids show:") and "What changes" not in by_name["grids_shown"]


class TestTheWishes:
    def test_the_question_replaces_the_demand_for_a_grid(self, tiny_tokenizer, small_task):
        stub = module(tiny_tokenizer, Runner([]))
        prompt = dr.wishes_prompt(dr.wishes_builder(stub), small_task)
        assert prompt.endswith("and nothing else.") and "NAME | what it would print" in prompt
        assert "Return *only* the predicted output grid" not in prompt
        assert "Return *only* the predicted output grid" in stub.builder.build(small_task, context={
            "grid_repr_type": "concise", "test_input_grid": small_task.test_inp}, assistant_prefix="")

    def test_the_wishes_are_read_out_of_the_reply(self, tiny_tokenizer, small_task):
        runner = Runner(["1. NUMBERED_GRIDS | grids with row numbers | I count rows by hand\n- DIFF | changed cells | long"])
        stub = module(tiny_tokenizer, runner)
        result = dr.ask_wishes(stub, dr.wishes_builder(stub), small_task)
        assert list(result["wishes"]) == ["NUMBERED_GRIDS", "DIFF"] and "NAME |" in runner.prompts[0]


class TestParse:
    def test_items_and_the_most_useful(self):
        items, most = dr.parse_wishes("1. ROW_INDEX | each row with its number | Row 1: 0033\nMOST: ROW_INDEX")
        assert items == {"ROW_INDEX": {"prints": "each row with its number", "by_hand": "Row 1: 0033"}} and most == "ROW_INDEX"

    def test_a_table_rule_and_prose_are_not_items(self):
        assert list(dr.parse_wishes("Here is the list:\n---|---|---\n**COUNTS** | per colour | n")[0]) == ["COUNTS"]


class TestRunningTheMenu:
    def test_the_requests_come_back_in_order_with_what_the_model_said(self, tiny_tokenizer, small_task):
        runner = Runner(["REQUEST: cell_changes", "REQUEST: symmetry", "REQUEST: objects", "3,3:\n1 000\n2 040\n3 000"])
        result = dr.run_menu(module(tiny_tokenizer, runner), small_task, dr.menu_tools(), max_requests=3)
        assert result["calls"] == [{"tool": "cell_changes", "answered": True}, {"tool": "symmetry", "answered": False},
                                   {"tool": "objects", "answered": True}]
        assert result["replies"][0] == "REQUEST: cell_changes" and result["solution"].startswith("3,3:")
        assert "(r2,c2) 0->4" in runner.prompts[1] and "nothing found" in runner.prompts[2]
        assert "pieces" in runner.prompts[3] and "REQUEST: <tool>" not in runner.prompts[3]


class TestTheRuleHint:
    def test_a_rule_that_fits_every_example_is_named_with_its_length(self):
        pairs = [(np.array([[0, 5, 0], [0, 0, 0]]), np.array([[0, 5, 0], [0, 7, 0]])),
                 (np.array([[5, 0, 0], [0, 0, 0]]), np.array([[5, 0, 0], [7, 0, 0]]))]
        task = ARCTask("r", [ARCSubtask(str(i), a, b) for i, (a, b) in enumerate(pairs)], pairs[0][0], pairs[0][1])
        text = dr.rule_hint(task)
        assert text.startswith("Rule of length") and "7" in text

    def test_a_size_change_has_no_rule(self, small_task):
        assert dr.rule_hint(small_task).startswith("No rule")

    def test_it_is_among_the_aids(self):
        assert "rule_hint" in dr.AIDS
