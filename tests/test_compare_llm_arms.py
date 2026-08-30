"""Tests for scripts/compare_llm_arms.py - the parts that decide what the
comparison says, kept apart from wandb.

The script exists because five per-shard tests answered nothing: with 1, 4,
1, 1 and 0 discordant pairs there was no shard on which a test could speak.
So the pooling and the exactness of the test it pools into are the load-
bearing pieces, along with the run lookup, whose one failure mode is silent
- a filter on the wrong field returns an empty set rather than an error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "compare_llm_arms", REPO_ROOT / "scripts" / "compare_llm_arms.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


class TestTheExactTest:
    """Chi-square is what McNemar is usually written with and it reads
    optimistic on a handful of pairs - which is all a binary outcome at a 7%
    base rate ever produces, however many tasks are run."""

    def test_no_discordant_pairs_is_no_evidence(self):
        assert script.two_sided_binomial(0, 0) == 1.0

    def test_the_measured_case_does_not_reach_significance(self):
        """5 gained against 2 lost, pooled over five shards - the result the
        script was written to state plainly rather than imply."""
        assert script.two_sided_binomial(5, 2) == pytest.approx(0.453, abs=0.001)

    def test_a_clean_sweep_of_seven_does(self):
        assert script.two_sided_binomial(7, 0) == pytest.approx(0.0156, abs=0.001)

    def test_it_is_symmetric(self):
        assert script.two_sided_binomial(2, 5) == script.two_sided_binomial(5, 2)

    def test_chi_square_would_have_called_the_sweep_of_five_significant(self):
        """Why exact: the continuity-corrected chi-square puts 5-0 at 3.2
        and 6-0 at 4.17, crossing 3.84 between them, while the exact test
        puts 6-0 at 0.031 - close, but they disagree about 5-0, which is
        the size of result this comparison actually produces."""
        exact = script.two_sided_binomial(5, 0)

        assert exact > 0.05
        assert (abs(5 - 0) - 1) ** 2 / 5 == pytest.approx(3.2)


class TestReadingWhatTheArmsDifferBy:
    def test_an_added_block_is_reported(self):
        before = "instruction\nexamples\noutput"
        after = "instruction\nexamples\n<SUMMARY>\nwhat changes\n</SUMMARY>\noutput"

        added = script.prompt_difference(before, after)

        assert [line[1:] for line in added] == ["<SUMMARY>", "what changes", "</SUMMARY>"]
        assert all(line.startswith("+") for line in added)

    def test_identical_prompts_differ_by_nothing(self):
        assert script.prompt_difference("same\ntext", "same\ntext") == []

    def test_a_difference_the_arms_were_not_supposed_to_have_still_shows(self):
        """The reason to diff rather than assume: if the arms differ
        anywhere beyond the block under test, the comparison is not
        measuring what it says, and this is where that surfaces."""
        added = script.prompt_difference("temperature 0.2\nexamples",
                                          "temperature 0.9\nexamples")

        assert any(line.startswith("-") for line in added)
        assert any(line.startswith("+") for line in added)


class TestReadingACheckpoint:
    def test_a_wrapped_generation_is_unwrapped(self):
        """SubsymbolicModule.solve returns a dict; older records hold the
        string itself, and both turn up in checkpoints."""
        assert script.generation_text({"generation_result": {"solution": "grid"}}) == "grid"
        assert script.generation_text({"generation_result": "grid"}) == "grid"
        assert script.generation_text({}) == ""

    def test_tasks_are_keyed_by_string_id(self):
        """solved_tasks and prompts_data do not agree on the type of a task
        id, and an int key silently intersects with nothing."""
        indexed = script.index_tasks({"prompts_data": [
            {"task_id": 12345, "primary_score": 0.5, "prompt_length": 10,
             "generation_result": {"solution": "x"}, "processing_time_min": 1.0}]})

        assert set(indexed) == {"12345"}
        assert indexed["12345"]["score"] == 0.5

    def test_a_missing_field_does_not_become_None_arithmetic(self):
        indexed = script.index_tasks({"prompts_data": [{"task_id": "a"}]})

        assert indexed["a"] == {"score": 0.0, "prompt_text": "", "prompt": 0,
                                 "generated": "", "minutes": 0.0}


class _FakeRun:
    def __init__(self, name, run_id, model="Qwen3-30B", description="with summary"):
        self.name, self.id = name, run_id
        self.config = {"model": model, "run_description": description}


class _FakeApi:
    """Records the filter it was given, so the field names can be checked -
    the one failure mode of this lookup is silent."""

    def __init__(self, runs):
        self._runs, self.filters = runs, None

    def runs(self, path, filters=None):
        self.filters = filters
        return self._runs


class TestFindingAnArm:
    def test_runs_are_keyed_by_shard(self):
        api = _FakeApi([_FakeRun("easy", "a1"), _FakeRun("medium", "a2")])

        found = script.find_runs(api, "e/p", "with summary")

        assert set(found) == {"easy", "medium"}
        assert found["easy"].id == "a1"

    def test_the_arm_is_matched_as_a_substring_not_for_equality(self):
        """Descriptions are free text typed per run and they drift - "with
        summary" against "with summary, without knowledge injection". An
        exact match is a query that returns nothing and says nothing."""
        assert script.arm_filters("with summary") == {
            "config.run_description": {"$regex": "with summary"}}

    def test_a_rerun_does_not_replace_the_newer_run(self):
        """api.runs returns newest first; a shard run twice should report
        the newer result, not whichever came last out of the iterator."""
        api = _FakeApi([_FakeRun("easy", "new"), _FakeRun("easy", "old")])

        assert script.find_runs(api, "e/p", "x")["easy"].id == "new"

    def test_a_model_filter_is_a_substring_match(self):
        assert script.arm_filters("x", model="Qwen3-30B")["config.model"] == {
            "$regex": "Qwen3-30B"}

    def test_one_description_spanning_two_models_is_refused(self):
        """The failure this exists for: quietly keeping the first run would
        compare one model's baseline against another model's summary arm,
        and the result would look entirely ordinary."""
        api = _FakeApi([_FakeRun("easy", "q", model="Qwen3-30B"),
                        _FakeRun("easy", "g", model="gemma-4-31B")])

        with pytest.raises(SystemExit) as raised:
            script.find_runs(api, "e/p", "with summary")

        assert "--model" in str(raised.value)

    def test_narrowing_by_model_resolves_it(self):
        api = _FakeApi([_FakeRun("easy", "q", model="Qwen3-30B")])

        assert script.find_runs(api, "e/p", "with summary", model="Qwen")["easy"].id == "q"


class TestTheReportIsOneFile:
    """Written because the grids are otherwise never seen: a figure drawn
    under !python has nowhere to appear, and a directory of PNGs beside a
    terminal dump is three things to send instead of one."""

    @staticmethod
    def _side(text="0 1", prompt="a\nb", score=0.5):
        return {"score": score, "prompt_text": prompt, "prompt": len(prompt),
                "generated": text, "minutes": 0.3}

    def _write(self, tmp_path, flips, name="report.html", **kwargs):
        import collections

        path = tmp_path / name
        script.write_report(
            path, "arm A", "arm B",
            per_shard=[("easy", 95, 4, 3, 0, 1, 31, 27)],
            cost=[("easy", 2592, 3086, 187, 187, 2.8, 1.9)],
            totals=collections.Counter(tasks=95, solved_a=4, solved_b=3),
            gained=["x"], lost=["y"], p_value=0.45, flips=flips,
            with_images=False, **kwargs)
        return path.read_text(encoding="utf-8")

    def test_it_holds_the_numbers_and_the_flips(self, tmp_path):
        page = self._write(tmp_path, [("2c0b0aff", "gained by B",
                                       self._side(), self._side())])

        assert "2c0b0aff" in page and "gained by B" in page
        assert "not significant" in page
        assert "easy" in page

    def test_a_generation_that_looks_like_markup_is_escaped(self, tmp_path):
        """Answers are model output and prompts carry <SUMMARY> tags - one
        unescaped bracket silently eats the rest of the report."""
        page = self._write(tmp_path, [("t", "gained by B",
                                       self._side(text="<script>bad()</script>"),
                                       self._side(prompt="<SUMMARY>\nfindings"))])

        assert "<script>bad()</script>" not in page
        assert "&lt;script&gt;" in page
        assert "&lt;SUMMARY&gt;" in page

    def test_the_prompt_difference_is_marked_up_by_direction(self, tmp_path):
        page = self._write(tmp_path, [("t", "gained by B",
                                       self._side(prompt="a"),
                                       self._side(prompt="a\nb"))])

        assert "class=add" in page

    def test_identical_prompts_say_so_rather_than_showing_an_empty_block(self, tmp_path):
        page = self._write(tmp_path, [("t", "lost by B",
                                       self._side(prompt="same"),
                                       self._side(prompt="same"))])

        assert "the prompts are identical" in page

    def test_no_flips_still_produces_a_readable_report(self, tmp_path):
        page = self._write(tmp_path, [])

        assert "0 tasks that flipped" in page
        assert "Cost" in page


class TestTheSameReportAsMarkdown:
    """VS Code opens .html as source - it has no built-in HTML preview the
    way it has one for Markdown - so a report meant to be read where the
    code is has to be .md. Chosen by the file's suffix rather than a flag,
    because which one is wanted depends only on where it will be read."""

    @staticmethod
    def _side(text="0 1", prompt="a\nb", score=0.5):
        return {"score": score, "prompt_text": prompt, "prompt": len(prompt),
                "generated": text, "minutes": 0.3}

    @staticmethod
    def _write(tmp_path, flips, name):
        import collections

        path = tmp_path / name
        script.write_report(
            path, "arm A", "arm B",
            per_shard=[("easy", 95, 4, 3, 0, 1, 31, 27)],
            cost=[("easy", 2592, 3086, 187, 187, 2.8, 1.9)],
            totals=collections.Counter(tasks=95, solved_a=4, solved_b=3),
            gained=["x"], lost=["y"], p_value=0.45, flips=flips,
            with_images=False)
        return path.read_text(encoding="utf-8")

    def test_the_suffix_picks_the_format(self, tmp_path):
        markdown = self._write(tmp_path, [], "report.md")
        page = self._write(tmp_path, [], "report.html")

        assert markdown.lstrip().startswith("# ")
        assert "<style>" not in markdown
        assert "<style>" in page

    def test_tables_survive_the_translation(self, tmp_path):
        markdown = self._write(tmp_path, [], "report.md")

        assert "| shard | tasks |" in markdown
        assert "|---|" in markdown

    def test_a_prompt_difference_becomes_a_diff_block(self, tmp_path):
        """```diff is what makes an editor colour the two directions - the
        leading +/- alone is invisible in a proportional font."""
        markdown = self._write(tmp_path, [("t", "gained by B",
                                            self._side(prompt="a"),
                                            self._side(prompt="a\nb"))], "report.md")

        assert "```diff" in markdown
        assert "+b" in markdown

    def test_generations_are_fenced_so_a_grid_keeps_its_rows(self, tmp_path):
        """Markdown folds consecutive lines into one paragraph, which turns
        a grid into a sentence."""
        markdown = self._write(tmp_path, [("t", "gained by B", self._side(),
                                            self._side(text="0 1\n1 0"))], "report.md")

        assert "```\n0 1\n1 0\n```" in markdown
