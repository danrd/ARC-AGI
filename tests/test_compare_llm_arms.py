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

    def test_prompt_text_that_looks_like_markup_is_escaped(self, tmp_path):
        """Prompts carry <SUMMARY> tags and the diff prints them verbatim -
        one unescaped bracket silently eats the rest of the report."""
        page = self._write(tmp_path, [("t", "gained by B",
                                       self._side(prompt="plain"),
                                       self._side(prompt="plain\n<SUMMARY>"))])

        assert "&lt;SUMMARY&gt;" in page
        assert "<SUMMARY>" not in page

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

    def test_images_go_beside_the_report_rather_than_into_it(self, tmp_path):
        """A data URI put the whole PNG on one 74,715-character line and did
        not appear in the editor's preview, which runs in a webview under a
        content policy that can refuse data: sources. A relative link to a
        file renders there, on GitHub, and anywhere else."""
        import collections

        path = tmp_path / "arms.md"
        script.write_report(
            path, "A", "B", per_shard=[("easy", 95, 4, 3, 0, 1, 31, 27)],
            cost=[("easy", 2592, 3086, 187, 187, 2.8, 1.9)],
            totals=collections.Counter(tasks=95, solved_a=4, solved_b=3),
            gained=["x"], lost=[], p_value=0.45,
            flips=[("2c0b0aff", "gained by B", self._side(), self._side())],
            with_images=True)
        text = path.read_text(encoding="utf-8")

        assert "![2c0b0aff](arms_images/2c0b0aff.png)" in text
        assert "data:image" not in text
        assert (tmp_path / "arms_images" / "2c0b0aff.png").exists()
        assert max(len(line) for line in text.splitlines()) < 1000

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

    def test_the_answers_are_scored_not_reprinted_as_digits(self, tmp_path):
        """The picture above shows the task and its answer; a second copy as
        digits is the same thing in the form that is harder to read. The
        terminal's --detail still prints them - it has no picture to defer
        to."""
        markdown = self._write(tmp_path, [("t", "gained by B", self._side(),
                                            self._side(text="0 1\n1 0"))], "report.md")

        assert "**Answer A**" in markdown and "**Answer B**" in markdown
        assert "scored 0.500" in markdown
        assert "0 1\n1 0" not in markdown

    def test_the_diff_says_which_arm_each_side_is(self, tmp_path):
        """`+` and `-` are readable from context and that is exactly the
        problem - nothing marks which arm is which, so the eye has to
        reconstruct it every time."""
        markdown = self._write(tmp_path, [("t", "gained by B",
                                            self._side(prompt="a"),
                                            self._side(prompt="a\nb"))], "report.md")

        assert "`-` is A only (arm A)" in markdown
        assert "`+` is B only (arm B)" in markdown


class TestCleaningUpTheDownloads:
    """Comparing five shards downloads ten checkpoints, each holding every
    prompt and every generation of its run, and wandb unpacks them under
    ./artifacts/ where they stay. Removed when the script finishes - but it
    is a recursive delete driven by a path that came from outside, so what
    it will remove is checked first."""

    @staticmethod
    def _checkpoint_dir(root, name="checkpoint-abc123:v0"):
        directory = root / "artifacts" / name
        directory.mkdir(parents=True)
        (directory / "checkpoint.json").write_text("{}")
        return directory

    def test_a_checkpoint_directory_is_removed(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path)

        script.remove_downloads([directory])

        assert not directory.exists()

    def test_the_empty_artifacts_folder_goes_with_it(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path)

        script.remove_downloads([directory])

        assert not (tmp_path / "artifacts").exists()

    def test_an_artifacts_folder_still_holding_something_stays(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path)
        (tmp_path / "artifacts" / "something-else").mkdir()

        script.remove_downloads([directory])

        assert (tmp_path / "artifacts" / "something-else").exists()

    def test_a_directory_that_is_not_a_checkpoint_is_left_alone(self, tmp_path):
        """The guard that matters: these paths come back from wandb, and a
        recursive delete should not act on one that does not look like what
        it was told to clean up."""
        directory = tmp_path / "artifacts" / "somebody-elses-data"
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("x")

        script.remove_downloads([directory])

        assert (directory / "keep.txt").exists()

    def test_the_same_directory_twice_is_removed_once(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path)

        script.remove_downloads([directory, directory])

        assert not directory.exists()

    def test_a_directory_that_was_never_created_is_not_an_error(self, tmp_path):
        script.remove_downloads([tmp_path / "artifacts" / "checkpoint-gone:v0"])


class TestGatheringOnePromptBlock:
    """A block is built per task and stored inside that task's prompt, so
    checking whether it says the right thing means opening one artifact per
    task and reading past everything else. --section gathers them, which is
    what turns "are the summaries any good" into a question with an answer.
    """

    PROMPT = ("<GENERAL_INSTRUCTION>\nSolve it.\n</GENERAL_INSTRUCTION>\n"
              "<EXAMPLES>\n0 1\n</EXAMPLES>\n"
              "<SUMMARY>\nWhat changes:\n  - shapes are duplicated\n</SUMMARY>\n"
              "<OUTPUT_FORMAT>\nGrid only\n</OUTPUT_FORMAT>")

    def test_a_block_is_read_out_whole(self):
        block = script.extract_section(self.PROMPT, "SUMMARY")

        assert block == "What changes:\n  - shapes are duplicated"

    def test_the_name_is_matched_however_it_is_typed(self):
        assert script.extract_section(self.PROMPT, "summary") is not None
        assert script.extract_section(self.PROMPT, "Summary") is not None

    def test_a_block_that_is_not_there_is_None_not_empty(self):
        """The summary resolver omits itself when it found nothing, so "the
        block was empty" and "the block was not built for this task" are
        different facts about a run and must not collapse."""
        assert script.extract_section(self.PROMPT, "HINTS") is None
        assert script.extract_section("<SUMMARY>\n</SUMMARY>", "SUMMARY") == ""

    def test_only_the_named_block_comes_back(self):
        block = script.extract_section(self.PROMPT, "EXAMPLES")

        assert block == "0 1"
        assert "SUMMARY" not in block

    def test_the_tags_a_prompt_carries_can_be_listed(self):
        """What to offer when the block that was asked for is in none of
        them - a filter that matches nothing otherwise says nothing."""
        assert script.section_names(self.PROMPT) == [
            "GENERAL_INSTRUCTION", "EXAMPLES", "SUMMARY", "OUTPUT_FORMAT"]

    def test_collecting_separates_the_tasks_that_have_it_from_those_that_do_not(self):
        rows = {"a": {"prompt_text": self.PROMPT},
                "b": {"prompt_text": "<EXAMPLES>\n0\n</EXAMPLES>"}}

        found, missing, tags = script.collect_section(rows, "SUMMARY")

        assert set(found) == {"a"}
        assert missing == ["b"]
        assert tags["EXAMPLES"] == 2

    def test_a_report_of_the_gathered_blocks_names_every_task(self, tmp_path):
        path = tmp_path / "summaries.md"
        blocks = {"with summary": ({("easy", "2c0b0aff"): "findings here",
                                     ("medium", "59341089"): "other findings"},
                                    ["009d5c81"], {})}

        script.write_section_report(path, "SUMMARY", blocks)
        text = path.read_text(encoding="utf-8")

        assert "<SUMMARY> in with summary" in text
        assert "2c0b0aff" in text and "59341089" in text
        assert "findings here" in text
        assert "absent on 1" in text

    def test_the_html_form_escapes_what_it_prints(self, tmp_path):
        path = tmp_path / "summaries.html"
        blocks = {"arm": ({("easy", "t"): "<script>bad()</script>"}, [], {})}

        script.write_section_report(path, "SUMMARY", blocks)
        text = path.read_text(encoding="utf-8")

        assert "<script>bad()</script>" not in text
        assert "&lt;script&gt;" in text
