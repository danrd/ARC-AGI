"""Sending the prompt variants to a model: finding them, grading the replies, not asking twice."""
import json

import numpy as np

from scripts import run_prompt_variants as rpv

ANSWER = "grid shape: 2,2\n1 12\n2 34\n"
GOOD = "2,2:\n1 12\n2 34"


def write_task(tmp_path, task="t", names=("p0", "p3_perm")):
    folder = tmp_path / task
    folder.mkdir()
    (folder / "answer.txt").write_text(ANSWER)
    (folder / "answer_perm.txt").write_text("grid shape: 2,2\n1 21\n2 43\n")
    for name in names:
        (folder / f"{name}.txt").write_text("prompt " + name)
    return folder


class Fake:
    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def generate(self, prompt):
        self.asked.append(prompt)
        return self.reply


class TestGrading:
    def test_an_answer_file_is_a_grid(self, tmp_path):
        (tmp_path / "a.txt").write_text(ANSWER)
        assert (rpv.read_answer(tmp_path / "a.txt") == np.array([[1, 2], [3, 4]])).all()

    def test_equal_grid_is_solved_and_nothing_else_is(self):
        answer = np.array([[1, 2], [3, 4]])
        assert rpv.grade(GOOD, answer)["solved"]
        wrong = rpv.grade("2,2:\n1 12\n2 35", answer)
        assert wrong["parsed"] and not wrong["solved"] and wrong["cells_right"] == 0.75
        assert rpv.grade("no grid here", answer) == {"parsed": False, "solved": False, "shape_ok": False,
                                                     "cells_right": 0.0}
        assert not rpv.grade("1,1:\n1 1", answer)["shape_ok"]

    def test_thinking_is_not_the_answer(self):
        assert rpv.grade("<think>2,2:\n1 99\n2 99</think>\n" + GOOD, np.array([[1, 2], [3, 4]]))["solved"]


class TestRunning:
    def test_prompts_are_found_by_task_and_variant_and_answers_are_not_prompts(self, tmp_path):
        write_task(tmp_path)
        (tmp_path / "t" / "p0.m.response.txt").write_text("old")
        found = rpv.prompts(tmp_path)
        assert [(t, v, p) for t, v, p, _, _ in found] == [("t", "p0", False), ("t", "p3", True)]
        assert found[1][4].name == "answer_perm.txt"
        assert [v for _, v, _, _, _ in rpv.prompts(tmp_path, variants=["p3"])] == ["p3"]
        assert rpv.prompts(tmp_path, tasks=["other"]) == []

    def test_a_reply_is_kept_graded_and_not_asked_for_twice(self, tmp_path):
        write_task(tmp_path)
        fake = Fake(GOOD)
        records = rpv.run(tmp_path, ["a/b"], lambda model: fake)
        assert [r["solved"] for r in records] == [True, False]  # the shuffled answer is another grid
        assert (tmp_path / "t" / "p0.a-b.response.txt").read_text() == GOOD
        assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2
        assert rpv.run(tmp_path, ["a/b"], lambda model: fake) == [] and len(fake.asked) == 2
        assert len(rpv.run(tmp_path, ["a/b", "c/d"], lambda model: fake)) == 2

    def test_a_failed_request_leaves_no_reply_to_stand_in_for_it(self, tmp_path):
        write_task(tmp_path, names=("p0",))

        class Broken:
            def generate(self, prompt):
                raise RuntimeError("down")

        assert rpv.run(tmp_path, ["m"], lambda model: Broken()) == []
        assert not list((tmp_path / "t").glob("*.response.txt"))

    def test_the_report_marks_solved_wrong_and_unparsed(self, tmp_path):
        write_task(tmp_path, names=("p0", "p3"))
        rows = [("p0", True, True), ("p3", True, False)]
        (tmp_path / "results.jsonl").write_text("".join(
            json.dumps({"task": "t", "variant": v, "perm": False, "model": "m", "parsed": p, "solved": s}) + "\n"
            for v, p, s in rows + [("p5", False, False)]))
        text = rpv.report(tmp_path)
        assert "| t | S | x | - |" in text and "1 of 3 solved" in text
