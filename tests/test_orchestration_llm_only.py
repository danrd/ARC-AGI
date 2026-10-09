"""The model on its own: no grammar, no summary, no hints; random order; a run that is cut is continued."""
import json

import numpy as np

from orchestration import llm_only
from rl.arc_task import ARCSubtask, ARCTask

MODEL = "repo/model-GGUF:model-Q4.gguf"


def make_task(label="t1"):
    inp = np.array([[1, 0], [0, 2]])
    return ARCTask(label=label, subtasks=[ARCSubtask(f"{label}_0", inp, inp[::-1])],
                   test_inp=inp, test_out=inp[::-1])


class Module:
    def __init__(self, replies):
        self.replies, self.asked = replies, []

    def solve(self, task, context=None):
        self.asked.append(task.label)
        return {"solution": self.replies[task.label], "module_results": {"prompt_tokens": 100, "reply_tokens": 5}}


class TestWhatTheModelIsGiven:
    def test_no_grammar_and_no_summary_or_hints(self):
        params = llm_only.llm_only_params(MODEL, "tok/model", 900)
        assert "grammar" not in params["generation"] and params["generation"]["max_tokens"] == 900
        assert "summary" not in params["prompt"]["blocks"] and "search_hints" not in params["prompt"]["blocks"]
        assert params["prompt"]["resolvers"] == ["examples"]
        assert params["llm"]["model"] == "repo/model-GGUF"

    def test_the_notebooks_own_parameters_are_not_changed(self):
        from subsymbolic.local_config import gpu_notebook_params
        before = gpu_notebook_params(MODEL, None, 1200)
        llm_only.llm_only_params(MODEL)
        after = gpu_notebook_params(MODEL, None, 1200)
        assert before == after and "grammar" in after["generation"] and "summary" in after["prompt"]["blocks"]

    def test_the_prompt_it_makes_holds_the_examples_and_neither_summary_nor_hints(self):
        from orchestration.configs import ExperimentConfig
        from subsymbolic.prompt_builder import ApproxTokenizer, PromptBuilder
        from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY
        config = ExperimentConfig.from_dict(llm_only.llm_only_params(MODEL))
        builder = PromptBuilder(config.prompt, ApproxTokenizer(), resolver_registry=RESOLVER_REGISTRY,
                                filter_registry=FILTER_REGISTRY)
        task = make_task()
        prompt = builder.build(task, context={"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp})
        assert "Training example 1" in prompt and "<TASK_REPR>" in prompt and "<OUTPUT_FORMAT>" in prompt
        assert "hint" not in prompt.lower() and "summary" not in prompt.lower()


class TestTheTasks:
    def test_the_tasks_are_the_370_of_the_earlier_runs_by_difficulty(self):
        import collections
        difficulties = llm_only.task_difficulties()
        assert len(difficulties) == 370
        assert collections.Counter(difficulties.values()) == {"easy": 95, "medium": 53, "hard": 67,
                                                              "impossible": 68, "very_hard": 87}
        everything = json.loads((llm_only.ROOT / "data/datasets/ARC/evaluation_difficulty.json").read_text())
        assert all(everything[t] == d for t, d in difficulties.items())
        assert not {"symbolic", "oversized"} & set(difficulties.values())

    def test_a_fixed_shuffle_that_is_not_the_files_order(self):
        difficulties = llm_only.task_difficulties()
        first, again = llm_only.shuffled_tasks(difficulties, 0), llm_only.shuffled_tasks(difficulties, 0)
        assert first == again and sorted(first) == sorted(difficulties) and first != sorted(first)
        assert llm_only.shuffled_tasks(difficulties, 1) != first


class TestGrading:
    def test_the_answer_a_grid_of_the_wrong_shape_and_prose(self):
        task = make_task()
        right = llm_only.grade(task, "2,2:\n1 02\n2 10")
        assert right["solved"] and right["parsed"] and right["shape_ok"]
        wrong_shape = llm_only.grade(task, "1,2:\n1 02")
        assert wrong_shape["parsed"] and not wrong_shape["shape_ok"] and not wrong_shape["solved"]
        prose = llm_only.grade(task, "I think the rule is to flip it.")
        assert not prose["parsed"] and not prose["solved"]


class TestARun:
    def loader(self, split, task_id):
        return make_task(task_id)

    def test_every_task_is_asked_once_and_recorded_with_its_reply(self, tmp_path):
        module = Module({"a": "2,2:\n1 02\n2 10", "b": "nonsense"})
        out = tmp_path / "o.jsonl"
        records = llm_only.run(["a", "b"], module, str(out), loader=self.loader)
        assert [r["solved"] for r in records] == [True, False]
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        assert lines[1]["reply"] == "nonsense" and lines[0]["prompt_tokens"] == 100

    def test_a_run_that_stops_is_continued_and_what_others_did_is_skipped(self, tmp_path):
        module = Module({"a": "x", "b": "x", "c": "x"})
        out, other = tmp_path / "o.jsonl", tmp_path / "other_1.jsonl"
        other.write_text(json.dumps({"task": "c", "solved": False}) + "\n")
        llm_only.run(["a"], module, str(out), loader=self.loader)
        llm_only.run(["a", "b", "c"], module, str(out), skip=[str(tmp_path / "other_*.jsonl")], loader=self.loader)
        assert module.asked == ["a", "b"]

    def test_it_stops_starting_tasks_after_the_hours(self, tmp_path):
        times = iter([0, 100, 3000, 4000])
        module = Module({t: "x" for t in "abc"})
        llm_only.run(["a", "b", "c"], module, str(tmp_path / "o.jsonl"), hours=1, clock=lambda: next(times),
                     loader=self.loader)
        assert module.asked == ["a", "b"]

    def test_the_summary_counts_what_was_solved(self):
        text = llm_only.summarise([{"solved": True, "shape_ok": True, "parsed": True, "seconds": 3600,
                                    "prompt_tokens": 10, "reply_tokens": 2, "difficulty": "easy"},
                                   {"solved": False, "shape_ok": False, "parsed": False, "seconds": 0,
                                    "difficulty": "hard"}])
        assert text.splitlines()[0].startswith("solved 1 of 2")
        assert "easy       solved 1 of 1" in text and "hard       solved 0 of 1" in text
        assert "medium" not in text

    def test_the_difficulty_of_a_task_is_kept_in_its_line(self, tmp_path):
        module = Module({"a": "x"})
        records = llm_only.run(["a"], module, str(tmp_path / "o.jsonl"), loader=self.loader,
                               difficulties={"a": "very_hard"})
        assert records[0]["difficulty"] == "very_hard"
