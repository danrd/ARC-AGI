"""The driver: a task through the system and what is written about it."""
import json

import numpy as np

from orchestration import run as driver
from orchestration.configs import OrchestrationOptions

from tests.test_orchestration_assemble import GOOD, WRONG_SHAPE, judge, make_task, module


def test_a_task_of_the_dataset_is_loaded_with_its_answer():
    task = driver.load_task("training", "007bbfb7")
    assert task.label == "007bbfb7" and len(task.subtasks) >= 2 and task.test_out.shape == (9, 9)


def test_a_run_is_recorded_with_its_answer_its_source_and_where_the_time_went(tiny_tokenizer):
    record = driver.run_task(make_task(), OrchestrationOptions(), module(tiny_tokenizer, [GOOD]), judge,
                             lambda task: None)
    assert record["task"] == "t" and record["solved"] is True and record["source"] == "llm"
    assert {"symbolic", "llm", "verify", "decide"} <= set(record["trace"])
    assert record["trace"]["llm"]["tokens_in"] > 0 and record["seconds"] > 0
    json.dumps(record)


def test_a_wrong_answer_that_nothing_accepted_is_recorded_as_not_solved(tiny_tokenizer):
    record = driver.run_task(make_task(), OrchestrationOptions(), module(tiny_tokenizer, [WRONG_SHAPE]), judge,
                             lambda task: None, max_iterations=1)
    assert record["solved"] is False and record["source"] != "llm"


def test_the_refinement_and_the_decisions_are_in_the_record(tiny_tokenizer):
    m = module(tiny_tokenizer, [WRONG_SHAPE, GOOD])
    record = driver.run_task(make_task(), OrchestrationOptions(refine_rounds=3), m, judge, lambda task: None)
    assert record["solved"] and record["refinements"][0]["attempts"] == 2 and record["refinements"][0]["accepted"]
    assert record["refinements"][0]["notes"][0] and record["refinements"][0]["notes"][1] == []
    decider = module(tiny_tokenizer, ['{"status": "INVALID", "action": "give_up"}'])
    both = driver.run_task(make_task(), OrchestrationOptions(decision="llm"), module(tiny_tokenizer, [WRONG_SHAPE]),
                           judge, lambda task: None)
    assert both["decisions"] and decider is not None


def test_the_answer_is_read_from_text_or_from_a_grid():
    grid = np.array([[1, 2]])
    assert (driver.answer_of({"solution": grid}) == grid).all()
    assert (driver.answer_of({"solution": "1,2:\n1 12"}) == grid).all()
    assert driver.answer_of({"solution": ""}) is None and driver.answer_of({"solution": "nonsense"}) is None


def test_a_run_is_kept_once_per_task_and_options(tmp_path):
    one, other = OrchestrationOptions(), OrchestrationOptions(feedback=True)
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"key": driver.key_of("a", one, True)}) + "\n{torn")
    keys = driver.done_keys(path)
    assert driver.key_of("a", one, True) in keys
    assert driver.key_of("a", other, True) not in keys and driver.key_of("a", one, False) not in keys
    assert driver.key_of("b", one, True) not in keys and driver.done_keys(tmp_path / "none") == set()


def test_the_summary_adds_the_phases_over_the_runs_and_counts_the_solved(tiny_tokenizer):
    lines = []
    for replies in ([GOOD], [WRONG_SHAPE]):
        r = driver.run_task(make_task(), OrchestrationOptions(), module(tiny_tokenizer, replies), judge,
                            lambda task: None, max_iterations=1)
        r["options"] = vars(OrchestrationOptions())
        lines.append(r)
    text = driver.summarise(lines)
    assert "solved 1 of 2" in text and "llm" in text and "share" in text
