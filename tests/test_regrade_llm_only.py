"""orchestration.regrade_llm_only: the kept replies of a run graded again."""
import numpy as np

import orchestration.regrade_llm_only as regrade
from rl.arc_task import ARCSubtask, ARCTask


def task(split, task_id):
    inp = np.array([[1, 0], [0, 2]])
    return ARCTask(label=task_id, subtasks=[ARCSubtask("s", inp, inp[::-1])], test_inp=inp, test_out=inp[::-1])


def test_a_reply_in_the_asked_for_form_is_counted_now_and_the_old_count_is_kept():
    rows = [{"task": "a", "solved": False, "reply": "<|channel>thought\nhm<channel|>2,2:\n1 0 2\n2 1 0"},
            {"task": "b", "solved": True, "reply": "2,2:\n1 02\n2 10"},
            {"task": "c", "solved": False, "reply": "<|channel>thought\n2,2:\n1 0 2\n2 1 0"}]
    out = regrade.regrade(rows, loader=task)
    assert (out["runs"], out["solved_before"], out["solved_now"]) == (3, 1, 2)
    assert (out["read_now"], out["read_by_old_parser"]) == (2, 1)
    assert out["gained"] == ["a"]
