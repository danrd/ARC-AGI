"""The rule probe: which tasks it searches and how it counts a solved one."""
import numpy as np

from scripts import rule_probe as rp


def between(width, row, left, right):
    grid = np.zeros((5, width), dtype=int)
    grid[row, left] = grid[row, right] = 2
    out = grid.copy()
    out[row, left + 1:right] = 4
    return grid.tolist(), out.tolist()


def test_only_tasks_that_keep_their_shape_are_searched():
    same = {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0, 0]]}]}
    assert rp.same_shape(same, [[[1, 1]]])
    assert not rp.same_shape(same, [[[1]]])
    grown = {"train": [{"input": [[0]], "output": [[1, 1]]}], "test": [{"input": [[0]]}]}
    assert not rp.same_shape(grown, [[[1]]])


def test_a_task_is_solved_when_every_test_is_right_in_two_attempts_and_its_rule_is_kept():
    train = [dict(zip(("input", "output"), between(9, 1, 1, 5))), dict(zip(("input", "output"), between(9, 3, 2, 7)))]
    test_in, test_out = between(10, 4, 0, 3)
    record = rp.probe("t", {"train": train, "test": [{"input": test_in}]}, [test_out], ["core"])
    assert record["rule"] and record["feature_set"] == "core" and record["second"] == [True]
    nines = (np.asarray(test_in) * 0 + 9).tolist()
    wrong = rp.probe("t", {"train": train, "test": [{"input": test_in}]}, [nines], ["core"])
    assert wrong["rule"] and wrong["second"] == [False] and wrong["first"] == [False]
