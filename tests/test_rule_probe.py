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


def test_a_shard_takes_every_nth_task_and_the_strategy_reaches_the_search(monkeypatch, tmp_path):
    import json
    import sys
    train = [dict(zip(("input", "output"), between(9, 1, 1, 5))), dict(zip(("input", "output"), between(9, 3, 2, 7)))]
    test_in, test_out = between(10, 4, 0, 3)
    challenge = {"train": train, "test": [{"input": test_in}]}
    names = ["a", "b", "c", "d", "e"]
    (tmp_path / "x_challenges.json").write_text(json.dumps({n: challenge for n in names}))
    (tmp_path / "x_solutions.json").write_text(json.dumps({n: [test_out] for n in names}))
    monkeypatch.setattr(rp, "DATA", tmp_path)
    monkeypatch.setattr(rp, "ProcessPoolExecutor", lambda workers: __import__("concurrent.futures").futures.ThreadPoolExecutor(1))
    seen = []
    original = rp.rule_search.attempts
    monkeypatch.setattr(rp.rule_search, "attempts",
                        lambda rules, grid, tries=2, strategy="shortest", **kw: seen.append(strategy) or
                        original(rules, grid, tries, strategy, **kw))
    for shard in (0, 1):
        out = tmp_path / f"out{shard}.json"
        monkeypatch.setattr(sys, "argv", ["rule_probe", "--split", "x", "--sets", "core", "--shard", str(shard), "--shards", "2",
                                          "--strategy", "vote", "--workers", "1", "--out", str(out)])
        rp.main()
        report = json.loads(out.read_text())
        got = [r["task"] for r in report["splits"]["x"]["records"]]
        assert got == names[shard::2] and report["strategy"] == "vote" and report["shard"] == [shard, 2]
    assert set(seen) == {"vote"}
    assert rp.rule_search.THRESHOLD_COST == 0.25
    rp.probe("a", challenge, [test_out], ["core"], threshold_cost=0.75)
    assert rp.rule_search.THRESHOLD_COST == 0.75
    rp.rule_search.THRESHOLD_COST = 0.25


def test_the_threshold_cost_of_the_command_line_reaches_the_search(monkeypatch, tmp_path):
    import json
    import sys
    train = [dict(zip(("input", "output"), between(9, 1, 1, 5))), dict(zip(("input", "output"), between(9, 3, 2, 7)))]
    test_in, test_out = between(10, 4, 0, 3)
    (tmp_path / "x_challenges.json").write_text(json.dumps({"a": {"train": train, "test": [{"input": test_in}]}}))
    (tmp_path / "x_solutions.json").write_text(json.dumps({"a": [test_out]}))
    monkeypatch.setattr(rp, "DATA", tmp_path)
    monkeypatch.setattr(rp, "ProcessPoolExecutor", lambda workers: __import__("concurrent.futures").futures.ThreadPoolExecutor(1))
    monkeypatch.setattr(rp.rule_search, "THRESHOLD_COST", 0.25)
    monkeypatch.setattr(sys, "argv", ["rule_probe", "--split", "x", "--sets", "core", "--threshold-cost", "0.75",
                                      "--out", str(tmp_path / "o.json")])
    rp.main()
    assert rp.rule_search.THRESHOLD_COST == 0.75
    assert json.loads((tmp_path / "o.json").read_text())["threshold_cost"] == 0.75


def test_the_relational_flag_and_the_width_reach_the_search(monkeypatch):
    seen = {}
    monkeypatch.setattr(rp.rule_search, "search", lambda train, feature_set: seen.update(
        relational=rp.rule_search.RELATIONAL, wide=rp.rule_search.WIDE, cap=rp.rule_search.SUPERSET_CAP,
        keep=rp.rule_search.KEEP) or [])
    saved = (rp.rule_search.RELATIONAL, rp.rule_search.WIDE, rp.rule_search.SUPERSET_CAP, rp.rule_search.KEEP)
    try:
        rp.probe("a", {"train": [], "test": []}, [], ["core"], relational=True, wide=1000, keep=24)
        assert seen == {"relational": True, "wide": 1000, "cap": 300, "keep": 24}
        rp.probe("a", {"train": [], "test": []}, [], ["core"])
        assert seen["relational"] is False
    finally:
        rp.rule_search.RELATIONAL, rp.rule_search.WIDE, rp.rule_search.SUPERSET_CAP, rp.rule_search.KEEP = saved
