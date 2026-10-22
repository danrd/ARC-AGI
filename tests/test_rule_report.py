"""The rule report: shards of a variant merged, and the confidence gate between the rules and the LLM."""
import json

from orchestration import rule_report as rr


def record(task, cost, right=None):
    out = {"task": task, "rule": None if right is None else "r", "first": [], "second": []}
    if right is not None:
        out.update(cost=cost, first=[right], second=[right])
    return out


def test_the_shards_of_a_variant_are_merged_and_other_files_stay_apart(tmp_path):
    for name, tasks in (("job_vote_0.json", ["a", "b"]), ("job_vote_1.json", ["c"]), ("other.json", ["d"])):
        (tmp_path / name).write_text(json.dumps({"splits": {"training": {"records": [record(t, 1.0, True) for t in tasks]}}}))
    found = rr.variants(tmp_path.glob("*.json"))
    assert sorted(found) == ["job_vote", "other"]
    assert sorted(found["job_vote"]["training"]) == ["a", "b", "c"] and list(found["other"]["training"]) == ["d"]


def test_tasks_with_a_rule_are_counted_by_attempt():
    records = {"a": record("a", 1.0, True), "b": record("b", 2.0, False), "c": record("c", 0, None),
               "d": {**record("d", 3.0, False), "second": [True]}}
    assert rr.tally(records) == (3, 1, 2)


def test_the_gate_takes_a_cheap_rules_answer_even_when_wrong_and_the_llm_for_the_rest():
    records = {"a": record("a", 1.0, True), "b": record("b", 2.0, False), "c": record("c", 5.0, True),
               "d": record("d", 0, None), "e": record("e", 0, None)}
    asked = {"a", "b", "c", "d", "e"}
    llm = {"b", "c", "d"}                                         # right where a rule is wrong, and elsewhere
    rows = {limit: row for limit, *row in rr.gate(records, llm, asked, costs=(1.0, 3.0, float("inf")))}
    assert rows[1.0] == [1, 1, 3, 4]        # a by rule; the LLM on b, c and d
    assert rows[3.0] == [2, 1, 2, 3]        # a right, b wrong by rule; the LLM on c and d (b is lost to the rule)
    assert rows[float("inf")] == [3, 2, 1, 3]   # a and c right by rule; the LLM on d only
    assert rr.gate({}, set(), set(), costs=(1.0,)) == [(1.0, 0, 0, 0, 0)]
