"""The measurement of a model's speed: three timings, each with the limits it is meant to have."""
from orchestration import bench


class Runner:
    """A runner that takes time in proportion to what it is asked to write, and more with a grammar."""

    def __init__(self, extra=None):
        self.generation_kwargs = {"max_tokens": 1200, "extra_body": extra or {}}
        self.clock = []

    def generate(self, prompt):
        body = self.generation_kwargs["extra_body"]
        self.clock.append((self.generation_kwargs["max_tokens"], "grammar" in body))
        return "x" * min(self.generation_kwargs["max_tokens"], 50)


def test_each_timing_asks_the_runner_for_what_it_is_named():
    runner = Runner({"grammar": "root ::= x", "top_k": 3})
    seen = []

    class Spy(Runner):
        def generate(self, prompt):
            seen.append((self.generation_kwargs["max_tokens"], "grammar" in self.generation_kwargs["extra_body"]))
            return "ab"

    spy = Spy({"grammar": "g", "top_k": 3})
    result = bench.run(spy, ["p1", "p2"], tokens=160)
    assert seen == [(1, False)] * 2 + [(160, False)] * 2 + [(160, True)] * 2
    assert set(result) == {"prefill", "free", "grammar"} and result["free"]["calls"] == 2
    assert spy.generation_kwargs["max_tokens"] == 1200 and "grammar" in spy.generation_kwargs["extra_body"]
    assert runner.generation_kwargs["extra_body"]["top_k"] == 3


def test_a_runner_without_a_grammar_cannot_be_measured_with_one():
    import pytest
    with pytest.raises(ValueError):
        bench.with_limits(Runner(), 10, grammar=True)


def test_the_summary_and_the_report_say_what_the_grammar_costs():
    rows = [{"seconds": 1.0, "prompt_chars": 100, "reply_chars": 10}, {"seconds": 3.0, "prompt_chars": 300, "reply_chars": 30}]
    summary = bench.summarise(rows)
    assert summary["mean_seconds"] == 2.0 and summary["median_seconds"] == 2.0 and summary["mean_reply_chars"] == 20
    result = {"prefill": {**summary, "mean_seconds": 1.0}, "free": {**summary, "mean_seconds": 3.0},
              "grammar": {**summary, "mean_seconds": 9.0}}
    text = bench.render("m", result, 160)
    assert "the grammar adds +6.00 s (3.0x the free reply" in text and "writing the reply (free minus prefill): 2.00 s" in text
