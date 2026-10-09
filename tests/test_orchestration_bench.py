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
    assert seen == [(1, False)] * 2 + [(160, False)] * 2 + [(160, True)] * 2 * (1 + len(bench.SIMPLE_GRAMMARS))
    assert set(result) == {"prefill", "free", "grammar", *bench.SIMPLE_GRAMMARS} and result["free"]["calls"] == 2
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


def test_a_simpler_grammar_replaces_the_runners_own_in_the_copy_only():
    runner = Runner({"grammar": "root ::= long", "top_k": 3})
    flat = bench.with_limits(runner, 7, grammar="root ::= [0-9]+")
    assert flat.generation_kwargs["extra_body"] == {"grammar": "root ::= [0-9]+", "top_k": 3}
    assert flat.generation_kwargs["max_tokens"] == 7
    assert runner.generation_kwargs["extra_body"]["grammar"] == "root ::= long"
    assert all(isinstance(text, str) and text.startswith("root ::=") for text in bench.SIMPLE_GRAMMARS.values())


def test_the_report_compares_each_simple_grammar_with_free_and_with_the_systems():
    one = {"calls": 1, "mean_seconds": 1.0, "median_seconds": 1.0, "mean_reply_chars": 10, "mean_prompt_chars": 100}
    result = {"prefill": one, "free": {**one, "mean_seconds": 2.0}, "grammar": {**one, "mean_seconds": 8.0},
              "flat": {**one, "mean_seconds": 4.0}, "rows": {**one, "mean_seconds": 6.0}}
    text = bench.render("m", result, 160)
    assert "flat grammar: +2.00 s against free (2.0x), -4.00 s against the system's grammar" in text
    assert "rows grammar: +4.00 s against free (3.0x), -2.00 s against the system's grammar" in text
