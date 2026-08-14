"""Tests for subsymbolic/subsymbolic_module.py's runner construction /
injection - the lazy-build-by-default, inject-if-you-have-one-already
contract described in the module's own docstring.
"""
from __future__ import annotations

from unittest.mock import patch

from orchestration.configs import ExperimentConfig
from subsymbolic.subsymbolic_module import SubsymbolicModule


class _FakeRunner:
    def __init__(self, label="fake"):
        self.label = label
        self.closed = False

    def generate(self, prompt):
        return f"{self.label}:{prompt}"

    def close(self):
        self.closed = True


def test_runner_is_built_lazily_when_not_injected(tiny_tokenizer):
    """No runner passed - build_runner() shouldn't be called until the
    `runner` property is actually touched, and only once after that."""
    module = SubsymbolicModule(ExperimentConfig(), tiny_tokenizer)

    with patch("subsymbolic.subsymbolic_module.build_runner", return_value=_FakeRunner()) as mock_build:
        mock_build.assert_not_called()
        first = module.runner
        second = module.runner

    mock_build.assert_called_once_with(module.experiment_config)
    assert first is second


def test_injected_runner_is_used_as_is_without_calling_build_runner(tiny_tokenizer):
    """An already-built runner (e.g. build_openrouter_runner(...), or a
    fake for testing) is used directly - experiment_config.llm is never
    consulted to build it, so a hosted and a local module can share one
    config without either mutating it."""
    injected = _FakeRunner(label="injected")
    module = SubsymbolicModule(ExperimentConfig(), tiny_tokenizer, runner=injected)

    with patch("subsymbolic.subsymbolic_module.build_runner") as mock_build:
        assert module.runner is injected

    mock_build.assert_not_called()


def test_solve_uses_the_injected_runner(tiny_tokenizer, arc_task):
    from subsymbolic.prompt_builder import PromptingConfig

    pconf = PromptingConfig(blocks=["output_format"], token_limit=4096)
    experiment_config = ExperimentConfig(prompt=pconf)
    module = SubsymbolicModule(experiment_config, tiny_tokenizer, runner=_FakeRunner(label="hosted"))

    result = module.solve(arc_task)

    assert result["solution"].startswith("hosted:")


def test_close_tears_down_an_injected_runner_and_forgets_it(tiny_tokenizer):
    injected = _FakeRunner()
    module = SubsymbolicModule(ExperimentConfig(), tiny_tokenizer, runner=injected)

    module.close()

    assert injected.closed is True
    assert module._runner is None
