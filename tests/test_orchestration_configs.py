"""Tests for orchestration/configs.py."""
from __future__ import annotations

from orchestration.configs import ExperimentConfig
from subsymbolic.llm_runtime import GenerationConfig
from subsymbolic.llm_setup import LlmConfig
from subsymbolic.prompt_builder import PromptingConfig


def test_from_dict_applies_logging_overrides():
    """Regression test: from_dict used to build base/llm/generation/prompt/rl
    from the input dict but never read "logging", so any wandb project/group/
    plot override there was silently dropped in favour of WandbLogConfig's
    defaults - no error, just the wrong project name at runtime."""
    config = ExperimentConfig.from_dict({
        "logging": {"project": "ARC-1", "log_result_plot": True},
    })

    assert config.logging.project == "ARC-1"
    assert config.logging.log_result_plot is True


def test_from_dict_defaults_logging_when_absent():
    config = ExperimentConfig.from_dict({})

    assert config.logging.project == "llm-run"


def test_to_wandb_config_flattens_llm_and_generation_only():
    """Not base/prompt/rl/system/logging/project - those aren't "what was
    queried", they're run mechanics or unrelated subsystems."""
    config = ExperimentConfig(llm=LlmConfig(model="my-model"),
                               generation=GenerationConfig(temperature=0.7))

    wandb_config = config.to_wandb_config()

    assert wandb_config["model"] == "my-model"
    assert wandb_config["temperature"] == 0.7
    assert "blocks_dir" not in wandb_config  # that's PromptingConfig, not llm/generation
    assert "seed" not in wandb_config  # that's BaseConfig


def test_to_wandb_config_last_field_wins_on_name_collision():
    """llm and generation are dumped separately then merged - if a name
    ever collides between them, generation's value should be the one
    that survives, since it's merged in second."""
    config = ExperimentConfig()
    llm_dump = config.llm.model_dump()
    generation_dump = config.generation.model_dump()
    shared_keys = set(llm_dump) & set(generation_dump)

    wandb_config = config.to_wandb_config()

    for key in shared_keys:
        assert wandb_config[key] == generation_dump[key]


def test_to_chat_completions_forwards_grammar_backend_to_generation():
    """llm_setup passes grammar_backend explicitly (it just started either
    a llama.cpp or a vllm server, so it knows which) - this thin wrapper
    has to actually forward it, not just seed."""
    config = ExperimentConfig(generation=GenerationConfig(grammar='root ::= "a"'))

    params = config.to_chat_completions(grammar_backend="vllm")

    assert params["extra_body"]["guided_grammar"] == 'root ::= "a"'


def test_chat_template_kwargs_syncs_from_generation_to_prompt():
    config = ExperimentConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.prompt.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_syncs_from_prompt_to_generation():
    config = ExperimentConfig(prompt=PromptingConfig(chat_template_kwargs={"enable_thinking": False}))

    assert config.generation.chat_template_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_untouched_when_neither_side_set():
    config = ExperimentConfig()

    assert config.prompt.chat_template_kwargs == {}
    assert config.generation.chat_template_kwargs == {}


def test_chat_template_kwargs_left_alone_when_both_sides_set_independently():
    config = ExperimentConfig(
        generation=GenerationConfig(chat_template_kwargs={"a": 1}),
        prompt=PromptingConfig(chat_template_kwargs={"b": 2}),
    )

    assert config.generation.chat_template_kwargs == {"a": 1}
    assert config.prompt.chat_template_kwargs == {"b": 2}


def test_chat_template_kwargs_sync_does_not_alias_the_dict():
    config = ExperimentConfig(generation=GenerationConfig(chat_template_kwargs={"enable_thinking": False}))

    config.prompt.chat_template_kwargs["extra"] = True

    assert "extra" not in config.generation.chat_template_kwargs


def test_str_is_readable_nested_yaml_not_the_default_single_line_repr():
    config = ExperimentConfig()

    text = str(config)

    assert "base:" in text
    assert "generation:" in text
    assert "prompt:" in text
    assert "\n" in text  # multi-line, unlike pydantic's default repr


def test_str_reflects_field_overrides():
    config = ExperimentConfig()
    config.base.seed = 123

    assert "123" in str(config)


def test_repr_is_left_untouched():
    config = ExperimentConfig()

    assert repr(config).startswith("ExperimentConfig(")
