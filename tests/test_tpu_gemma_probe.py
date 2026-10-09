"""The TPU probe, run with a stand-in for keras_hub and jax: its steps are bound to their results, and a run that
finds nothing it can load still writes its report."""
import json
import sys
import types

import pytest

from scripts import tpu_gemma_probe as probe


@pytest.fixture(autouse=True)
def fresh_report(monkeypatch):
    monkeypatch.setattr(probe, "report", {"steps": []})


def fake_modules(monkeypatch, presets):
    class GemmaCausalLM:
        pass

    GemmaCausalLM.presets = presets
    hub = types.ModuleType("keras_hub")
    hub.models = types.SimpleNamespace(GemmaCausalLM=GemmaCausalLM)
    jax = types.ModuleType("jax")
    jax.devices = lambda: [types.SimpleNamespace(device_kind="fake", memory_stats=lambda: {"bytes_limit": 1})]
    monkeypatch.setitem(sys.modules, "keras_hub", hub)
    monkeypatch.setitem(sys.modules, "jax", jax)


def test_a_step_is_bound_to_its_result_and_a_failed_one_to_none():
    @probe.step("fine")
    def fine():
        return {"a": 1}

    @probe.step("broken")
    def broken():
        raise RuntimeError("no")

    assert fine == {"a": 1} and broken is None
    assert [(s["step"], s["ok"]) for s in probe.report["steps"]] == [("fine", True), ("broken", False)]


def test_a_run_that_finds_no_preset_that_fits_still_writes_its_report(monkeypatch, tmp_path):
    fake_modules(monkeypatch, {"gemma3_instruct_27b": object(), "gemma3_instruct_4b": object()})
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--out", str(out), "--max-gb", "1"])
    probe.main()
    written = json.loads(out.read_text())
    assert written["chosen"] is None
    assert ["GemmaCausalLM", "gemma3_instruct_27b"] in written["options"]
    assert all(step["ok"] for step in written["steps"])


def test_the_largest_preset_under_the_budget_is_chosen(monkeypatch, tmp_path):
    fake_modules(monkeypatch, {"gemma3_instruct_27b": object(), "gemma3_instruct_4b": object(),
                               "gemma3_instruct_12b": object()})
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--out", str(out), "--max-gb", "30"])
    probe.main()
    assert json.loads(out.read_text())["chosen"] == ["GemmaCausalLM", "gemma3_instruct_12b"]


def test_only_plain_instruction_tuned_gemma_presets_are_candidates_and_the_default_budget_holds_the_27b(
        monkeypatch, tmp_path):
    fake_modules(monkeypatch, {"translategemma_12b_it": object(), "medgemma_instruct_27b": object(),
                               "gemma3_instruct_27b_text": object(), "function_gemma_instruct_270m": object(),
                               "embedding_gemma3_300m": object(), "gemma3_12b": object(),
                               "gemma3_instruct_27b": object(), "gemma3_instruct_1b": object()})
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--out", str(out)])
    probe.main()
    written = json.loads(out.read_text())
    assert sorted(p for _, p in written["options"]) == ["gemma3_instruct_1b", "gemma3_instruct_27b"]
    assert written["chosen"] == ["GemmaCausalLM", "gemma3_instruct_27b"]


def test_the_layout_map_shards_the_attention_and_the_feed_forward_over_the_model_axis(monkeypatch):
    import types

    class LayoutMap(dict):
        def __init__(self, mesh):
            super().__init__()

    keras = types.ModuleType("keras")
    keras.distribution = types.SimpleNamespace(LayoutMap=LayoutMap)
    monkeypatch.setitem(sys.modules, "keras", keras)
    layout = probe.gemma_layout_map("mesh")
    assert layout["decoder_block.*attention.*(query|key|value)/kernel"][0] == "model"
    assert set(layout) >= {"token_embedding/embeddings", "decoder_block.*ffw_gating.*/kernel"}
