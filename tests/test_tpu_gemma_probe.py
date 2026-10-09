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
