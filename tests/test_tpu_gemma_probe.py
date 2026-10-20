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
    import re
    experts = [key for key in layout if re.search(key, "decoder_block_3/moe_expert_bank/down_proj")]
    assert [layout[key] for key in experts] == [("model", "batch", None)]
    assert not any(re.search(key, "decoder_block_3/moe_expert_bank/per_expert_scale") for key in layout)


def test_the_largest_instruct_preset_is_the_gemma_4_31b_and_the_default_budget_holds_it(monkeypatch, tmp_path):
    fake_modules(monkeypatch, {"gemma3_instruct_27b": object(), "gemma4_instruct_31b": object(),
                               "gemma4_instruct_26b_a4b": object(), "gemma4_instruct_31b_assistant": object()})
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--out", str(out)])
    probe.main()
    written = json.loads(out.read_text())
    assert written["chosen"] == ["GemmaCausalLM", "gemma4_instruct_31b"]
    assert sorted(p for _, p in written["options"]) == ["gemma3_instruct_27b", "gemma4_instruct_31b"]


def test_an_attached_model_is_found_by_its_folder_and_a_name_alone_finds_nothing(tmp_path):
    folder = tmp_path / "gemma4" / "keras" / "gemma4_instruct_31b" / "2"
    folder.mkdir(parents=True)
    other = tmp_path / "gemma4" / "keras" / "gemma4_instruct_12b" / "1"
    other.mkdir(parents=True)
    for place in (folder, other):
        (place / "config.json").write_text("{}")
        (place / "model.weights.h5").write_text("")
    assert probe.attached_preset("gemma4_instruct_31b", tmp_path) == str(folder)
    assert probe.attached_preset("gemma4_instruct_27b", tmp_path) is None
    assert probe.attached_preset("gemma4_instruct_31b", tmp_path / "nowhere") is None


def test_a_folder_without_weights_is_not_a_model(tmp_path):
    folder = tmp_path / "gemma4_instruct_31b"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    assert probe.attached_preset("gemma4_instruct_31b", tmp_path) is None


def test_the_tree_lists_folders_to_a_depth(tmp_path):
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    assert probe.tree(tmp_path, depth=2) == [str(tmp_path / "a"), str(tmp_path / "a" / "b")]


def test_keras_is_told_to_use_jax_before_anything_imports_it():
    import subprocess
    import sys
    code = ("import os; os.environ.pop('KERAS_BACKEND', None); import sys; sys.path.insert(0, 'scripts'); "
            "import tpu_gemma_probe; print(os.environ['KERAS_BACKEND'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(probe.__file__).rsplit("/scripts/", 1)[0])
    assert out.stdout.strip() == "jax", out.stderr[-300:]
    code = code.replace("os.environ.pop('KERAS_BACKEND', None)", "os.environ['KERAS_BACKEND'] = 'tensorflow'")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(probe.__file__).rsplit("/scripts/", 1)[0])
    assert out.stdout.strip() == "tensorflow", "a backend chosen by the caller is left alone"


def test_the_mesh_cuts_the_weights_over_the_model_axis_and_leaves_the_rest_to_the_batch():
    assert probe.mesh_shape(8, 4) == (2, 4)
    assert probe.mesh_shape(8, 8) == (1, 8)
    assert probe.mesh_shape(8, 3) == (4, 2)          # 3 does not divide 8: the largest divisor below it
    assert probe.mesh_shape(1, 4) == (1, 1)


def test_replicating_the_embedding_takes_out_every_rule_that_names_one_and_nothing_else():
    layout = {"token_embedding/embeddings": (None, "model"), "decoder_block.*ffw_linear/kernel": ("model", "batch"),
              "vision.*patch_embedding/kernel": ("model",)}
    removed = probe.without_embedding_split(layout)
    assert sorted(removed) == ["token_embedding/embeddings", "vision.*patch_embedding/kernel"]
    assert list(layout) == ["decoder_block.*ffw_linear/kernel"]


def test_a_preset_named_on_the_command_line_finds_its_class_or_falls_back_to_gemma_4():
    found = {"Gemma3CausalLM": ["gemma3_instruct_12b"], "Gemma4CausalLM": ["gemma4_instruct_2b", "gemma4_instruct_31b"]}
    assert probe.preset_class(found, "gemma3_instruct_12b") == "Gemma3CausalLM"
    assert probe.preset_class(found, "gemma4_instruct_2b") == "Gemma4CausalLM"
    assert probe.preset_class({}, "gemma4_instruct_2b") == "Gemma4CausalLM"


def test_the_assistant_drafter_class_is_never_taken_for_the_model():
    found = {"Gemma4AssistantCausalLM": ["gemma4_instruct_2b", "gemma4_instruct_31b"],
             "Gemma4CausalLM": ["gemma4_instruct_2b", "gemma4_instruct_31b"]}
    assert probe.preset_class(found, "gemma4_instruct_31b") == "Gemma4CausalLM"
    assert probe.preset_class({"Gemma4AssistantCausalLM": ["gemma4_instruct_2b"]}, "gemma4_instruct_2b") == "Gemma4CausalLM"
    assert probe.drafter("Gemma4AssistantCausalLM") and not probe.drafter("Gemma4CausalLM")


def test_a_preset_folder_is_read_without_a_model(tmp_path):
    import json

    import numpy as np
    import pytest
    h5py = pytest.importorskip("h5py")
    (tmp_path / "metadata.json").write_text(json.dumps({"keras_hub_version": "0.40.0"}))
    (tmp_path / "config.json").write_text(json.dumps({"class_name": "Gemma4Backbone"}))
    with h5py.File(tmp_path / "model.weights.h5", "w") as handle:
        handle.create_dataset("layers/token_embedding/vars/0", data=np.zeros((8, 4), dtype="float16"))
        handle.create_dataset("layers/decoder_block_0/attention/vars/0", data=np.zeros((4, 4)))
    found = probe.weight_entries(tmp_path)
    assert found["metadata.json"] == {"keras_hub_version": "0.40.0"} and found["config.json"]["class_name"] == "Gemma4Backbone"
    assert found["datasets"] == 2 and "model.weights.h5" in found["files"]
    assert found["matching"] == [{"file": "model.weights.h5", "path": "layers/token_embedding/vars/0",
                                  "shape": [8, 4], "dtype": "float16"}]


def test_a_failed_step_keeps_where_the_error_came_from():
    probe.report["steps"].clear()

    @probe.step("boom")
    def boom():
        raise ValueError("no embeddings")

    entry = probe.report["steps"][-1]
    assert boom is None and entry["ok"] is False and entry["error"] == "ValueError: no embeddings"
    assert "raise ValueError" in entry["traceback"]
    probe.report["steps"].clear()


# Repairing a load that lost its embedding: tpu_gemma4h failed on every rung with token_embedding (31B) or
# per_layer_token_embedding (2B) "expected 1 variables, but received 0", though the file holds both.

ERROR_31B = ("A total of 1 objects could not be loaded. Example error message for object <ReversibleEmbedding "
             "name=token_embedding, built=True>:\n\nLayer 'token_embedding' expected 1 variables, but received 0 "
             "variables during loading. Expected: ['embeddings']\n\nList of objects that could not be loaded:\n"
             "[<ReversibleEmbedding name=token_embedding, built=True>]")


def test_the_layers_that_did_not_load_are_read_from_the_list_and_not_from_the_example():
    two = ERROR_31B.replace("[<ReversibleEmbedding name=token_embedding, built=True>]",
                            "[<ReversibleEmbedding name=token_embedding, built=True>, "
                            "<Embedding name=per_layer_token_embedding, built=True>]")
    assert probe.failed_layers(ERROR_31B) == ["token_embedding"]
    assert probe.failed_layers(two) == ["token_embedding", "per_layer_token_embedding"]
    assert probe.failed_layers("RESOURCE_EXHAUSTED: out of memory") == []


def test_a_layer_is_found_in_the_file_by_its_class_and_its_place_among_the_layers_of_that_class():
    def layer(cls_name, name):
        return type(cls_name, (), {"name": name})()

    layers = [layer("InputLayer", "token_ids"), layer("ReversibleEmbedding", "token_embedding"),
              layer("Dense", "per_layer_model_projection"), layer("Embedding", "per_layer_token_embedding"),
              layer("RMSNormalization", "per_layer_projection_norm"), layer("InputLayer", "padding_mask"),
              layer("Gemma4TextDecoderBlock", "decoder_block_0"), layer("Gemma4TextDecoderBlock", "decoder_block_1"),
              layer("RMSNormalization", "final_normalization")]
    assert probe.weight_groups(layers) == {
        "token_ids": "layers/input_layer", "token_embedding": "layers/reversible_embedding",
        "per_layer_model_projection": "layers/dense", "per_layer_token_embedding": "layers/embedding",
        "per_layer_projection_norm": "layers/rms_normalization", "padding_mask": "layers/input_layer_1",
        "decoder_block_0": "layers/gemma4_text_decoder_block", "decoder_block_1": "layers/gemma4_text_decoder_block_1",
        "final_normalization": "layers/rms_normalization_1"}


def test_bfloat16_records_come_back_as_the_numbers_they_hold():
    import numpy as np
    numbers = np.array([[1.0, -2.5], [0.15625, 3.0e38]], dtype=np.float32)
    upper = (numbers.view(np.uint32) >> 16).astype(np.uint16)     # what bfloat16 keeps of each
    records = upper.view("V2")
    back = probe.bfloat16_from_raw(records)
    assert back.dtype == np.float32 and back.shape == (2, 2)
    assert np.array_equal(back.view(np.uint32), upper.astype(np.uint32) << 16)
    assert back[0, 0] == 1.0 and back[0, 1] == -2.5 and back[1, 0] == 0.15625
    plain = np.ones((2, 3), dtype=np.float16)
    assert probe.bfloat16_from_raw(plain) is not None and probe.bfloat16_from_raw(plain).dtype == np.float16


def test_the_shard_of_a_group_comes_from_the_weight_map(tmp_path):
    (tmp_path / "model.weights.json").write_text(json.dumps({"weight_map": {
        "/layers/reversible_embedding/vars": "model_00000.weights.h5",
        "/layers/embedding/vars": ["model_00001.weights.h5"]}}))
    assert probe.weight_file(tmp_path, "layers/reversible_embedding") == tmp_path / "model_00000.weights.h5"
    assert probe.weight_file(tmp_path, "layers/embedding") == tmp_path / "model_00001.weights.h5"
    with pytest.raises(KeyError):
        probe.weight_file(tmp_path, "layers/dense")
    single = tmp_path / "one"
    single.mkdir()
    assert probe.weight_file(single, "layers/embedding") == single / "model.weights.h5"


class FakeVariable:
    def __init__(self, shape):
        import numpy as np
        self.value = np.zeros(shape, dtype=np.float32)

    def assign(self, value):
        import numpy as np
        assert np.shape(value) == self.value.shape
        self.value = np.asarray(value, dtype=np.float32)

    def __getitem__(self, index):
        return self.value[index]

    def __array__(self, dtype=None, copy=None):
        return self.value if dtype is None else self.value.astype(dtype)


def test_a_lost_embedding_is_assigned_from_its_shard(tmp_path):
    import numpy as np
    h5py = pytest.importorskip("h5py")
    table = np.arange(24, dtype=np.float32).reshape(6, 4) - 3.5         # exact in bfloat16
    with h5py.File(tmp_path / "model_00001.weights.h5", "w") as handle:
        handle.create_dataset("layers/reversible_embedding/vars/0",
                              data=(table.view(np.uint32) >> 16).astype(np.uint16).view("V2"))
    (tmp_path / "model.weights.json").write_text(json.dumps({"weight_map": {
        "/layers/reversible_embedding/vars": "model_00001.weights.h5"}}))
    layer = types.SimpleNamespace(name="token_embedding", weights=[FakeVariable((6, 4))])
    read = probe.assign_from_file(layer, tmp_path, "layers/reversible_embedding")
    assert np.array_equal(layer.weights[0].value, table)
    assert read == {"file": "model_00001.weights.h5", "group": "layers/reversible_embedding",
                    "variables": [{"shape": [6, 4], "stored_dtype": "|V2", "max_difference": 0.0}]}
    two = types.SimpleNamespace(name="token_embedding", weights=[FakeVariable((6, 4)), FakeVariable((6, 4))])
    with pytest.raises(ValueError, match="has 2 variables"):
        probe.assign_from_file(two, tmp_path, "layers/reversible_embedding")


def test_every_embedding_load_is_logged_with_what_the_store_held_and_what_went_wrong():
    class Store(dict):
        _h5_entry_path = "layers/reversible_embedding"
        current_shard_path = types.SimpleNamespace(name="model_00000.weights.h5")

    class Embedding:
        def __init__(self, name):
            self.name = name

        def load_own_variables(self, store):
            if len(store) != 1:
                raise ValueError(f"Layer '{self.name}' expected 1 variables, but received {len(store)}")

    class ReversibleEmbedding(Embedding):
        pass

    log = []
    probe.trace_loads({Embedding, ReversibleEmbedding, None}, log)
    ReversibleEmbedding("token_embedding").load_own_variables(Store({"0": 1}))
    with pytest.raises(ValueError):
        Embedding("per_layer_token_embedding").load_own_variables(Store())
    assert [(e["layer"], e["group"], e["shard"], e["variables_found"], e["ok"]) for e in log] == [
        ("token_embedding", "layers/reversible_embedding", "model_00000.weights.h5", 1, True),
        ("per_layer_token_embedding", "layers/reversible_embedding", "model_00000.weights.h5", 0, False)]
    assert "received 0" in log[1]["error"] and "raise ValueError" in log[1]["traceback"]


def test_a_prompt_is_generated_once_per_row_of_the_mesh_and_once_without_one():
    assert probe.generation_batch({"mode": "auto", "mesh": (2, 4)}) == 2
    assert probe.generation_batch({"mode": "auto", "mesh": [4, 2]}) == 4
    assert probe.generation_batch({"mode": "none"}) == 1
