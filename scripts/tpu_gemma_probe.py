#!/usr/bin/env python3
"""What a Kaggle TPU (v5e-8: eight chips of 16 GB) can hold and how fast it answers, for Gemma through Keras + JAX.

An earlier attempt to run an LLM on the TPU stopped at memory: one model on one chip has 16 GB, and a model of
twenty billion parameters is forty in bfloat16. The eight chips together have 128 GB, which is room if the model is
sharded across them. This tries it: it lists the Gemma presets keras-hub knows, loads the largest one that fits under
a budget with its weights split over the eight chips (model parallel), and times a short answer to a real prompt.
Every step reports what it did and what went wrong instead of stopping at the first error, so one run says how far
each idea gets.

    python scripts/tpu_gemma_probe.py --prompt PROMPT.txt --out result.json
"""
import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

# Keras picks its backend when it is first imported, and takes TensorFlow when nothing says otherwise: the TPU run that
# loaded Gemma 4 31B then failed at the first generation on a TensorFlow-backend call. This must come before any
# import of keras or keras_hub.
os.environ.setdefault("KERAS_BACKEND", "jax")

report = {"steps": []}


def step(name):
    """Runs the function it decorates at once; the name is then bound to its result (None if it failed)."""
    def wrap(fn):
        started = time.time()
        try:
            value = fn()
            report["steps"].append({"step": name, "ok": True, "seconds": round(time.time() - started, 1),
                                    "value": value if isinstance(value, (dict, list, str, int, float, bool, type(None))) else str(value)})
            print(f"[{name}] ok: {str(value)[:600]}", flush=True)
            return value
        except Exception as error:  # noqa: BLE001 - the point is to see how far each step gets
            report["steps"].append({"step": name, "ok": False, "seconds": round(time.time() - started, 1),
                                    "error": f"{type(error).__name__}: {error}"[:600],
                                    # A TPU session comes after a day in the queue: the report carries where
                                    # the error came from, not only what it said.
                                    "traceback": traceback.format_exc()[-2000:]})
            print(f"[{name}] FAILED: {type(error).__name__}: {str(error)[:600]}\n{traceback.format_exc()[-1500:]}", flush=True)
            return None
    return wrap


def mesh_shape(chips, model_axis):
    """(batch, model) of the device mesh: the model axis as asked, or the largest divisor of `chips` below it
    when it does not divide them, and the chips left over on the batch axis. An axis the attention's heads do
    not divide fails the load (a 4-head key/value projection cut 8 ways: IndivisibleError)."""
    model = max(m for m in range(1, min(model_axis, chips) + 1) if chips % m == 0)
    return (chips // model, model)


def sequence_length(prompt_tokens, new_tokens, step=128):
    """As long as the prompt and its answer need (and a few tokens of markers), up to a multiple of `step`, rather than
    a fixed 4096: the cache of keys and values is laid out for the whole length, and on the flat mesh every chip holds
    all of it."""
    return -(-(prompt_tokens + new_tokens + 8) // step) * step


def generation_batch(layout):
    """How many copies of the prompt to generate at once: the batch axis of the mesh, which shards the inputs and
    must divide them (one prompt on a (2, 4) mesh: IndivisibleError, "dimension size is 1"), or one without a mesh.
    The copies are the same prompt, so the speed per sequence is the one a single prompt would get, and the batch's
    total says what the spare chips add."""
    return (layout.get("mesh") or [1])[0]


def gemma_layout_map(mesh):
    """How to split a Gemma decoder over the `model` axis when the backbone has no `get_layout_map` of its own (Gemma 3
    and 4 have none): the attention projections by head, the feed-forward by its hidden width, the embedding by its
    width - the recipe keras-hub gives for Gemma 2, by the names of the weights - and a mixture-of-experts bank by
    expert."""
    import keras
    layout = keras.distribution.LayoutMap(mesh)
    layout["token_embedding/embeddings"] = (None, "model")
    layout["decoder_block.*attention.*(query|key|value)/kernel"] = ("model", None, None)
    layout["decoder_block.*attention_output/kernel"] = ("model", None, None)
    layout["decoder_block.*ffw_gating.*/kernel"] = ("batch", "model")
    layout["decoder_block.*ffw_linear/kernel"] = ("model", "batch")
    # Gemma 4 26B-A4B's 128 experts per layer are most of its weights: the experts over the model axis and their
    # width over the batch one, so the bank is cut over all eight chips.
    layout["decoder_block.*moe_expert_bank/(gate_proj|up_proj|down_proj)"] = ("model", "batch", None)
    return layout


def flat_layout_map(mesh):
    """Every large weight cut eight ways over the `model` axis of a (1, 8) mesh, always along a dimension eight
    divides in all three Gemma 4 models: the hidden width of the attention projections rather than their heads (2B has
    one key/value head, 31B four global ones, 26B-A4B two), the feed-forward's inner width, the embeddings' tokens and
    the experts of a mixture bank. tpu_gemma4i cut 31B four ways and copied it over the other axis - 15.6 of 15.75 GB
    a chip, so the embedding found no room ("RESOURCE_EXHAUSTED ... 2.62G", reported by Keras as "received 0
    variables") - and put all of 2B on one chip."""
    import keras
    layout = keras.distribution.LayoutMap(mesh)
    # The tables by token, not by width: the logits over 262144 tokens are then cut eight ways as well, where a table
    # cut by width gives every chip all of them.
    layout["(token_embedding|per_layer_token_embedding)/embeddings"] = ("model", None)
    layout["per_layer_model_projection/kernel"] = (None, "model")
    layout["decoder_block.*attention.*(query|key|value)/kernel"] = (None, "model", None)
    layout["decoder_block.*attention_output/kernel"] = (None, None, "model")
    layout["decoder_block.*ffw_gating.*/kernel"] = (None, "model")
    layout["decoder_block.*ffw_linear/kernel"] = ("model", None)
    layout["decoder_block.*moe_expert_bank/(gate_proj|up_proj|down_proj)"] = ("model", None, None)
    return layout


LAYOUTS = ("auto", "none", "replicate_embedding", "flat")


def without_embedding_split(layout):
    """`layout` with every rule that names an embedding taken out, so the embedding table is replicated on every chip
    rather than cut over the model axis; returns the keys it removed. Loaded split, Gemma 4 31B's table came back
    empty ("Layer 'token_embedding' expected 1 variables, but received 0") where every other layer loaded."""
    removed = [key for key in list(layout.keys()) if "embedding" in key]
    for key in removed:
        del layout[key]
    return removed


def drafter(cls_name):
    """Whether `cls_name` is an assistant class: the small drafter of speculative decoding, its own architecture
    with no Backbone in keras_hub.models. Its presets list the main models' names too, and it sorts first, so taking
    it loads Gemma 4 into the wrong network (tpu_gemma4f: every rung failed on it)."""
    return "Assistant" in cls_name


def preset_class(found, preset):
    """The CausalLM class whose presets list `preset`, or the Gemma 4 one when none does (a preset attached as a folder
    need not be in the installed keras-hub's list)."""
    return next((cls for cls, names in found.items() if preset in names and not drafter(cls)), "Gemma4CausalLM")


def attached_preset(preset, root="/kaggle/input"):
    """The folder of `preset` among the models attached to the notebook, or None.

    A model attached through the notebook's metadata is a folder under /kaggle/input, and keras-hub loads a preset
    from a folder as it does from a name. The name goes to the hub, which answers a notebook started through the API
    with 401 (it asks for a JWT only a session opened in the browser has)."""
    root = Path(root)
    if not root.is_dir():
        return None
    for config in sorted(root.rglob("config.json")):
        folder = config.parent
        if preset in str(folder).lower().replace("-", "_") and any(folder.glob("*.weights.*")):
            return str(folder)
    return None


def weight_entries(folder, needle="embed", limit=60):
    """What a preset folder holds about its weights, without loading a model: its metadata and config, every
    file with its size, and the datasets of its .h5 weight files whose path contains `needle`, with their shapes and
    dtypes - where to look when a layer "expected 1 variables, but received 0" (tpu_gemma4f/g: token_embedding of
    31B, per_layer_token_embedding of 2B, with and without a layout)."""
    import h5py

    folder = Path(folder)
    found = {"files": {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()}}
    for name in ("metadata.json", "config.json"):
        if (folder / name).exists():
            found[name] = json.loads((folder / name).read_text())
    datasets, total = [], 0
    for path in sorted(folder.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            def visit(key, item, file=path.name):
                nonlocal total
                if isinstance(item, h5py.Dataset):
                    total += 1
                    if needle in key.lower() and len(datasets) < limit:
                        datasets.append({"file": file, "path": key, "shape": list(item.shape), "dtype": str(item.dtype)})
            handle.visititems(visit)
    found["datasets"] = total
    found["matching"] = datasets
    return found


def to_snake_case(name):
    """Keras's own snake case (keras.src.utils.naming), which names a layer's group in a weights file."""
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()


def weight_groups(layers):
    """layer name -> its group in the weights file, for the layers of a functional model in order. Keras saves the
    list by class, not by name - the first ReversibleEmbedding is layers/reversible_embedding, a second one
    layers/reversible_embedding_1 - so token_embedding is found where the file has it whatever it is called."""
    used, groups = {}, {}
    for layer in layers:
        name = to_snake_case(type(layer).__name__)
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0
        groups[layer.name] = f"layers/{name}"
    return groups


def failed_layers(message):
    """The names of the layers a Keras loading error lists as not loaded ("List of objects that could not be
    loaded: [<ReversibleEmbedding name=token_embedding, built=True>]")."""
    listed = message.rsplit("could not be loaded:", 1)[-1]
    return re.findall(r"name=([\w.\-/]+)", listed)


def weight_file(folder, group):
    """The .h5 file of a preset folder that holds `group`'s variables: the one model.weights.json names for it when
    the weights are sharded, else model.weights.h5."""
    folder = Path(folder)
    index = folder / "model.weights.json"
    if not index.exists():
        return folder / "model.weights.h5"
    shard = json.loads(index.read_text())["weight_map"].get(f"/{group}/vars")
    if shard is None:
        raise KeyError(f"{group} is not in {index}")
    return folder / (shard[0] if isinstance(shard, list) else shard)


def bfloat16_from_raw(raw):
    """float32 values of an array the file holds as bfloat16. h5py has no bfloat16 and hands such a dataset back as
    opaque 2-byte records (|V2, what tpu_gemma4h saw for both embeddings); a bfloat16 is the upper half of a float32,
    so shifting its bits up by sixteen gives the same number. Anything else comes back as it is."""
    import numpy as np

    raw = np.asarray(raw)
    if raw.dtype.kind != "V" or raw.dtype.itemsize != 2:
        return raw
    return (np.ascontiguousarray(raw).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def host_value(raw, dtype):
    """What to hand a variable of `dtype` from a dataset's raw values: bfloat16 records kept as bfloat16 (a view, no
    copy) when the variable is bfloat16 and ml_dtypes is there, else float32 (bfloat16_from_raw). tpu_gemma4i's repair
    made float32 of 31B's embedding - 5.25 GB where the chip had 5.22 free."""
    import numpy as np

    raw = np.asarray(raw)
    if raw.dtype.kind == "V" and raw.dtype.itemsize == 2 and "bfloat16" in str(dtype):
        try:
            import ml_dtypes
            return np.ascontiguousarray(raw).view(np.uint16).view(ml_dtypes.bfloat16)
        except ImportError:
            pass
    return bfloat16_from_raw(raw)


def placed(value, variable):
    """`value` put on the devices the way `variable` is laid out, shard by shard, when it is laid out at all: assigned
    whole, a table is first copied to one chip and only then cut, which is the copy that does not fit."""
    sharding = getattr(getattr(variable, "value", None), "sharding", None)
    if sharding is None or not hasattr(sharding, "mesh"):
        return value
    import jax
    return jax.device_put(value, sharding)


def assign_from_file(layer, folder, group):
    """Reads `layer`'s variables from `group` of the preset's weights and assigns them, the way Keras would have;
    returns what it read and how far the layer's first rows are from the file's afterwards (0 when it took)."""
    import h5py
    import numpy as np

    path = weight_file(folder, group)
    read = {"file": path.name, "group": group, "variables": []}
    with h5py.File(path, "r") as handle:
        stored = handle[group]["vars"]
        if len(stored) != len(layer.weights):
            raise ValueError(f"{layer.name} has {len(layer.weights)} variables, {group} holds {len(stored)}")
        for index, variable in enumerate(layer.weights):
            dataset = stored[str(index)]
            value = host_value(dataset[()], getattr(variable, "dtype", None))
            variable.assign(placed(value, variable))
            rows = min(4, value.shape[0]) if value.ndim else 0
            back = np.asarray(variable[:rows] if rows else variable, dtype=np.float32)
            want = np.asarray(value[:rows] if rows else value, dtype=np.float32)
            read["variables"].append({"shape": list(dataset.shape), "stored_dtype": str(dataset.dtype),
                                      "max_difference": float(np.max(np.abs(back - want))) if back.size else 0.0})
    return read


def trace_loads(classes, log):
    """Wraps load_own_variables of each class that defines one so every call is logged: the layer, the group Keras
    looked in, the shard it had open, how many variables it found there and what went wrong. Keras keeps only the
    last error per layer, so a layer visited twice can hide the first failure behind "received 0 variables"."""
    for cls in classes:
        if cls is None or "load_own_variables" not in vars(cls):
            continue

        def traced(self, store, _original=vars(cls)["load_own_variables"]):
            entry = {"layer": self.name, "group": getattr(store, "_h5_entry_path", None),
                     "shard": getattr(getattr(store, "current_shard_path", None), "name", None)}
            try:
                entry["variables_found"] = len(store)
            except Exception as error:  # noqa: BLE001 - the log is for seeing what the store does
                entry["variables_found"] = f"{type(error).__name__}: {error}"
            try:
                _original(self, store)
                entry["ok"] = True
            except Exception as error:
                entry.update(ok=False, error=f"{type(error).__name__}: {error}"[:600],
                             traceback=traceback.format_exc()[-1200:])
                raise
            finally:
                log.append(entry)

        cls.load_own_variables = traced


def tree(root="/kaggle/input", depth=4):
    """The folders under `root` down to `depth`, for the report: where an attached model landed."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= depth)[:80]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="data/experiments/runs/kaggle/prompts_gemma_0/009d5c81/p0.txt")
    parser.add_argument("--out", default="tpu_probe.json")
    parser.add_argument("--max-gb", type=float, default=70.0, help="the largest model to load, in parameters x 2 bytes")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--model-axis", type=int, default=4, help="chips the weights are cut over; the rest hold copies "
                        "(Gemma 4 31B has 4 key/value heads, which 8 cannot divide)")
    parser.add_argument("--preset", help="this preset instead of the largest that fits --max-gb")
    parser.add_argument("--inspect", action="store_true",
                        help="read the attached preset's metadata, config and embedding weights and stop: no model is "
                             "loaded")
    parser.add_argument("--layout", choices=LAYOUTS, default="auto",
                        help="auto: the backbone's own layout map, or ours when it has none; none: no distribution "
                             "(everything on one chip); replicate_embedding: auto with the embedding left whole; "
                             "flat: every large weight cut eight ways on a (1, 8) mesh (flat_layout_map)")
    parser.add_argument("--repair", action="store_true",
                        help="load the weights without keras-hub's loader, and assign the layers Keras could not load "
                             "straight from the file (the embedding every Gemma 4 load so far has lost)")
    args = parser.parse_args()

    @step("devices")
    def devices():
        import jax
        return [{"kind": d.device_kind, "bytes_limit": (d.memory_stats() or {}).get("bytes_limit")} for d in jax.devices()]

    @step("inputs")
    def inputs():
        return tree()

    @step("presets")
    def presets():
        import keras_hub
        found = {}
        for name in sorted(dir(keras_hub.models)):
            cls = getattr(keras_hub.models, name)
            if "emma" in name and hasattr(cls, "presets") and name.endswith("CausalLM") and not drafter(name):
                found[name] = sorted(cls.presets)
        return found

    found = presets or {}
    options = []
    for cls_name, names in found.items():
        for preset in names:
            if re.fullmatch(r"gemma\d*_instruct_\d+(\.\d+)?b", preset):     # not the translate, med, function, MoE or text-only presets
                options.append((cls_name, preset))
    report["options"] = options
    print("instruction-tuned presets:", options, flush=True)

    def size_of(preset):
        match = re.search(r"(\d+(?:\.\d+)?)b", preset.lower())
        return float(match.group(1)) if match else None

    # the newest family first at equal size: Gemma 4 over Gemma 3
    sized = sorted(((size_of(p), p, c) for c, p in options if size_of(p)), reverse=True)
    sized = [(s, c, p) for s, p, c in sized]
    chosen = next(((c, p) for s, c, p in sized if s * 2 <= args.max_gb), None)
    if args.preset:
        chosen = (preset_class(found, args.preset), args.preset)
    report["chosen"] = chosen
    print("chosen:", chosen, flush=True)
    if chosen is None:
        Path(args.out).write_text(json.dumps(report, indent=1))
        return

    cls_name, preset = chosen
    if args.inspect:
        @step("versions")
        def versions():
            import importlib.metadata as meta
            return {name: meta.version(name) for name in ("keras", "keras-hub", "jax", "h5py")}

        @step("inspect weights")
        def inspected():
            folder = attached_preset(preset)
            if folder is None:
                raise FileNotFoundError(f"{preset} is not among the models under /kaggle/input")
            return {"folder": folder, **weight_entries(folder)}

        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        return
    prompt = Path(args.prompt).read_text() if Path(args.prompt).exists() else "Say hello."

    @step("shard and load")
    def load():
        import jax
        import keras
        import keras_hub
        keras.config.set_dtype_policy("bfloat16")
        devs = keras.distribution.list_devices()
        report["layout"] = {"mode": args.layout}
        if args.layout != "none":
            shape = (1, len(devs)) if args.layout == "flat" else mesh_shape(len(devs), args.model_axis)
            mesh = keras.distribution.DeviceMesh(shape, ["batch", "model"], devs)
            backbone = getattr(keras_hub.models, cls_name.replace("CausalLM", "Backbone"))
            own = hasattr(backbone, "get_layout_map") and args.layout != "flat"
            layout = backbone.get_layout_map(mesh) if own else \
                (flat_layout_map(mesh) if args.layout == "flat" else gemma_layout_map(mesh))
            removed = without_embedding_split(layout) if args.layout == "replicate_embedding" else []
            report["layout"].update({"mesh": shape, "source": "backbone" if own else "ours", "removed": removed,
                                     "rules": {key: str(value) for key, value in layout.items()}})
            print("layout:", report["layout"], flush=True)
            keras.distribution.set_distribution(keras.distribution.ModelParallel(layout_map=layout,
                                                                                 batch_dim_name="batch"))
        source = attached_preset(preset) or preset
        print("loading from", source, flush=True)
        report["embedding_loads"] = []
        trace_loads({keras.layers.Embedding, getattr(keras.layers, "ReversibleEmbedding", None),
                     getattr(keras_hub.layers, "ReversibleEmbedding", None)}, report["embedding_loads"])
        if not args.repair:
            model = getattr(keras_hub.models, cls_name).from_preset(source)
        else:
            model = getattr(keras_hub.models, cls_name).from_preset(source, load_weights=False)
            weights = Path(source) / "model.weights.json"
            if not weights.exists():
                weights = Path(source) / "model.weights.h5"
            try:
                model.backbone.load_weights(str(weights))
                report["load_error"] = None
            except ValueError as error:
                # Keras raises only after every other layer is loaded; what it lists is all that is missing.
                report["load_error"] = str(error)[:1500]
                missing = failed_layers(str(error))
                if not missing:
                    raise
                groups = weight_groups(model.backbone.layers)
                report["repaired"] = {name: assign_from_file(model.backbone.get_layer(name), source, groups[name])
                                      for name in missing}
                print("repaired:", report["repaired"], flush=True)
        report["_model"] = model
        used = [(d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()]
        return {"preset": preset, "gb_in_use_per_chip": [round(u / 1e9, 2) for u in used]}

    if load is not None:
        model = report.pop("_model")

        @step("generate")
        def generate():
            tokenizer = model.preprocessor.tokenizer
            prompt_tokens = len(tokenizer.tokenize(prompt))
            model.preprocessor.sequence_length = sequence_length(prompt_tokens, args.new_tokens)
            batch = generation_batch(report["layout"])
            prompts = [prompt] * batch
            model.generate(prompts, max_length=64)                 # compile
            # generate compiles again for a new max_length, so the first call at this length is timed apart from the
            # second, which is the speed.
            started = time.time()
            model.generate(prompts, max_length=prompt_tokens + args.new_tokens)
            first = time.time() - started
            started = time.time()
            texts = model.generate(prompts, max_length=prompt_tokens + args.new_tokens)
            seconds = time.time() - started
            text = str(texts[0]) if isinstance(texts, (list, tuple)) else str(texts)
            new_tokens = max(0, len(tokenizer.tokenize(text)) - prompt_tokens)
            speed = round(new_tokens / seconds, 2) if seconds else None
            return {"seconds": round(seconds, 2), "first_call_seconds": round(first, 2), "batch": batch,
                    "sequence_length": model.preprocessor.sequence_length,
                    "prompt_tokens": prompt_tokens, "new_tokens_asked": args.new_tokens, "new_tokens": new_tokens,
                    "tokens_per_second": speed, "batch_tokens_per_second": speed and round(speed * batch, 2),
                    "reply_tail": text[-400:]}

    report.pop("_model", None)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
