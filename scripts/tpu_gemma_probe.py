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
                                    "error": f"{type(error).__name__}: {error}"[:600]})
            print(f"[{name}] FAILED: {type(error).__name__}: {str(error)[:600]}\n{traceback.format_exc()[-1500:]}", flush=True)
            return None
    return wrap


def mesh_shape(chips, model_axis):
    """(batch, model) of the device mesh: the model axis as asked, or the largest divisor of `chips` below it
    when it does not divide them, and the chips left over on the batch axis. An axis the attention's heads do
    not divide fails the load (a 4-head key/value projection cut 8 ways: IndivisibleError)."""
    model = max(m for m in range(1, min(model_axis, chips) + 1) if chips % m == 0)
    return (chips // model, model)


def gemma_layout_map(mesh):
    """How to split a Gemma decoder over the `model` axis when the backbone has no `get_layout_map` of its own (Gemma 3
    has none): the attention projections by head, the feed-forward by its hidden width, the embedding by its width -
    the recipe keras-hub gives for Gemma 2, by the names of the weights."""
    import keras
    layout = keras.distribution.LayoutMap(mesh)
    layout["token_embedding/embeddings"] = (None, "model")
    layout["decoder_block.*attention.*(query|key|value)/kernel"] = ("model", None, None)
    layout["decoder_block.*attention_output/kernel"] = ("model", None, None)
    layout["decoder_block.*ffw_gating.*/kernel"] = ("batch", "model")
    layout["decoder_block.*ffw_linear/kernel"] = ("model", "batch")
    return layout


LAYOUTS = ("auto", "none", "replicate_embedding")


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
                             "(everything on one chip); replicate_embedding: auto with the embedding left whole")
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
            shape = mesh_shape(len(devs), args.model_axis)
            mesh = keras.distribution.DeviceMesh(shape, ["batch", "model"], devs)
            backbone = getattr(keras_hub.models, cls_name.replace("CausalLM", "Backbone"))
            own = hasattr(backbone, "get_layout_map")
            layout = backbone.get_layout_map(mesh) if own else gemma_layout_map(mesh)
            removed = without_embedding_split(layout) if args.layout == "replicate_embedding" else []
            report["layout"].update({"mesh": shape, "source": "backbone" if own else "ours", "removed": removed,
                                     "rules": {key: str(value) for key, value in layout.items()}})
            print("layout:", report["layout"], flush=True)
            keras.distribution.set_distribution(keras.distribution.ModelParallel(layout_map=layout,
                                                                                 batch_dim_name="batch"))
        source = attached_preset(preset) or preset
        print("loading from", source, flush=True)
        model = getattr(keras_hub.models, cls_name).from_preset(source)
        report["_model"] = model
        used = [(d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()]
        return {"preset": preset, "gb_in_use_per_chip": [round(u / 1e9, 2) for u in used]}

    if load is not None:
        model = report.pop("_model")

        @step("generate")
        def generate():
            model.preprocessor.sequence_length = 4096
            model.generate(prompt, max_length=64)                 # compile
            started = time.time()
            text = model.generate(prompt, max_length=4096 if False else len(prompt) // 3 + args.new_tokens)
            seconds = time.time() - started
            return {"seconds": round(seconds, 2), "new_tokens_asked": args.new_tokens, "reply_tail": str(text)[-200:]}

    report.pop("_model", None)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
