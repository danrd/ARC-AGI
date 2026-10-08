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
import time
import traceback
from pathlib import Path

report = {"steps": []}


def step(name):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="data/experiments/runs/kaggle/prompts_gemma_0/009d5c81/p0.txt")
    parser.add_argument("--out", default="tpu_probe.json")
    parser.add_argument("--max-gb", type=float, default=24.0, help="the largest model to load, in parameters x 2 bytes")
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    @step("devices")
    def devices():
        import jax
        return [{"kind": d.device_kind, "bytes_limit": (d.memory_stats() or {}).get("bytes_limit")} for d in jax.devices()]

    @step("presets")
    def presets():
        import keras_hub
        found = {}
        for name in sorted(dir(keras_hub.models)):
            cls = getattr(keras_hub.models, name)
            if "emma" in name and hasattr(cls, "presets") and name.endswith("CausalLM"):
                found[name] = sorted(cls.presets)
        return found

    found = presets() or {}
    options = []
    for cls_name, names in found.items():
        for preset in names:
            if "instruct" in preset or "_it" in preset or preset.endswith("it"):
                options.append((cls_name, preset))
    report["options"] = options
    print("instruction-tuned presets:", options, flush=True)

    def size_of(preset):
        import re
        match = re.search(r"(\d+(?:\.\d+)?)b", preset.lower())
        return float(match.group(1)) if match else None

    sized = sorted(((size_of(p), c, p) for c, p in options if size_of(p)), reverse=True)
    chosen = next(((c, p) for s, c, p in sized if s * 2 <= args.max_gb), None)
    report["chosen"] = chosen
    print("chosen:", chosen, flush=True)
    if chosen is None:
        Path(args.out).write_text(json.dumps(report, indent=1))
        return

    cls_name, preset = chosen
    prompt = Path(args.prompt).read_text() if Path(args.prompt).exists() else "Say hello."

    @step("shard and load")
    def load():
        import jax
        import keras
        import keras_hub
        keras.config.set_dtype_policy("bfloat16")
        devs = keras.distribution.list_devices()
        mesh = keras.distribution.DeviceMesh((1, len(devs)), ["batch", "model"], devs)
        backbone = getattr(keras_hub.models, cls_name.replace("CausalLM", "Backbone"))
        layout = backbone.get_layout_map(mesh)
        keras.distribution.set_distribution(keras.distribution.ModelParallel(layout_map=layout, batch_dim_name="batch"))
        model = getattr(keras_hub.models, cls_name).from_preset(preset)
        report["_model"] = model
        used = [(d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()]
        return {"preset": preset, "gb_in_use_per_chip": [round(u / 1e9, 2) for u in used]}

    if load() is not None:
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
