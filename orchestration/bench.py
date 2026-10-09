"""How fast a model answers, and what the answer's grammar costs.

A run of the system says where the time went by phase; when the model is the phase, the next question is
whether the time is the card or something around it. This measures one model on real prompts three ways:

    prefill     max_tokens=1, no grammar: the cost of reading the prompt
    free        a reply of up to `tokens` tokens, no grammar: reading and writing
    grammar     the same, held to the grid grammar the system uses
    flat, rows  the same, held to a simpler grammar (SIMPLE_GRAMMARS)

so that prefill speed, writing speed and what the grammar adds each have a number. A grammar is checked token by
token against the whole vocabulary, which is a lot of checking for a model with a vocabulary of 250 thousand;
the simpler grammars ask whether it is the grammar's shape or the checking itself that costs.

    python -m orchestration.bench --model REPO:FILE --tokenizer REPO --prompts data/experiments/runs/kaggle/prompts_gemma_0
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


#: Grammars that hold the reply to the characters of a grid and say less about its shape than the system's own
#: (subsymbolic.utils.build_grid_grammar): one loop over a character class, and the same per line.
SIMPLE_GRAMMARS = {
    "flat": "root ::= [0-9 ,:\\n]+\n",
    "rows": "root ::= row (\"\\n\" row)*\nrow ::= [0-9 ,:]+\n",
}


def timed(generate: Callable[[str], str], prompts: List[str], repeats: int = 1) -> List[Dict[str, Any]]:
    rows = []
    for prompt in prompts:
        for _ in range(repeats):
            started = time.perf_counter()
            reply = generate(prompt)
            rows.append({"seconds": time.perf_counter() - started, "prompt_chars": len(prompt),
                         "reply_chars": len(reply) if isinstance(reply, str) else 0})
    return rows


def with_limits(runner, max_tokens: int, grammar=False):
    """A copy of `runner` that generates at most `max_tokens`: without its grammar (False), with it (True), or
    with the grammar text given in its place."""
    from orchestration.tools import GRAMMAR_KEYS, without_grammar

    kept = runner if grammar else without_grammar(runner)
    copied = copy.copy(kept)
    kwargs = dict(kept.generation_kwargs)
    kwargs["max_tokens"] = max_tokens
    body = dict(kwargs.get("extra_body") or {})
    keys = [k for k in GRAMMAR_KEYS if k in body]
    if grammar and not keys:
        raise ValueError("this runner has no grammar to measure")
    if isinstance(grammar, str):
        body[keys[0]] = grammar
        kwargs["extra_body"] = body
    copied.generation_kwargs = kwargs
    return copied


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    seconds = [r["seconds"] for r in rows]
    return {"calls": len(rows), "mean_seconds": statistics.mean(seconds), "median_seconds": statistics.median(seconds),
            "mean_reply_chars": statistics.mean(r["reply_chars"] for r in rows),
            "mean_prompt_chars": statistics.mean(r["prompt_chars"] for r in rows)}


def run(runner, prompts: List[str], tokens: int = 160, repeats: int = 1) -> Dict[str, Dict[str, float]]:
    """prefill, free and grammar timings of `runner` on `prompts`, and one for each of SIMPLE_GRAMMARS."""
    result = {}
    cases = {"prefill": (1, False), "free": (tokens, False), "grammar": (tokens, True),
             **{name: (tokens, text) for name, text in SIMPLE_GRAMMARS.items()}}
    for name, (limit, grammar) in cases.items():
        result[name] = summarise(timed(with_limits(runner, limit, grammar).generate, prompts, repeats))
    return result


def render(model: str, result: Dict[str, Dict[str, float]], tokens: int) -> str:
    lines = [f"{model}", f"{'':<9}{'calls':>6}{'mean s':>9}{'reply chars':>13}"]
    for name, row in result.items():
        lines.append(f"{name:<9}{row['calls']:>6.0f}{row['mean_seconds']:>9.2f}{row['mean_reply_chars']:>13.0f}")
    pre, free, grammar = (result[k]["mean_seconds"] for k in ("prefill", "free", "grammar"))
    chars = max(result["free"]["mean_reply_chars"], 1.0)
    lines.append(f"writing the reply (free minus prefill): {max(free - pre, 0):.2f} s; "
                 f"the grammar adds {grammar - free:+.2f} s ({grammar / free:.1f}x the free reply, up to {tokens} tokens)")
    for name in SIMPLE_GRAMMARS:
        if name in result:
            lines.append(f"{name} grammar: {result[name]['mean_seconds'] - free:+.2f} s against free "
                         f"({result[name]['mean_seconds'] / free:.1f}x), {result[name]['mean_seconds'] - grammar:+.2f} s "
                         f"against the system's grammar")
    lines.append(f"reading a prompt of {result['prefill']['mean_prompt_chars']:.0f} chars: {pre:.2f} s; "
                 f"{1000 * max(free - pre, 0) / chars:.0f} ms per character written free")
    return "\n".join(lines)


def main():
    from orchestration.configs import local_experiment
    from subsymbolic.llm_setup import build_runner

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompts", required=True, help="a folder of <task>/p0.txt prompt files")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--out", default="bench.jsonl")
    args = parser.parse_args()
    files = sorted(Path(args.prompts).glob("*/p0.txt"))[:args.n]
    prompts = [f.read_text() for f in files]
    runner = build_runner(local_experiment(args.model, args.tokenizer, args.tokens))
    try:
        run(runner, prompts[:1], args.tokens)                 # the first call pays for loading and warming up
        result = run(runner, prompts, args.tokens)
    finally:
        runner.close()
    print(render(args.model, result, args.tokens), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as handle:
        handle.write(json.dumps({"model": args.model, "tokens": args.tokens, "result": result}) + "\n")


if __name__ == "__main__":
    main()
