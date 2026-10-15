#!/usr/bin/env python3
"""Ask a model what it worked out by hand in its own reasoning, and could have been handed.

A reasoning model solving a puzzle copies the grid out row by row, numbers the columns, counts cells,
compares two rows, lists what changed between an example's input and output. Each of those is something a
program could print before the model starts. The reasoning shows which ones it did; this asks the model
itself to name them, as requests to a program, so that the names can be collected over many tasks into a
menu of what is worth giving (and then each entry measured: does the reasoning get shorter, does it finish).

The traces are the result files of orchestration.llm_only run with --thinking (the `reply` of each line is
the whole reasoning). Each is cut to its head and tail, put to the model with the question, and the reply is
read for lines of the form `NAME | what the program prints | where you did it by hand`, and for `MOST: NAME`.
The model is loaded the way the GPU notebooks load it, thinking off, no grammar.

    python scripts/elicit_trace_needs.py --model REPO:FILE --tokenizer REPO --traces runs/*.jsonl --out needs.jsonl

A trace already answered in `--out` is not asked again, so a run that stops is continued by running it again.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: (opening, closing) of the reasoning a model writes before its answer: Qwen's and Gemma 4's.
THINKING_MARKS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))

QUESTION = """Below is your own reasoning while you were solving a puzzle on grids of coloured cells (digits are colours). \
Part of it is cut out where it says [...].

You are not solving the puzzle now. Read how you worked, and find what you had to work out by hand: copying a grid \
out row by row, numbering rows or columns, counting cells, comparing two rows or columns position by position, \
listing which cells changed between an example's input and output, finding objects and where they are, \
checking your answer again. For each, say what a program could have printed for you before you started, so that \
you would not have needed to.

Answer with at most 8 lines, one per thing, each in the form

NAME | what the program prints, for one example pair, in a few words or one short sample | a few words you wrote when doing it by hand

Then one last line: MOST: NAME of the one that would have saved you the most reasoning.

--- YOUR REASONING ---
{reasoning}
--- END ---
"""


def reasoning_of(reply):
    """The reasoning inside a reply: without its opening mark, and up to its closing one when it has one."""
    for opening, closing in THINKING_MARKS:
        if opening in reply or closing in reply:
            return reply.split(opening, 1)[-1].split(closing, 1)[0].strip()
    return reply.strip()


def shorten(text, head, tail):
    """The first `head` and last `tail` characters, with a mark where the middle was; the text as it is when short."""
    if len(text) <= head + tail:
        return text
    return text[:head].rstrip() + "\n[...]\n" + text[-tail:].lstrip()


def parse(reply):
    """({name: {'prints', 'by_hand'}}, name of MOST or None) of the model's reply."""
    items, most = {}, None
    for line in reply.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        found = re.match(r"MOST\s*:\s*(.+)$", line, re.I)
        if found:
            most = found.group(1).strip(" *`.")
            continue
        parts = [part.strip(" *`") for part in line.split("|")]
        if len(parts) >= 2 and parts[0] and not set(parts[0]) <= set("-: "):
            items[parts[0]] = {"prints": parts[1], "by_hand": parts[2] if len(parts) > 2 else ""}
    return items, most


def read_traces(paths):
    """(source, record) of every line that has a reasoning, source being the file it came from."""
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("reply") and "task" in record:
                yield Path(path).stem, record


def answered(out):
    """{(source, task)} already in `out`."""
    if not Path(out).exists():
        return set()
    return {(r["source"], r["task"]) for r in map(json.loads, Path(out).read_text().splitlines()) if "source" in r}


def run(traces, ask, out, head=5000, tail=2500, model=None, limit=None):
    """Put every trace not yet answered to `ask(prompt) -> reply`, appending a line per trace to `out`."""
    done = answered(out)
    records = []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for source, trace in traces:
        if (source, trace["task"]) in done:
            continue
        if limit is not None and len(records) >= limit:
            break
        prompt = QUESTION.format(reasoning=shorten(reasoning_of(trace["reply"]), head, tail))
        started = time.perf_counter()
        reply = ask(prompt)
        items, most = parse(reply)
        record = {"source": source, "task": trace["task"], "model": model, "trace_solved": trace.get("solved"),
                  "trace_tokens": trace.get("reply_tokens"), "items": items, "most": most, "reply": reply,
                  "seconds": round(time.perf_counter() - started, 1)}
        with open(out, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(f"{source} {trace['task']}: {len(items)} items, most={most}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="GGUF repository and file: REPO:FILE")
    parser.add_argument("--tokenizer")
    parser.add_argument("--traces", nargs="+", required=True, help="result files of orchestration.llm_only --thinking")
    parser.add_argument("--out", default="needs.jsonl")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--head", type=int, default=5000, help="characters of the start of a reasoning kept")
    parser.add_argument("--tail", type=int, default=2500, help="characters of its end kept")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from subsymbolic.llm_setup import build_runner
    from subsymbolic.local_config import local_model_config

    runner = build_runner(local_model_config(args.model, args.tokenizer, args.max_tokens, grammar=False))
    try:
        run(read_traces(args.traces), runner.generate, args.out, args.head, args.tail, args.model, args.limit)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
