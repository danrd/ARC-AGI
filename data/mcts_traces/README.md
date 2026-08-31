# Solving traces harvested from the MCTS scan

`traces_ARC_0_262.json` is what the search found on the 262 shape-preserving
ARC-AGI-1 training tasks: for each task it solved, the action sequences that
reach the target, with the action index table they are written in.

How it was produced, and what each step of that costs to redo:

    # ~15 hours over five machines, one span each
    python scripts/compare_reward_approaches.py --approaches 2 --tasks 0-53 \
        --repeats 3 --rounds 1 --out shard_0.json
    # seconds
    python scripts/harvest_traces.py shard_*.json --out traces_ARC_0_262.json

What is in the file has been replayed in a fresh env, one sequence at a time,
and reaches the target there - not only inside the search that recorded it.
Every step that can be removed with the sequence still solving has been
removed, which took 735 recorded steps down to 374 and 219 recorded sequences
down to 69 distinct ones. Lengths run 1 to 5, median 2.

Twenty-four tasks of 260 scored. A twenty-fifth, `ea786f4a`, was solved inside
a playout the search never committed to, so it has no trace here.

The vocabulary is the one `build_actions(["red", "blue"], ["N", "E"])` makes:
two colours and two directions reach every branch of every transform, so the
names are not the full 2926-name space and a trace re-indexed against a
differently built vocabulary means something else entirely. Use
`action_names` from the file itself.
