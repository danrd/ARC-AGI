#!/usr/bin/env python3
"""Which cell features make the coordinate tasks solvable at all - measured without RL.

Every coordinate run (coord_b to coord_f) fitted some training pairs and solved no test: the policy paints where the
answer was on the training grids, not where it is on the test one. Its cells carry their colour, their absolute
position and a convolutional map that RL learns from a sparse reward. This asks the question underneath, with no
policy and no search: from the training pairs alone, can a small decision tree over a cell's features say which cells
change and to what, so that the test output comes out right? A feature set on which a tree cannot do it is one a
policy cannot learn the rule from either; one on which it can is worth giving the pointer instead of the position.

Each cell of each training input is an example. The tree first says whether the cell changes; then what it becomes,
either as a colour or as a role - "the colour of the nearest cell to the left", "the colour of the cell's object" -
since 11 of the 23 tasks need a colour on the test that no training output has. The tree grows until its leaves
are pure, to a cap on depth, so a simple rule stays a shallow tree.

More features also give a greedy tree more ways to fit the training pairs by accident (the rich set wins three tasks
over the structure set and loses three others), so a task also picks its set the way it could without its test:
by how its trees do on each training pair left out in turn ("chosen").

    python scripts/feature_probe.py                       # the 23 coordinate tasks, every feature set
    python scripts/feature_probe.py --tasks 3eda0437 --out probe.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "datasets" / "ARC"
COORDINATE_TASKS = ("0ca9ddb6", "11852cab", "1bfc4729", "29623171", "29c11459", "2bcee788", "363442ee", "39e1d7f9",
                    "3eda0437", "4612dd53", "496994bd", "72322fa7", "73251a56", "7b6016b9", "7df24a62", "82819916",
                    "97999447", "a65b410d", "a699fb00", "af902bf9", "ba26e723", "e8593010", "f15e1fac")
OUTSIDE = -1                     # the colour of a neighbour off the grid
DIRECTIONS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0),
              "up_left": (-1, -1), "up_right": (-1, 1), "down_left": (1, -1), "down_right": (1, 1)}
FEATURE_SETS = ("position", "neighbours", "structure", "structure_and_position", "rich", "rich_and_position")


def background(grid):
    """The most common colour of a grid, 0 on a tie with it."""
    counts = np.bincount(grid.ravel(), minlength=10)
    return 0 if counts[0] == counts.max() else int(counts.argmax())


def neighbour_colours(grid):
    """name -> (rows, cols) array: the colour of the cell one step away in each of the eight directions."""
    padded = np.pad(grid, 1, constant_values=OUTSIDE)
    rows, cols = grid.shape
    return {f"next_{name}": padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
            for name, (dr, dc) in DIRECTIONS.items()}


def nearest_along(grid, bg):
    """For each of the eight directions, the colour of the first cell that is not background, walking from each
    cell (OUTSIDE when there is none), and how many steps away it is (0 when there is none)."""
    rows, cols = grid.shape
    found = {}
    for name, (dr, dc) in DIRECTIONS.items():
        colour = np.full(grid.shape, OUTSIDE)
        distance = np.zeros(grid.shape, dtype=int)
        for r in range(rows):
            for c in range(cols):
                rr, cc, steps = r + dr, c + dc, 1
                while 0 <= rr < rows and 0 <= cc < cols:
                    if grid[rr, cc] != bg:
                        colour[r, c], distance[r, c] = grid[rr, cc], steps
                        break
                    rr, cc, steps = rr + dr, cc + dc, steps + 1
        found[f"nearest_{name}"] = colour
        found[f"distance_{name}"] = distance
    return found


def object_features(grid, bg):
    """Each cell's object - the 4-connected cells of its colour, background included as objects of its own - by its
    size, the height and width of its box, where the cell sits in the box and whether it is on the box's border."""
    out = {key: np.zeros(grid.shape, dtype=int) for key in
           ("object_size", "box_height", "box_width", "from_top", "from_bottom", "from_left", "from_right",
            "on_box_border", "object_touches_edge")}
    for colour in np.unique(grid):
        labels, count = ndimage.label(grid == colour)
        for index, box in enumerate(ndimage.find_objects(labels), start=1):
            cells = labels == index
            top, bottom, left, right = box[0].start, box[0].stop - 1, box[1].start, box[1].stop - 1
            rr, cc = np.nonzero(cells)
            out["object_size"][cells] = cells.sum()
            out["box_height"][cells] = bottom - top + 1
            out["box_width"][cells] = right - left + 1
            out["from_top"][rr, cc] = rr - top
            out["from_bottom"][rr, cc] = bottom - rr
            out["from_left"][rr, cc] = cc - left
            out["from_right"][rr, cc] = right - cc
            out["on_box_border"][rr, cc] = (rr == top) | (rr == bottom) | (cc == left) | (cc == right)
            out["object_touches_edge"][cells] = int(top == 0 or left == 0 or bottom == grid.shape[0] - 1
                                                    or right == grid.shape[1] - 1)
    return out


def cell_features(grid, feature_set):
    """name -> (rows, cols) array of one feature per cell, and the names of those that are colours (split by
    equality, and candidates for the role a new colour is taken from)."""
    grid = np.asarray(grid)
    bg = background(grid)
    rows, cols = np.indices(grid.shape)
    features = {"colour": grid, "is_background": (grid == bg).astype(int)}
    colours = {"colour"}
    if feature_set.endswith("position"):
        features.update(row=rows, col=cols)
    if feature_set != "position":
        near = neighbour_colours(grid)
        features.update(near)
        colours |= set(near)
    if feature_set.startswith(("structure", "rich")):
        along = nearest_along(grid, bg)
        features.update(along)
        colours |= {name for name in along if name.startswith("nearest_")}
        features.update(object_features(grid, bg))
        for a, b in (("left", "right"), ("up", "down"), ("up_left", "down_right"), ("up_right", "down_left")):
            same = (along[f"nearest_{a}"] == along[f"nearest_{b}"]) & (along[f"nearest_{a}"] != OUTSIDE)
            features[f"between_{a}_{b}"] = same.astype(int)
        for colour in range(10):
            features[f"row_has_{colour}"] = np.broadcast_to((grid == colour).any(axis=1, keepdims=True), grid.shape)
            features[f"col_has_{colour}"] = np.broadcast_to((grid == colour).any(axis=0, keepdims=True), grid.shape)
        features["row_foreground"] = np.broadcast_to((grid != bg).sum(axis=1, keepdims=True), grid.shape)
        features["col_foreground"] = np.broadcast_to((grid != bg).sum(axis=0, keepdims=True), grid.shape)
    if feature_set.startswith("rich"):
        extra, extra_colours = rich_features(grid, bg, features)
        features.update(extra)
        colours |= extra_colours
    return {name: np.asarray(value).astype(int) for name, value in features.items()}, colours


def blob_features(grid, bg):
    """Each cell's blob - the 8-connected cells that are not background, of any colour - by its size, how many
    colours it has, its commonest and its rarest colour (OUTSIDE for background cells): where "recolour the shape in
    its other colour" reads its colour from."""
    out = {"blob_size": np.zeros(grid.shape, dtype=int), "blob_colours": np.zeros(grid.shape, dtype=int),
           "blob_majority": np.full(grid.shape, OUTSIDE), "blob_minority": np.full(grid.shape, OUTSIDE)}
    labels, _ = ndimage.label(grid != bg, structure=np.ones((3, 3)))
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        cells = labels == index
        counts = np.bincount(grid[cells], minlength=10)
        present = np.nonzero(counts)[0]
        out["blob_size"][cells] = cells.sum()
        out["blob_colours"][cells] = len(present)
        out["blob_majority"][cells] = present[counts[present].argmax()]
        out["blob_minority"][cells] = present[counts[present].argmin()]
    return out


def rich_features(grid, bg, structure):
    """What the structure set lacked on the tasks it missed: how many neighbours of each colour a cell has (a rule
    "next to red" is one split, not an "or" of eight), the colour of its mirror images (symmetry tasks), its position
    modulo 2 and 3 (every third column), whether markers close it in on all four sides, its blob, and the grid's
    rarest and commonest colours as sources for a new colour."""
    rows, cols = grid.shape
    near = {name: value for name, value in structure.items() if name.startswith("next_")}
    orthogonal = [near[f"next_{d}"] for d in ("left", "right", "up", "down")]
    diagonal = [near[f"next_{d}"] for d in ("up_left", "up_right", "down_left", "down_right")]
    extra, colours = {}, set()
    for colour in range(10):
        extra[f"orthogonal_{colour}"] = sum((n == colour).astype(int) for n in orthogonal)
        extra[f"diagonal_{colour}"] = sum((n == colour).astype(int) for n in diagonal)
    extra["orthogonal_foreground"] = sum(((n != bg) & (n != OUTSIDE)).astype(int) for n in orthogonal)
    extra["diagonal_foreground"] = sum(((n != bg) & (n != OUTSIDE)).astype(int) for n in diagonal)
    extra["mirror_left_right"] = grid[:, ::-1]
    extra["mirror_up_down"] = grid[::-1, :]
    extra["mirror_both"] = grid[::-1, ::-1]
    extra["mirror_diagonal"] = grid.T if rows == cols else np.full(grid.shape, OUTSIDE)
    colours |= {"mirror_left_right", "mirror_up_down", "mirror_both", "mirror_diagonal"}
    r, c = np.indices(grid.shape)
    for k in (2, 3):
        extra[f"row_mod_{k}"], extra[f"col_mod_{k}"] = r % k, c % k
        extra[f"row_from_bottom_mod_{k}"], extra[f"col_from_right_mod_{k}"] = (rows - 1 - r) % k, (cols - 1 - c) % k
    sides = [structure[f"nearest_{d}"] for d in ("left", "right", "up", "down")]
    extra["closed_in"] = np.logical_and.reduce([side != OUTSIDE for side in sides]).astype(int)
    extra["closed_in_by_one_colour"] = (extra["closed_in"].astype(bool)
                                        & np.logical_and.reduce([side == sides[0] for side in sides])).astype(int)
    extra.update(blob_features(grid, bg))
    colours |= {"blob_majority", "blob_minority"}
    counts = np.bincount(grid[grid != bg], minlength=10) if (grid != bg).any() else np.zeros(10, dtype=int)
    present = np.nonzero(counts)[0]
    extra["grid_majority"] = np.full(grid.shape, present[counts[present].argmax()] if len(present) else OUTSIDE)
    extra["grid_minority"] = np.full(grid.shape, present[counts[present].argmin()] if len(present) else OUTSIDE)
    colours |= {"grid_majority", "grid_minority"}
    return extra, colours


def table(grids, feature_set):
    """(cells, features) matrix over every cell of every grid, the feature names, and which are colours."""
    blocks, names, colours = [], None, None
    for grid in grids:
        features, colours = cell_features(grid, feature_set)
        names = sorted(features)
        blocks.append(np.stack([features[name].ravel() for name in names], axis=1))
    return np.concatenate(blocks), names, colours


# ---------------------------------------------------------------------------
# a small decision tree: numeric features split by threshold, colours by equality
# ---------------------------------------------------------------------------

def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    share = counts / counts.sum()
    return 1.0 - float((share * share).sum())


def best_split(x, y, categorical):
    """(feature, value, equality) of the split that lowers impurity most, or None when nothing does."""
    best, best_score = None, gini(y) - 1e-12
    n = len(y)
    for feature in range(x.shape[1]):
        column = x[:, feature]
        values = np.unique(column)
        if len(values) < 2:
            continue
        equality = categorical[feature]
        for value in (values if equality else values[:-1]):
            left = column == value if equality else column <= value
            k = left.sum()
            score = (k * gini(y[left]) + (n - k) * gini(y[~left])) / n
            if score < best_score:
                best, best_score = (feature, int(value), equality), score
    return best


def grow(x, y, categorical, depth):
    """A tree as nested tuples: ("leaf", label) or ("split", feature, value, equality, yes, no)."""
    labels, counts = np.unique(y, return_counts=True)
    majority = labels[counts.argmax()]
    if depth == 0 or len(labels) == 1:
        return ("leaf", majority)
    split = best_split(x, y, categorical)
    if split is None:
        return ("leaf", majority)
    feature, value, equality = split
    yes = x[:, feature] == value if equality else x[:, feature] <= value
    return ("split", feature, value, equality, grow(x[yes], y[yes], categorical, depth - 1),
            grow(x[~yes], y[~yes], categorical, depth - 1))


def predict(tree, x):
    out = np.empty(len(x), dtype=object)
    stack = [(tree, np.arange(len(x)))]
    while stack:
        node, rows = stack.pop()
        if node[0] == "leaf":
            out[rows] = node[1]
            continue
        _, feature, value, equality, yes, no = node
        go = x[rows, feature] == value if equality else x[rows, feature] <= value
        stack += [(yes, rows[go]), (no, rows[~go])]
    return out


def describe(tree, names):
    if tree[0] == "leaf":
        return str(tree[1])
    _, feature, value, equality, yes, no = tree
    return f"({names[feature]} {'==' if equality else '<='} {value} ? {describe(yes, names)} : {describe(no, names)})"


def fitted_tree(x, y, categorical, max_depth):
    """The tree grown to pure leaves or `max_depth`, and whether it reproduces `y`. Growth stops at a pure leaf, so
    a rule that a shallow tree holds comes out shallow."""
    tree = grow(x, y, categorical, max_depth)
    return tree, bool((predict(tree, x) == y).all())


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

def new_colour_labels(x, names, colours, targets):
    """What each changed cell becomes, as a label the test can be read with: a role "src:<feature>" - the colour a
    colour feature holds there, which carries over to a test whose colours differ - or the colour itself "abs:<c>".
    The labels are chosen greedily, the one that covers most of the cells not yet labelled first and a role before a
    colour on a tie, so a rule like "the nearest colour on the left, or on the right past the middle" becomes two
    roles the tree can tell apart."""
    candidates = {f"src:{name}": x[:, names.index(name)] == targets for name in sorted(colours)}
    candidates.update({f"abs:{c}": targets == c for c in np.unique(targets)})
    labels = np.empty(len(targets), dtype=object)
    left = np.ones(len(targets), dtype=bool)
    while left.any():
        best = max(candidates, key=lambda key: ((candidates[key] & left).sum(), key.startswith("src:")))
        take = candidates[best] & left
        labels[take] = best
        left &= ~take
    return labels


def apply_colours(labels, x, names):
    out = np.zeros(len(labels), dtype=int)
    for i, label in enumerate(labels):
        kind, what = label.split(":", 1)
        out[i] = int(what) if kind == "abs" else x[i, names.index(what)]
    return out


def fit(train, feature_set, max_depth=6):
    """The two trees for a task's training pairs over `feature_set`: which cells change, and what they become."""
    inputs = [np.asarray(p["input"]) for p in train]
    outputs = [np.asarray(p["output"]) for p in train]
    x, names, colours = table(inputs, feature_set)
    categorical = [name in colours for name in names]
    before = np.concatenate([g.ravel() for g in inputs])
    after = np.concatenate([g.ravel() for g in outputs])
    changed = after != before
    change_tree, change_fits = fitted_tree(x, changed.astype(int), categorical, max_depth)
    labels = new_colour_labels(x[changed], names, colours, after[changed])
    colour_tree, colour_fits = fitted_tree(x[changed], labels, categorical, max_depth)
    return {"set": feature_set, "names": names, "change": change_tree, "colour": colour_tree, "labels": labels,
            "fits": bool(change_fits and colour_fits)}


def apply(model, grid):
    """The output the trees give for `grid`, and the cells they change."""
    grid = np.asarray(grid)
    x, _, _ = table([grid], model["set"])
    will_change = predict(model["change"], x).astype(int) == 1
    guess = grid.ravel().copy()
    if will_change.any():
        guess[will_change] = apply_colours(predict(model["colour"], x[will_change]), x[will_change], model["names"])
    return guess.reshape(grid.shape), will_change.reshape(grid.shape)


def held_out_pairs(train, feature_set, max_depth=6):
    """(pairs reproduced, mean cell accuracy) when each training pair in turn is predicted from the others: how well
    a feature set's trees generalise, measured on the training pairs alone."""
    if len(train) < 2:
        return (0, 0.0)
    right, accuracy = 0, []
    for i, pair in enumerate(train):
        guess, _ = apply(fit(train[:i] + train[i + 1:], feature_set, max_depth), pair["input"])
        want = np.asarray(pair["output"])
        right += int(guess.shape == want.shape and (guess == want).all())
        accuracy.append(float((guess == want).mean()))
    return (right, round(float(np.mean(accuracy)), 4))


def probe_task(train, test_input, test_output, feature_set, max_depth=6):
    """How a tree over `feature_set` does on one task: whether it reproduces the training outputs, how it does on
    each training pair left out in turn, and how much of the test output it gets right."""
    model = fit(train, feature_set, max_depth)
    test_input, test_output = np.asarray(test_input), np.asarray(test_output)
    guess, will_change = apply(model, test_input)
    truly_changed = test_output != test_input
    return {"fits_train": model["fits"], "held_out_pairs": held_out_pairs(train, feature_set, max_depth),
            "test_solved": bool((guess == test_output).all()),
            "test_changed_cells_right": bool((will_change == truly_changed).all()),
            "test_cell_accuracy": round(float((guess == test_output).mean()), 4),
            "changed_cells_right": round(float((guess[truly_changed] == test_output[truly_changed]).mean()), 4)
            if truly_changed.any() else 1.0,
            "colour_labels": sorted(set(model["labels"].tolist())),
            "change_rule": describe(model["change"], model["names"])[:400],
            "colour_rule": describe(model["colour"], model["names"])[:300]}


def chosen(results, candidates=("neighbours", "structure", "rich")):
    """The feature set a task would pick without seeing its test: the one whose trees reproduce most training pairs
    held out in turn, then the best cell accuracy on them, then the simplest. More features also give a greedy tree
    more ways to fit the training pairs by accident; the held-out pairs are what tells the two apart."""
    present = [name for name in candidates if name in results]
    return max(present, key=lambda name: (tuple(results[name]["held_out_pairs"]), -present.index(name)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=list(COORDINATE_TASKS))
    parser.add_argument("--sets", nargs="*", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "experiments" / "feature_probe.json"))
    args = parser.parse_args()
    challenges = json.loads((DATA / "training_challenges.json").read_text())
    solutions = json.loads((DATA / "training_solutions.json").read_text())
    results = {}
    for task in args.tasks:
        challenge = challenges[task]
        results[task] = {name: probe_task(challenge["train"], challenge["test"][0]["input"], solutions[task][0],
                                          name, args.max_depth) for name in args.sets}
    for task in results:
        if any(name in results[task] for name in ("neighbours", "structure", "rich")):
            pick = chosen(results[task])
            results[task]["chosen"] = {"set": pick, **results[task][pick]}
    names = [name for name in args.sets] + (["chosen"] if any("chosen" in r for r in results.values()) else [])
    summary = {name: {"fits_train": sum(results[t][name]["fits_train"] for t in results),
                      "test_solved": sum(results[t][name]["test_solved"] for t in results),
                      "test_changed_cells_right": sum(results[t][name]["test_changed_cells_right"] for t in results),
                      "solved": sorted(t for t in results if results[t][name]["test_solved"])}
               for name in names}
    Path(args.out).write_text(json.dumps({"max_depth": args.max_depth, "summary": summary, "tasks": results},
                                         indent=1, ensure_ascii=False) + "\n")
    print(f"{'feature set':24} {'fits train':>10} {'which cells':>12} {'test solved':>12}")
    for name, row in summary.items():
        print(f"{name:24} {row['fits_train']:>10} {row['test_changed_cells_right']:>12} {row['test_solved']:>12}  "
              f"{' '.join(row['solved'])}")


if __name__ == "__main__":
    main()
