"""What a cell of an ARC grid is, beyond its colour and where it sits: its neighbours, the nearest colour along each
direction and how far it is, its object and blob, its mirror images, its row's and column's colours, and more.

Measured first without RL (scripts/feature_probe.py): a small decision tree over these carries the rule of 7 of the 23
coordinate tasks from the training pairs to the test, where colour and absolute position carry none. The coordinate
policy reads them for each cell (`encoded`, the 'cell_features' observation) in place of the cell's position.

Everything here is numpy over one grid; a feature set names which groups a cell gets.
"""
import numpy as np
from scipy import ndimage

OUTSIDE = -1                     # the colour of a neighbour off the grid
DIRECTIONS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0),
              "up_left": (-1, -1), "up_right": (-1, 1), "down_left": (1, -1), "down_right": (1, 1)}
FEATURE_SETS = ("position", "neighbours", "structure", "structure_and_position", "rich", "rich_and_position", "core")
#: The plainest 25 of the rich set: the cell's colour and its eight neighbours', the nearest colour and how far it is in
#: the four straight directions, whether the cell lies between two of one colour, its object's size and whether it is
#: on the object's border, and its mirror images. The rules the probe's trees found for the 7 tasks they solved used 19
#: features between them, a handful each; 103 give a policy as many ways to fit 2-5 training pairs by accident.
CORE = ("colour", "is_background", *(f"next_{d}" for d in ("left", "right", "up", "down", "up_left", "up_right",
                                                            "down_left", "down_right")),
        *(f"nearest_{d}" for d in ("left", "right", "up", "down")),
        *(f"distance_{d}" for d in ("left", "right", "up", "down")),
        "between_left_right", "between_up_down", "object_size", "on_box_border",
        "mirror_left_right", "mirror_up_down", "mirror_both")


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
    if feature_set == "core":
        features, colours = cell_features(grid, "rich")
        return {name: features[name] for name in CORE}, colours & set(CORE)
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


#: What the coordinate policy reads of each cell: the rich set, which has no absolute position.
POLICY_SET = "rich"


def observed_set(key):
    """The feature set an observation key names: 'cell_features' the rich set, 'cell_features_<set>' that set."""
    return POLICY_SET if key == "cell_features" else key[len("cell_features_"):]


def feature_key(keys):
    """The cell-feature key among an observation's keys, or None."""
    return next((key for key in keys if key == "cell_features" or key.startswith("cell_features_")), None)


def schema(feature_set=POLICY_SET):
    """(names, colours): a feature set's features in the order `encoded` lays them out, and which of them are
    colours. The names do not depend on the grid."""
    features, colours = cell_features(np.zeros((3, 3), dtype=int), feature_set)
    return sorted(features), sorted(colours)


def encoded(grid, shape=None, feature_set=POLICY_SET):
    """(rows, cols, features) uint8 of every cell's features in `schema` order: a colour as one more than itself
    (0 for OUTSIDE), anything else a count clipped to 255. Padded with zeros to `shape` when one is given, so cells
    past the grid's own size read as nothing."""
    grid = np.asarray(grid)
    features, colours = cell_features(grid, feature_set)
    names = sorted(features)
    stacked = np.stack([features[name] + 1 if name in colours else features[name] for name in names], axis=-1)
    stacked = np.clip(stacked, 0, 255).astype(np.uint8)
    if shape is None:
        return stacked
    out = np.zeros((*shape, len(names)), dtype=np.uint8)
    rows, cols = min(grid.shape[0], shape[0]), min(grid.shape[1], shape[1])
    out[:rows, :cols] = stacked[:rows, :cols]
    return out
