"""A small library of whole-grid transformations, each taking a grid to another grid (or to None when it does not apply).

The cell rules of symbolic.rule_search change cells in place, so they only reach tasks whose output has the input's
shape. Many of the rest are a rearrangement - the grid turned, mirrored, cut to what it holds, repeated or shrunk -
followed by a recolouring that a cell rule can say. `TRANSFORMS` is that first step; symbolic.transformed_rules puts
the rule search after it.
"""
from typing import Callable, Dict, Optional

import numpy as np
from scipy import ndimage


def background(grid: np.ndarray) -> int:
    """The colour that covers most of the grid."""
    return int(np.bincount(grid.ravel()).argmax())


def box(mask: np.ndarray) -> Optional[tuple]:
    """(row start, row stop, column start, column stop) of the cells in `mask`, or None when there are none."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def crop_to_content(grid):
    found = box(grid != background(grid))
    return None if found is None else grid[found[0]:found[1], found[2]:found[3]]


def crop_to_piece(grid, largest: bool):
    """The bounding box of the largest (or smallest) piece of non-background cells, touching at corners too."""
    labelled, count = ndimage.label(grid != background(grid), structure=np.ones((3, 3)))
    if not count:
        return None
    sizes = np.bincount(labelled.ravel())[1:]
    pick = int(sizes.argmax() if largest else sizes.argmin()) + 1
    found = box(labelled == pick)
    return grid[found[0]:found[1], found[2]:found[3]]


def downscale(grid, k: int):
    """Every k-th cell, when the grid is made of k x k blocks of one colour."""
    height, width = grid.shape
    if height % k or width % k or min(height, width) < k * 2:
        return None
    small = grid[::k, ::k]
    return small if np.array_equal(np.kron(small, np.ones((k, k), dtype=grid.dtype)), grid) else None


def trim(grid):
    return grid[1:-1, 1:-1] if min(grid.shape) > 2 else None


def pad(grid):
    return np.pad(grid, 1, constant_values=background(grid))


def half(grid, side: str):
    height, width = grid.shape
    if side in ("left", "right") and width < 2 or side in ("top", "bottom") and height < 2:
        return None
    return {"left": grid[:, :width // 2], "right": grid[:, width - width // 2:],
            "top": grid[:height // 2], "bottom": grid[height - height // 2:]}[side]


def mirrored(grid, horizontal: bool, vertical: bool):
    out = grid
    if horizontal:
        out = np.hstack([out, out[:, ::-1]])
    if vertical:
        out = np.vstack([out, out[::-1]])
    return out


def dedup(grid, axis: int):
    """The grid with consecutive equal rows (axis 0) or columns (axis 1) merged into one: the grid a stretched one was
    made from. None when nothing merges."""
    moved = grid if axis == 0 else grid.T
    keep = [0] + [i for i in range(1, len(moved)) if not np.array_equal(moved[i], moved[i - 1])]
    out = moved[keep]
    if len(keep) == len(moved):
        return None
    return out if axis == 0 else out.T


def dedup_both(grid):
    rows = dedup(grid, 0)
    rows = grid if rows is None else rows
    both = dedup(rows, 1)
    both = rows if both is None else both
    return None if both.shape == grid.shape else both


def crop_to_colour(grid, rarest: bool):
    """The bounding box of all cells of the rarest (or commonest) colour that is not the background."""
    background_colour = background(grid)
    colours, counts = np.unique(grid[grid != background_colour], return_counts=True)
    if not len(colours):
        return None
    colour = colours[counts.argmin() if rarest else counts.argmax()]
    found = box(grid == colour)
    return grid[found[0]:found[1], found[2]:found[3]]


def parts(grid, vertical: bool):
    """The two halves of the grid side by side (or one above the other) with a divider line of one colour between them
    left out, or None when the grid cannot be cut in two equal parts."""
    moved = grid.T if vertical else grid
    width = moved.shape[1]
    if width < 2:
        return None
    half_width = width // 2
    if width % 2 == 1 and len(np.unique(moved[:, half_width])) > 1:
        return None                                      # an odd grid is cut at a line of one colour, or not at all
    first, second = moved[:, :half_width], moved[:, width - half_width:]
    return (first.T, second.T) if vertical else (first, second)


def combined(grid, vertical: bool, how: str):
    """The two parts of the grid laid over each other cell by cell: `or_first`/`or_second` keep the cell of the named
    part that is not background (the first part's wins in `or_first`), `and` the first part's cell where both have one,
    `xor` the cell of the one part that has one, `nor` a mark where neither does."""
    cut = parts(grid, vertical)
    if cut is None:
        return None
    first, second = cut
    bg = background(grid)
    a, b = first != bg, second != bg
    if how == "or_first":
        return np.where(a, first, second)
    if how == "or_second":
        return np.where(b, second, first)
    if how == "and":
        return np.where(a & b, first, bg)
    if how == "xor":
        return np.where(a & ~b, first, np.where(b & ~a, second, bg))
    return np.where(~a & ~b, (bg + 1) % 10, bg)


#: name -> function of a grid. Applying one never changes its argument.
TRANSFORMS: Dict[str, Callable[[np.ndarray], Optional[np.ndarray]]] = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "flip_rows": lambda g: g[::-1],
    "flip_columns": lambda g: g[:, ::-1],
    "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
    "up2": lambda g: np.kron(g, np.ones((2, 2), dtype=g.dtype)),
    "up3": lambda g: np.kron(g, np.ones((3, 3), dtype=g.dtype)),
    "up4": lambda g: np.kron(g, np.ones((4, 4), dtype=g.dtype)),
    "down2": lambda g: downscale(g, 2),
    "down3": lambda g: downscale(g, 3),
    "tile_2x1": lambda g: np.tile(g, (2, 1)),
    "tile_1x2": lambda g: np.tile(g, (1, 2)),
    "tile_2x2": lambda g: np.tile(g, (2, 2)),
    "tile_3x3": lambda g: np.tile(g, (3, 3)),
    "mirror_right": lambda g: mirrored(g, True, False),
    "mirror_down": lambda g: mirrored(g, False, True),
    "mirror_both": lambda g: mirrored(g, True, True),
    "crop_content": crop_to_content,
    "crop_largest": lambda g: crop_to_piece(g, True),
    "crop_smallest": lambda g: crop_to_piece(g, False),
    "left_half": lambda g: half(g, "left"),
    "right_half": lambda g: half(g, "right"),
    "top_half": lambda g: half(g, "top"),
    "bottom_half": lambda g: half(g, "bottom"),
    "trim": trim,
    "pad": pad,
    "down4": lambda g: downscale(g, 4),
    "down5": lambda g: downscale(g, 5),
    "dedup_rows": lambda g: dedup(g, 0),
    "dedup_columns": lambda g: dedup(g, 1),
    "dedup_both": dedup_both,
    "crop_rarest_colour": lambda g: crop_to_colour(g, True),
    "crop_commonest_colour": lambda g: crop_to_colour(g, False),
}
for _axis, _vertical in (("side", False), ("stack", True)):
    for _how in ("or_first", "or_second", "and", "xor", "nor"):
        TRANSFORMS[f"{_axis}_{_how}"] = (lambda g, v=_vertical, h=_how: combined(g, v, h))


def transformed(name: str, grid) -> Optional[np.ndarray]:
    """`grid` after the named transformation, or None when it does not apply to this grid or gives an empty one."""
    out = TRANSFORMS[name](np.asarray(grid))
    return None if out is None or out.size == 0 else np.ascontiguousarray(out)
