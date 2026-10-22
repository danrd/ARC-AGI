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
}


def transformed(name: str, grid) -> Optional[np.ndarray]:
    """`grid` after the named transformation, or None when it does not apply to this grid or gives an empty one."""
    out = TRANSFORMS[name](np.asarray(grid))
    return None if out is None or out.size == 0 else np.ascontiguousarray(out)
