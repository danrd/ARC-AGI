"""Addressable coordinates: what a slot names when it does not name an object.

This is not a representation level. A representation level answers "what is
on this grid" - components, shapes, their relations - and every one of them
produces GridObjects. A coordinate answers "where can a rectangle's corner
go", which is a question about addressing, not about content, and has no
object in it.

The distinction is not pedantry. The earlier version of this made every
anchor a one-cell GridObject so that the env's existing slot machinery
would accept it, and measured what that cost: of the 25 fields in
OBJECT_SCHEMA, 8 are constant across every anchor of every grid because
they describe the shape of a 1x1 cell, 6 position fields carry two numbers
between them, and 10 colour fields carry one category. Four quantities
wearing twenty-five, at 16.6ms per reset on a 10x10 grid against 0.04ms
for the same information as integers.

So the anchor has its own schema, and every field in it is a reason to
choose this point as a corner:

    position    where it is, normalised by the grid
    colour      what is under it
    boundary    whether its row/column differs from the neighbouring one,
                on each side - this is what made it an anchor at all
    edge        whether it sits on one of the grid's four edges

Which points. Measured over the 400 training tasks against a greedy
single-colour rectangle cover of what each task changes: the bounding-box
corners of the level 1 objects contain both corners of 21.7% of the
rectangles a cover needs, the intersections of the boundary lines 81.7%.
Free cells would contain all of them and cost the grid's area - a median
of 100 slots and up to 900, against a median of 65 for the boundary
intersections.
"""
from __future__ import annotations

import numpy as np

#: (name, width). The vector a slot carries in an anchor-addressed
#: observation, in this order. Bumped alongside ANCHOR_SCHEMA_VERSION.
ANCHOR_SCHEMA = (
    ("i", 1),               # row, normalised to [0, 1]
    ("j", 1),               # column, normalised to [0, 1]
    ("colour", 10),         # one-hot of the cell under the point
    ("row_differs_above", 1),
    ("row_differs_below", 1),
    ("col_differs_left", 1),
    ("col_differs_right", 1),
    ("on_top_edge", 1),
    ("on_bottom_edge", 1),
    ("on_left_edge", 1),
    ("on_right_edge", 1),
)

ANCHOR_DIM = sum(width for _name, width in ANCHOR_SCHEMA)

#: Bumped whenever the composition or order of ANCHOR_SCHEMA changes, for
#: the same reason OBJECT_SCHEMA_VERSION exists: the vector is a model
#: input, so a checkpoint trained against one version reads a different
#: meaning out of the same slot under another.
ANCHOR_SCHEMA_VERSION = 1

#: The palette an ARC grid draws from. The one-hot is this wide whatever
#: the task uses, so an observation keeps its shape across subtasks.
COLOURS = 10


def boundary_lines(grid: np.ndarray, axis: int) -> list:
    """Rows (axis 0) or columns (axis 1) where the grid's content changes,
    plus the two edges - the lines a rectangle's side can usefully fall on.

    Both sides of each change are kept: a rectangle that stops before a new
    colour begins ends on the line before it, and one that starts with it
    begins on the line itself, and those are different edges. Keeping one
    of them makes half the rectangles inexpressible.
    """
    lines = grid if axis == 0 else grid.T
    marks = {0, lines.shape[0] - 1}
    for index in range(1, lines.shape[0]):
        if not np.array_equal(lines[index], lines[index - 1]):
            marks |= {index - 1, index}
    return sorted(marks)


def anchor_points(grid: np.ndarray) -> np.ndarray:
    """The points a slot may name, as an (n, 2) array of (row, column),
    ordered row-major so a slot index is stable for a given grid."""
    rows = boundary_lines(grid, 0)
    cols = boundary_lines(grid, 1)
    return np.array([(i, j) for i in rows for j in cols], dtype=np.int64)


def anchor_embeddings(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n, ANCHOR_DIM), one row per point, in ANCHOR_SCHEMA's order.

    Built with array arithmetic rather than a per-point object, which is
    the whole point of the separation: the previous version constructed a
    GridObject per anchor and asked it for symmetry, contour and holes.
    """
    rows, cols = grid.shape
    points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    i, j = points[:, 0], points[:, 1]

    #: Which rows/columns differ from their neighbour, computed once for the
    #: grid rather than per point.
    row_differs = np.zeros(rows, dtype=bool)
    if rows > 1:
        row_differs[1:] = np.any(grid[1:] != grid[:-1], axis=1)
    col_differs = np.zeros(cols, dtype=bool)
    if cols > 1:
        col_differs[1:] = np.any(grid[:, 1:] != grid[:, :-1], axis=0)

    block = np.zeros((len(points), ANCHOR_DIM), dtype=np.float32)
    # Normalised by the last index, not by the count, so the far edge is
    # exactly 1.0 and a one-row grid is 0.0 rather than a division by zero.
    block[:, 0] = i / max(rows - 1, 1)
    block[:, 1] = j / max(cols - 1, 1)
    colours = np.clip(grid[i, j], 0, COLOURS - 1)
    block[np.arange(len(points)), 2 + colours] = 1.0
    block[:, 12] = row_differs[i]
    block[:, 13] = np.where(i + 1 < rows, row_differs[np.minimum(i + 1, rows - 1)], False)
    block[:, 14] = col_differs[j]
    block[:, 15] = np.where(j + 1 < cols, col_differs[np.minimum(j + 1, cols - 1)], False)
    block[:, 16] = i == 0
    block[:, 17] = i == rows - 1
    block[:, 18] = j == 0
    block[:, 19] = j == cols - 1
    return block
