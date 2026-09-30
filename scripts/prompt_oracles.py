"""The rules of ten evaluation tasks, as text and as code, for prompt tests.

Each entry holds what is needed to ask "does knowing the rule help a model
that does not solve this task?": the rule in one or two sentences, a
plausible but wrong rule to see whether the model trusts prose over the
examples, and a function that applies the rule to a grid. The function is
what makes the sentence trustworthy - tests/test_prompt_oracles.py holds it
to every training pair and the test pair of its task - and it is what
scripts/prompt_variants.py takes a draft answer and a walk-through from.

The ten are the evaluation tasks that graded easy or medium by hand, that
no LLM run solved, and whose rule was known from a search or a solver
(data/datasets/ARC/evaluation_difficulty.json; see prompt_variants.py).

Rules are written against a colour function `col`, so a run with the
colours permuted - to defeat memorisation of a public task - names the
permuted colours and not the original ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

EIGHT = np.ones((3, 3))


def _components(grid, colour, structure=EIGHT):
    labels, count = ndimage.label(grid == colour, structure=structure)
    return [labels == index for index in range(1, count + 1)]


def _holes(mask):
    """Number of enclosed empty regions of a shape."""
    return ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)[1]


def _crop(mask):
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


@dataclass(frozen=True)
class Oracle:
    task: str
    rule: Callable[[Callable[[int], str]], str]
    wrong: Callable[[Callable[[int], str]], str]
    solve: Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------

def _solve_6ea4a07e(grid):
    background_to = {8: 2, 3: 1, 5: 4}
    return np.where(grid == 0, background_to[int(grid.max())], 0)


def _solve_ae58858e(grid):
    out = grid.copy()
    for mask in _components(grid, 2):
        if mask.sum() >= 4:
            out[mask] = 6
    return out


def _solve_64a7c07e(grid):
    out = np.zeros_like(grid)
    for colour in np.unique(grid):
        if colour == 0:
            continue
        for mask in _components(grid, colour):
            rows, cols = np.where(mask)
            width = cols.max() - cols.min() + 1
            out[rows, cols + width] = colour
    return out


def _solve_84f2aca1(grid):
    out = grid.copy()
    for colour in np.unique(grid):
        if colour == 0:
            continue
        for mask in _components(grid, colour):
            labels, count = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)
            for index in range(1, count + 1):
                hole = labels == index
                out[hole] = {1: 5, 2: 7}.get(int(hole.sum()), out[hole][0])
    return out


#: The key shape of each training pair of 009d5c81 and the colour it names.
_KEYS_009D5C81 = {
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): 2,
    ((1, 0, 1), (0, 1, 0), (1, 1, 1)): 3,
    ((1, 1, 1), (1, 0, 1), (0, 1, 0)): 7,
}


def _solve_009d5c81(grid):
    key = tuple(map(tuple, _crop(grid == 1).astype(int)))
    out = grid.copy()
    out[grid == 8] = _KEYS_009D5C81[key]
    out[grid == 1] = 0
    return out


def _by_holes(mapping):
    def solve(grid):
        out = grid.copy()
        for mask in _components(grid, 8):
            out[mask] = mapping[_holes(mask)]
        return out
    return solve


def _solve_00dbd492(grid):
    out = grid.copy()
    for mask in _components(grid, 2):
        if mask.sum() < 2:
            continue
        interior = ndimage.binary_fill_holes(mask) & ~mask & (grid == 0)
        if interior.any():
            height = int(np.ptp(np.where(interior)[0]) + 1)
            out[interior] = {3: 8, 5: 4, 7: 3}[height]
    return out


def _solve_60a26a3e(grid):
    out = grid.copy()
    labels, count = ndimage.label(grid == 2, structure=EIGHT)
    centres = [(int(np.mean(np.where(labels == k)[0])), int(np.mean(np.where(labels == k)[1])))
               for k in range(1, count + 1)]
    for row, col in centres:
        right = [c for r, c in centres if r == row and c > col]
        if right:
            out[row, col + 2:min(right) - 1] = 1
        below = [r for r, c in centres if c == col and r > row]
        if below:
            out[row + 2:min(below) - 1, col] = 1
    return out


def _solve_54db823b(grid):
    out = grid.copy()
    labels, count = ndimage.label(grid > 0, structure=EIGHT)
    nines = [(int((grid[labels == k] == 9).sum()), k) for k in range(1, count + 1)]
    out[labels == min(nines)[1]] = 0
    return out


# ---------------------------------------------------------------------------
# the entries
# ---------------------------------------------------------------------------

ORACLES = {oracle.task: oracle for oracle in (
    Oracle(
        "6ea4a07e",
        lambda c: (f"The input holds one non-zero colour. Every cell of that colour becomes {c(0)}, and every "
                   f"cell that was {c(0)} becomes a colour that depends on the input's colour: "
                   f"{c(8)} gives {c(2)}, {c(3)} gives {c(1)}, {c(5)} gives {c(4)}."),
        lambda c: (f"The two colours swap: every {c(0)} becomes the input's non-zero colour and "
                   f"every cell of that colour becomes {c(0)}."),
        _solve_6ea4a07e),
    Oracle(
        "ae58858e",
        lambda c: (f"Every connected group of {c(2)} cells (cells touching sideways or diagonally) that has "
                   f"4 or more cells becomes {c(6)}. Smaller groups stay {c(2)}."),
        lambda c: f"The largest group of {c(2)} cells becomes {c(6)}; every other group stays {c(2)}.",
        _solve_ae58858e),
    Oracle(
        "64a7c07e",
        lambda c: ("Every object (a group of same-coloured cells touching sideways or diagonally) moves to "
                   "the right by its own width in cells, keeping its shape and colour."),
        lambda c: "Every object moves one cell to the right, keeping its shape and colour.",
        _solve_64a7c07e),
    Oracle(
        "84f2aca1",
        lambda c: (f"Each closed ring has a hole of empty ({c(0)}) cells inside. A hole of one cell is filled "
                   f"with {c(5)}; a hole of two cells is filled with {c(7)}."),
        lambda c: f"Every hole inside a closed ring is filled with {c(5)}.",
        _solve_84f2aca1),
    Oracle(
        "009d5c81",
        lambda c: (f"The small object of colour {c(1)} is a key: its cells become {c(0)}. The larger object "
                   f"of colour {c(8)} is recoloured according to the key's shape - the plus shape gives "
                   f"{c(2)}, the shape 101/010/111 gives {c(3)}, the shape 111/101/010 gives {c(7)}."),
        lambda c: f"The object of colour {c(8)} becomes {c(2)} and the object of colour {c(1)} is removed.",
        _solve_009d5c81),
    Oracle(
        "0a2355a6",
        lambda c: (f"Every shape of colour {c(8)} is recoloured by how many holes (enclosed empty regions) it "
                   f"has: 1 hole gives {c(1)}, 2 holes give {c(3)}, 3 holes give {c(2)}, 4 holes give {c(4)}."),
        lambda c: (f"A shape of colour {c(8)} that has any hole becomes {c(1)}; a shape without holes "
                   f"stays {c(8)}."),
        _by_holes({1: 1, 2: 3, 3: 2, 4: 4})),
    Oracle(
        "37d3e8b2",
        lambda c: (f"Every shape of colour {c(8)} is recoloured by how many holes (enclosed empty regions) it "
                   f"has: 1 hole gives {c(1)}, 2 holes give {c(2)}, 3 holes give {c(3)}, 4 holes give {c(7)}."),
        lambda c: (f"Every shape of colour {c(8)} is recoloured to the number of its holes: 1 hole gives "
                   f"{c(1)}, 2 holes give {c(2)}, 3 holes give {c(3)}, 4 holes give {c(4)}."),
        _by_holes({1: 1, 2: 2, 3: 3, 4: 7})),
    Oracle(
        "00dbd492",
        lambda c: (f"Each square ring of colour {c(2)} has an empty interior, which is filled by the ring's "
                   f"size: a 3x3 interior gets {c(8)}, a 5x5 interior gets {c(4)}, a 7x7 interior gets {c(3)}. "
                   f"The single centre cell of colour {c(2)} stays."),
        lambda c: f"The interior of every square ring of colour {c(2)} is filled with {c(8)}.",
        _solve_00dbd492),
    Oracle(
        "60a26a3e",
        lambda c: (f"The {c(2)} shapes are small diamonds. Where two neighbouring diamonds share a row or a "
                   f"column, with no diamond between them, the empty cells between their nearest tips are "
                   f"filled with {c(1)}."),
        lambda c: (f"The empty cells between every two diamonds that share a row or a column are filled "
                   f"with {c(1)}, including two diamonds that have another diamond between them."),
        _solve_60a26a3e),
    Oracle(
        "54db823b",
        lambda c: (f"The grid holds several separate patterns of colours {c(3)} and {c(9)}. The pattern with "
                   f"the fewest {c(9)} cells is erased (all its cells become {c(0)}); the others stay."),
        lambda c: (f"The pattern with the fewest cells of any colour is erased (all its cells become "
                   f"{c(0)}); the others stay."),
        _solve_54db823b),
)}
