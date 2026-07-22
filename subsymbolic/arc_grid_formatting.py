"""ARC-specific grid textual representations, used by PromptBuilder's `grid`
filter. Consolidates what used to be four near-duplicate functions
(prepare_grid_for_prompt + concise/ascii/color_text/array_to_ascii) into one
dispatcher, so adding a representation means adding one branch here instead
of a new top-level function plus a new if/elif in the caller.
"""
import numpy as np

REPR_TYPES = ("concise", "ascii", "color_text", "text_ascii")

DEFAULT_COLORS_MAPPING = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown',
}

# Single-letter codes for the compact 'text_ascii' representation.
_ASCII_LETTER_MAPPING = {
    0: 'b', 1: 'B', 2: 'R', 3: 'G', 4: 'Y',
    5: 'g', 6: 'M', 7: 'O', 8: 'S', 9: 'W',
}


def format_grid(grid: np.ndarray, repr_type: str = "concise",
                 colors_mapping: dict = None) -> str:
    if repr_type == "concise":
        return _concise(grid)
    if repr_type == "ascii":
        return _ascii(grid)
    if repr_type == "color_text":
        return _color_text(grid, colors_mapping or DEFAULT_COLORS_MAPPING)
    if repr_type == "text_ascii":
        return _text_ascii(grid)
    raise ValueError(f"Unknown grid repr_type: {repr_type!r}. Expected one of {REPR_TYPES}.")


def _concise(grid: np.ndarray) -> str:
    lines = [f"grid shape: {grid.shape[0]},{grid.shape[1]}"]
    for i in range(grid.shape[0]):
        row = "".join(str(int(v)) for v in grid[i])
        lines.append(f"{i + 1} {row}")
    return "\n".join(lines) + "\n"


def _ascii(grid: np.ndarray) -> str:
    lines = [f"grid shape: {grid.shape[0]},{grid.shape[1]}"]
    for i in range(grid.shape[0]):
        lines.append("|".join(str(int(v)) for v in grid[i]))
    return "\n".join(lines) + "\n"


def _color_text(grid: np.ndarray, colors_mapping: dict) -> str:
    lines = [f"grid shape: {grid.shape[0]},{grid.shape[1]}"]
    for i in range(grid.shape[0]):
        lines.append(" ".join(colors_mapping[int(v)] for v in grid[i]))
    return "\n".join(lines) + "\n"


def _text_ascii(grid: np.ndarray, empty_symbol: str = ".", include_coordinates: bool = True) -> str:
    header = f"grid shape: {grid.shape[0]},{grid.shape[1]}"
    rows = ["".join(_ASCII_LETTER_MAPPING.get(int(v), empty_symbol) for v in row) for row in grid]
    if include_coordinates:
        rows = [f"{i + 1:2} {line}" for i, line in enumerate(rows)]
    return header + "\n" + "\n".join(rows)
