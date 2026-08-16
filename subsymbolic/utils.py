import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def lev_sim(s1: str, s2: str) -> float:
    """Calculate a normalized similarity score (0 to 1) between two strings using Levenshtein distance.
    - 1.0 = identical
    - 0.0 = completely different
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len) if max_len != 0 else 1.0

def prompts_length_dist(dataset, tokenizer, plot=False, percentiles=False):
    lens = []
    for prompt in dataset['text']:
        token_len = len(tokenizer(prompt)['input_ids'])
        lens.append(token_len)
    counter = sorted(Counter(lens), reverse=True)
    if plot:
        plt.hist(sorted(counter, reverse=True))
    if percentiles:
        percs = {}
        for p in range(10, 100, 10):
            percs[p] = np.percentile(counter, p)
        percs[95] = np.percentile(counter, 95)
        percs[99] = np.percentile(counter, 99)
        return counter, percs
    else:
        return counter

def parse_concise_grid(grid_str: str) -> np.array:
    """Parse a grid from concise LLM output representation into a NumPy array.
    """
    lines = grid_str.strip().split('\n')

    # Extract shape from the first line
    shape_line = lines[0]
    shape = tuple(map(int, shape_line.split('(')[1].split(')').split(',')))

    # Parse grid values
    grid = []
    for line in lines[1:]:
        # Ignore the row index at the start of each line
        row_values = list(map(int, line.split()[1]))
        grid.append(row_values)

    return np.array(grid).reshape(shape)

def parse_ascii_grid(grid_str: str) -> np.array:
    """Parse a grid from ASCII LLM output representation into a NumPy array.
    """
    lines = grid_str.strip().split('\n')

    # Extract shape from the first line
    shape_line = lines[0]
    shape = tuple(map(int, shape_line.split('(')[1].split(')').split(',')))

    # Parse grid values
    grid = []
    for line in lines[1:]:
        row_values = list(map(int, line.split('|')))
        grid.append(row_values)

    return np.array(grid).reshape(shape)

def check_module_devices(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Device: {param.device}")

def parse_llm_output(text, colors_str=False, max_grid_dim=30):
    """Parse a string in the format:
    n,m:
    1 x_1 ... x_m
    ...
    n x_1 ... x_m

    Where n is the number of rows, m is the number of columns and x_i are the cell values.

    Args:
        text (str): The text to parse

    Returns:
        numpy.ndarray: A NumPy array of shape (n, m)
    """
    inverse_colors_mapping_short = {
    'b':0, 'B':1, 'R':2, 'G':3, 'Y':4,
    'g':5, 'M':6, 'O':7, 'S':8, 'W':9
}
    # Split the text into lines
    lines = text.strip().split('\n')

    # Parse the header to get dimensions
    dimensions = lines[0].strip().rstrip(':').split(',')
    try:
        if len(dimensions) == 2 and int(dimensions[0]) in range(1, max_grid_dim + 1) and int(dimensions[1]) in range(1, max_grid_dim + 1):
            n_rows = int(dimensions[0])
            n_cols = int(dimensions[1])
        else:
            return ""
    except ValueError:
        return ""

    # Initialize the result array
    result = np.zeros((n_rows, n_cols), dtype=int)

    # Parse each row
    for i in range(1, n_rows + 1):
        if i < len(lines):
            line = lines[i].strip()
            # Split by whitespace and remove the row number
            parts = line.split()
            if len(parts) < 1:
                return ""
            try:
                # Identify current row
                row_num = int(parts[0])
            except ValueError:
                return ""
            # A row line with no data at all (just the row number), or a
            # row number outside [1, n_rows] - out of range would otherwise
            # either IndexError on the result[] write below, or (for
            # row_num <= 0) silently wrap around and overwrite an
            # unrelated row via Python's negative indexing.
            if len(parts) < 2 or not (1 <= row_num <= n_rows):
                return ""
            row_data = parts[1]
            if len(parts) > 2:
                return ""
            # Check if there's a single string of values or separate values
            elif len(parts) == 2 and len(row_data) == n_cols:
                # Single string of values (like "000020000")
                for j in range(n_cols):
                    try:
                        result[row_num-1, j] = int(row_data[j]) if not colors_str else inverse_colors_mapping_short[row_data[j]]
                    except (ValueError, KeyError):
                        return ""
            else:
                # Multiple separate values (like "0 0 0 0 2 0 0 0 0")
                for j in range(min(n_cols, len(row_data) - 1)):
                    try:
                        result[row_num-1, j] = int(row_data[j]) if not colors_str else inverse_colors_mapping_short[row_data[j]]
                    except (ValueError, KeyError):
                        return ""
    return result


def build_grid_grammar(colors_str: bool = False, max_dim: int = 30) -> str:
    """GBNF grammar (llama.cpp grammar-constrained decoding) matching the
    format parse_llm_output actually accepts - not the "1 x_1 ... x_m"
    space-separated notation the format instructions describe. In practice
    `line.split()` producing more than 2 parts (any row with 2+ columns
    written as separate space-separated values) hits `if len(parts) > 2:
    return ""` and is rejected outright - only a single concatenated digit
    string per row (e.g. "1 012") ever actually parses for n_cols > 1. This
    grammar only forces that working shape.

    What it does NOT enforce: that the row count matches the declared n, that
    row numbers are sequential, or that each row's digit count matches the
    declared m - GBNF is a static, context-free grammar and n/m are only
    known once the model itself generates them, so exact structural
    consistency can't be forced this way. What it DOES guarantee: every
    generated token is a digit, comma, colon, space or newline - no
    reasoning/preamble text is representable at all. Row-count/shape
    consistency is still parse_llm_output's job, same as before.

    Args:
        colors_str: match parse_llm_output(colors_str=True)'s letter-coded
            cell values (b/B/R/G/Y/g/M/O/S/W) instead of digits 0-9.
        max_dim: only bounds the header's digit count (1-2 digits covers up
            to 99) - parse_llm_output separately rejects anything outside
            [1, max_dim].
    """
    dim_rule = "[1-9] [0-9]?" if max_dim <= 99 else "[1-9] [0-9]? [0-9]?"
    cell_char_class = "[0-9]" if not colors_str else "[bBRGYgMOSW]"
    return (
        "root   ::= header \"\\n\" row (\"\\n\" row)*\n"
        f"header ::= dim \",\" dim (\":\")?\n"
        f"dim    ::= {dim_rule}\n"
        f"row    ::= dim \" \" cell+\n"
        f"cell   ::= {cell_char_class}\n"
    )
