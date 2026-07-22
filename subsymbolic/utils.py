import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between two strings.
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
    """
    Calculate a normalized similarity score (0 to 1) between two strings using Levenshtein distance.
    - 1.0 = identical
    - 0.0 = completely different
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len) if max_len != 0 else 1.0

def prompts_length_dist(dataset, tokenizer, plot=False, percentiles=False):
    lens = []
    for prompt in dataset['text']:
        l = len(tokenizer(prompt)['input_ids'])
        lens.append(l)
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
    """
    Parse a grid from concise LLM output representation into a NumPy array.
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
    """
    Parse a grid from ASCII LLM output representation into a NumPy array.
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
    """
    Parse a string in the format:
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
            try:
                # Identify current row
                row_num = int(parts[0])
            except ValueError:
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