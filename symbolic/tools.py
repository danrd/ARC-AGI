# UPSCALE
import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from typing import Dict, List
from itertools import permutations, product
from rl.ARC_task import ARCTask
from symbolic.summaries import GridSummary 
from symbolic.utils import find_upper_left_corner, coords_transform
from symbolic.patterns import retrieve_shapes, find_connected_components_with_color

def solve_pattern_upscale(task, font_color=0):
    """
    General pattern upscale solver that:
    - Analyzes all training examples
    - Handles replacement of all elements (including zeros)
    - Considers pattern inversions (swapping background and foreground)
    - Extracts pattern from outputs if needed
    Args:
        task: ARCTask instance with subtasks and test_subtask attributes
        
    Returns:
        numpy.ndarray: The predicted output for test_subtask.train_inp
    """
    subtasks = task.subtasks
    test_subtask = task.test_subtask
    scales = define_examples_scales(subtasks)
    pattern_candidates, replacement_map = analyze_training_examples(subtasks, test_subtask, scales, font_color)
    return apply_upscale(test_subtask.train_inp, subtasks, scales, pattern_candidates, replacement_map, font_color)

def define_examples_scales(subtasks):
    scales = defaultdict(list)
    for idx, subtask in enumerate(subtasks):
        inp = subtask.train_inp
        out = subtask.train_out
        scale_i = out.shape[0] / inp.shape[0]
        scale_j = out.shape[1] / inp.shape[1]
        scales[idx] = (scale_i, scale_j)
    return scales
    
def analyze_training_examples(subtasks, test_subtask, scales, font_color):
    """
    Analyzes all training examples to determine pattern_candidates and infer replacement map
    """
    pattern_candidates = extract_pattern_from_output(subtasks, scales, font_color)
    replacement_map = build_replacement_map(subtasks, test_subtask, scales, pattern_candidates, font_color)
    if replacement_map is not None:
        return pattern_candidates, replacement_map
    else:
        return defaultdict(list), {}


def extract_pattern_from_output(subtasks, scales, font_color):
    """
    Extracts the repeating pattern from output grid.
    Takes the top-left scale×scale block as the pattern.
    """
    patterns_candidates = {}
    for idx, subtask in enumerate(subtasks):
        inp = subtask.train_inp 
        out = subtask.train_out 
        scale_i, scale_j = scales[idx]
        scale_i = int(scale_i) 
        scale_j = int(scale_j) 
        subtask_pattern = None
        for j in range (inp.shape[1]):
            for i in range(inp.shape[0]):
                cell_val = inp[i,j]
                i_out, j_out = i * scale_i, j * scale_j
                block = out[i_out:i_out+scale_i, j_out:j_out+scale_j]
                if (block != font_color).any() and (block != cell_val).all() and (inp.shape != block.shape):
                    if (idx in patterns_candidates.keys()) and (patterns_candidates[idx] != block).any():
                        return {}
                    elif idx not in patterns_candidates.keys():
                        patterns_candidates[idx] = block
    return patterns_candidates

def get_mappings(subtasks, test_subtask, scales, pattern_candidates, font_color):
    """
    For each subtask builds a mapping from input colors to some pattern.
    
    Returns:
        list [{input_color: pattern_version}]
    """
    # Collect mappings from all examples
    use_pattern = False
    all_mappings = []
    if len(pattern_candidates.keys()) == len(subtasks):
        use_pattern = True
        
    for idx, subtask in enumerate(subtasks):
        inp = subtask.train_inp
        out = subtask.train_out
        scale_i, scale_j = scales[idx]
        scale_i = int(scale_i)
        scale_j = int(scale_j)
        
        unique_colors = np.unique(inp)
        example_mapping = {}
        pattern = pattern_candidates[idx] if (use_pattern and idx in pattern_candidates.keys()) else inp
        for color in unique_colors:
            # Find all cells with this color
            positions = np.argwhere(inp == color)
            
            # Check what pattern they all map to
            blocks = []
            for pos in positions:
                i, j = pos
                out_i, out_j = i * scale_i, j * scale_j
                block = out[out_i:out_i+scale_i, out_j:out_j+scale_j]
                blocks.append(block)
            
            # Check if all blocks are the same
            if len(blocks) > 0:
                first_block = blocks[0]
                if all(np.array_equal(b, first_block) for b in blocks):
                    # All blocks are identical - this is a valid mapping
                    # Check if block is all background color (no replacement)
                    if np.all(first_block == font_color):
                        example_mapping[color] = 'font'
                    elif np.array_equal(first_block, inp):
                        example_mapping[color] = 'original'
                    # Check if it matches the pattern
                    elif np.array_equal(first_block, pattern):
                        example_mapping[color] = 'output_pattern'
                    # Check if it matches inverted pattern
                    elif np.array_equal(first_block, invert_pattern(inp, font_color)) or np.array_equal(first_block, invert_pattern(pattern, font_color)):
                        example_mapping[color] = 'inverted'
                    elif np.unique(first_block)[0] == color:
                        example_mapping[color] = 'color_upscale'
                    else:
                        # Doesn't match any known pattern
                        example_mapping[color] = None

        all_mappings.append(example_mapping)
    return all_mappings if all_mappings else None

def build_replacement_map(subtasks, test_subtask, scales, pattern_candidates, font_color):
    """
    Builds a unified mapping from input colors to pattern for test array.
    
    Returns:
        dict: {input_color: pattern_version} or None if can't determine
    """
    replacement_map = {}
    all_mappings = get_mappings(subtasks, test_subtask, scales, pattern_candidates, font_color)

    # Get colors that appear in all examples
    all_colors = set(all_mappings[0].keys())
    for mapping in all_mappings[1:]:
        all_colors |= set(mapping.keys())
    
    test_inp = test_subtask.train_inp
    test_unique_colors = np.unique(test_inp)

    if all('color_upscale' in mapping.values() for mapping in all_mappings):
        for color in test_unique_colors:
            replacement_map[color] = "color_upscale"
    elif need_ranking(all_mappings):
        replacement_map = ranking_mapping(subtasks, test_inp, all_mappings, font_color)
    else:
       for color in all_colors:
            mappings = [m[color] for m in all_mappings if color in m.keys()]
            replacement_map[color] = mappings[0]
    
    return replacement_map if replacement_map else None

def ranking_mapping(subtasks, test_inp, all_mappings, font_color):
    replacement_map = {}
    used_ranks = []
    all_non_font_replacements = []
    test_colors_ranking = create_ranking_dict(test_inp, font_color)
    test_unique_colors = np.unique(test_inp)
    colors_rankings = [create_ranking_dict(subtask.train_inp, font_color) for subtask in subtasks]
    for idx, mapping in enumerate(all_mappings):
        non_font_ranks = [colors_rankings[idx][k] for k,v in mapping.items() if v != 'font' and k != font_color]
        non_font_replacements = [v for v in mapping.values() if v != 'font']
        used_ranks.extend(non_font_ranks)
        all_non_font_replacements.extend(non_font_replacements)
    
    unique_ranks = set(used_ranks)
    unique_non_font_replacements = set(all_non_font_replacements)
    if len(unique_ranks) == 1 and len(unique_non_font_replacements) == 1:
        target_rank = unique_ranks.pop()
        test_colors_ranking_inverted = {v:k for k,v in test_colors_ranking.items()}
        target_color = test_colors_ranking_inverted[target_rank]
        for unique_color in test_unique_colors:
            if unique_color == target_color:
                replacement_map[unique_color] = unique_non_font_replacements.pop()
            else:
                replacement_map[unique_color] = 'font'
    
    return replacement_map if replacement_map else None

def apply_upscale(inp, subtasks, scales, pattern_candidates, replacement_map, font_color):
    """
    Applies the upscaling transformation to input grid.
    Only replaces colors that are in the replacement_map.
    """ 
    h, w = inp.shape
    
    unique_colors = np.unique(inp)
    if 'output_pattern' in replacement_map.values():
        pattern = pattern_candidates[0]
        scale_i, scale_j = pattern.shape
    elif 'color_upscale' in replacement_map.values():
        pattern = copy(inp)
        unique_scalings = list(set(scales.values()))
        if not scales[0][0].is_integer():
            return non_int_scaling(inp, subtasks)
        elif len(unique_scalings) == 1:
            scale_i, scale_j = unique_scalings[0]
        elif all([(len(np.unique(subtask.train_inp))-1)==scales[idx][0] for idx, subtask in enumerate(subtasks)]):
            scale_i, scale_j = len(unique_colors)-1, len(unique_colors)-1
        elif all([len(np.unique(subtask.train_inp))==scales[idx][0] for idx, subtask in enumerate(subtasks)]):
            scale_i, scale_j = len(unique_colors), len(unique_colors)
        scale_i = int(scale_i)
        scale_j = int(scale_j)
    else:
        pattern = copy(inp)
        scale_i, scale_j = pattern.shape

    output = np.full((h * scale_i, w * scale_j), font_color, dtype=inp.dtype)
    
    for i in range(h):
        for j in range(w):
            color = inp[i, j]
            out_i, out_j = i * scale_i, j * scale_j
            if color in replacement_map:
                mapping = replacement_map[color]
                if mapping == 'original':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = inp
                elif mapping == 'output_pattern':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = pattern
                elif mapping == 'inverted':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = invert_pattern(pattern, font_color)
                elif mapping == 'font':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = font_color
                elif mapping == 'color_upscale':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = color                  
                elif isinstance(mapping, tuple) and mapping[0] == 'recolored':
                    output[out_i:out_i+scale_i, out_j:out_j+scale_j] = mapping[1]
            # else: leave as background color (already initialized)
    
    return output    

def non_int_scaling(inp, subtasks):
    # scales = non_eq_scales(subtasks)
    h, w = inp.shape
    middle = subtasks[0].train_out.shape[0] // subtasks[0].train_out.shape[0]
    non_middle = middle + 1
    output = np.zeros(subtasks[0].train_out.shape)
    base_i = 0
    for i in range(h):
        base_j = 0
        if i == middle:
            step_i = middle
        else:
            step_i = non_middle
        for j in range(w):
            color = inp[i, j]     
            if j == middle:
                step_j = middle
            else:
                step_j = non_middle
            output[base_i:base_i+step_i, base_j:base_j+step_j] = color 
            base_j += step_j
        base_i += step_i
    return output.astype(int)


def invert_pattern(pattern, font_color):
    """
    Inverts the pattern: swaps background with foreground.
    """
    inverted = pattern.copy()
    fg_colors = np.unique(pattern[pattern != font_color])
    
    if len(fg_colors) == 0:
        return inverted
    
    fg_color = fg_colors[0] if len(fg_colors) == 1 else fg_colors[0]
    
    inverted[pattern == font_color] = fg_color
    inverted[pattern != font_color] = font_color
    
    return inverted

def create_ranking_dict(array, font_color=0):
    unique_elements, counts = np.unique(array, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Indices for descending frequency
    sorted_elements = list(unique_elements[sorted_indices])
    ranking_dict = {}
    if font_color in sorted_elements:
        sorted_elements.remove(font_color)
    for idx, el in enumerate(sorted_elements):
        value = idx + 1
        if value == len(unique_elements):
            value = "rarest"
        ranking_dict[el] = value
    return ranking_dict

def need_ranking(mappings):
    key_lens = [len(mapping) for mapping in mappings]
    # eq_keys_size = all([all([len(mapping_1) == len(mapping_2) for mapping_1 in mappings]) for mapping_2 in mappings])
    if len(set(key_lens)) > 1:
        max_len = max(key_lens)
        smaller_idxs = [key_len < max_len for key_len in key_lens]
        max_idxs = [key_len == max_len for key_len in key_lens]
        if all([all([set(mappings[smaller_idx].keys()) <= set(mappings[max_idx].keys()) for smaller_idx in smaller_idxs]) for max_idx in max_idxs]):
            return False
    elif len(set(key_lens)) == 1:
        non_font_ranks = [k for mapping in mappings for k,v in mapping.items() if v != 'font']
        if key_lens[0] == 2 and len(set(non_font_ranks)) > 1:
            return True
        else:
            return False
    unique_keys = list(set().union(*mappings))
    for key in unique_keys:
        if not all(key in mapping.keys() for mapping in mappings):
            return True
        curent_val = None
        for mapping in mappings:
            if key in mapping.keys() and curent_val is None:
                curent_val = mapping[key]
            elif key in mapping.keys() and curent_val != mapping[key]:
                return True           
    return False

def non_eq_scales(subtasks):
    scales = []
    for idx, subtask in enumerate(subtasks):
        inp = subtask.train_inp
        out = subtask.train_out
        p = inp.shape[0]
        n = out.shape[0]

        base = n // p
        remainder = n % p
        
        result = [base] * p
        
        # Distribute remainder to ends, keeping middle smallest
        for i in range(remainder):
            if i % 2 == 0:
                result[i // 2] += 1  # Add to left side
            else:
                result[p - 1 - i // 2] += 1  # Add to right side
    
        scales.append(result)
    if all([scale==scales[0] for scale in scales]):
        return scales[0]
    
# COVERING
def solve_pattern_planting(task, font_color=0):
    """
    Solves pattern planting tasks where the input pattern is placed
    in the output (possibly rotated/inverted) following a some planting strategy
    Args:
        task: ARCTask instance with subtasks and test_subtask attributes
        
    Returns:
        numpy.ndarray: The predicted output for test_subtask.train_inp
    """
    subtasks = task.subtasks
    test_subtask = task.test_subtask
    # Analyze training examples to determine planting strategy and scaling rule
    strategy, scaling_rule = analyze_planting_strategy(subtasks, font_color)
    return apply_planting_strategy(test_subtask.train_inp, strategy, scaling_rule, font_color)

def analyze_planting_strategy(subtasks, font_color):
    """
    Analyzes training examples to determine the planting pattern and scaling rule.
    Keeps all possible transformations for each position and filters to consistent ones.
    
    Returns:
        tuple: (strategy, scaling_rule)
            - strategy: list of transformations per position
            - scaling_rule: dict with scaling pattern information
    """
    all_possible_strategies = []
    scaling_rules = []
    
    for idx, subtask in enumerate(subtasks):
        possible_strategies_per_position = []
        inp = subtask.train_inp 
        out = subtask.train_out 
        
        # Calculate number of tiles in each direction
        i_steps = int(out.shape[0] // inp.shape[0])
        j_steps = int(out.shape[1] // inp.shape[1])
        
        # Count unique colors in input
        num_colors = len(np.unique(inp))
        
        # Store scaling rule for this example
        scaling_rules.append({
            'input_shape': inp.shape,
            'output_shape': out.shape,
            'i_steps': i_steps,
            'j_steps': j_steps,
            'num_colors': num_colors
        })
        
        pattern_variants = generate_pattern_variants(inp, font_color)
        
        # Iterate through output grid to find ALL matching patterns per position
        for i in range(i_steps):
            for j in range(j_steps):
                # Calculate block position in output
                i_out = i * inp.shape[0]
                j_out = j * inp.shape[1]
                
                # Extract block from output
                block = out[i_out:i_out+inp.shape[0], j_out:j_out+inp.shape[1]]
                
                # Find ALL variants that match this block
                matching_mods = []
                for mod, var in pattern_variants.items():
                    if np.array_equal(block, var):
                        matching_mods.append(mod)
                
                if len(matching_mods) == 0:
                    matching_mods.append('unknown')
                
                possible_strategies_per_position.append(matching_mods)
        
        all_possible_strategies.append(possible_strategies_per_position)
    
    # Infer scaling rule from training examples
    inferred_scaling = infer_scaling_rule(scaling_rules)
    
    # Find strategies that are consistent across all examples
    if len(all_possible_strategies) == 0:
        return [], inferred_scaling
    
    # Get number of positions
    num_positions = len(all_possible_strategies[0])
    
    # For each position, find transformations that appear in all examples
    consistent_strategy = []
    for pos_idx in range(num_positions):
        # Get possible transformations for this position across all examples
        possible_at_pos = [
            set(example[pos_idx]) 
            for example in all_possible_strategies 
            if pos_idx < len(example)
        ]
        
        if len(possible_at_pos) == 0:
            consistent_strategy.append('unknown')
            continue
        
        # Find intersection - transformations valid in ALL examples
        consistent_mods = possible_at_pos[0]
        for pos_set in possible_at_pos[1:]:
            consistent_mods = consistent_mods.intersection(pos_set)
        
        # Pick the first consistent transformation (or 'unknown' if none)
        if len(consistent_mods) > 0:
            # Prefer 'original' if it's in the set, otherwise pick deterministically
            if 'original' in consistent_mods:
                consistent_strategy.append('original')
            else:
                consistent_strategy.append(sorted(list(consistent_mods))[0])
        else:
            consistent_strategy.append('unknown')
    
    return consistent_strategy, inferred_scaling

def apply_planting_strategy(inp, strategy, scaling_rule, font_color):
    """
    Places pattern sequentially following the detected pattern and scaling rule.
    """
    h_in, w_in = inp.shape
    
    # Count unique colors in input for color-based scaling
    num_colors = len(np.unique(inp))
    
    # Calculate output dimensions based on scaling rule
    i_steps, j_steps = calculate_output_dimensions(inp.shape, scaling_rule, num_colors)
    
    # Create output with correct dimensions
    output = np.zeros((h_in * i_steps, w_in * j_steps), dtype=inp.dtype)
    
    pattern_variants = generate_pattern_variants(inp, font_color)
    
    # If strategy is shorter than needed, cycle/repeat it or use 'original' as default
    total_needed = i_steps * j_steps
    
    idx = 0
    for i in range(i_steps):
        for j in range(j_steps):
            # Calculate position in output
            i_out = i * h_in
            j_out = j * w_in
            
            # Get the transformation for this position
            if len(strategy) > 0:
                # If strategy is shorter, cycle through it
                transform = strategy[idx % len(strategy)]
                if transform in pattern_variants:
                    output[i_out:i_out+h_in, j_out:j_out+w_in] = pattern_variants[transform]
                else:
                    # Fallback to original if transform not found
                    output[i_out:i_out+h_in, j_out:j_out+w_in] = pattern_variants['original']
            else:
                # No strategy available, use original
                output[i_out:i_out+h_in, j_out:j_out+w_in] = pattern_variants['original']
            
            idx += 1
    
    return output

def generate_pattern_variants(pattern, font_color):
    """
    Generates all variants of the pattern: rotations and inversions.
    
    Returns:
        dict: {variant_name: variant_pattern}
    """
    variants = {
        'original': pattern.copy(),
        'rot90': np.rot90(pattern, 1),
        'rot180': np.rot90(pattern, 2),
        'rot270': np.rot90(pattern, 3),
        'flip_h': np.fliplr(pattern),
        'flip_v': np.flipud(pattern),
        'inverted': invert_pattern(pattern, font_color)
    }
    
    return variants

def invert_pattern(pattern, font_color):
    """
    Inverts the pattern: swaps background with foreground.
    """
    inverted = pattern.copy()
    fg_colors = np.unique(pattern[pattern != font_color])
    
    if len(fg_colors) == 0:
        return inverted
    
    fg_color = fg_colors[0] if len(fg_colors) == 1 else fg_colors[0]
    
    inverted[pattern == font_color] = fg_color
    inverted[pattern != font_color] = font_color
    
    return inverted

def infer_scaling_rule(scaling_rules):
    """
    Infers the scaling rule from training examples.
    Tries to find patterns in how input dimensions relate to output dimensions.
    
    Returns:
        dict: Scaling rule configuration
    """
    if len(scaling_rules) == 0:
        return {'type': 'fixed', 'i_steps': 1, 'j_steps': 1}
    
    # Check if all examples have the same fixed scaling
    first_rule = scaling_rules[0]
    if all(rule['i_steps'] == first_rule['i_steps'] and 
           rule['j_steps'] == first_rule['j_steps'] 
           for rule in scaling_rules):
        return {
            'type': 'fixed',
            'i_steps': first_rule['i_steps'],
            'j_steps': first_rule['j_steps']
        }
    
    # Check for color-based scaling (tiles = number of unique colors)
    color_based_total = []
    color_based_per_dim = []
    for rule in scaling_rules:
        num_colors = rule['num_colors']
        total_tiles = rule['i_steps'] * rule['j_steps']
        
        # Check if total tiles equals number of colors
        if total_tiles == num_colors:
            color_based_total.append(True)
        else:
            color_based_total.append(False)
        
        # Check if tiles per dimension equals number of colors
        if rule['i_steps'] == num_colors and rule['j_steps'] == num_colors:
            color_based_per_dim.append(True)
        else:
            color_based_per_dim.append(False)
    
    if all(color_based_total):
        return {'type': 'color_based_total'}
    
    if all(color_based_per_dim):
        return {'type': 'color_based_per_dim'}
    
    # Check for dimension-based scaling (e.g., output tiles = input dimension)
    # Pattern: if input is HxW, output has H tiles vertically and W tiles horizontally
    dimension_based = []
    for rule in scaling_rules:
        h_in, w_in = rule['input_shape']
        if rule['i_steps'] == h_in and rule['j_steps'] == w_in:
            dimension_based.append(True)
        else:
            dimension_based.append(False)
    
    if all(dimension_based):
        return {'type': 'dimension_based'}
    
    # Check for area-based scaling (total tiles = input area)
    area_based = []
    for rule in scaling_rules:
        h_in, w_in = rule['input_shape']
        total_tiles = rule['i_steps'] * rule['j_steps']
        if total_tiles == h_in * w_in:
            area_based.append(True)
        else:
            area_based.append(False)
    
    if all(area_based):
        # Default to square-ish layout
        return {'type': 'area_based'}
    
    # Default to using the first example's scaling
    return {
        'type': 'fixed',
        'i_steps': first_rule['i_steps'],
        'j_steps': first_rule['j_steps']
    }

def calculate_output_dimensions(input_shape, scaling_rule, num_colors=None):
    """
    Calculates output dimensions based on input shape and scaling rule.
    
    Args:
        input_shape: tuple (height, width)
        scaling_rule: dict with scaling rule type and parameters
        num_colors: int, number of unique colors in input (optional)
    
    Returns:
        tuple: (i_steps, j_steps) - number of tiles in each dimension
    """
    h_in, w_in = input_shape
    
    if scaling_rule['type'] == 'fixed':
        return scaling_rule['i_steps'], scaling_rule['j_steps']
    
    elif scaling_rule['type'] == 'color_based_total':
        # Total tiles = number of colors, arrange in square-ish layout
        if num_colors is None:
            return 1, 1
        total_tiles = num_colors
        # Try to make it square-ish
        j_steps = int(np.sqrt(total_tiles))
        while total_tiles % j_steps != 0 and j_steps > 1:
            j_steps -= 1
        i_steps = total_tiles // j_steps
        return i_steps, j_steps
    
    elif scaling_rule['type'] == 'color_based_per_dim':
        # Number of tiles per dimension = number of colors
        if num_colors is None:
            return 1, 1
        return num_colors, num_colors
    
    elif scaling_rule['type'] == 'dimension_based':
        # Number of tiles = input dimensions
        return h_in, w_in
    
    elif scaling_rule['type'] == 'area_based':
        # Total tiles = input area, arrange in square-ish layout
        total_tiles = h_in * w_in
        # Try to make it square-ish
        j_steps = int(np.sqrt(total_tiles))
        while total_tiles % j_steps != 0 and j_steps > 1:
            j_steps -= 1
        i_steps = total_tiles // j_steps
        return i_steps, j_steps
    
    else:
        # Default fallback
        return 1, 1
    
# MIXER
def np_logical_both_not(array_1:np.array, array_2:np.array) -> np.array:
    "Return array where element is False for both arrays."
    return np.logical_and(np.logical_not(array_1), np.logical_not(array_2))

def np_logical_fill(arr_1, arr_2):
    xor = np.logical_xor(arr_1, arr_2)
    if xor.all():
        return np.logical_and(arr_1, arr_2)
    else: 
        return arr_1

logic_funcs = {"AND": np.logical_and,
               "OR": np.logical_or,
               "XOR": np.logical_xor,
               "BNOT": np_logical_both_not,
               "FILL": np_logical_fill}

augs = {
    "ID": lambda x: x,
    "LR": np.fliplr,
    "UD": np.flipud,
    "90": lambda x: np.rot90(x, k=1, axes=(0, 1)),
    "180": lambda x: np.rot90(x, k=2, axes=(0, 1)),
    "270": lambda x: np.rot90(x, k=3, axes=(0, 1))}

class NoAnswer(Exception):
    """Exception raised when there is no answer from agent."""
    def __init__(self, message="Task is not solvable by the agent"):
        self.message = message
        super().__init__(self.message)

class WrongCheck(Exception):
    """Exception raised when answer obtained for previous subtask is nor valid for the next."""
    def __init__(self, message="Contradiction in answer searching"):
        self.message = message
        super().__init__(self.message)

def color_analysis(task:ARCTask, patterns, font_val)->str:
    """Define a type of transformation for test subtask."""
    subtasks = task.subtasks
    flattened_inp_vals = []
    flattened_out_vals = []
    for subtask in subtasks:               
        flattened_inp_vals.extend(subtask.train_inp.flatten().tolist())
        flattened_out_vals.extend(subtask.train_out.flatten().tolist())
    uniq_inp = len(set(flattened_inp_vals))
    uniq_out = len(set(flattened_out_vals))
    if patterns != defaultdict(list, {}):
        uniq_inp -= 1
    if font_val in flattened_inp_vals:
       uniq_inp -= 1 
    if font_val in flattened_out_vals:
       uniq_out -= 1 
    if uniq_inp == uniq_out and uniq_out > 2: # if all colors from inputs are in outputs
        return 'color_mix'
    else:
        return 'logical_ops'

def main_color(grid:np.array, font_val=0, pad_val=10):
    i, j = np.where((grid!=pad_val)*1 *  (grid!=font_val)*1)
    coords = list(zip(i, j))
    return grid[coords[0]]

def arr_diff(arr_1:np.array, arr_2:np.array):
    """Return mask with element 1 in position where the element from array 1 is not equal to the element from array 2.""" 
    shape = arr_1.shape
    i_new, j_new = np.where(arr_1!=arr_2)
    new_coords = list(zip(i_new, j_new))
    new_mask = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i, j ) in new_coords:
                new_mask[i, j] = 1
    return new_mask.astype(bool)

def mixer(task:ARCTask, font_val=0, pad_val=10) -> np.array:
    """Sybmolic agent for solving tasks involving grid segments interraction."""
    test_input = task.test_subtask.train_inp
    segmentation = []
    solution = []
    colors_mapper = {}
    for idx, subtask in enumerate(task.subtasks):
        grid = subtask.train_inp
        subtask_input = subtask.train_inp
        subtask_target = subtask.train_out
        grid_shape = subtask.train_inp_shape
        summary = GridSummary(grid , grid_shape, levels=[3])
        patterns = retrieve_shapes(grid, grid_shape, ('markup', 'partition_lines'), 0)
        if idx == 0:
            transf_type = color_analysis(task, patterns, font_val)
        segments = get_segments(grid, patterns)
        if segments == []:
            continue
        elif transf_type == 'logical_ops':
            segment_color = main_color(segments[0])
            target_color = main_color(subtask_target) # check what color to use after operator application
            colors_mapper[segment_color] = target_color
        try:
            pos_solution = solver(segments, transf_type, solution, subtask_target, font_val, pad_val) # find params for solution or check already found
        except WrongCheck:
            raise NoAnswer
        if pos_solution:
            solution = copy(pos_solution)
    if solution == []:
        raise NoAnswer 
    answer = infer_grid(test_input, transf_type, solution, colors_mapper, font_val=0, pad_val=10) # apply approch tested on training examples
    return answer

def infer_grid(test_input, transf_type, solution, colors_mapper, font_val=0, pad_val=1):
    grid = test_input
    cropped_grid = grid
    grid_shape = cropped_grid.shape
    patterns = retrieve_shapes(grid, grid_shape, ('markup', 'partition_lines'), 0)
    segments = get_segments(grid, patterns)
    
    if transf_type == "logical_ops":  
        res_mask = segments[0]
        segments_color = main_color(res_mask, font_val, pad_val)
        aug_name = solution[0]
        func_name = solution[1]
        aug = augs[aug_name]
        func = logic_funcs[func_name]
        for i in range(1, len(segments)):
            res_mask = func(res_mask, aug(segments[i]))
        target_color = colors_mapper[segments_color]
        answer = res_mask * target_color
    
    elif transf_type == "color_mix":
        segments_masks = [grid!=font_val for grid in segments]
        segment_shape = segments[0].shape
        res_mask_prev = np.zeros(segment_shape)
        answer = np.zeros(segment_shape)
        aug_name = solution[0]
        aug = augs[aug_name]
        perm = list(solution[1])
        res_mask_prev = segments_masks[perm[0]]
        answer += res_mask_prev * segments[perm[0]]
        for idx in perm[1:]:
            res_mask = np.logical_or(res_mask_prev, aug(segments_masks[idx]))
            new_mask = arr_diff(res_mask, res_mask_prev)
            answer += new_mask * aug(segments[idx])
            res_mask_prev = copy(res_mask)
            
    elif transf_type == "conjunction":
        segments_masks = [grid!=font_val for grid in segments]   
        n_segments = len(segments_masks)   
        unique_coords_list = []
        answer = segments[0]
        aug_name = solution[0]
        aug = augs[aug_name]
        aug_segmented_masks = [aug(segment) for segment in segments_masks]
        segments_colors = solution[1]
        for i in range(n_segments):
            segment = segments[i]
            j = (i+1) % (n_segments)
            unique_mask = aug_segmented_masks[i] != aug_segmented_masks[j]
            seg_i, seg_j = np.where(unique_mask)
            unique_coords = list(zip(seg_i, seg_j))
            for coord in unique_coords:
                if segment[coord] != font_val:
                    answer[coord] = segments_colors[i]
    return answer.astype(int) 

def logical_transf_search(segments:List[np.array], target:np.array, solution, font_val, pad_val):
    """Try different logical operations on segments if solution was not found before. Otherwise check given solution."""
    target_color = main_color(target)
    segments_masks = [grid!=font_val for grid in segments]
    if solution != []:
        res_mask = segments_masks[0]
        aug_name = solution[0]
        func_name = solution[1]
        aug = augs[aug_name]
        func = logic_funcs[func_name]
        for i in range(1, len(segments)):
            res_mask = func(res_mask, aug(segments_masks[i]))
        answer = res_mask * target_color
        if np.equal(answer, target).all():
            return (aug_name, func_name)
    for func_name, func in logic_funcs.items(): 
        res_mask = segments_masks[0]
        for aug_name, aug in augs.items():
            if aug(res_mask).shape == target.shape:
                for i in range(1, len(segments)):
                    res_mask = func(res_mask, aug(segments_masks[i]))
                answer = res_mask * target_color
                if np.equal(answer, target).all(): 
                    return (aug_name, func_name)
    return False

def color_mix_search(segments:List[np.array], target:np.array, solution:List[str], font_val, pad_val):
    """Try different order of segment addition if solution was not found before. Otherwise check given solution."""   
    segments_masks = [grid!=font_val for grid in segments]
    segment_shape = segments[0].shape
    res_mask_prev = np.zeros(segment_shape)
    if solution != []:
        answer = np.zeros(segment_shape)
        aug_name = solution[0]
        aug = augs[aug_name]
        perm = solution[1]
        res_mask_prev = segments_masks[perm[0]]
        answer += res_mask_prev * segments[perm[0]]
        for idx in perm[1:]:
            res_mask = np.logical_or(res_mask_prev, aug(segments_masks[idx]))
            new_mask = arr_diff(res_mask, res_mask_prev)
            answer += new_mask * aug(segments[idx])
            res_mask_prev = copy(res_mask)
        if np.equal(answer, target).all():
            return (aug_name, perm)
    for perm in list(permutations(range(len(segments)))):
        perm = list(perm)
        for aug_name, aug in augs.items():
            answer = np.zeros(segment_shape)
            res_mask_prev = segments_masks[perm[0]]
            if aug(res_mask_prev).shape == target.shape:
                answer += res_mask_prev * segments[perm[0]]
                for idx in perm[1:]:
                    res_mask = np.logical_or(res_mask_prev, aug(segments_masks[idx]))
                    new_mask = arr_diff(res_mask, res_mask_prev)
                    answer += new_mask * aug(segments[idx])
                    res_mask_prev = copy(res_mask)
                if np.equal(answer, target).all():
                    return (aug_name, perm)                   
    return False

def conjunction_search(segments:List[np.array], target:np.array, solution:List[str], font_val, pad_val):
    """Find specific color for uncommon elements for each segment."""   
    segments_masks = [grid!=font_val for grid in segments]   
    n_segments = len(segments_masks)   
    unique_coords_list = []
    if solution != []:
        aug_name = solution[0]
        aug = augs[aug_name]
        aug_segmented_masks = [aug(segment) for segment in segments_masks]
        segments_colors = solution[1]
        answer = aug(segments[0])
        for i in range(n_segments):
            segment = aug(segments[i])
            j = (i+1) % n_segments
            unique_mask = aug_segmented_masks[i] != aug_segmented_masks[j]
            seg_i, seg_j = np.where(unique_mask)
            unique_coords = list(zip(seg_i, seg_j))
            for coord in unique_coords:
                if segment[coord] != font_val:
                    answer[coord] = segments_colors[i]
        if np.equal(answer, target).all():
            return (aug_name, segments_colors)
    for aug_name, aug in augs.items():
        if aug(segments[0]).shape == target.shape and aug_name == "ID":
            aug_segmented_masks = [aug(segment) for segment in segments_masks]
            segments_colors = []
            for i in range(n_segments):
                segment_colors = []
                j = (i+1) % n_segments
                unique_mask = aug_segmented_masks[i] != aug_segmented_masks[j]
                seg_i, seg_j = np.where(unique_mask)
                unique_coords = list(zip(seg_i, seg_j))
                for coord in unique_coords:
                    if aug(target)[coord] != font_val and aug(segments[i])[coord] != font_val:
                        segment_colors.append(aug(target)[coord])
                if len(set(segment_colors)) != 1:
                    break
                else:
                    segments_colors.append(segment_colors[0])
                    unique_coords_list.append(unique_coords)
                    correct_aug = aug_name      
    if segments_colors != []:
        answer = augs[correct_aug](segments[0])
        for i in range(n_segments):
            segment = augs[correct_aug](segments[i])
            for coord in unique_coords_list[i]:
                if segment[coord] != font_val:
                    answer[coord] = segments_colors[i]
        if np.equal(answer, target).all():
            return (correct_aug, segments_colors)
        else:
            return False
    else:
        return False
        
def solver(segments:List[np.array], transf_type:str, solution:List[str], target:np.array, font_val, pad_val):
    """Route segments to proper function for answer search."""
    if transf_type == "logical_ops": 
        return logical_transf_search(segments, target, solution, font_val, pad_val)
    elif transf_type == "color_mix":
        return color_mix_search(segments, target, solution, font_val, pad_val)
    elif transf_type == "conjunction":
        return conjunction_search(segments, target, solution, font_val, pad_val)        
    else:
        raise ValueError(f'Unsupported transformation type:{transf_type}. Expected: "logical_ops" or "color_mix"')

def homog_colored(segment:np.array):
    return True if len(set(segment.flatten().tolist()))==2 else False

def get_segments(grid:np.array, markups:Dict[str, List[List[tuple]]]):
    unpadded_grid = grid
    shape = unpadded_grid.shape
    part_coords = []
    segments = []
    if markups['markup'] != []: # if matrix pattern was identidied
        markups = markups['markup']
        ul = find_upper_left_corner(shape)
        markup_i_coords, markup_j_coords = coords_transform(markups[0])
        n_lines = sum(np.array(markup_i_coords)==ul[0])
        n_segments = n_lines + 1
        step_i = shape[0] // n_segments
        step_j = shape[1] // n_segments
        i_offset = -1
        for i in range(n_segments):
            i_offset += 1
            j_offset = 0
            for j in range(n_segments):
                segments.append(unpadded_grid[i*step_i+i_offset:(i+1)*step_i+i_offset, j*step_j+j_offset:(j+1)*step_j+j_offset])
                j_offset += 1    
    elif markups['partition_lines'] != []: # if lines pattern was identified
        markups = markups['partition_lines']
        # tuple_of_objects = [tuple(sublist.coords) for sublist in markups]
        # markups = [list(coords) for coords in set(tuple_of_objects)]
        cur_coord = 0
        for idx, markup in enumerate(markups):
            dim = 0 if markup[0][0] == markup[1][0] else 1
            part_coords.append(markup[0][dim])
        part_coords.sort()
        if dim == 0:
            for part_coord in part_coords:
                segment = grid[cur_coord:part_coord, :]
                segments.append(segment)
                cur_coord = part_coord + 1
            segments.append(grid[cur_coord:, :])
        else:
            for part_coord in part_coords:
                segment = grid[:,cur_coord:part_coord]
                segments.append(segment)
                cur_coord = part_coord + 1
            segments.append(grid[:,cur_coord:])
    else:
        proposed_segments = []
        break_flag = False
        if shape[0] == shape[1] and shape[0] >= 4: # if split matrix-like without given pattern
           for n_segments in range(2, shape[0]//2):
               step = int(shape[0] / 2)
               if shape[0] % n_segments == 0:
                   for i in range(n_segments):
                       if break_flag:
                           break_flag = False
                           break 
                       else:
                           for j in range(n_segments):
                               segment = unpadded_grid[i*step:(i+1)*step, j*step:(j+1)*step]
                               if homog_colored(segment):
                                   proposed_segments.append(segment)
                               else:
                                   proposed_segments = []
                                   break_flag = True
        else:
            dim = 0 if shape[0] > shape[1] else 1 # if split with lines pattern without given pattern
            for n_segments in range(2, (shape[dim]//2)+1):
                cache = []
                if shape[dim] % n_segments == 0:
                    step = int(shape[dim] / n_segments)
                    for i in range(n_segments):
                        segment = unpadded_grid[i*step:(i+1)*step, :] if dim==0 else unpadded_grid[:, i*step:(i+1)*step]
                        if homog_colored(segment):
                            cache.append(segment)
                        else:
                            break
                    if i == n_segments-1 and cache != []:
                        proposed_segments.append(cache)
            if len(proposed_segments) > 1:
                lens = [len(el) for el in proposed_segments]
                idx = lens.index(min(lens))
                proposed_segments = proposed_segments[idx]
            else:
                proposed_segments = proposed_segments[0] if len(proposed_segments) > 0 else proposed_segments
        segments = proposed_segments
    return segments 

# COLOR RESTORE
def symmetry_patch_filler(task:ARCTask, font_val=0, pad_val=10):
    train_subtask = deepcopy(task.subtasks[-1])
    test_subtask = deepcopy(task.test_subtask)
    train_inp_grid = train_subtask.train_inp
    train_out_grid = train_subtask.train_out
    test_inp_grid = test_subtask.train_inp
    test_out_grid = test_subtask.train_out
    shape_correspondence = train_inp_grid.shape == train_out_grid.shape
    restored_grid = copy(test_inp_grid)
    if not shape_correspondence: # if output is patch - restore output with this patch for training purposes
        patch = find_connected_components_with_color(train_inp_grid, font_val)[0]
        i_1, i_2, j_1, j_2 = segment2slice(patch)
        train_inp_grid[i_1:i_2, j_1:j_2] = train_out_grid
        train_out_grid = train_inp_grid
    symmetry = check_symmetry(train_out_grid)
    if symmetry:
        restored_grid = restore_with_symmetry(grid=test_inp_grid, symmetry_type=symmetry, font_val=font_val)
    else:   
        symmetry_shape = find_symmetry_shape(train_out_grid)
        if symmetry_shape:
            i_1, i_2, j_1, j_2 = symmetry_shape
            symmetry_type = check_symmetry(train_out_grid[i_1:i_2, j_1:j_2])
            if symmetry_type == "lr&ud":
                restored_grid = restore_with_edges(restored_grid, symmetry_shape, font_val=font_val)
            restored_patch = restore_with_symmetry(grid=test_inp_grid[i_1:i_2, j_1:j_2], symmetry_type=symmetry_type,  font_val=font_val) 
            restored_grid[i_1:i_2, j_1:j_2] = restored_patch
    if not shape_correspondence:
        test_patch = find_connected_components_with_color(test_inp_grid, font_val)[0]
        i_1, i_2, j_1, j_2 = segment2slice(test_patch)
        restored_grid = restored_grid[i_1:i_2, j_1:j_2]
    return restored_grid

def restore_with_slices(grid:np.array, symmetry_type:str, font_val=0.0):
    shape = grid.shape
    mid_i = shape[0] // 2 
    max_i = shape[0]
    mid_j = shape[1] // 2
    max_j = shape[1]  
    restored_grid = copy(grid)
    for _ in range(5):
        # left-right slices
        for j in range(mid_j):
            left_slice_verif = None
            right_slice_verif = None
            # left
            left_slice = copy(restored_grid)[:, :mid_j-j]
            if font_val not in left_slice:
                left_slice_verif = left_slice
            # right 
            right_slice = copy(restored_grid)[:, mid_j+j:]
            if font_val not in right_slice:
                right_slice_verif = right_slice   
            # copying pattern if have the biggest slice
            if left_slice_verif is not None and right_slice_verif is None:
                if symmetry_type == "lr&ud":
                    restored_grid[:, mid_j+j:] = np.fliplr(left_slice) 
                    restored_grid[:mid_j-j, :] = np.rot90(left_slice, k=1, axes=(1,0))
                    restored_grid[mid_j+j:, :] = np.rot90(left_slice, k=1, axes=(0,1))
                elif symmetry_type == "lr":
                    restored_grid[:, mid_j+j:] = np.fliplr(left_slice) 
            if left_slice_verif is None and right_slice_verif is not None:
                if symmetry_type == "lr&ud":
                    restored_grid[:, :mid_j-j] = np.fliplr(right_slice) 
                    restored_grid[:mid_j-j, :] = np.rot90(right_slice, k=1, axes=(0,1))
                    restored_grid[mid_j+j:, :] = np.rot90(right_slice, k=1, axes=(1,0))
                elif symmetry_type == "lr":
                    restored_grid[:, :mid_j-j] = np.fliplr(right_slice) 
            left_slice_verif = None
            right_slice_verif = None
        # top-bottom slices
        for i in range(mid_i):
            top_slice_verif = None
            bottom_slice_verif = None
            # top
            top_slice = copy(restored_grid)[:mid_i-i, :]
            if font_val not in top_slice:
                top_slice_verif = top_slice
            # bottom
            bottom_slice = copy(restored_grid)[mid_i+i:, :]
            if font_val not in bottom_slice:
                bottom_slice_verif = bottom_slice   
            # copying pattern if have the biggest slice
            if top_slice_verif is not None and bottom_slice_verif is None:
                if symmetry_type == "lr&ud":
                    restored_grid[mid_i+i:, :] = np.flipud(top_slice) 
                    restored_grid[:, mid_i+i:] = np.rot90(top_slice, k=1, axes=(1,0))
                    restored_grid[:, :mid_i-i] = np.rot90(top_slice, k=1, axes=(0,1))
                elif symmetry_type == "ud":
                    restored_grid[mid_i+i:, :] = np.flipud(top_slice) 
            if top_slice_verif is None and bottom_slice_verif is not None:
                if symmetry_type == "lr&ud":
                    restored_grid[:mid_i-i, :] = np.flipud(bottom_slice) 
                    restored_grid[:, mid_i+i:] = np.rot90(bottom_slice, k=1, axes=(0,1))
                    restored_grid[:, :mid_i-i] = np.rot90(bottom_slice, k=1, axes=(1,0))
                elif symmetry_type == "ud":
                    restored_grid[:, :mid_j-j] = np.flipud(bottom_slice)    
            top_slice_verif = None
            bottom_slice_verif = None       
    return restored_grid

def find_symmetry_shape(grid:np.array, max_slice:int=4):
    increments = [i for i in range(max_slice)]
    negative_increments = [-i for i in range(max_slice)]
    max_i, max_j = grid.shape
    top_left_increment_tuples = list(product(increments, increments))[1:]
    top_right_increment_tuples = list(product(increments, negative_increments))[1:]
    bottom_left_increment_tuples = list(product(negative_increments, increments))[1:]
    bottom_right_increment_tuples = list(product(negative_increments, negative_increments))[1:]   
    for incr in top_left_increment_tuples:
       i_1 = incr[0]
       j_1 = incr[1]
       i_2 = max_i
       j_2 = max_j
       cropped_grid = copy(grid)[i_1:i_2, j_1:j_2]
       if check_symmetry(cropped_grid) == "lr&ud":
           return (i_1, i_2, j_1, j_2)
    for incr in top_right_increment_tuples:
       i_1 = incr[0]
       j_1 = 0
       i_2 = max_i
       j_2 = max_j + incr[1]
       cropped_grid = copy(grid)[i_1:i_2, j_1:j_2]
       if check_symmetry(cropped_grid) == "lr&ud":
           return (i_1, i_2, j_1, j_2)
    for incr in bottom_left_increment_tuples:
       i_1 = 0
       j_1 = incr[1]
       i_2 = max_i - incr[0]
       j_2 = max_j
       cropped_grid = copy(grid)[i_1:i_2, j_1:j_2]
       if check_symmetry(cropped_grid) == "lr&ud":
           return (i_1, i_2, j_1, j_2)
    for incr in bottom_right_increment_tuples:
       i_1 = 0
       j_1 = 0
       i_2 = max_i - incr[0]
       j_2 = max_j - incr[1]
       cropped_grid = copy(grid)[i_1:i_2, j_1:j_2]
       if check_symmetry(cropped_grid) == "lr&ud":
           return (i_1, i_2, j_1, j_2)
    return False 

def restore_with_symmetry(grid:np.array, symmetry_type:str, font_val=0, n_full_sects=0):
    shape = grid.shape
    mid_i = shape[0] // 2 
    max_i = shape[0]
    mid_j = shape[1] // 2
    max_j = shape[1]
    restored_grid = copy(grid)
    # analize halves    
    h_left = (0, max_i, 0, mid_j)
    h_top = (0, mid_i, 0, max_j)
    h_right = (0, max_i, mid_j, max_j)
    h_bottom =  (mid_i, max_i, 0, max_j)
    hs = [h_left, h_top, h_right, h_bottom] # keep halves coordinates
    h_idxs = []
    halves = []
    for idx in range(4):
        half = grid[hs[idx][0]:hs[idx][1], hs[idx][2]:hs[idx][3]]
        if font_val not in half:
            h_idxs.append(idx) # keep position of the half
            halves.append(half)
    # analize quarters        
    q_0 = (0, mid_i, 0, mid_j)
    q_1 = (0, mid_i, mid_j, max_j)
    q_2 = (mid_i, max_i, mid_i, max_j)
    q_3 = (mid_i, max_i, 0, mid_j)
    qs = [q_0, q_1, q_2, q_3] # keep quarters coordinates
    q_idxs = []   
    quarters = []
    for idx in range(4):
        quartet = grid[qs[idx][0]:qs[idx][1], qs[idx][2]:qs[idx][3]]
        if font_val not in quartet:
            q_idxs.append(idx) # keep position of the quarter
            quarters.append(quartet)
            
    if n_full_sects == len(h_idxs) + len(q_idxs): # if initial number of fully colored segments unchanged 
        restored_grid = restore_with_slices(grid=restored_grid, symmetry_type=symmetry_type, font_val=font_val) # try to restore iteratively from center based om symmetry type 
        if (restored_grid == grid).all(): # if unchanged - failed to restore
            raise NoAnswer
        else:  #  otherwise proceed with restoration
            restore_with_symmetry(grid=restored_grid, symmetry_type=symmetry_type, font_val=font_val, n_full_sects=len(h_idxs)+len(q_idxs))
    else: 
        n_full_sects = len(h_idxs) + len(q_idxs) # keep number of fully colored segments
    # try to use specified type of symmetry
    if symmetry_type == "lr&ud": 
        if q_idxs != []: # try to use fully colored quarters 
            q_idx = q_idxs[0]
            quarter = quarters[0]
            seq = [0, 1, 2, 3][q_idx:] + [0, 1, 2, 3][:q_idx] # take first and replace the remainder by rotating 
            for i, idx in enumerate(seq):
                coords = qs[idx]
                i_1, i_2, j_1, j_2 = coords
                restored_grid[i_1:i_2, j_1:j_2] = np.rot90(quarter, k=i, axes=(1,0))
        elif h_idxs != []: # otherwise try to use fully colored halves
            h_idx = h_idxs[0]
            half = halves[0]
            seq = [0, 1, 2, 3][h_idx:] + [0, 1, 2, 3][:h_idx] # take first and replace the remainder  by rotating 
            for i, idx in enumerate(seq):
                coords = hs[idx]
                i_1, i_2, j_1, j_2 = coords
                restored_grid[i_1:i_2, j_1:j_2] = np.rot90(half, k=i, axes=(1,0))   
            
    elif symmetry_type == "lr":
        if q_idxs != []: # try to use fully colored quarters 
            q_idx = q_idxs[0]
            quarter = quarters[0]
            seq = [0, 1, 2, 3][q_idx:] + [0, 1, 2, 3][:q_idx] # take first and replace the remainder  by rotating 
            for i, idx in enumerate(seq):
                coords = qs[idx]
                i_1, i_2, j_1, j_2 = coords
                restored_grid[i_1:i_2, j_1:j_2] = np.rot90(quarter, k=i, axes=(1,0))
        elif h_idxs != [] and h_idxs[0] in [0, 2]: # otherwise try to use suitable fully colored halves
            h_idx = h_idxs[0]
            half = halves[0]
            idx = 2 if h_idx == 0 else 0
            coords = hs[idx]
            i_1, i_2, j_1, j_2 = coords
            restored_grid[i_1:i_2, j_1:j_2] = np.rot90(half, k=2, axes=(1,0))
        if font_val in restored_grid: # if need to proceed
           restore_with_symmetry(grid, symmetry_type, symmetric_shape, font_val, n_full_sects)
            
    elif symmetry_type == "ud":
        if q_idxs != []:  # try to use fully colored quarters  
            q_idx = q_idxs[0]
            quarter = quarters[0]
            seq = [0, 1, 2, 3][q_idx:] + [0, 1, 2, 3][:q_idx] # take first and replace the remainder  
            for i, idx in enumerate(seq):
                coords = qs[idx]
                i_1, i_2, j_1, j_2 = coords
                restored_grid[i_1:i_2, j_1:j_2] = np.rot90(quarter, k=i, axes=(1,0))
        elif h_idxs != [] and h_idxs[0] in [1, 3]: # otherwise try to use suitable fully colored halves
            h_idx = h_idxs[0]
            half = halves[0]
            idx = 1 if h_idx == 3 else 1
            coords = hs[idx]
            i_1, i_2, j_1, j_2 = coords
            restored_grid[i_1:i_2, j_1:j_2] = np.rot90(half, k=2, axes=(1,0))
        if font_val in restored_grid: # if need to proceed
           restore_with_symmetry(grid, symmetry_type, symmetric_shape, font_val, n_full_sects)
    if font_val in restored_grid: # failed to restore
        raise NoAnswer
    else:
        return restored_grid

def restore_with_edges(restored_grid:np.array, symmetry_shape:tuple, font_val):
    min_i = 0
    min_j = 0
    max_i, max_j = restored_grid.shape
    patch_min_i, patch_max_i, patch_min_j, patch_max_j = symmetry_shape 
    offset = (patch_min_i-min_i, max_i-patch_max_i, patch_min_j-min_j, max_j-patch_max_j)
    segments = []
    full_segment = -1
    # if offset == 0 then segment at the edge amd we can't use corresponding information
    if offset[0] > 0 and offset[2] > 0:
        edges_type = 'top_left'
        top = restored_grid[0:patch_min_i, :]
        left = restored_grid[:, 0:patch_min_j]
        if font_val not in top:
            restored_grid[:, 0:patch_min_j] = np.fliplr(np.rot90(top, k=1, axes=(1,0)))
        elif font_val not in left:
            restored_grid[0:patch_min_i, :] = np.fliplr(np.rot90(left, k=1, axes=(1,0)))
    elif offset[0] > 0 and offset[3] > 0:
        edges_type = 'top_right'
        top = restored_grid[0:patch_min_i, :]
        right = restored_grid[:, patch_max_j:]
        if font_val not in top:
            restored_grid[:, 0:patch_min_j] = np.rot90(top, k=1, axes=(1,0))
        elif font_val not in right:
            restored_grid[0:patch_min_i, :] = np.rot90(right, k=1, axes=(0,1))
    elif offset[1] > 0 and offset[2] > 0:
        edges_type = 'bottom_left'
        bottom = restored_grid[patch_max_i:, :]
        left = restored_grid[:, 0:patch_min_j]
        if font_val not in bottom:
            restored_grid[:, 0:patch_min_j] = np.rot90(bottom, k=1, axes=(1,0))
        elif font_val not in left:
            restored_grid[0:patch_min_i, :] = np.rot90(left, k=1, axes=(0,1))
    elif offset[1] > 0 and offset[3] > 0:
        edges_type = 'bottom_right'
        bottom = restored_grid[patch_max_i:, :]
        right = restored_grid[:, patch_max_j:]
        if font_val not in bottom:
            restored_grid[:, 0:patch_min_j] = np.fliplr(np.rot90(bottom, k=1, axes=(1,0)))
        elif font_val not in right:
            restored_grid[0:patch_min_i, :] = np.fliplr(np.rot90(right, k=1, axes=(1,0)))
    return restored_grid 

def segment2slice(coords:List[tuple]):
    max_i, max_j = 0, 0 
    min_i, min_j = 30, 30
    for coord in coords:
        max_i = max(max_i, coord[0])
        max_j = max(max_j, coord[1])
        min_i = min(min_i, coord[0])
        min_j = min(min_j, coord[1])
    return min_i, max_i+1, min_j, max_j+1

def check_symmetry(grid:np.array):
    shape = grid.shape
    if shape[0] % 2 != 0 or shape[1] % 2 != 0:
        return False
    mid_i = shape[0] // 2 
    mid_j = shape[1] // 2
    lr = np.equal(np.fliplr(grid[:, :mid_j]), grid[:, mid_j:]).all()
    ud = np.equal(np.flipud(grid[:mid_i, :]), grid[mid_i:, :]).all()
    if lr and ud:
        return "lr&ud"
    elif lr:
        return "lr"
    elif ud:
        return "ud"
    else:
        return False