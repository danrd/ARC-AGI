import random
import numpy as np
from symbolic.utils import crop_pad, adjust_grid_shape
from rl.ARC_task import ARCTask

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: A function that takes the progress (from 0 to 1) and returns the learning rate.
    """
    def func(progress: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return initial_value + (final_value - initial_value) * (1 - progress)

    return func

def vote_for_cell(votes, tie_break='random'):
    """
    Given a 1D array of integer votes, return the most frequent value.
    
    Parameters:
    - votes (np.array): 1D array of integers.
    - tie_break (str): 'lowest' to choose the smallest value in a tie,
                       'random' to randomly choose among tied values.
                       
    Returns:
    - int: The voted value for this cell.
    """
    # Count the frequency of each vote
    # Using np.unique with return_counts gives us the unique values and their frequencies
    values, counts = np.unique(votes, return_counts=True)
    
    # Find the maximum frequency
    max_count = counts.max()
    
    # Find the values that have the maximum frequency
    candidates = values[counts == max_count]
    
    if len(candidates) == 1:
        return candidates[0]
    else:
        # In case of a tie, choose based on the tie_break policy
        if tie_break == 'lowest':
            return candidates.min()
        elif tie_break == 'random':
            return random.choice(candidates.tolist())
        else:
            raise ValueError(f"Unknown tie_break strategy: {tie_break}")

def vote_grid(arrays, tie_break='random'):
    """
    Perform cell-wise voting over a list of numpy arrays.
    
    Parameters:
    - arrays (list of np.array): A list of 2D numpy arrays of the same shape.
    - tie_break (str): Tie-breaker strategy. Options: 'lowest' or 'random'.
    
    Returns:
    - np.array: A 2D array of the same shape where each cell is the voted value.
    """
    if not arrays:
        raise ValueError("The list of arrays is empty.")
    
    # Ensure all arrays have the same shape
    shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError("All arrays must have the same shape.")
    
    # Stack arrays into a single 3D array of shape (N, H, W)
    stacked = np.stack(arrays, axis=0)
    
    # Prepare an output array of shape (H, W)
    H, W = shape
    voted_grid = np.empty(shape, dtype=arrays[0].dtype)
    
    # Iterate over each cell position and apply the voting function
    for i in range(H):
        for j in range(W):
            # Extract the votes for cell (i, j) from all arrays
            cell_votes = stacked[:, i, j]
            voted_grid[i, j] = vote_for_cell(cell_votes, tie_break=tie_break)
    
    return voted_grid

def get_vec_info(vec_env):
    right_placements = []
    max_ints = []
    agent_pos = []
    envs = vec_env.envs
    for env in envs:
        right_placements.append(env.subtask.right_placement)
        max_ints.append(env.subtask.max_int)
    return right_placements, max_ints

def get_position(obs_pos_vector):
    positions = []
    for pos in obs_pos_vector:
        positions.append(pos.nonzero())
    return positions

def repad(subtask, max_shape=(15,15), pad_val=-0.1):
    subtask.train_inp = adjust_grid_shape(crop_pad(subtask.train_inp, pad_val=1), 
                                                    target_shape=max_shape, pad_value=pad_val, normalize=False)
    subtask.train_out = adjust_grid_shape(crop_pad(subtask.train_out, pad_val=1), 
                                          target_shape=max_shape, pad_value=pad_val, normalize=False)
    return subtask

def define_padding(task:ARCTask):
    """Define max padding for specific task"""
    max_i = 0
    max_j = 0
    for subtask in task.subtasks:
        inp_shape = subtask.train_inp_shape
        out_shape = subtask.train_out_shape
        if inp_shape[0] > max_i:
            max_i = inp_shape[0]
        if inp_shape[1] > max_j:
            max_j = inp_shape[1]
        if out_shape[0] > max_i:
            max_i = out_shape[0]
        if out_shape[1] > max_j:
            max_j = out_shape[1]
    return (max_i, max_j)

def calculate_eval_freq(n_envs, total_steps, n_evaluations):
    """Calculate evaluation frequency value based on number of environments, training steps and desired number of evaluations during training."""
    return (total_steps//n_evaluations//n_envs)

def solution_description(actions, env):
    direction2name = {'N': 'north',
                      'NE': 'north-east',
                      'E': 'east',
                      'SE': 'south-east',
                      'S': 'south',
                      'SW': 'south-west',
                      'W': 'west',
                      'NW': 'north-west'
                     }
    env = vec_env.envs[0]
    sorted_actions = []
    actions_dict = env.action_name_to_idx
    idx2name = {v:k for k,v in actions_dict.items()}
    all_actions =  [action[0] for action in actions]
    action_names = list(set([idx2name[action] for action in all_actions]))
    description = ""
    for idx, action in enumerate(actions):
        if action[1] == action[2]:      
            description += f'{idx2name[action[0]].replace("_", " ")} for object_{action[1]}'
        else:
            description += f'{idx2name[action[0]].replace("_", " ")} for object_{action[1]} and object_{action[2]}'
        if idx != len(actions) - 1:
            description += " -> "
    for k, v in direction2name.items():
        description = description.replace(k, direction2name[k])
    return description 

def define_feasible_actions(action_types, colors, directions, color_dependent_actions, double_color_dependent_actions, direction_dependent_actions):
    """Creates the dict of all possible actions {idx:action}."""
    final_action_list = []
    for action in action_types:
        if action in color_dependent_actions:
            # Create variants for each feasible color
            colored_actions = [f"{color}_{action}" for color in colors]
            if action in direction_dependent_actions:
                colored_actions = [f"{action}_{direction}" for action in colored_actions for direction in directions]
            if action in double_color_dependent_actions:
                if action in ["emission_with_recolor_collision", "emission_with_contour_collision", "emission_with_object_recolor"]:
                   colored_actions =  [f'{action.split("_with_")[0]}_with_{color}_{action.split("_with_")[1]}' 
                                       for action in colored_actions for color in colors]
                else:
                    colored_actions = [f"{action}_{color}" for action in colored_actions for color in colors] 
            final_action_list.extend(colored_actions)
        elif action in direction_dependent_actions and action not in color_dependent_actions:
            # Create variants for each feasible color
            directed_actions = [f"{action}_{direction}" for direction in directions]
            final_action_list.extend(directed_actions)
        elif action != "submit": # Exclude submit if handled separately
             final_action_list.append(action)
    # Add the submit action if it's intended to be part of the main action space
    final_action_list.append("submit")

    # Create mapping from index to action name string
    actions_dict = {idx: name for idx, name in enumerate(final_action_list)}       
    return actions_dict

def get_action_description(action, action_mapping):
    """
    Generate a textual description of an action.
    
    Args:
        action (array-like): 5-dimensional action array [action_type, row, col, offset_row, offset_col].
        action_mapping (dict): dictionary mapping action numbers to action names, e.g., {'copy': 0}.
        
    Returns:
        str: Textual description of the action
    """
    if action is None:
        return "Unknown action"
    
    action_type, object_1_idx, object_2_idx = action
    action_type = action_mapping[action_type]
    if object_1_idx == object_2_idx:
        # Generate description
        description = f"{action_type} for {object_1_idx}"

    else:
        # Generate description
        description = f"{action_type} between {object_1_idx} and {object_2_idx}"    
    return description

def get_step_description(step_idx, observation, action, reward, action_mapping, info=None):
    """
    Generate a textual description for a step in the rollout.
    
    Args:
        step_idx (int): the index of the step in the rollout.
        observation (dict): the observation at this step, including grid state.
        action (array-like): the 5-dimensional action taken
        reward (float): the reward received.
        action_mapping (dict): dictionary mapping action numbers to action names.
        info (dict), optional: additional information about the step.
        
    Returns:
        str: textual description of the step.
    """
    action_desc = get_action_description(action, action_mapping)
    
    description = f"Step {step_idx}: {action_desc}"
    
    # Add any additional info if available
    if info and isinstance(info, dict):
        for key, value in info.items():
            if key != 'action_mapping' and not isinstance(value, (dict, list, np.ndarray)) and value is not None:
                description += f"\n{key}: {value}"
    
    return description

def task_colors(task, colors_mapping):
    uniq_colors = []
    for subtask in task.subtasks:
        subtask_vals = np.unique(np.vstack([crop_pad(subtask.train_inp, pad_val=10), crop_pad(subtask.train_out, pad_val=10)]))
        uniq_colors.extend([colors_mapping[val] for val in subtask_vals if val!=0])
    test_vals = np.unique(crop_pad(subtask.train_inp, pad_val=10))
    uniq_colors.extend([colors_mapping[val] for val in test_vals if val!=0])   
    return uniq_colors