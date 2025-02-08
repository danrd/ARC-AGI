import random
import numpy as np
from symbolic.utils import crop_pad, adjust_grid_shape

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

def repad(task, max_shape=(15,15)):
    for subtask in task.subtasks:
        subtask.train_inp = adjust_grid_shape(crop_pad(subtask.train_inp, pad_val=1), target_shape=max_shape, pad_value=-0.1, normalize=False)
        subtask.train_out = adjust_grid_shape(crop_pad(subtask.train_inp, pad_val=1), target_shape=max_shape, pad_value=-0.1, normalize=False)
    task.test_subtask.train_inp = adjust_grid_shape(crop_pad(task.test_subtask.train_inp, pad_val=1), 
                                                    target_shape=max_shape, pad_value=-0.1, normalize=False)
    task.test_subtask.train_out = adjust_grid_shape(crop_pad(task.test_subtask.train_out, pad_val=1), target_shape=max_shape, pad_value=-0.1, normalize=False)
    return task