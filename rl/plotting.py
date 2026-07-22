
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rl.utils import crop_pad, get_step_description, grid_formatting


def plot_rollout_grid_trace(rollout, num_steps_to_plot=None, action_mapping=None, 
                            figsize=(15, 10), include_descriptions=True, include_info=True):
    """
    Plots the grid states from a rollout to monitor a trace with textual descriptions.
    
    Args:
        rollout (dict): a dictionary with keys ['observations', 'actions', 'rewards', 'dones', 'infos', 'total_reward', 'length'].
        where 'observations' contains states with 'grid' attributes representing grid states.
        num_steps_to_plot (int), optional: number of steps to plot. If None, plots all steps.
        plot_rewards (bool), optional: whether to plot rewards alongside the grids.
        figsize (tuple), optional: figure size (width, height).
        include_descriptions (bool), optional: whether to include textual descriptions of each step.
        
    Returns:
        fig (matplotlib.figure.Figure): the figure containing the grid trace plots.
    """
    # Extract the grid states and other data from the rollout
    states = rollout['observations']
    grids = [state['grid'] for state in states if 'grid' in state]
    actions = rollout['actions'] if 'actions' in rollout else []
    rewards = rollout['rewards'] if 'rewards' in rollout else []
    infos = rollout['infos'] if 'infos' in rollout else {}
    
    # Determine number of steps to plot
    total_steps = len(grids)
    if num_steps_to_plot is None:
        num_steps_to_plot = total_steps
    else:
        num_steps_to_plot = min(num_steps_to_plot, total_steps)
    
    # Create evenly spaced indices if we're not plotting all steps
    if num_steps_to_plot < total_steps:
        indices = np.linspace(0, total_steps - 1, num_steps_to_plot, dtype=int)
        grids = [grids[i] for i in indices]
        rewards_to_plot = [rewards[i] if i < len(rewards) else None for i in indices]
        actions_to_plot = [actions[i] if i < len(actions) else None for i in indices]
    else:
        indices = range(num_steps_to_plot)
        rewards_to_plot = rewards[:num_steps_to_plot]
        actions_to_plot = actions[:num_steps_to_plot]
    
    # Determine grid dimensions for plotting
    n_cols = min(5, num_steps_to_plot)
    n_rows = (num_steps_to_plot + n_cols - 1) // n_cols
    
    # Create figure and GridSpec to organize subplots
    fig = plt.figure(figsize=figsize)
    
    # Calculate row heights based on whether we're including descriptions and rewards
    row_heights = []
    for _ in range(n_rows):
        row_heights.append(3)  # Grid height
        if include_descriptions:
            row_heights.append(1)  # Description height
    
    gs = GridSpec(len(row_heights), n_cols, figure=fig, height_ratios=row_heights)
    cmap = matplotlib.colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff', '#002f1f'])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=11)
    # Plot each grid state
    for i in range(num_steps_to_plot):
        row, col = divmod(i, n_cols)
        row_idx = row * (1 + int(include_descriptions))
        
        # Plot the grid
        ax_grid = fig.add_subplot(gs[row_idx, col])
        grid = grids[i]
        grid = crop_pad(grid_formatting(grid))
        im = ax_grid.imshow(grid, cmap=cmap, norm=norm)
        ax_grid.set_title(f'Step {indices[i]}, Reward: {rewards_to_plot[i] if rewards_to_plot[i] is not None else 0:.2f}')
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        # Add textual description if enabled
        if include_descriptions:
            desc_row = row_idx + 1 
            ax_desc = fig.add_subplot(gs[desc_row, col])
            
            # Get step info
            
            step_idx = indices[i]
            observation = states[step_idx] if step_idx < len(states) else None
            action = actions_to_plot[i] if i < len(actions_to_plot) else None
            reward = rewards_to_plot[i] if i < len(rewards_to_plot) else None
            
            # Get info for this step
            step_info = {k:v for k, v in infos[step_idx].items() if k!='TimeLimit.truncated'}
            if not include_info:
                step_info = None
            # Create description text
            description = get_step_description(step_idx, observation, action, 
                                             reward if reward is not None else 0, 
                                             action_mapping, step_info)
            
            # Remove axis ticks and labels
            ax_desc.set_xticks([])
            ax_desc.set_yticks([])
            ax_desc.set_frame_on(False)
            
            # Add description text
            ax_desc.text(0.5, 1.9, description, 
                       ha='center', va='top', 
                       fontsize=6, wrap=True)
    
    # Add a main title with rollout information
    plt.suptitle(f'Rollout Grid Trace (Total Reward: {sum(rewards):.2f}, Length: {len(actions)})', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

def plot_grids_comparison(grid_1, grid_2, target_grid=None):
    # Ensure the arrays are 2D
    if grid_1.ndim != 2 or grid_2.ndim != 2:
        raise ValueError("Both arrays must be 2D.")

    grid_1 = crop_pad(grid_formatting(grid_1))
    grid_2 = crop_pad(grid_formatting(grid_2))    
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cmap = matplotlib.colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    # Plot the first grid
    axes[0, 0].imshow(grid_1, cmap=cmap, norm=norm)
    axes[0, 0].set_title('Grid 1')
    axes[0, 0].set_xticks(np.arange(-0.5, grid_1.shape[1], 1), minor=True)
    axes[0, 0].set_yticks(np.arange(-0.5, grid_1.shape[0], 1), minor=True)
    axes[0, 0].grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Plot the second grid
    axes[1, 0].imshow(grid_2, cmap=cmap, norm=norm)
    axes[1, 0].set_title('Grid 2')
    axes[1, 0].set_xticks(np.arange(-0.5, grid_2.shape[1], 1), minor=True)
    axes[1, 0].set_yticks(np.arange(-0.5, grid_2.shape[0], 1), minor=True)
    axes[1, 0].grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Find the cells in the second grid that are not in the first grid
    unique_cells = np.setdiff1d(grid_2, grid_1)
    
    # Create a mask for the unique cells
    mask = np.isin(grid_2, unique_cells)
    
    # Create a new grid with the same shape as array2, filled with zeros
    unique_grid = np.zeros_like(grid_2, dtype=np.int32)
    
    # Set the unique cells to 1 (or any other value to highlight them)
    unique_grid[mask] = grid_2[mask]
    
    # Plot the unique cells grid
    axes[0, 1].imshow(unique_grid, cmap=cmap, norm=norm)
    axes[0, 1].set_title('New Cells in Grid 2')
    axes[0, 1].set_xticks(np.arange(-0.5, unique_grid.shape[1], 1), minor=True)
    axes[0, 1].set_yticks(np.arange(-0.5, unique_grid.shape[0], 1), minor=True)
    axes[0, 1].grid(which='minor', color='w', linestyle='-', linewidth=1)

    if target_grid is not None:
        target_grid = crop_pad(grid_formatting(target_grid)) 
        axes[1, 1].imshow(target_grid, cmap=cmap, norm=norm)
        axes[1, 1].set_title('Target grid')
        axes[1, 1].set_xticks(np.arange(-0.5, target_grid.shape[1], 1), minor=True)
        axes[1, 1].set_yticks(np.arange(-0.5, target_grid.shape[0], 1), minor=True)
        axes[1, 1].grid(which='minor', color='w', linestyle='-', linewidth=1)

    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')
    
    plt.tight_layout()
    plt.show()

def plot_rewards(path_to_logs:str):
    file = pd.read_csv(path_to_logs)
    plt.plot(file['time/total_timesteps'], file['rollout/ep_rew_mean'], label='Training mean reward')
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.legend()
    if os.path.exists(path_to_logs):
          os.remove(path_to_logs)
    plt.savefig(os.getcwd()+'/plot.png')
    plt.show()
    plt.close('all')
    return