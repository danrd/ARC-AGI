import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import copy
import os
import pandas as pd
from typing import List, Union
from utils.utils import load_json
from data.dataset.ARC_dataset import ARCDataset
from symbolic.utils import coords_transform, grid_formatting, crop_pad

def plot_task(task_id:str, dataset:ARCDataset):
    """Plots the train and test pairs of a specified task, using same color scheme as the ARC app."""   
    all_challenges = dataset.training_challenges
    all_solutions = dataset.training_solutions
    task = all_challenges[task_id]
    task_solution = all_solutions[task_id][0]
    num_train = len(task['train'])
    num_test  = len(task['test'])
    w = num_train + num_test
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'Task #{task_id}', fontsize=20, fontweight='bold', y=1)
    
    for j in range(num_train):     
        plot_one(axs[0, j], j, task, 'train', 'input')
        plot_one(axs[1, j], j, task, 'train', 'output')        
    
    plot_one(axs[0, j+1], 0, task, 'test', 'input')

    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    answer = task_solution
    
    axs[1, j+1].imshow(answer, cmap=cmap, norm=norm)
    axs[1, j+1].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    axs[1, j+1].set_yticks([x-0.5 for x in range(1 + len(answer))])
    axs[1, j+1].set_xticks([x-0.5 for x in range(1 + len(answer[0]))])     
    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1].set_title('TEST OUTPUT', color = 'green', fontweight='bold')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    plt.show()  
    
    print()
    print()
     
def plot_one(ax, i, task, train_or_test, input_or_output):
    """Auxilary function for plot_task function."""
    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])   
    ax.set_title(train_or_test + ' ' + input_or_output, fontweight='bold')

def plot_multiple_tasks(task_ids: List[str], dataset: ARCDataset):
    """
    Plots the training and test pairs for multiple tasks, each in its own figure,
    using the same color scheme as the ARC app.
    Args:
        task_ids (list[str]): List of task IDs to plot.
        dataset (ARCDataset): The dataset containing training challenges and solutions.
    """
    for task_id in task_ids:
        print(task_id)
        plot_task(dataset.tasks[task_id].label, dataset)

def plot_grid(grid):
    grid = crop_pad(grid_formatting(grid))
    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff', '#002f1f'])
    norm = colors.Normalize(vmin=0, vmax=11)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5) 
    plt.xticks(np.arange(-0.5, grid.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid.shape[0]), [])
    plt.xlim(-0.5, grid.shape[1]-0.5)  

def plot_multiple_grids(grids: List[np.array]):
    """
    Plots each grid from given list.
    Args:
        grids (List[np.array]): List of grids.
    """
    for grid in grids:
        plot_grid(grid)  

def plot_preds(predictions: List[tuple], task_idxs: List[int], dataset):
    """
    Plots tiplet input_grid-predicton-output_grid in a single row.
    
    Args:
        predictions (List[List[np.array, float]]): List of prediction grids with similarity score.
        task_idxs (List[int]): List task idxs.
        dataset: ARC dataset
    
    Raises:
        ValueError: If the lengths of the input lists differ.
    """
    if len(predictions) != len(task_idxs):
        raise ValueError("Prediction and target grids lists must have the same length.")
    input_grids =[crop_pad(dataset.tasks[idx].test_subtask.train_inp, pad_val=1) for idx in task_idxs]
    prediction_grids =[pred[0] for pred in predictions]
    target_grids = [crop_pad(dataset.tasks[idx].test_subtask.train_out, pad_val=1) for idx in task_idxs]
    n = len(prediction_grids)
    for i in range(n):
        # Create a figure with one row and two columns
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        
        # Plot prediction grid on the left
        plt.sca(axes[0])  # Set current axis to the first subplot
        plot_grid(input_grids[i])
        plt.title(f"Task {task_idxs[i]} input")

        # Plot prediction grid on the left
        plt.sca(axes[1])  # Set current axis to the first subplot
        plot_grid(prediction_grids[i])
        plt.title(f"Prediction with similarity {predictions[i][1]}")
        
        # Plot target grid on the right
        plt.sca(axes[2])  # Set current axis to the second subplot
        plot_grid(target_grids[i])
        plt.title(f"Task {task_idxs[i]} target")
        
        plt.tight_layout()
        plt.show() 
        if n == 1:
            return fig   
 
def evaluate_grid(correct_grid, predicted_grids):
    """Calculate metrics based on predicted grid and correct grid."""
    correct_grid = np.array(correct_grid)
    metrics = dict(accuracy=0, correct_pixels=0, correct_size=0, unanswered=(2 - len(predicted_grids))/2)
    for predicted_grid in predicted_grids:
        predicted_grid = np.array(predicted_grid)
        if correct_grid.shape == predicted_grid.shape:
            metrics['accuracy'] = max(metrics['accuracy'], np.all(predicted_grid == correct_grid))
            metrics['correct_pixels'] = max(metrics['correct_pixels'], np.mean(predicted_grid == correct_grid))
            metrics['correct_size'] = max(metrics['correct_size'], correct_grid.shape == predicted_grid.shape)
    return metrics     

def plot_shape(shape:List[tuple]):
    """Plot a figure which is a list of tuples with coordinates."""
    i, j = coords_transform(shape)
    min_coord = min(min(i), min(j))
    i_shape = max(i) - min(i) + 1
    j_shape = max(j) - min(j) + 1
    grid = np.zeros((i_shape, j_shape))
    i_shifted = [i_coord-min_coord for i_coord in i]
    j_shifted = [j_coord-min_coord for j_coord in j]
    shifted_shape = list(zip(i_shifted, j_shifted))
    for coord in shifted_shape:
        grid[coord] = 11
    plot_grid(grid)
    
def plot_intersection(grid:np.array, shape:Union[List[tuple], List[List[tuple]]]):
    """Plot intersection with defined shape."""
    grid = copy.deepcopy(grid)
    if type(shape) == list:
        shape_union = []
        for sh in shape:
            shape_union.extend(sh)
            shape = shape_union
    i, j = coords_transform(shape)
    grid[i, j] = 11
    grid = crop_pad(grid_formatting(grid))
    plot_grid(grid)
    
def plot_rewards(path_to_logs:str):
    """Plot rewards for RL agent."""
    file = pd.read_csv(path_to_logs)
    plt.plot(file['time/total_timesteps'], file['rollout/ep_rew_mean'], label=f'Training mean reward')
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.legend()
    if os.path.exists(path_to_logs)==True:
          os.remove(path_to_logs)
    plt.savefig(os.getcwd()+'/plot.png')
    plt.show()
    plt.close('all')
    return

def plot_grids_comparison(grid_1, grid_2, target_grid=None):
    # Ensure the arrays are 2D
    if grid_1.ndim != 2 or grid_2.ndim != 2:
        raise ValueError("Both arrays must be 2D.")

    grid_1 = crop_pad(grid_formatting(grid_1))
    grid_2 = crop_pad(grid_formatting(grid_2))    
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    
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
class TaskIterator:
    def __init__(self, start=0, end=0, tasks_keys=False):
        self.current = start
        self.end = end
        self.tasks_keys = tasks_keys
        if tasks_keys:
            self.end = len(tasks_keys)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.end:
            if self.tasks_keys:
                value = self.tasks_keys[self.current]
            else:
                value = self.current
            self.current += 1
            return value
        else:
            raise StopIteration