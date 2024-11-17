import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import copy
import os
import pandas as pd
from typing import List
from utils.utils import load_json
from symbolic.utils import coords_transform, find_upper_left_corner

training_challenges = load_json('data/dataset/training_challenges.json')
training_solutions = load_json('data/dataset/training_solutions.json')
evaluation_challenges = load_json('data/dataset/evaluation_challenges.json')
evaluation_solutions = load_json('data/dataset/evaluation_solutions.json')
test_challenges = load_json('data/dataset/test_challenges.json')
tasks_keys = list(training_challenges.keys())+list(evaluation_challenges.keys())
all_challenges = training_challenges | evaluation_challenges 

def plot_task(task_id):
    """Plots the train and test pairs of a specified task, using same color scheme as the ARC app."""   
    task = all_challenges[task_id]
    task_solution = all_challenges[task_id][0]
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
            
def plot_grid(grid):
    grid = np.array(grid)
    i, j = np.where(grid!=1.0)
    i_size = max(i) - min(i) + 1
    j_size = max(j) - min(j) + 1
    grid_without_pad = grid[coords_transform(list(zip(i, j)))].reshape((i_size, j_size))*10
    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40', '#FFDC00', '#AAAAAA', 
                                 '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    plt.imshow(grid_without_pad, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5) 
    plt.xticks(np.arange(-0.5, grid_without_pad.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid_without_pad.shape[0]), [])
    plt.xlim(-0.5, grid_without_pad.shape[1]-0.5)    
 
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
        grid[coord] = 0.2
    plot_grid(grid)
    
def plot_intersection(grid:np.array, shape:List[tuple]):
    """Plot intersection with defined shape."""
    grid = copy.deepcopy(grid)
    i, j =  np.where(grid!=0.0)
    colors = grid[i, j] # identify all colors to use different color for intersection
    new_color = 0
    for c in range(1, 10,):
        if c/10 not in colors:
            new_color = c/10
            break
    i, j = coords_transform(shape)
    grid[i, j] = new_color
    i, j = np.where(grid!=10.0)
    i_shape = max(i) - min(i) + 1
    j_shape = max(j) - min(j) + 1
    grid_without_pad = grid[coords_transform(list(zip(i, j)))].reshape((i_shape, j_shape)) # exclude padding
    plot_grid(grid_without_pad)
    
def plot_intersection_with_replace(grid:np.array, shape:List[tuple], grid_size:tuple):
    """Plot intersection for several figeres at once."""
    i, j =  np.where(grid!=0.0)
    colors = grid[i, j] # identify all colors to use different color for intersection
    new_color = 0
    for c in range(1, 10,):
        if c/10 not in colors:
            new_color = c/10
            break
    i, j = coords_transform(shape)
    grid[i, j] = new_color
    i, j = np.where(grid!=10.0)
    i_shape = max(i) - min(i) + 1
    j_shape = max(j) - min(j) + 1
    grid_without_pad = grid[coords_transform(list(zip(i, j)))].reshape((i_shape, j_shape)) # exclude padding
    plot_grid(grid_without_pad)
    
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