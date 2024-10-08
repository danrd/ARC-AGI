import matplotlib.pyplot as plt
from matplotlib import colors
import copy

training_challenges = load_json('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json')
training_solutions = load_json('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json')
evaluation_challenges = load_json('/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json('/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json')
test_challenges = load_json('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json')
tasks_keys = list(training_challenges.keys())+list(evaluation_challenges.keys())
all_challenges = training_challenges | evaluation_challenges 

def plot_task(task_id):
    """Plots the train and test pairs of a specified task, using same color scheme as the ARC app."""   
    try:
        task = training_challenges[task_id]
        task_solution = training_solutions[task_id][0]
    except Exception:
        task = evaluation_challenges[task_id]
        task_solution = evaluation_solutions[task_id][0]   
    num_train = len(task['train'])
    num_test  = len(task['test'])
    w = num_train + num_test
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'Task #{task_id}', fontsize=20, fontweight='bold', y=1)
    
    for j in range(num_train):     
        plot_one(axs[0, j], j, task, 'train', 'input')
        plot_one(axs[1, j], j, task, 'train', 'output')        
    
    plot_one(axs[0, j+1], 0, task, 'test', 'input')

    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
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
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])   
    ax.set_title(train_or_test + ' ' + input_or_output, fontweight='bold')
            
def plot_grids(grids):
    for plot_idx, grid in enumerate(grids):
        plt.subplot(1, len(grids), plot_idx + 1)
        plot_grid(grid)
            
def plot_grid(grid):
    grid = np.array(grid)
    cmap = colors.ListedColormap(
        ['#FFFFFF', '#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=-1, vmax=9)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5) 
    plt.xticks(np.arange(-0.5, grid.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid.shape[0]), [])
    plt.xlim(-0.5, grid.shape[1]-0.5)    
    
def evaluate_grid(correct_grid, predicted_grids):
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
    grid = np.zeros((30,30))
    for coord in shape:
        grid[coord] = 2
    plot_grid(grid)
    
def plot_intersection(grid, figure):
    i, j = coords_tranform(figure)
    grid = copy.deepcopy(grid)
    old_color = grid[i, j][0]
    new_color =  old_color-1 if old_color>1 else old_color+1
    grid[i, j] = new_color
    plot_grid(grid)
    
def plot_intersection_with_replace(grid, figure):
    i, j = coords_tranform(figure)
    old_color = grid[i, j][0]
    new_color =  old_color-1 if old_color>1 else old_color+1
    grid[i, j] = new_color
    plot_grid(grid)