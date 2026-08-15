import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
from copy import deepcopy
import os
import pandas as pd
from typing import List, Union
from data.datasets.ARC.arc_dataset import ARCDataset
from symbolic.utils import coords_transform, grid_formatting, crop_pad

# The official ARC-AGI palette (background + 9 colors) plus a white
# "padding" marker at index 10 - shared by every grid-plotting function
# below instead of each keeping its own copy (they used to: 5 separate
# inline ListedColormap literals, all meant to be the same palette).
ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA',
              '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff']
ARC_CMAP = colors.ListedColormap(ARC_COLORS)
ARC_NORM = colors.Normalize(vmin=0, vmax=len(ARC_COLORS) - 1)

# Same palette plus one more marker color (index 11), used to draw an
# arbitrary highlighted shape/intersection on top of a grid.
ARC_COLORS_HIGHLIGHT = ARC_COLORS + ['#002f1f']
ARC_CMAP_HIGHLIGHT = colors.ListedColormap(ARC_COLORS_HIGHLIGHT)
ARC_NORM_HIGHLIGHT = colors.Normalize(vmin=0, vmax=len(ARC_COLORS_HIGHLIGHT) - 1)

# Match/mismatch/shape-mismatch overlay for plot_grids_comparison's
# correctness panel - not an ARC color grid, so it gets its own 3-color map.
DIFF_COLORS = ['#FF4136', '#2ECC40', '#AAAAAA']  # wrong, correct, shape mismatch
DIFF_CMAP = colors.ListedColormap(DIFF_COLORS)
DIFF_NORM = colors.Normalize(vmin=0, vmax=len(DIFF_COLORS) - 1)


def plot_grid(grid, ax=None, cmap=ARC_CMAP_HIGHLIGHT, norm=ARC_NORM_HIGHLIGHT):
    """Draw one grid with ARC's color scheme and cell gridlines.

    Draws on `ax` if given, else the current axes. Nothing is shown/saved
    here - that's the caller's decision (plt.show()/savefig()/
    wandb.Image(fig), ...), same as every other function in this module.
    """
    grid = crop_pad(grid_formatting(grid))
    if ax is None:
        ax = plt.gca()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, grid.shape[1]), [])
    ax.set_yticks(np.arange(-0.5, grid.shape[0]), [])
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    return ax


def plot_task(task_id:str, dataset:ARCDataset):
    """Plots the train and test pairs of a specified task, using same color scheme as the ARC app."""
    all_challenges = dataset.training_challenges
    all_solutions = dataset.training_solutions
    if "_" in task_id:
        main,test_idx = task_id.split('_')
        task = all_challenges[main]
        task_solutions = all_solutions[main]
    else:
        task = all_challenges[task_id]
        task_solutions = all_solutions[task_id]
    num_train = len(task['train'])
    num_test  = len(task['test'])
    w = num_train + num_test
    # squeeze=False: plt.subplots(2, w) collapses to a 1D array when w == 1,
    # breaking every axs[row, col] index below.
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2), squeeze=False)
    plt.suptitle(f'Task #{task_id}', fontsize=20, fontweight='bold', y=1)

    j = -1  # so j+1+inc below still starts test columns at 0 when num_train == 0
    for j in range(num_train):
        plot_one(axs[0, j], j, task, 'train', 'input')
        plot_one(axs[1, j], j, task, 'train', 'output')

    for inc in range(num_test):
        answer = task_solutions[inc]
        plot_one(axs[0, j+1+inc], 0+inc, task, 'test', 'input')
        plot_grid(np.array(answer), ax=axs[1, j+1+inc], cmap=ARC_CMAP, norm=ARC_NORM)
        axs[1, j+1+inc].set_xticklabels([])
        axs[1, j+1+inc].set_yticklabels([])
        axs[1, j+1+inc].set_title(f'Test {inc+1} output', fontweight='bold')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()
    return fig

def plot_one(ax, i, task, train_or_test, input_or_output):
    """Auxilary function for plot_task function."""
    input_matrix = task[train_or_test][i][input_or_output]
    plot_grid(np.array(input_matrix), ax=ax, cmap=ARC_CMAP, norm=ARC_NORM)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    title_prefix = f'Example {i+1}' if train_or_test == "train" else f'Test {i+1}'
    ax.set_title(title_prefix + ' ' + input_or_output, fontweight='bold')

def plot_multiple_tasks(task_ids: List[str], dataset: ARCDataset):
    """Plots the training and test pairs for multiple tasks, each in its own figure,
    using the same color scheme as the ARC app.
    Args:
        task_ids (list[str]): List of task IDs to plot.
        dataset (ARCDataset): The dataset containing training challenges and solutions.
    """
    for task_id in task_ids:
        print(task_id)
        plot_task(task_id, dataset)
        plt.show()

def plot_multiple_grids(grids: List[np.array]):
    """Plots each grid from given list, one grid per figure.

    Args:
        grids (List[np.array]): List of grids.

    Returns:
        List[matplotlib.figure.Figure]: one figure per grid.
    """
    figs = []
    for grid in grids:
        fig = plt.figure()
        plot_grid(grid)
        figs.append(fig)
    return figs

def plot_preds(predictions: List[tuple], task_idxs: List[int], dataset):
    """Plots tiplet input_grid-predicton-output_grid in a single row.

    Args:
        predictions (List[List[np.array, float]]): List of prediction grids with similarity score.
        task_idxs (List[int]): List task idxs.
        dataset: ARC dataset

    Raises:
        ValueError: If the lengths of the input lists differ.

    Returns:
        List[matplotlib.figure.Figure]: one figure per prediction.
    """
    if len(predictions) != len(task_idxs):
        raise ValueError("Prediction and target grids lists must have the same length.")
    input_grids = [dataset.tasks[idx].test_subtask.train_inp for idx in task_idxs]
    prediction_grids = [pred[0] for pred in predictions]
    target_grids = [dataset.tasks[idx].test_subtask.train_out for idx in task_idxs]
    n = len(prediction_grids)
    figs = []
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        plot_grid(input_grids[i], ax=axes[0])
        axes[0].set_title(f"Task {task_idxs[i]} input")

        plot_grid(prediction_grids[i], ax=axes[1])
        axes[1].set_title(f"Prediction with similarity {predictions[i][1]}")

        plot_grid(target_grids[i], ax=axes[2])
        axes[2].set_title(f"Task {task_idxs[i]} target")

        plt.tight_layout()
        figs.append(fig)
    return figs

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
    min_i, min_j = min(i), min(j)
    i_shape = max(i) - min_i + 1
    j_shape = max(j) - min_j + 1
    grid = np.zeros((i_shape, j_shape))
    i_shifted = [i_coord-min_i for i_coord in i]
    j_shifted = [j_coord-min_j for j_coord in j]
    shifted_shape = list(zip(i_shifted, j_shifted))
    for coord in shifted_shape:
        grid[coord] = 11
    return plot_grid(grid)

def plot_intersection(grid:np.array, shape:Union[List[tuple], List[List[tuple]]]):
    """Plot intersection with defined shape."""
    grid = deepcopy(grid)
    # A single shape (List[tuple]) and a list of shapes (List[List[tuple]])
    # are both `list` at the top level - isinstance(shape, list) alone
    # can't tell them apart. shape[0] can: a tuple means a single shape,
    # a list means shape is already a list of shapes to flatten.
    if shape and isinstance(shape[0], list):
        shape_union = []
        for sh in shape:
            shape_union.extend(sh)
        shape = shape_union
    i, j = coords_transform(shape)
    grid[i, j] = 11
    return plot_grid(grid)

def plot_rewards(path_to_logs:str):
    """Plot rewards for RL agent."""
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

def _correctness_overlay(predicted: np.ndarray, reference: np.ndarray):
    """Per-cell match (1, green) / mismatch (0, red) between `predicted`
    and `reference`. A shape mismatch is reported as-is (2, grey) rather
    than silently cropped/padded to compare - a wrong output shape is
    itself the finding for an ARC prediction, not something to paper over.

    Returns (overlay_grid, is_fully_correct).
    """
    if predicted.shape != reference.shape:
        return np.full(reference.shape, 2), False
    match = predicted == reference
    return match.astype(int), bool(match.all())

def plot_grids_comparison(grid_1, grid_2, target_grid=None):
    """Compare two grids side by side, plus a per-cell correctness overlay
    (green=match, red=mismatch, grey=shape mismatch): grid_2 against
    target_grid if given, else grid_2 against grid_1 directly. Typical use:
    grid_1=model prediction, target_grid=the actual answer.
    """
    # Ensure the arrays are 2D
    if grid_1.ndim != 2 or grid_2.ndim != 2:
        raise ValueError("Both arrays must be 2D.")

    grid_1 = crop_pad(grid_formatting(grid_1))
    grid_2 = crop_pad(grid_formatting(grid_2))
    reference = crop_pad(grid_formatting(target_grid)) if target_grid is not None else grid_1

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plot_grid(grid_1, ax=axes[0, 0], cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0, 0].set_title('Grid 1')

    plot_grid(grid_2, ax=axes[1, 0], cmap=ARC_CMAP, norm=ARC_NORM)
    axes[1, 0].set_title('Grid 2')

    overlay, is_correct = _correctness_overlay(grid_2, reference)
    axes[0, 1].imshow(overlay, cmap=DIFF_CMAP, norm=DIFF_NORM)
    axes[0, 1].grid(True, which='both', color='white', linewidth=0.5)
    axes[0, 1].set_xticks(np.arange(-0.5, overlay.shape[1]), [])
    axes[0, 1].set_yticks(np.arange(-0.5, overlay.shape[0]), [])
    against = "target" if target_grid is not None else "Grid 1"
    axes[0, 1].set_title(f"Grid 2 vs {against}: {'MATCH' if is_correct else 'mismatch'}")

    if target_grid is not None:
        plot_grid(reference, ax=axes[1, 1], cmap=ARC_CMAP, norm=ARC_NORM)
        axes[1, 1].set_title('Target grid')
    else:
        axes[1, 1].axis('off')

    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()
    return fig

def plot_task_result(task, predicted_grid, eval_result=None, raw_text=None):
    """The plot_preds format (input / prediction-with-similarity / target,
    one row) applied to a single (task, prediction) pair instead of a
    dataset-indexed batch - task input, the model's prediction, and the
    known target, each its own panel. If `task` carries an `index`
    attribute (e.g. ARCDataset stamps one on - the 0..N-1 numbering
    task.id itself doesn't carry), panel titles lead with that instead of
    the bare id, same as plot_preds' own "Task {idx}" convention.

    `raw_text`: the model's raw generation, shown in the prediction panel
    instead of a blank box when `predicted_grid` is None (didn't parse
    into a grid) - otherwise there's nothing to look at to tell a genuine
    non-answer from a near-miss format the parser just doesn't handle yet.

    Unlike plot_task_with_prediction, this reads straight off an in-memory
    ARCTask (task.test_subtask) - no ARCDataset lookup by id needed, so it
    works from anywhere a task object is already at hand (e.g.
    subsymbolic.arc_evaluators.arc_result_plotter, mid-inference).
    """
    task_id = getattr(task, "id", getattr(task, "label", "?"))
    task_index = getattr(task, "index", None)
    task_number = task_index if task_index is not None else task_id

    test_input = np.array(task.test_subtask.train_inp)
    target_grid = np.array(task.test_subtask.train_out)
    predicted = np.asarray(predicted_grid) if predicted_grid is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_grid(test_input, ax=axes[0], cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title(f"Task {task_number} input")

    if predicted is not None and predicted.ndim == 2:
        plot_grid(predicted, ax=axes[1], cmap=ARC_CMAP, norm=ARC_NORM)
        similarity = eval_result.primary_score if eval_result is not None else None
        axes[1].set_title(f"Prediction with similarity {similarity}")
    else:
        axes[1].set_title("Prediction did not parse into a grid")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        for spine in axes[1].spines.values():
            spine.set_color("lightgrey")
        axes[1].set_facecolor("#fdf2f2")
        if raw_text:
            import textwrap
            # Wrap each original line on its own - textwrap.wrap(whole_text)
            # treats '\n' as ordinary whitespace and collapses it, which
            # flattens exactly the row breaks that matter most for a
            # botched grid attempt.
            wrapped_lines = []
            for line in raw_text.split("\n"):
                wrapped_lines.extend(textwrap.wrap(line, width=48) or [""])
            max_lines = 22
            if len(wrapped_lines) > max_lines:
                wrapped_lines = wrapped_lines[:max_lines] + ["... (truncated)"]
            axes[1].text(0.5, 0.5, "\n".join(wrapped_lines), transform=axes[1].transAxes,
                         ha="center", va="center", fontsize=7, family="monospace", color="#555555")

    plot_grid(target_grid, ax=axes[2], cmap=ARC_CMAP, norm=ARC_NORM)
    axes[2].set_title(f"Task {task_number} target")

    plt.tight_layout()
    return fig


def plot_task_with_prediction(task_id: str, dataset: ARCDataset, predicted_grid: np.ndarray, test_idx: int = 0):
    """The two views someone reviewing "did we solve this task" needs
    together: the whole task (plot_task) plus how the prediction for
    test[test_idx] compares to the actual answer, cell by cell
    (plot_grids_comparison - grid_2 is the one checked against target_grid,
    so the prediction has to be grid_2, not grid_1).

    Returns (task_figure, comparison_figure).
    """
    fig_task = plot_task(task_id, dataset)

    all_challenges = dataset.training_challenges
    all_solutions = dataset.training_solutions
    main = task_id.split('_')[0] if "_" in task_id else task_id
    test_input = np.array(all_challenges[main]['test'][test_idx]['input'])
    target = np.array(all_solutions[main][test_idx])

    fig_comparison = plot_grids_comparison(test_input, np.array(predicted_grid), target_grid=target)
    return fig_task, fig_comparison

def plot_objects(grid: np.array, objects: List, colormap_name='gist_ncar', max_distinct_colors=50):
    """Enhanced version with better color distinction for many objects.
    Fixed grid display to prevent cutting off and ensure even cell sizes.
    """
    grid = deepcopy(grid)

    # Prepare the plot
    fig = plt.figure(figsize=(12, 10))

    object_values = {}
    legend_handles = []

    n_objects = len(objects)

    # Generate distinct colors using multiple strategies
    def generate_distinct_colors(n_colors):
        if n_colors <= 20:
            # Use tab20 colormap for small numbers - most distinct
            cmap = plt.colormaps['tab20']
            return [cmap(i % 20) for i in range(n_colors)]
        elif n_colors <= max_distinct_colors:
            # Combine multiple qualitative colormaps
            colors_list = []
            qualitative_maps = ['tab20', 'Set3', 'Pastel1', 'Pastel2', 'Dark2', 'Paired']
            for cmap_name in qualitative_maps:
                cmap = plt.colormaps[cmap_name]
                colors_list.extend([cmap(i) for i in range(cmap.N)])
                if len(colors_list) >= n_colors:
                    break
            return colors_list[:n_colors]
        else:
            # For very large numbers, generate colors with varied hue,
            # saturation and brightness rather than sampling colormaps -
            # spreading hue by the golden ratio keeps adjacent objects
            # visually distinct even in the hundreds.
            colors_list = []

            import colorsys
            for i in range(n_colors):
                # Vary hue systematically, but also vary saturation and value
                hue = (i * 0.618033988749895) % 1.0  # Golden ratio conjugate for distribution
                saturation = 0.6 + 0.3 * (i % 3) / 3  # Vary saturation
                value = 0.7 + 0.2 * (i % 4) / 4      # Vary brightness
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors_list.append(rgb)

            return colors_list

    # Get distinct colors for all objects
    distinct_colors = generate_distinct_colors(n_objects)

    # Process each object
    for idx, obj in enumerate(objects):
        obj_value = 12 + idx
        color = distinct_colors[idx]
        hex_color = colors.to_hex(color)
        object_values[obj_value] = (obj.label, hex_color)

        coords = obj.coords
        for coord in coords:
            x, y = coord
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = obj_value

        legend_handles.append(mpatches.Patch(color=hex_color, label=obj.label))

    # Create colormap
    all_colors = ARC_COLORS_HIGHLIGHT + [object_values[val][1] for val in sorted(object_values.keys())]

    cmap = colors.ListedColormap(all_colors)
    norm = colors.Normalize(vmin=0, vmax=len(all_colors)-1)

    height, width = grid.shape

    # Create coordinate arrays for pcolormesh
    # pcolormesh needs the edges of each cell, so we create arrays from -0.5 to dim+0.5
    x_edges = np.arange(-0.5, width + 0.5, 1)
    y_edges = np.arange(-0.5, height + 0.5, 1)

    # Create meshgrid for the edges
    X, Y = np.meshgrid(x_edges, y_edges)

    # Plot with pcolormesh - this ensures square cells
    plt.pcolormesh(X, Y, grid, cmap=cmap, norm=norm, edgecolor='lightgrey', linewidth=0.5)

    # Set equal aspect ratio to ensure square cells
    plt.gca().set_aspect('equal')

    # Set limits to show entire grid
    plt.xlim(-0.5, width - 0.5)
    plt.ylim(-0.5, height - 0.5)

    # Set ticks at integer positions
    plt.xticks(np.arange(0, width, 1))
    plt.yticks(np.arange(0, height, 1))

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    # Enhanced legend for many objects
    if n_objects > 30:
        n_cols = max(2, min(6, n_objects // 10))
        legend_fontsize = 'x-small' if n_objects > 50 else 'small'
    else:
        n_cols = max(2, min(4, n_objects // 15))
        legend_fontsize = 'small'

    plt.legend(handles=legend_handles,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               ncol=n_cols,
               fontsize=legend_fontsize,
               framealpha=0.9)

    plt.tight_layout()
    return fig

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
