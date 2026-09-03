import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from rl.utils import crop_pad, get_step_description, grid_formatting
from utils.plotting import ARC_CMAP_HIGHLIGHT, ARC_NORM_HIGHLIGHT, plot_grids_comparison, plot_rewards  # noqa: F401 - plot_grids_comparison/plot_rewards re-exported, RL-specific callers import them from here


def _select_step_indices(total_steps, num_steps_to_plot, rewards, prioritize_eventful):
    """Which of the 0..total_steps-1 real steps to actually draw.

    Uniform spacing (np.linspace) is the simple default, but on a long
    episode it can step right over the one action that mattered - a big
    reward swing (or penalty) a few steps wide, sitting between two evenly
    spaced samples. When prioritize_eventful is on (the default), the
    first/last step are always kept and the remaining slots go to the
    steps with the largest |reward| instead, so an interesting moment
    can't be silently skipped just because of where the even spacing fell.
    """
    if num_steps_to_plot >= total_steps:
        return list(range(total_steps))
    if not prioritize_eventful or not rewards:
        return list(np.linspace(0, total_steps - 1, num_steps_to_plot, dtype=int))

    must_include = {0, total_steps - 1}
    remaining_slots = max(0, num_steps_to_plot - len(must_include))
    reward_padded = list(rewards) + [0.0] * (total_steps - len(rewards))
    candidates = sorted(
        (i for i in range(total_steps) if i not in must_include),
        key=lambda i: abs(reward_padded[i]),
        reverse=True,
    )
    chosen = must_include | set(candidates[:remaining_slots])
    return sorted(chosen)


def _match_pct(grid, target):
    if target is None:
        return None
    target = crop_pad(grid_formatting(target))
    if grid.shape != target.shape:
        return None
    return float(np.mean(grid == target))


def _highlight_changed_cells(ax, grid, prev_grid):
    """Outline every cell that changed since the previous *real* step (not
    the previous displayed one, even when downsampled) - the grids
    themselves only show two full snapshots, this points at exactly where
    the action in between actually acted."""
    if prev_grid is None or grid.shape != prev_grid.shape:
        return
    for y, x in np.argwhere(grid != prev_grid):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))


def plot_rollout_grid_trace(rollout, num_steps_to_plot=None, action_mapping=None,
                            figsize=(15, 10), include_descriptions=True, include_info=True,
                            target_grid=None, highlight_changes=True, prioritize_eventful_steps=True):
    """Plots the grid states from a rollout to monitor a trace with textual descriptions.

    Args:
        rollout (dict): a dictionary with keys ['observations', 'actions', 'rewards', 'dones', 'infos', 'total_reward', 'length'].
        where 'observations' contains states with 'grid' attributes representing grid states.
        num_steps_to_plot (int), optional: number of steps to plot. If None, plots all steps.
        plot_rewards (bool), optional: whether to plot rewards alongside the grids.
        figsize (tuple), optional: figure size (width, height).
        include_descriptions (bool), optional: whether to include textual descriptions of each step.
        target_grid (np.ndarray), optional: the goal grid (e.g. subtask.train_out) - when given,
            each step's title also shows its % match to it.
        highlight_changes (bool): outline cells that changed since the previous step.
        prioritize_eventful_steps (bool): when downsampling, prefer the steps with the
            largest |reward| over uniform spacing - see _select_step_indices.

    Returns:
        fig (matplotlib.figure.Figure): the figure containing the grid trace plots.
    """
    # Extract the grid states and other data from the rollout
    states = rollout['observations']
    grids = [crop_pad(grid_formatting(state['grid'])) for state in states if 'grid' in state]
    actions = rollout['actions'] if 'actions' in rollout else []
    rewards = rollout['rewards'] if 'rewards' in rollout else []
    infos = rollout['infos'] if 'infos' in rollout else []

    # Determine number of steps to plot
    total_steps = len(grids)
    if num_steps_to_plot is None:
        num_steps_to_plot = total_steps
    else:
        num_steps_to_plot = min(num_steps_to_plot, total_steps)

    if num_steps_to_plot == 0:
        fig = plt.figure(figsize=figsize)
        fig.text(0.5, 0.5, "Rollout has no steps to plot", ha='center', va='center')
        return fig

    indices = _select_step_indices(total_steps, num_steps_to_plot, rewards, prioritize_eventful_steps)
    rewards_to_plot = [rewards[i] if i < len(rewards) else None for i in indices]
    actions_to_plot = [actions[i] if i < len(actions) else None for i in indices]
    num_steps_to_plot = len(indices)  # _select_step_indices is the source of truth now

    # Determine grid dimensions for plotting
    n_cols = min(5, num_steps_to_plot)
    n_rows = (num_steps_to_plot + n_cols - 1) // n_cols

    # Create figure and GridSpec to organize subplots
    fig = plt.figure(figsize=figsize)

    # Calculate row heights: one summary row for reward-vs-step (if any
    # rewards exist), then one grid row (+ optional description row) per
    # row of the grid trace itself.
    show_reward_summary = len(rewards) > 0
    row_heights = [1.2] if show_reward_summary else []
    for _ in range(n_rows):
        row_heights.append(3)  # Grid height
        if include_descriptions:
            row_heights.append(1)  # Description height

    gs = GridSpec(len(row_heights), n_cols, figure=fig, height_ratios=row_heights)
    row_offset = 0

    if show_reward_summary:
        ax_reward = fig.add_subplot(gs[0, :])
        ax_reward.plot(range(len(rewards)), rewards, marker='o', markersize=3, color='tab:blue')
        ax_reward.axhline(0, color='grey', linewidth=0.5)
        for idx in indices:
            ax_reward.axvline(idx, color='tab:orange', linestyle=':', linewidth=1, alpha=0.6)
        ax_reward.set_xlabel('step')
        ax_reward.set_ylabel('reward')
        ax_reward.set_title('Reward per step (dotted lines mark the steps shown below)', fontsize=9)
        row_offset = 1

    # Plot each grid state
    for i in range(num_steps_to_plot):
        row, col = divmod(i, n_cols)
        row_idx = row_offset + row * (1 + int(include_descriptions))

        # Plot the grid
        ax_grid = fig.add_subplot(gs[row_idx, col])
        step_idx = indices[i]
        grid = grids[step_idx]
        ax_grid.imshow(grid, cmap=ARC_CMAP_HIGHLIGHT, norm=ARC_NORM_HIGHLIGHT)
        if highlight_changes and step_idx > 0:
            _highlight_changed_cells(ax_grid, grid, grids[step_idx - 1])

        reward_i = rewards_to_plot[i]
        title = f'Step {step_idx}, Reward: {reward_i if reward_i is not None else 0:.2f}'
        match_pct = _match_pct(grid, target_grid)
        if match_pct is not None:
            title += f', match: {match_pct:.0%}'
        ax_grid.set_title(title, fontsize=9)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        # Add textual description if enabled
        if include_descriptions:
            desc_row = row_idx + 1
            ax_desc = fig.add_subplot(gs[desc_row, col])

            # Get step info
            observation = states[step_idx] if step_idx < len(states) else None
            action = actions_to_plot[i] if i < len(actions_to_plot) else None
            reward = rewards_to_plot[i] if i < len(rewards_to_plot) else None

            # Get info for this step
            raw_info = infos[step_idx] if step_idx < len(infos) else {}
            step_info = {k:v for k, v in raw_info.items() if k!='TimeLimit.truncated'}
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
    suptitle = f'Rollout Grid Trace (Total Reward: {sum(rewards):.2f}, Length: {len(actions)})'
    final_match = _match_pct(grids[-1], target_grid) if grids else None
    if final_match is not None:
        suptitle += f', Final match to target: {final_match:.0%}'
    plt.suptitle(suptitle, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig
