import numpy as np
from collections import defaultdict
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
from rl.utils import vote_grid

def evaluate_ARC_policy(
    model: "base_class.BaseAlgorithm",
    vec_env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    debug: bool = False, 
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent to evaluate
    :param env: The gym VecEnv.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    episode_rewards = []
    episode_lengths = []
    episode_accs = []
    n_envs = vec_env.num_envs if hasattr(vec_env, 'envs') else 1
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = vec_env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)
    predicted_grids = [None for _ in range(n_envs)]
    monitor = defaultdict(list)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = vec_env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        if debug:
            monitor['actions'].append(actions)
            monitor['states'].append(states)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
    
                if dones[i]:
                    predicted_grid = info['terminal_observation']['grid']
                    predicted_grids[i] = predicted_grid
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    env = vec_env.envs[i]
                    acc = (env.max_int - env.base_int) / (env.target_int - env.base_int)
                    episode_accs.append(acc)
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
    
        observations = new_observations
    
    mean_reward = np.mean(episode_rewards)
    max_reward = max(episode_rewards)
    mean_acc = f'{np.mean(episode_accs):.3f}'
    max_acc = f'{max(episode_accs):.3f}'
    grid_pred = vote_grid(predicted_grids)
    env = vec_env.envs[0]
    acc = (env.maximal_intersection(grid_pred) - env.base_int) / (env.target_int - env.base_int)
    acc = max(acc, 0)
    std_reward = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)
    min_len = min(episode_lengths)
    if debug:
        return acc, mean_len, grid_pred, monitor
    else:
        return acc, mean_len, grid_pred