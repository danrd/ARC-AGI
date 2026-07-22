import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def evaluate_ARC_policy(
    model: "base_class.BaseAlgorithm",
    vec_env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None, 
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
    n_envs = vec_env.num_envs + 1 if hasattr(vec_env, 'envs') else 1 # 1 for train + 1 for test
    test_env = vec_env.test_env
    episode_rewards = [[] for _ in range(n_envs)]
    episode_lengths = [[] for _ in range(n_envs)]
    episode_accs = [[] for _ in range(n_envs)]
    predicted_grids = [[] for _ in range(n_envs)]
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = vec_env.reset()
    test_observation = test_env.reset()
    states = None
    while (episode_counts < episode_count_targets).any() and not test_done:
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            deterministic=deterministic,
        )
        test_action, test_states = model.predict(
            test_observation,  # type: ignore[arg-type]
            state=test_states,
            deterministic=deterministic,
        )       
        new_observations, rewards, dones, infos = vec_env.step(actions)
        new_test_observation, test_reward, test_done, test_info = test_env.step(test_action)
        current_rewards[:-1] += rewards
        dones.append(test_done) # for unification
        infos.append(test_info) # for unification
        current_rewards[-1] = test_reward
        current_lengths += 1
        for i in range(n_envs+1):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                if dones[i]:
                    predicted_grids[i].append(info['terminal_observation']['grid'])
                    episode_rewards[i].append(current_rewards[i])
                    episode_lengths[i].append(current_lengths[i])
                    env = vec_env.envs[i] if i < n_envs-1 else test_env
                    acc = (env.max_int - env.base_int) / (env.target_int - env.base_int)
                    episode_accs[i].append(acc)
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0 
        observations = new_observations
    return episode_accs, episode_lengths, grid_pred