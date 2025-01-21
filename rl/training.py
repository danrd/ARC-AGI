import gym
import os
import copy
import numpy as np
import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common import base_class
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from data.dataset.ARC_dataset import ARCDataset
from rl.ARC_task import ARCTask
from utils.utils import seed_everything

dataset = ARCDataset()

def create_env(subtask, seed, target_grid=False):
    """Auxiliary function for creating environments to create vectorized environment."""
    env = gym.make('ARC-Gridworld-v0', seed=seed, target_grid=target_grid)
    subtask = copy.deepcopy(subtask)
    env.set_subtask(subtask)
    return env

def train_on_dataset(dataset, tasks_interval:list=[], tasks_subset:list=[]):
    seed_everything()
    accuracy = {}
    envs = [functools.partial(create_env, subtask=dataset.tasks[0].subtasks[0], actions=[], seed=42+i, target_grid=False) for i in range(16)] # creating envs for vectorizing
    vec_env = VecMonitor(DummyVecEnv(envs))
    p = (128,128,128)
    agent = PPO("MultiInputPolicy", vec_env, verbose=0, batch_size=256, gamma=0.99, learning_rate=0.0006, clip_range=0.2, 
            policy_kwargs=dict(net_arch=dict(pi=p, vf=p)))
    if tasks_subset == []:
        if tasks_interval != []:
            start_index = tasks_interval[0]
            end_index = tasks_interval[1]
            tasks = dataset.tasks[start_index:end_index]
        else:
            tasks = dataset.tasks
    for idx_t, task in enumerate(tasks):
        for idx_s, subtask in enumerate(task.subtasks):
            label = f'{idx_t}_{idx_s}'
            envs = [functools.partial(create_env, subtask=subtask, actions=[], seed=42+i, target_grid=False) for i in range(16)] # creating envs for vectorizing
            vec_env = VecMonitor(DummyVecEnv(envs))
            agent.env = vec_env
            agent.learn(400000)
            mean_reward, _ = evaluate_policy(agent, vec_env, n_eval_episodes=10)
            print(f'reward {mean_reward} for subtask {label}')
            accuracy[label] = mean_reward
    return accuracy

def train_on_subtask(subtask):
    seed_everything()
    envs = [functools.partial(create_env, subtask=subtask, actions=[], seed=42+i, target_grid=False) for i in range(16)] # creating envs for vectorizing
    vec_env = VecMonitor(DummyVecEnv(envs))
    p = (128,128,128)
    agent = PPO("MultiInputPolicy", vec_env, verbose=1, batch_size=256, gamma=0.99, learning_rate=0.0006, clip_range=0.2, 
            policy_kwargs=dict(net_arch=dict(pi=p, vf=p)))
    agent.learn(400000)
    mean_reward, _ = evaluate_policy(agent, vec_env, n_eval_episodes=10)
    return mean_reward

def train_on_task(task:ARCTask):
    seed_everything()
    accuracy = {}
    subtasks = task.subtasks
    envs = [functools.partial(create_env, subtask=subtasks[0], actions=[], seed=42+i, target_grid=False) for i in range(16)] # creating envs for vectorizing
    vec_env = VecMonitor(DummyVecEnv(envs))
    p = (128,128,128)
    agent = PPO("MultiInputPolicy", vec_env, verbose=1, batch_size=256, gamma=0.99, learning_rate=0.0006, clip_range=0.2, 
            policy_kwargs=dict(net_arch=dict(pi=p, vf=p)))
    for idx, subtask in enumerate(subtasks):
        envs = [functools.partial(create_env, subtask=subtask, actions=[], seed=42+i, target_grid=False) for i in range(16)] # creating envs for vectorizing
        vec_env = VecMonitor(DummyVecEnv(envs))
        agent.env = vec_env
        agent.learn(400000)
        mean_reward, _ = evaluate_policy(agent, vec_env, n_eval_episodes=10)
        accuracy[idx] = mean_reward
    return accuracy

def evaluate_ARC_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    last_states = []
    actions = []
    episode_rewards, episode_lengths = [], []
    not_reseted = True
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            obs, _ = env.reset()
            not_reseted = False
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_actions = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            episode_actions.append(action)
            obs, reward, done, info, _ = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
        last_states.append(obs['grid'])
        actions.append(episode_actions)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    max_reward = max(episode_rewards)
    best_grid = last_states[np.argmax(episode_rewards)]
    std_reward = np.std(episode_rewards)
    return max_reward, actions, best_grid 

def plot_rewards(path_to_logs:str):
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