import os
import gymnasium as gym
import functools
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO, A2C, DQN
from copy import copy, deepcopy
from rl.ARC_task import ARCTask, ARCSubtask
from rl.evaluation import evaluate_ARC_policy
from rl.callbacks import MonitorCallback, ARCLogger
from data.configs.rl_configs import load_PPO_config
from utils.utils import seed_everything
from utils.plotting import plot_grid

def create_agent(rl_config:dict, vec_env, model_config:dict=None, path_to_pretrained:str=None, agent_init=None):
    if agent_init:
        agent = agent_init
    elif path_to_pretrained:
        agent = agent.load(path_to_pretrained)
    else:
        if rl_config['model_type'] == 'PPO':
            PPO_config = load_PPO_config()
            if model_config:
                PPO_config.update(model_config)
            policy = PPO_config['policy'] if PPO_config['policy'] != 'default' else "MultiInputPolicy"
            policy_kwargs = {'net_arch':dict(pi=PPO_config['actor_arch'], vf=PPO_config['critic_arch']), 'activation_fn':PPO_config['activation_fn'],
                             'features_extractor_kwargs':{'pos_enc_dim': PPO_config['pos_enc_dim'], 'cnn_arch': PPO_config['cnn_arch']}}
            agent = PPO(policy, vec_env, batch_size=PPO_config['batch_size'], n_steps=PPO_config['n_steps'], verbose=PPO_config['verbose'], 
                        n_epochs=PPO_config['n_epochs'], gamma=PPO_config['gamma'], max_grad_norm=PPO_config['max_grad_norm'],
                        learning_rate=PPO_config['learning_rate'], clip_range=PPO_config['clip_range'], ent_coef=PPO_config['ent_coef'], 
                        vf_coef=PPO_config['vf_coef'], use_sde = PPO_config['use_sde'], policy_kwargs=policy_kwargs)
        if rl_config['model_type'] == 'A2C':
            policy_kwargs = {'net_arch':dict(pi=A2C_config['actor_arch'], vf=A2C_config['critic_arch']), 'activation_fn':A2C_config['activation_fn']}
            agent = A2C("MultiInputPolicy", vec_env, verbose=1, gamma=A2C_config['gamma'], max_grad_norm=A2C_config['max_grad_norm'],
                    learning_rate=A2C_config['learning_rate'], policy_kwargs=policy_kwargs)
        if rl_config['model_type'] == 'DQN':
            policy_kwargs = {'net_arch':DQN_config['net_arch'], 'activation_fn':DQN_config['activation_fn']}
            agent = DQN("MultiInputPolicy", vec_env, verbose=1, buffer_size=DQN_config['buffer_size'], exploration_fraction=DQN_config['exploration_fraction'],
                        batch_size=DQN_config['batch_size'], gamma=DQN_config['gamma'], max_grad_norm=DQN_config['max_grad_norm'],
                        learning_rate=DQN_config['learning_rate'], policy_kwargs=policy_kwargs)
    return agent

def create_ARC_env(subtask:ARCSubtask, max_episode_len, right_placement_scale, wrong_placement_scale,
                   seed:int=42, font_color=0.0, padding=False, random_start=False, input_pattern=False,
                   milestones_rewards=[2, 3, 4, 5], pad_val=1.0, int_colors=False):
    """Auxiliary function for creating environments to create vectorized environment."""
    gym.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='rl.ARC_env:create_env',
     kwargs={})
    env = gym.make('ARC-Gridworld-v0', max_steps=max_episode_len, right_placement_scale=right_placement_scale, 
                   wrong_placement_scale=wrong_placement_scale, seed=seed, font_color=font_color, padding=padding, random_start=random_start,
                   input_pattern=input_pattern, milestones_rewards=milestones_rewards, pad_val=pad_val, int_colors=int_colors)
    env.set_subtask(subtask)
    env.action_space.seed(seed)
    return env

def create_vec_env(subtask:ARCSubtask, n_envs:int, max_episode_len:int=1000,
                   right_placement_scale:float=1.0, wrong_placement_scale:float=1.0,
                   seed:int=42, font_color:float=0.0, padding=False, random_start=False,
                   input_pattern=False, milestones_rewards=[2, 3, 4, 5], pad_val=1.0,
                   int_colors=True):
    """Auxiliary function for creating vectorized environment."""
    subtask = deepcopy(subtask)
    envs = [functools.partial(create_ARC_env, subtask=subtask, max_episode_len=max_episode_len, right_placement_scale=right_placement_scale, 
                              wrong_placement_scale=wrong_placement_scale, seed=seed+i, font_color=font_color, padding=padding, random_start=random_start,
                              input_pattern=input_pattern, milestones_rewards=milestones_rewards, pad_val=pad_val, int_colors=int_colors) for i in range(n_envs)]
    vec_env = VecMonitor(DummyVecEnv(envs))
    return vec_env

def train_on_subtask(subtask:ARCSubtask, rl_config:dict, PPO_config:dict=None, agent_init=None, 
                     path_to_pretrained=None, verbose=False, plot_grid_pred=False, debug=False):
    seed = rl_config['seed']
    seed_everything(seed)    
    vec_env = create_vec_env(subtask, n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'], 
                             right_placement_scale=rl_config['right_placement_scale'], wrong_placement_scale=rl_config['wrong_placement_scale'], 
                             seed=seed, font_color=rl_config['font_color'], padding=rl_config['padding'], random_start=rl_config['random_start'], 
                             input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], int_colors=rl_config['int_colors'])
    callback = MonitorCallback(vec_env, eval_freq=rl_config['eval_freq'], n_eval_episodes=rl_config['n_eval_episodes'],  
                                   log_path=rl_config['log_path'], debug=debug)
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config, 
                     path_to_pretrained=path_to_pretrained, agent_init=agent_init)
    metrics_list = ['train/loss', 'train/value_loss', 'train/clip_fraction', 'train/approx_kl', 'train/explained_variance', 'rollout/ep_rew_mean']
    logger = ARCLogger('/kaggle/working/arc-challenge/data', ["stdout", "csv"], metrics_list)
    agent.set_logger(logger)
    agent.learn(rl_config['total_steps'], callback=callback)
    acc, mean_len, grid_pred = evaluate_ARC_policy(agent, vec_env, n_eval_episodes=rl_config['n_eval_episodes'])
    if verbose:
        print(f'Accuracy for {subtask.label}: {acc}. Mean episode length: {mean_len}')
    if plot_grid_pred:
        plot_grid(grid_pred)
    return acc, mean_len, agent, callback

def train_on_task(task:ARCTask, rl_config:dict, PPO_config:dict=None, agent_init=None, verbose=False, plot_grid_pred=False):
    seed = rl_config['seed']
    seed_everything(seed)
    train_metrics = {}
    accs_for_subtasks = {}
    lens_for_subtasks = {}
    expl_vars = {}
    subtasks = task.subtasks
    vec_env = create_vec_env(subtasks[0], n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'], 
                             right_placement_scale=rl_config['right_placement_scale'], wrong_placement_scale=rl_config['wrong_placement_scale'], 
                             seed=seed, font_color=rl_config['font_color'], padding=rl_config['padding'], random_start=rl_config['random_start'], 
                             input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], int_colors=rl_config['int_colors'])
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config, agent_init=agent_init)
    for idx, subtask in enumerate(subtasks):
        acc, mean_len, agent, callback = train_on_subtask(subtask=subtask, rl_config=rl_config, agent_init=agent, verbose=verbose, plot_grid_pred=plot_grid_pred)
        accs_for_subtasks[idx] = acc
        lens_for_subtasks[idx] = mean_len
        expl_vars[idx] = round(callback.explained_variances[-1], 3)
    print(expl_vars)
    train_metrics['expl_vars'] = list(expl_vars.values())
    print(f'Accuracies for task: {list(accs_for_subtasks.values())}, Mean episode lengths for task: {list(lens_for_subtasks.values())}')
    return accs_for_subtasks, lens_for_subtasks, agent, train_metrics

def train_on_dataset(dataset, rl_config:dict, tasks_interval:list=[], tasks_subset:list=[], verbose=False, plot_grid_pred=False):
    seed = rl_config['seed']
    seed_everything(seed)
    accs_for_tasks = {}
    lens_for_tasks = {}
    if tasks_subset == []:
        if tasks_interval != []:
            start_index = tasks_interval[0]
            end_index = tasks_interval[1]
            tasks = dataset.tasks[start_index:end_index]
        else:
            tasks = dataset.tasks
    else:
        tasks = tasks_subset
    for idx, task in enumerate(tasks):
        accs_for_subtasks, lens_for_subtasks, agent = train_on_task(task, rl_config, agent, verbose, plot_grid_pred)
        accs_for_tasks[idx] = accs_for_subtasks
        lens_for_subtasks[idx] = lens_for_subtasks
    return accs_for_tasks, lens_for_tasks, agent

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