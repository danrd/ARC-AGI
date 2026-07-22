import gymnasium as gym
import functools
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
from rl.evaluation import evaluate_ARC_policy
from rl.callbacks import MonitorCallback, ARCLogger
from utils.utils import seed_everything
from utils.plotting import plot_grid
from data.configs.rl_configs import load_PPO_config

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
                             'action_heads':PPO_config['action_heads'], 'action_types':PPO_config['action_types'],
                             'features_extractor_kwargs':{'extr_arch': PPO_config['extr_arch'], 'shapes_match':vec_env.shapes_match}}
            agent = PPO(policy, vec_env, batch_size=PPO_config['batch_size'], n_steps=PPO_config['n_steps'], verbose=PPO_config['verbose'], 
                        n_epochs=PPO_config['n_epochs'], gamma=PPO_config['gamma'], max_grad_norm=PPO_config['max_grad_norm'],
                        learning_rate=PPO_config['learning_rate'], clip_range=PPO_config['clip_range'], ent_coef=PPO_config['ent_coef'], 
                        vf_coef=PPO_config['vf_coef'], use_sde = PPO_config['use_sde'], policy_kwargs=policy_kwargs)
    return agent


def create_ARC_env(subtask, max_episode_len=50, right_placement_reward=5.0, action_penalty=1.0, 
                   repetitive_actions_penalty=1.0, seed=None, font_color=0, padding=False, input_pattern=False, 
                   milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                   feasible_actions={0:"submit"}, observation_space_elements = ["objects_emb", "relations_emb"], 
                  ):
    """Auxiliary function for creating environments to create vectorized environment."""
    gym.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='rl.ARC_env:create_env',
     kwargs={})
    env = gym.make('ARC-Gridworld-v0', max_episode_len=max_episode_len, right_placement_reward=right_placement_reward,
                   action_penalty=action_penalty, repetitive_actions_penalty=repetitive_actions_penalty,
                   seed=seed, font_color=font_color, padding=padding, input_pattern=input_pattern, repr_level=repr_level,
                   reward_approach=reward_approach, milestones_rewards=milestones_rewards, pad_val=pad_val,
                   feasible_actions=feasible_actions, observation_space_elements=observation_space_elements,
                  )
    env.set_subtask(subtask)
    env.action_space.seed(seed)
    return env

def create_vec_env(subtasks, n_envs:int, max_episode_len=50, right_placement_reward=5.0, action_penalty=1.0, 
                   repetitive_actions_penalty=1.0, seed=None, font_color=0, padding=False, input_pattern=False, 
                   milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                   feasible_actions={0:"submit"}, observation_space_elements = ["objects_emb", "relations_emb"]):
    """Auxiliary function for creating vectorized environment."""
    envs = [functools.partial(create_ARC_env, subtask=subtask, max_episode_len=max_episode_len, right_placement_reward=right_placement_reward,
                              action_penalty=action_penalty, repetitive_actions_penalty=repetitive_actions_penalty,
                              seed=seed, font_color=font_color, padding=padding, input_pattern=input_pattern, repr_level=repr_level,
                              reward_approach=reward_approach, milestones_rewards=milestones_rewards, pad_val=pad_val, 
                              observation_space_elements=observation_space_elements,
                              feasible_actions=feasible_actions) for subtask in subtasks for i in range(n_envs)]
    vec_env = VecMonitor(DummyVecEnv(envs))
    return vec_env

def train_on_subtask(subtask, rl_config:dict, PPO_config:dict=None, agent_init=None, 
                     path_to_pretrained=None, verbose=False, plot_grid_pred=False, debug=False):
    seed = rl_config['seed']
    seed_everything(seed)    
    vec_env = create_vec_env(subtask, n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'], repr_level=rl_config['repr_level'],
                             right_placement_reward=rl_config['right_placement_reward'],  action_penalty=rl_config['action_penalty'], 
                             repetitive_actions_penalty=rl_config['repetitive_actions_penalty'], seed=seed, font_color=rl_config['font_color'], 
                             padding=rl_config['padding'], input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                             feasible_actions=rl_config['feasible_actions'], observation_space_elements=rl_config['observation_space_elements'])
    callback = MonitorCallback(vec_env, eval_freq=rl_config['eval_freq'], n_eval_episodes=rl_config['n_eval_episodes'],  
                                   log_path=rl_config['log_path'], debug=debug)
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config, 
                     path_to_pretrained=path_to_pretrained, agent_init=agent_init)
    metrics_list = ['train/loss', 'train/value_loss', 'train/clip_fraction', 'train/approx_kl', 'train/explained_variance', 'rollout/ep_rew_mean']
    logger = ARCLogger('/data/logs/rl', ["stdout", "csv"], metrics_list, load_PPO_config()['verbose'])
    agent.set_logger(logger)
    agent.learn(rl_config['total_steps'], callback=callback)
    acc, mean_len, grid_pred = evaluate_ARC_policy(agent, vec_env, n_eval_episodes=rl_config['n_eval_episodes'])
    if verbose:
        print(f'Accuracy for {subtask.label}: {acc}. Mean episode length: {mean_len}')
    if plot_grid_pred:
        plot_grid(grid_pred)
    return acc, mean_len, agent, callback, vec_env

def train_on_task(task, rl_config:dict, PPO_config:dict=None, agent_init=None, verbose=False, plot_grid_pred=False):
    seed = rl_config['seed']
    seed_everything(seed)
    train_metrics = {}
    accs_for_subtasks = {}
    lens_for_subtasks = {}
    expl_vars = {}
    subtasks = task.subtasks
    vec_env = create_vec_env(subtasks[0],n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'], repr_level=rl_config['repr_level'],
                             right_placement_reward=rl_config['right_placement_reward'],  action_penalty=rl_config['action_penalty'], 
                             repetitive_actions_penalty=rl_config['repetitive_actions_penalty'], seed=seed, font_color=rl_config['font_color'], 
                             padding=rl_config['padding'], input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                             feasible_actions=rl_config['feasible_actions'], observation_space_elements=rl_config['observation_space_elements'])
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config, agent_init=agent_init)
    for idx, subtask in enumerate(subtasks):
        acc, mean_len, agent, callback = train_on_subtask(subtask=subtask, rl_config=rl_config, agent_init=agent, verbose=verbose, plot_grid_pred=plot_grid_pred)
        accs_for_subtasks[idx] = acc
        lens_for_subtasks[idx] = mean_len
        expl_vars[idx] = round(callback.explained_variances[-1], 3)
    print(f'Explaines variances for subtasks: {expl_vars}')
    train_metrics['expl_vars'] = list(expl_vars.values())
    print(f'Accuracies for task: {list(accs_for_subtasks.values())}, Mean episode lengths for task: {list(lens_for_subtasks.values())}')
    return accs_for_subtasks, lens_for_subtasks, agent, train_metrics