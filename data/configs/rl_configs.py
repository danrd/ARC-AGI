import torch.nn as nn
from rl.utils import linear_schedule
from rl.policy import ARCCustomActorCriticPolicy

rl_config = {
    'model_type': 'PPO',
    'total_steps': 1000000,
    'n_eval_episodes': 1,
    'n_envs': 1,
    'seed' : 42,
    'eval_freq': 5,
    'log_path': ".data/logs/rl/",
    'max_episode_len': 25,
    'right_placement_reward': 5.0,
    'action_penalty': 1.0,
    'repetitive_actions_penalty': 1.0,
    'font_color': 0.0,
    'padding': False,
    'input_pattern': 'start',
    'milestones_rewards': [1,2,3,4],
    'reward_approach': 3,
    'pad_val': 10,
    'feasible_actions': {0:'submit'},
    'repr_level': 1,
    'observation_space_elements': ["objects_emb"] # ["objects_emb", "relations_emb"]
    }

def load_PPO_config():
    return {
    'verbose': 1,
    'batch_size': 256,
    'n_steps': 2048,
    'n_epochs' : 3,
    'gamma': 0.9,
    'gae_lambda': 0.9,
    'learning_rate': linear_schedule(0.0002),
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'use_sde': False,
    'policy': ARCCustomActorCriticPolicy,
    'actor_arch': [256, 256, 256],
    'critic_arch': [256, 256, 256],
    'activation_fn': nn.ReLU,
    'extr_arch': lin_arch,
    'action_heads': 3,
    }

def lin(act_func=nn.ReLU()):
    return nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: [batch, 16, 1, 1]
              nn.Flatten()                   # Output shape: [batch, 16]
            )
lin_arch = lin()
