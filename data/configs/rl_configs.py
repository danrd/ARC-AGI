import torch
import torch.nn as nn
from rl.utils import linear_schedule
from rl.policy import ARCCustomActorCriticPolicy

rl_config = {
    'model_type': 'PPO',
    'total_steps': 2500000,
    'n_eval_episodes': 16,
    'n_envs': 16,
    'seed' : 42,
    'eval_freq': 1000,
    'log_path': ".data/logs/rl/",
    'max_episode_len': 250,
    'right_placement_scale': 5.0,
    'wrong_placement_scale': 0.1,
    'font_color': 0.0,
    'padding': (15, 15),
    'random_start': True,
    'input_pattern': True,
    'milestones_rewards': [0,1,3,6],
    'pad_val': -0.1,
    'int_colors': False
    }

A2C_config = { 
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'learning_rate': 0.007,
    'max_grad_norm': 0.5,
    'actor_arch': [128, 128, 128],
    'critic_arch': [128, 128, 128],
    'activation_fn': torch.nn.Sigmoid
    }

DQN_config = {
    'gamma': 0.99,
    'buffer_size': 1000000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.5,
    'batch_size' : 32,
    'exploration_fraction': 0.1,
    'net_arch': [128, 128, 128],
    'activation_fn': torch.nn.Sigmoid 
}

def load_PPO_config():
    return {   
    'verbose': 1,
    'batch_size': 256,
    'n_steps': 2048,
    'n_epochs' : 3,
    'gamma': 0.9,
    'gae_lambda': 0.9,
    'learning_rate': linear_schedule(0.002),
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'use_sde': False, 
    'policy': ARCCustomActorCriticPolicy,
    'actor_arch': [128, 128, 128],
    'critic_arch': [128, 128, 128],
    'activation_fn': torch.nn.Tanh,
    'pos_enc_dim': 128,
    'cnn_arch': cnn
    }

cnn =  nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),)