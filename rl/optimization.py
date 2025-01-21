import os
import torch
import optuna
import functools
import copy
import functools
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from utils.utils import seed_everything
from data.dataset.ARC_dataset import ARCDataset
from rl.training import create_env

gym.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='ARCGridWorld_env:create_env',
     kwargs={}
)

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'gamma': trial.suggest_loguniform('gamma', 0.91, 0.99),
        'gae_lambda': trial.suggest_loguniform('gae_lambda', 0.9, 0.99),        
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.1),
        'batch_size': trial.suggest_uniform('batch_size', 64, 256),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'max_grad_norm': int(trial.suggest_uniform('max_grad_norm', 0.3, 0.5)),
        'net_arch': trial.suggest_categorical('net_arch', [[64,64,64], [128,128,128], [64,64,64,64], [128,128,128,128]]),
        'activation_fn': trial.suggest_categorical('activation_fn', [torch.nn.ReLU, torch.nn.Sigmoid])
    }    

def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)
    envs = [functools.partial(create_env, subtask=subtask, seed=42+i) for i in range(16)] # creating envs for vectorizing
    vec_env = VecMonitor(DummyVecEnv(envs))
    p = model_params['net_arch']
    model = PPO("MultiInputPolicy", vec_env, verbose=1, batch_size=256, policy_kwargs=dict(net_arch=dict(pi=p, vf=p)), gamma=model_params['gamma'],
               gae_lambda=model_params['gae_lambda'], learning_rate=0.0002, clip_range=model_params['clip_range'],
               max_grad_norm=model_params['max_grad_norm'])     
    eval_callback = EvalCallback(vec_env, log_path=".data/", eval_freq=100)
    model.learn(400000)
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return -1 * mean_reward

def run_study(subtask):
    seed_everything()
    subtask=subtask
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=8)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')