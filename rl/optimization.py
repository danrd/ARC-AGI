import optuna
import wandb
import os
import gc
import torch
from copy import copy 
from rl.ARC_task import ARCTask, ARCSubtask
from rl.training import train_on_subtask, train_on_task
from data.configs.rl_configs import load_PPO_config
from data.dataset.ARC_dataset import ARCDataset

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise."""
    return {
        'gamma': trial.suggest_categorical("gamma", [0.7, 0.8, 0.9, 0.95, 0.99]),
        'gae_lambda': trial.suggest_categorical("gae_lambda", [0.7, 0.8, 0.9, 0.95]),        
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_steps' : trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048]),
        # 'clip_range': trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        # 'n_epochs' : trial.suggest_categorical("n_epochs", [3, 4]),
        'max_grad_norm': trial.suggest_categorical("max_grad_norm", [0.6, 0.7, 0.8, 0.9]),
        # 'actor_arch': trial.suggest_categorical('actor_arch', [[128,128,128], [256, 256, 256], [128,128,128,128], [256, 256, 256, 256]]),
        # 'critic_arch': trial.suggest_categorical('actor_arch', [[128,128,128], [256, 256, 256], [128,128,128,128], [256, 256, 256, 256]]),       
        # 'activation_fn': trial.suggest_categorical('activation_fn', [torch.nn.ReLU, torch.nn.Sigmoid])
    }  

def rl_objective(trial, target:ARCSubtask, rl_config:dict):
    model_params = optimize_ppo(trial)
    PPO_config = load_PPO_config()
    PPO_config.update(model_params)
    acc, mean_len, agent, callback = train_on_subtask(subtask=target, rl_config=rl_config, PPO_config=PPO_config)
    del agent, callback
    if trial.should_prune():
        raise optuna.TrialPruned()
    return -1 * acc

def optimize_agent(target, rl_config):
    study = optuna.create_study()
    try:
        study.optimize(lambda trial: rl_objective(trial=trial, target=target, rl_config=rl_config), n_trials=100, n_jobs=8)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
    return study.best_params, study.best_trial

def hp_tuning(task_idx:int, subtasks_idxs:list, rl_config_params, model_config_params, base_rl_config, base_model_config):
    wandb.login(key='d71fac65cbcb94496f54123b10bf2d3d64fd8606')
    dataset = ARCDataset()
    for param in rl_config_params.keys():
        rl_config = copy(base_rl_config)
        for value in rl_config_params[param]:
            rl_config[param] = value
            for model_param in model_config_params.keys():
                model_config = copy(base_model_config) 
                for model_value in model_config_params[model_param]:
                    model_config[model_param] = model_value
                    merged_config = copy(rl_config)
                    merged_config.update(model_config)
                    merged_config['task_idx'] = task_idx
                    merged_config['subtasks_idxs'] = subtasks_idxs
                    run = wandb.init(project="ARC_RL", name='PPO', config=merged_config, group=str(merged_config['task_idx']))
                    if subtasks_idxs[0] == -1:
                        task = dataset.tasks[task_idx]
                        accs_for_subtasks, lens_for_subtasks, agent, train_metrics = train_on_task(task=task, rl_config=rl_config, PPO_config=model_config, 
                                                                                                   verbose=True, plot_grid_pred=True)
                        wandb.log({"accuracies": list(accs_for_subtasks.values()), "mean_ep_lens": list(lens_for_subtasks.values())})
                        for metric, val in train_metrics.items():
                            wandb.log({metric: val})
                        del agent  
                    else:
                        task = dataset.tasks[task_idx]
                        subtask = task.subtasks[subtasks_idxs[0]]
                        acc, mean_len, agent, callback = train_on_subtask(subtask=subtask, rl_config=rl_config, PPO_config=model_config, 
                                                                     verbose=True, plot_grid_pred=True, debug=False)
                        expl_var = round(callback.explained_variances[-1], 3)
                        wandb.log({"accuracies": [acc], "mean_ep_lens":[mean_len], "expl_vars": [expl_var]})
                        del agent, callback
                    print('All trials were finished')
                    wandb.finish()
                    torch.cuda.empty_cache()
                    gc.collect()
    return