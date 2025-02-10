import optuna
from rl.ARC_task import ARCTask, ARCSubtask
from rl.training import train_on_subtask
from data.configs.rl_configs import load_PPO_config

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