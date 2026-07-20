import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List,Tuple, Union
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.logger import Logger, KVWriter, make_output_format
from rl.evaluation import evaluate_ARC_policy 
from utils.plotting import plot_grid
class MonitorCallback(EvalCallback):
    def __init__(
            self,
            eval_env,
            eval_freq: Union[int, Tuple[int, str]] = 5000,
            eval_start: int = 1,
            n_eval_episodes: int = 1,
            max_episode_length: int = None,
            verbose = True,    
            n_envs: int = 1, 
            max_rewards: dict = {},
            log_path:str = None,
            debug:bool = False,
            **kwargs,
    ):
        super().__init__(eval_env, **kwargs)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_start = eval_start
        self.n_eval_episodes = n_eval_episodes
        self.last_eval: int = -1 
        self.n_eval = 0
        self.verbose = verbose      
        self.n_envs = eval_env.num_envs if hasattr(eval_env, 'envs') else 1
        self.log_path = log_path
        self.debug = debug
        self.episode_accs = []
        self.episode_mean_lens = []
        self.episode_grid_preds = []
        if self.debug:
            #values to track
            self.train_rewards = [[] for _ in range(self.n_envs)]
            self.grids = [[] for _ in range(self.n_envs)]
            self.next_grids = [[] for _ in range(self.n_envs)]
            self.actions = [[] for _ in range(self.n_envs)]
            self.values = [[] for _ in range(self.n_envs)]
            self.dones = [[] for _ in range(self.n_envs)]
            self.infos = [[] for _ in range(self.n_envs)]
            self.positions = [[] for _ in range(self.n_envs)]

    def update_metric(self, metric, new_values):
        """Update values for metric for each environment."""
        for i in range(self.n_envs):
            metric[i].append(new_values[i])
        return metric 
    
    def _on_step(self) -> bool:
        if self.debug:
            self.train_rewards = self.update_metric(self.train_rewards, self.locals['rewards'])
            self.grids = self.update_metric(self.grids, self.locals['obs_tensor']['grid'])
            self.next_grids = self.update_metric(self.next_grids, self.locals['new_obs']['grid']) 
            self.actions = self.update_metric(self.actions, self.locals['actions'])
            self.values = self.update_metric(self.values, self.locals['values'])
            self.dones = self.update_metric(self.dones, self.locals['dones'])    
            self.infos = self.update_metric(self.infos, self.locals['infos']) 
            self.positions = self.update_metric(self.positions, self.locals['obs_tensor']['agent_position']) 
        if self.n_calls > self.eval_start and self._do_evaluation():
            self._evaluate()
        return True

    def plot_metrics(self):
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        axs[0, 0].plot(self.ep_rew_mean)
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 1].plot(self.losses)
        axs[0, 1].set_title('Total Loss')
        axs[1, 0].plot(self.value_losses)
        axs[1, 0].set_title('Value Loss')
        axs[1, 1].plot(self.clip_fractions)
        axs[1, 1].set_title('Clip Fraction')
        axs[2, 0].plot(self.kl_divergences)
        axs[2, 0].set_title('KL Divergence')
        axs[2, 1].plot(self.explained_variances)
        axs[2, 1].set_title('Explained variance')
        plt.tight_layout()
        plt.show()

    def plot_action_dist(self, all_env:bool=False):
        mid = len(self.actions[0]) // 2
        n_envs = self.n_envs if all_env else 1
        if not all_env:
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].hist(x=self.actions[0][:mid], bins=14, edgecolor='black', density=True)
            axs[0].set_title('1 half actions dist')
            axs[1].hist(x=self.actions[0][mid:], bins=14, edgecolor='black', density=True)
            axs[1].set_title('2 half actions dist')
        else:
            for idx in range(n_envs):
                fig, axs = plt.subplots(n_envs, 2, figsize=(15, 10))
                axs[idx, 0].hist(x=self.actions[idx][:mid], bins=14, edgecolor='black', density=True)
                axs[idx, 0].set_title('1 half actions dist')
                axs[idx, 1].hist(x=self.actions[idx][mid:], bins=14, edgecolor='black', density=True)
                axs[idx, 1].set_title('2 half actions dist')
        plt.tight_layout()
        plt.show()


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass
    
    def _evaluate(self):
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                )
        acc, mean_len, grid_pred = evaluate_ARC_policy(
                                                        self.model,
                                                        self.eval_env,
                                                        n_eval_episodes=self.n_eval_episodes,
                                                        deterministic=self.deterministic,
                                                        callback=self._log_success_callback,
                                                      )

        self.episode_accs.append(acc)
        self.episode_mean_lens.append(mean_len)
        self.episode_grid_preds.append(grid_pred)
        
        if self.verbose:
            print(f'After {(self.num_timesteps/self.model._total_timesteps)*100:.2f}% of training: Accuracy: {acc:.2f}, Mean episode length: {mean_len:.2f}')
            plot_grid(grid_pred)

        try:
            self.env.stop_evaluation()
        except AttributeError:
            pass
    
        self.last_eval = self.num_timesteps
        self.n_eval += 1

    def _do_evaluation(self,) -> bool:
        return self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.model._total_timesteps >= self.num_timesteps
    
    def _reset(self):
        self.last_eval = 0
        self.n_eval = 0
        self.n_calls = 0
        pass
        
    def _on_training_end(self) -> None:
        """ This event is triggered before exiting the `learn()` method."""
        metrics = self.logger.name_to_value_track
        self.losses = metrics['train/loss']
        self.value_losses = metrics['train/value_loss']
        self.clip_fractions = metrics['train/clip_fraction']
        self.kl_divergences = metrics['train/approx_kl']
        self.explained_variances = metrics['train/explained_variance']
        self.ep_rew_mean = metrics['rollout/ep_rew_mean']
        self._reset()
        pass
class ARCLogger(Logger):
    def __init__(self, folder:str, output_formats, metrics_list:List[str]):
        super().__init__(folder, output_formats)
        self.name_to_value_track = defaultdict(list)
        self.metrics_list = metrics_list
        self.output_formats = [make_output_format(f, folder, "") for f in output_formats]

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == 50:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)
        for key in self.metrics_list:
            self.name_to_value_track[key].append(self.name_to_value[key])
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()