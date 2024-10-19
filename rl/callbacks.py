import os
from enum import Enum
from collections import defaultdict
from typing import NamedTuple, Union, Tuple, Optional

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

class EvalFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"

class EvalFreq(NamedTuple):
    frequency: Union[int, Tuple[int, int]]
    unit: EvalFrequencyUnit

def convert_to_eval_freq(eval_freq: Union[int, Tuple[int, str], EvalFreq]) -> EvalFreq:
    if not isinstance(eval_freq, EvalFreq):
        if not isinstance(eval_freq, tuple):
            eval_freq = (eval_freq, "episode")
        # type checking happens in the Enum class constructor
        return EvalFreq(eval_freq[0], EvalFrequencyUnit(eval_freq[1]))
    else:
        raise TypeError('Argument eval_freq does not match a supported type.')

class EpisodeEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_freq: Union[int, Tuple[int, str], EvalFreq] = (100, "step"),
            eval_freq_max: Optional[Union[int, Tuple[int, str], EvalFreq]] = None,
            eval_start: int = 0,
            max_episode_length: int = None,
            log_episode_lengths_and_rewards: bool = False,
            verbose = False,    
            verbose_prefix: str = '',
            reward_history = defaultdict(list),
            env_label: str = '', 
            n_envs: int = 1, 
            max_rewards: dict = {},  
            **kwargs,
    ):
        self.max_episode_length = max_episode_length

        if 'results' in kwargs:
            self.results = kwargs.pop('results')
        else:
            self.results = None

        super().__init__(**kwargs)
        self.env = kwargs['eval_env']
        self.eval_freq = convert_to_eval_freq(eval_freq)
        if eval_freq_max is not None:
            self.eval_freq_max = convert_to_eval_freq(eval_freq_max)
        else:
            self.eval_freq_max = None
        self.eval_start = eval_start
        self.last_eval: int = -1  # timesteps finished at last evaluation
        self.last_eval_ep: int = eval_start - self.eval_freq.frequency if self.eval_freq.unit == EvalFrequencyUnit.EPISODE else -1
        self.n_eval = 0
        self.evaluations_episodes = []
        self.log_episode_lengths_and_rewards = log_episode_lengths_and_rewards
        self.verbose = verbose      
        self.verbose_prefix = verbose_prefix
        self.pre_auc = 0
        self.auc = 0
        self.reward_history = reward_history
        self.env_label = env_label
        self.last_eval = 0
        self.n_envs = n_envs
        self.max_rewards = max_rewards
        
    def _convert_eval_freq(self):
        """Convert eval_freq to an EvalFreq instance."""
        if not isinstance(self.eval_freq, EvalFreq):
            eval_freq = self.eval_freq
            if not isinstance(eval_freq, tuple):
                eval_freq = (eval_freq, "episode")
            # type checking happens in the Enum class constructor
            self.eval_freq = EvalFreq(eval_freq[0], EvalFrequencyUnit(eval_freq[1]))

    def _on_step(self) -> bool:
        if self.n_calls > self.eval_start and self._do_evaluation():
            self._evaluate()
        return True

    def _evaluate(self):
        try:
            self.env.start_evaluation()
        except AttributeError:
            pass

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                )

            # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
#             max_episode_length=self.max_episode_length,
#             track_agent_actions_diversity=True,
        )

        assert len(episode_lengths) == len(episode_rewards)

        if self.log_episode_lengths_and_rewards and self.verbose > 0:

            for i, (episode_reward, episode_length) in enumerate(
                    zip(episode_rewards, episode_lengths)):
                print(
                    f'🔵 episode index = {i:-5d} \tepisode length = {episode_length:-10d} \tepisode reward = {episode_reward:-10.3f}'
                )
            print(
                f'🟢 n episodes = {len(episode_lengths):-5d}   ' +
                f'mean episode length = {np.mean(episode_lengths):-10.3f}        mean reward = {np.mean(episode_rewards):-10.3f}'
            )

        if self.results is not None:
            self.results.append_episode_lengths(episode_lengths)
            self.results.append_episode_rewards(episode_rewards)
            self.results.append_timesteps(self.num_timesteps)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            self.evaluations_episodes.append(self.model._episode_num)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                episodes=self.evaluations_episodes,
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose > 0:
            print(
                self.verbose_prefix + f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(
                f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps,
                           exclude="tensorboard")
        # self.logger.dump(self.num_timesteps)

        try:
            self.env.stop_evaluation()
        except AttributeError:
            pass

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(
                    os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback if needed
            if self.callback is not None:
                return self._on_event()
    
        self.pre_auc += self.last_mean_reward * (self.num_timesteps/self.n_envs - self.last_eval/self.n_envs)
        
        self.auc = float((self.pre_auc + self.last_mean_reward * (self.model._total_timesteps - self.num_timesteps/self.n_envs)) / self.model._total_timesteps)
        self.last_eval = self.num_timesteps
        self.last_eval_ep = self.model._episode_num
        self.n_eval += 1

    def _do_evaluation(self,) -> bool:
        do = False
        if self.eval_freq.unit == EvalFrequencyUnit.STEP:
            do = self.eval_freq.frequency > 0 and self.n_calls % self.eval_freq.frequency == 0 and self.model._total_timesteps >= self.num_timesteps
        elif self.eval_freq.unit == EvalFrequencyUnit.EPISODE:
            if self.eval_freq.frequency > 0:
                do = (self.model._episode_num - self.eval_start + 1) // self.eval_freq.frequency >= self.n_eval and (self.model._episode_num - self.last_eval_ep) >= self.eval_freq.frequency

        if do and self.eval_freq_max is not None and self.eval_freq_max.unit == EvalFrequencyUnit.STEP:
            do = (self.num_timesteps - self.last_eval) >= self.eval_freq_max.frequency
        elif do and self.eval_freq_max is not None and self.eval_freq_max.unit == EvalFrequencyUnit.EPISODE:
            do = (self.model._episode_num - self.last_eval_ep) >= self.eval_freq_max.frequency
        return do
    
    def _update_history(self):
        normalized_reward = self.auc / self.max_rewards[self.env_label]
        self.reward_history[self.env_label].append(normalized_reward)
        print(f'new reward for env {self.env_label} is {normalized_reward}')
        pass
    
    def _reset(self):
        self.last_eval = 0
        self.n_eval = 0
        self.evaluations_episodes = []
        self.pre_auc = 0
        self.auc = 0
        self.last_eval = 0
        self.n_calls = 0
        pass
        
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._update_history()
        self._reset()
        pass