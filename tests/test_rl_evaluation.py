"""Tests for rl/evaluation.py.

The previous version of this function could not have run: it read a
`test_env` attribute nothing sets, called `.append` on the numpy array of
dones, iterated one index past the arrays it indexed, and returned three
lists where every caller unpacks three scalars. So these tests pin the
contract the callers actually use - a mean accuracy, a mean length, and
the last grid - and the two things that are easy to get wrong in it: which
grid the accuracy is computed from, and when the loop stops.
"""
from __future__ import annotations

import numpy as np

from rl.evaluation import closed_fraction, evaluate_ARC_policy


class _Env:
    """The counters evaluation reads, and nothing else."""

    def __init__(self, base=0, target=10, max_int=0):
        self.base_int, self.target_int, self.max_int = base, target, max_int

    def maximal_intersection(self, grid):
        return int(np.asarray(grid).sum())


class _Wrapper:
    """What a vector actually holds: gymnasium's wrapper around the env."""

    def __init__(self, env):
        self.unwrapped = env


class _VecEnv:
    """Steps through a script of (rewards, dones, infos) tuples."""

    def __init__(self, script, envs):
        self.script, self.envs, self.num_envs = list(script), envs, len(envs)
        self.steps = 0

    def reset(self):
        return {}

    def step(self, actions):
        rewards, dones, infos = self.script[min(self.steps, len(self.script) - 1)]
        self.steps += 1
        return {}, np.array(rewards), np.array(dones), infos


class _Model:
    def predict(self, observations, state=None, deterministic=True):
        return np.zeros(1), None


def _done(grid):
    return {"terminal_observation": {"grid": np.array(grid)}}


class TestClosedFraction:
    def test_the_fraction_of_the_distance_a_grid_closed(self):
        assert closed_fraction(_Env(base=0, target=10), np.array([4])) == 0.4

    def test_a_wrapper_is_unwrapped(self):
        """A vector holds OrderEnforcing around the env; the counters live
        on the env."""
        assert closed_fraction(_Wrapper(_Env(base=0, target=10)),
                               np.array([4])) == 0.4

    def test_nothing_to_close_scores_one_rather_than_dividing_by_zero(self):
        assert closed_fraction(_Env(base=10, target=10), np.array([10])) == 1.0

    def test_without_a_grid_the_env_counter_is_used(self):
        assert closed_fraction(_Env(base=0, target=10, max_int=7)) == 0.7


class TestEvaluating:
    def test_the_accuracy_comes_from_the_terminal_grid(self):
        """Not from the env's max_int: by the time a done is visible the
        vector has reset that slot, and its counters describe the next
        episode."""
        env = _Env(base=0, target=10, max_int=0)
        vec = _VecEnv([([1.0], [True], [_done([8])])], [env])

        accuracy, _, _ = evaluate_ARC_policy(_Model(), vec, n_eval_episodes=1)

        assert accuracy == 0.8

    def test_it_stops_at_the_number_of_episodes_asked_for(self):
        env = _Env()
        vec = _VecEnv([([1.0], [True], [_done([5])])], [env])

        evaluate_ARC_policy(_Model(), vec, n_eval_episodes=3)

        assert vec.steps == 3

    def test_episode_length_counts_steps_to_the_done(self):
        env = _Env()
        vec = _VecEnv([([0.0], [False], [{}]),
                       ([0.0], [False], [{}]),
                       ([1.0], [True], [_done([5])])], [env])

        _, mean_length, _ = evaluate_ARC_policy(_Model(), vec, n_eval_episodes=1)

        assert mean_length == 3

    def test_the_last_grid_is_returned_for_plotting(self):
        env = _Env()
        vec = _VecEnv([([1.0], [True], [_done([3])])], [env])

        _, _, grid = evaluate_ARC_policy(_Model(), vec, n_eval_episodes=1)

        assert np.array_equal(grid, np.array([3]))

    def test_several_environments_are_all_scored(self):
        envs = [_Env(), _Env()]
        vec = _VecEnv([([1.0, 1.0], [True, True], [_done([2]), _done([8])])], envs)

        accuracy, _, _ = evaluate_ARC_policy(_Model(), vec, n_eval_episodes=2)

        assert accuracy == 0.5

    def test_the_callback_sees_each_step(self):
        """MonitorCallback's success logging reads `reward`, `done` and
        `info` out of this function's local scope - an odd contract, but a
        used one."""
        seen = []
        env = _Env()
        vec = _VecEnv([([1.0], [True], [_done([5])])], [env])

        evaluate_ARC_policy(_Model(), vec, n_eval_episodes=1,
                            callback=lambda local, glob: seen.append(local["done"]))

        assert seen == [True]
