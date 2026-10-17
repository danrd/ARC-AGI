"""The held-out pair scored by what the policy ends with, not by the first grid that matched the answer."""
import numpy as np
import pytest

from rl.arc_task import ARCSubtask
from rl.training import create_vec_env, evaluate_on_subtask

ACTIONS = {0: "submit", 1: "red_fill", 2: "blue_fill"}


def pair():
    grid = np.zeros((3, 3), dtype=int)
    target = grid.copy()
    target[0, 0] = 2
    return ARCSubtask("p", grid, target)


def env(stop_on_solve):
    vec_env = create_vec_env([pair()], n_envs=1, max_episode_len=4, feasible_actions=ACTIONS,
                             observation_space_elements=["delta_input"], observation_grid_shape=(3, 3),
                             repr_level=1, input_pattern="start", addressing="coordinates", coordinate_shape=(3, 3),
                             reward_approach=3)
    vec_env.envs[0].unwrapped.stop_on_solve = stop_on_solve
    return vec_env


class Scripted:
    """A model that plays the same actions every episode, one a step."""

    def __init__(self, actions):
        self.actions, self.step = actions, 0

    def predict(self, observations, state=None, deterministic=True):
        action = self.actions[self.step % len(self.actions)]
        self.step += 1
        return np.asarray([action]), state


SOLVE = [1, 0, 0, 0, 0]          # paint (0, 0) red: the answer
BREAK = [2, 1, 1, 1, 1]          # paint (1, 1) blue: a cell the answer leaves black


class TestTheEnv:
    def test_by_default_the_answer_ends_the_episode(self):
        vec_env = env(True)
        vec_env.reset()
        _obs, _reward, done, _info = vec_env.step(np.asarray([SOLVE]))
        assert done[0]

    def test_without_it_the_episode_goes_on_until_submit_or_the_limit(self):
        vec_env = env(False)
        vec_env.reset()
        _obs, _reward, done, _info = vec_env.step(np.asarray([SOLVE]))
        assert not done[0]
        dones = [vec_env.step(np.asarray([SOLVE]))[2][0] for _ in range(3)]
        assert dones == [False, False, True]            # the fourth step is the limit

    def test_without_it_submit_still_ends_it(self):
        vec_env = env(False)
        vec_env.reset()
        vec_env.step(np.asarray([SOLVE]))
        assert vec_env.step(np.asarray([[0, 0, 0, 0, 0]]))[2][0]


class TestTheScore:
    CONFIG = {"max_episode_len": 4, "repr_level": 1, "right_placement_reward": 5.0, "action_penalty": 1.0,
              "repetitive_actions_penalty": 1.0, "seed": 0, "font_color": 0, "padding": False,
              "input_pattern": "start", "milestones_rewards": (1, 2, 3, 4), "pad_val": 10, "reward_approach": 3,
              "feasible_actions": ACTIONS, "observation_space_elements": ["delta_input"],
              "observation_grid_shape": (3, 3), "addressing": "coordinates", "coordinate_shape": (3, 3),
              "n_eval_episodes": 1}

    def test_a_policy_that_reaches_the_answer_and_paints_over_it_is_solved_only_by_the_env_that_knows_it(self):
        paint_over = [SOLVE, BREAK, BREAK, BREAK]
        stopped, _, _ = evaluate_on_subtask(Scripted(paint_over), pair(), self.CONFIG)
        honest, _, _ = evaluate_on_subtask(Scripted(paint_over), pair(), self.CONFIG, stop_on_solve=False)
        assert stopped == pytest.approx(1.0) and honest < 1.0

    def test_a_policy_that_solves_and_submits_is_solved_both_ways(self):
        submits = [SOLVE, [0, 0, 0, 0, 0]]
        assert evaluate_on_subtask(Scripted(submits), pair(), self.CONFIG, stop_on_solve=False)[0] == pytest.approx(1.0)
        assert evaluate_on_subtask(Scripted(submits), pair(), self.CONFIG)[0] == pytest.approx(1.0)


class TestTrainingWithoutTheStop:
    """rl_config's stop_on_solve reaches the environments a policy is trained in."""

    def make(self, **flag):
        return create_vec_env([pair()], n_envs=2, max_episode_len=4, feasible_actions=ACTIONS,
                              observation_space_elements=["delta_input"], observation_grid_shape=(3, 3),
                              repr_level=1, input_pattern="start", addressing="coordinates", coordinate_shape=(3, 3),
                              reward_approach=3, **flag)

    def test_every_env_of_the_vector_takes_it(self):
        assert [e.unwrapped.stop_on_solve for e in self.make(stop_on_solve=False).envs] == [False, False]
        assert [e.unwrapped.stop_on_solve for e in self.make().envs] == [True, True]

    def test_training_builds_its_envs_with_the_configs_flag(self, monkeypatch):
        import rl.training
        seen = {}

        class Stop(Exception):
            pass

        def fake(subtasks, **kwargs):
            seen.update(kwargs)
            raise Stop

        monkeypatch.setattr(rl.training, "create_vec_env", fake)
        config = dict(TestTheScore.CONFIG, n_envs=1, total_steps=10, evaluations=1, log_path="/tmp/x/")
        for flag, expected in (({"stop_on_solve": False}, False), ({}, True)):
            with pytest.raises(Stop):
                rl.training.train_on_subtasks([pair()], dict(config, **flag), {})
            assert seen["stop_on_solve"] is expected
