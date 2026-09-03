"""Tests for rl/training.py's task-level entry points.

A task's subtasks are one rule shown several times, so what the agent
trains on is the whole question. train_on_task used to build a vec env
over subtasks[0], then walk the subtasks calling train_on_subtask - which
built an env per subtask that the agent never stepped in (see
tests/test_rl_env.py on set_env), and never touched the held-out pair at
all. So these pin what each mode puts in the rollout buffer, that the test
pair is scored and never trained on, and that a wrong grid is not handed
back as a solution.

Training is real PPO here, kept to a few hundred steps on a submit-only
vocabulary: what is asserted is which envs the steps came from, not
anything about what was learned.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from data.configs.rl_configs import load_PPO_config
from rl.arc_task import ARCSubtask, ARCTask
from rl.rl_module import RLModule, RlConfig
from rl.training import (create_agent, create_vec_env, train_on_subtasks,
                         train_on_task)

SUBMIT_ONLY = {0: "submit"}


@pytest.fixture
def task():
    """Three subtasks and a held-out pair, each distinguishable by label."""
    def pair(fill):
        return np.array([[fill, 0], [0, fill]]), np.array([[fill, fill], [0, 0]])

    subtasks = [ARCSubtask(f"t_{i}", *pair(i + 1)) for i in range(3)]
    test_inp, test_out = pair(7)
    return ARCTask(label="t", subtasks=subtasks, test_inp=test_inp, test_out=test_out)


@pytest.fixture
def config(tmp_path):
    return {
        "model_type": "PPO", "total_steps": 64, "n_eval_episodes": 1, "n_envs": 1,
        "seed": 42, "eval_freq": 10_000, "log_path": str(tmp_path), "max_episode_len": 5,
        "right_placement_reward": 5.0, "action_penalty": 1.0,
        "repetitive_actions_penalty": 1.0, "font_color": 0.0, "padding": False,
        "input_pattern": "start", "milestones_rewards": [1, 2, 3, 4],
        "reward_approach": 3, "pad_val": 10, "feasible_actions": SUBMIT_ONLY,
        "repr_level": 1, "observation_space_elements": ["objects_emb"],
    }


@pytest.fixture
def ppo():
    return {"n_steps": 16, "batch_size": 8, "verbose": 0, "action_heads": 5}


def labels_of(vec_env):
    return [env.unwrapped.subtask_label for env in vec_env.unwrapped.envs]


class TestWhatIsInTheRolloutBuffer:
    def test_mixed_training_steps_in_every_subtask(self, task, config, ppo):
        """One buffer over all of them: an update is made against the task,
        not against one example of it."""
        _, _, agent, _ = train_on_task(task, rl_config=config, PPO_config=ppo)

        assert labels_of(agent.get_env()) == ["t_0", "t_1", "t_2"]

    def test_sequential_training_ends_in_the_last_subtask(self, task, config, ppo):
        """The original walk: one env at a time, the agent carried forward -
        and it has to actually be pointed at each one, which is what used to
        be missing."""
        _, _, agent, _ = train_on_task(task, rl_config=config, PPO_config=ppo,
                                       mode="sequential")

        assert labels_of(agent.get_env()) == ["t_2"]

    def test_an_unknown_mode_names_itself(self, task, config, ppo):
        with pytest.raises(ValueError, match="sideways"):
            train_on_task(task, rl_config=config, PPO_config=ppo, mode="sideways")

    def test_train_on_subtasks_holds_all_of_them_at_once(self, task, config, ppo):
        _, _, agent, _, vec_env = train_on_subtasks(task.subtasks, rl_config=config,
                                                    PPO_config=ppo)

        assert labels_of(vec_env) == ["t_0", "t_1", "t_2"]
        assert agent.get_env() is vec_env


class TestTheHeldOutPair:
    @pytest.mark.parametrize("mode", ["mixed", "sequential"])
    def test_the_test_pair_is_scored(self, task, config, ppo, mode):
        _, _, _, metrics = train_on_task(task, rl_config=config, PPO_config=ppo,
                                         mode=mode)

        assert 0.0 <= metrics["test_acc"] <= 1.0

    @pytest.mark.parametrize("mode", ["mixed", "sequential"])
    def test_the_test_pair_is_never_trained_on(self, task, config, ppo, mode):
        """The whole point of holding it out. It is the only subtask whose
        label carries `_test`."""
        _, _, agent, _ = train_on_task(task, rl_config=config, PPO_config=ppo,
                                       mode=mode)

        assert not any("test" in label for label in labels_of(agent.get_env()))

    def test_the_training_accuracies_still_cover_every_subtask(self, task, config, ppo):
        accs, lens, _, _ = train_on_task(task, rl_config=config, PPO_config=ppo)

        assert sorted(accs) == [0, 1, 2]
        assert sorted(lens) == [0, 1, 2]


class TestTheConfigReachesTheAgent:
    """A hyperparameter the config names and create_agent forgets to pass is
    a setting that looks configured and is not. gae_lambda was one from the
    start: the config said 0.9 and every agent ever built here ran on PPO's
    own default of 0.95."""

    def test_gae_lambda_is_the_configured_one(self, task, config, ppo):
        model_config = {**load_PPO_config(), **ppo, "gae_lambda": 0.77}
        vec_env = create_vec_env([task.subtasks[0]], n_envs=1, max_episode_len=5,
                                  feasible_actions=SUBMIT_ONLY,
                                  observation_space_elements=["objects_emb"])
        try:
            agent = create_agent(rl_config=config, vec_env=vec_env,
                                 model_config=model_config)

            assert agent.gae_lambda == 0.77
        finally:
            vec_env.close()

    def test_no_key_of_the_ppo_config_goes_unread(self):
        """Structural, so a key added to the config later cannot sit there
        doing nothing without this failing."""
        source = inspect.getsource(create_agent)
        unread = {key for key in load_PPO_config()
                  if f"PPO_config['{key}']" not in source}

        assert unread == set()


class TestWhatCountsAsASolution:
    def _module_result(self, task, config, ppo, monkeypatch, test_acc, grid):
        module = RLModule(RlConfig(**{**config, "log_path": ".data/logs/rl/"}), ppo)

        def fake_train(**kwargs):
            return {}, {}, object(), {"test_acc": test_acc, "test_grid": grid,
                                      "test_len": 1.0, "expl_vars": []}

        monkeypatch.setattr("rl.training.train_on_task", fake_train)
        return module.solve(task)

    def test_a_solved_held_out_grid_is_the_solution(self, task, config, ppo, monkeypatch):
        grid = np.array([[7, 7], [0, 0]])

        result = self._module_result(task, config, ppo, monkeypatch, 1.0, grid)

        assert np.array_equal(result["solution"], grid)

    def test_a_grid_that_did_not_solve_it_is_not_a_solution(self, task, config, ppo,
                                                            monkeypatch):
        """0.99 of the distance closed is a wrong grid. Reported as a
        solution it would end the orchestration graph with a wrong answer."""
        result = self._module_result(task, config, ppo, monkeypatch, 0.99,
                                     np.array([[7, 0], [0, 0]]))

        assert result["solution"] is None
        assert result["module_results"]["train_metrics"]["test_acc"] == 0.99
