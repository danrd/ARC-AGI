"""What a whole episode pays, under every reward approach.

Each piece of the reward was tested on its own - _submit_reward's shape,
a step's normalisation - and the pieces did not add up. A step is divided
by max_reward, which counts the milestones as well as the cells, so the
steps of a whole solve summed to 0.2; a submit was paid raw, up to 4.0;
and a solve ends the episode on the step that reaches the target, so the
submit that pays for it never came. Under approach 4 stopping halfway and
submitting paid 2.1 against 0.2 for solving.

So these are asked of episodes: a solve pays 1.0 under every approach,
and nothing short of it pays as much.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.mcts import replay_solution

ACTIONS = {0: "submit", 1: "red_fill", 2: "red_line", 3: "red_triangle"}
SUBMIT = [0, 0, 0, 0, 0]


def env_for(approach, size):
    """A size x size block of red to paint on a 10x10 grid: two fills,
    half of it and then the rest, solve it."""
    inp = np.zeros((10, 10), dtype=int)
    out = inp.copy()
    out[:size, :size] = 2
    env = ARCGridWorld(max_episode_len=25, feasible_actions=ACTIONS,
                       reward_approach=approach, repr_level=1, input_pattern="start",
                       addressing="coordinates", coordinate_shape=(10, 10),
                       observation_space_elements=["delta_input"])
    env.set_subtask(ARCSubtask("block", inp, out))
    env.reset()
    return env


def half(size):
    return [1, 0, 0, size // 2 - 1, size - 1]


def rest(size):
    return [1, 0, 0, size - 1, size - 1]


def play(env, actions):
    total, done = 0.0, False
    for action in actions:
        _obs, reward, done, _truncated, _info = env.step(np.array(action))
        total += reward
        if done:
            break
    return total, done


APPROACHES = [1, 2, 3, 4]
SIZES = [4, 10]


@pytest.mark.parametrize("approach", APPROACHES)
@pytest.mark.parametrize("size", SIZES)
class TestWhatAnEpisodePays:
    def test_a_solve_pays_one(self, approach, size):
        total, done = play(env_for(approach, size), [half(size), rest(size)])
        assert done
        assert total == pytest.approx(1.0)

    def test_stopping_halfway_pays_less_than_solving(self, approach, size):
        """The one that failed: under 4 it paid ten times as much."""
        halfway, _ = play(env_for(approach, size), [half(size), SUBMIT])
        assert halfway < 1.0 - 0.1

    def test_giving_up_at_once_pays_no_more_than_stopping_halfway(self, approach, size):
        at_once, _ = play(env_for(approach, size), [SUBMIT])
        halfway, _ = play(env_for(approach, size), [half(size), SUBMIT])
        assert at_once <= halfway

    def test_the_simulator_pays_what_the_env_pays(self, approach, size):
        """MCTS values actions through simulate_action; a solve it valued
        differently from the env would steer the search elsewhere."""
        env = env_for(approach, size)
        grid, objects, reached = env.grid.copy(), env.objects, int(env.max_int)
        first = env.simulate_action(np.array(half(size)), objects, grid, reached, None)
        second = env.simulate_action(np.array(rest(size)), first[1], first[0], first[2],
                                     np.array(half(size)))
        submit = env.simulate_action(np.array(SUBMIT), first[1], first[0], first[2],
                                     np.array(half(size)))
        assert second[4]
        assert first[3] + second[3] == pytest.approx(
            play(env_for(approach, size), [half(size), rest(size)])[0])
        assert first[3] + submit[3] == pytest.approx(
            play(env_for(approach, size), [half(size), SUBMIT])[0])


@pytest.mark.parametrize("approach", APPROACHES)
def test_a_replayed_solution_counts_the_solve_once(approach):
    """replay_solution appends a submit after the solving step for the
    trace's sake; the solve was already paid on that step."""
    rollout = replay_solution(env_for(approach, 4), [half(4), rest(4)])
    assert rollout["rewards"][-1] == 0.0
    assert rollout["total_reward"] == pytest.approx(1.0)


def test_a_grid_that_starts_solved_pays_one_for_submitting_it():
    """initialize_targets builds max_reward differently when there is no
    distance to close; the scale has to hold there too."""
    inp = np.zeros((4, 4), dtype=int)
    for approach in APPROACHES:
        env = ARCGridWorld(max_episode_len=5, feasible_actions=ACTIONS,
                           reward_approach=approach, repr_level=1, input_pattern="start",
                           addressing="coordinates", coordinate_shape=(4, 4),
                           observation_space_elements=["delta_input"])
        env.set_subtask(ARCSubtask("done", inp, inp.copy()))
        env.reset()
        assert play(env, [SUBMIT])[0] == pytest.approx(1.0), approach
