"""Tests for the ARCGridWorld environment (rl/arc_env.py) - environment
init, basic lifecycle methods, and dispatching transformations through
World. Part 1 of the RL test plan (environment); MCTS rollout collection
and plotting are covered in their own test modules.

Same philosophy as the LLM smoke test: a handful of exact tests where the
right answer is known by construction, plus broader "does it crash" smoke
tests for everything else - this environment's surface (reward shaping,
observation assembly, transformation dispatch through World) is too large
to hand-verify exhaustively for every code path.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.training import create_agent, create_vec_env

SUBMIT_ONLY = {0: "submit"}
SUBMIT_AND_ROTATE = {0: "submit", 1: "rotate90"}


@pytest.fixture
def subtask(arc_task):
    return arc_task.subtasks[0]


def make_env(**kwargs) -> ARCGridWorld:
    kwargs.setdefault("max_episode_len", 5)
    kwargs.setdefault("feasible_actions", SUBMIT_ONLY)
    return ARCGridWorld(**kwargs)


# -- exact tests: known answer by construction -------------------------------

def test_maximal_intersection_exact():
    """Matching cells count positive, mismatching cells count negative,
    cells padded in either grid are excluded entirely."""
    env = make_env(pad_val=10)
    env.train_out = np.array([[1, 2, 10], [3, 10, 10]])
    grid = np.array([[1, 9, 10], [3, 10, 10]])
    # matches: (0,0), (1,0) = 2; mismatches: (0,1) = 1 (both non-pad);
    # rest excluded (train_out is padded there) -> 2 - 1 = 1
    assert env.maximal_intersection(grid) == 1


def test_step_intersection_tracks_delta():
    env = make_env(pad_val=10)
    env.train_out = np.array([[1, 1], [1, 1]])
    env.max_int = 0
    env.target_int = 4
    grid = np.array([[1, 1], [1, 1]])

    right_placement, done = env.step_intersection(grid)

    assert right_placement == 4  # went from 0 matches to 4
    assert bool(done)  # max_int reached target_int
    assert env.max_int == 4


# -- lifecycle: set_subtask / reset ------------------------------------------

def test_set_subtask_and_reset_produces_valid_observation(subtask):
    env = make_env()
    env.set_subtask(subtask)
    obs, info = env.reset()

    assert {"grid", "action_space"} <= obs.keys()
    assert obs["grid"].shape == subtask.train_out_shape
    assert len(env.objects) > 0
    assert env.action_space.nvec[1] == len(env.objects)
    assert env.action_space.nvec[2] == len(env.objects)


def test_reset_returns_to_the_same_starting_state(subtask):
    """reset() should bring episode-local state back to the same starting
    point every time, not accumulate state across resets."""
    env = make_env()
    env.set_subtask(subtask)
    obs1, _ = env.reset()
    env.step(np.array([0, 0, 0]))  # submit, ends the episode
    obs2, _ = env.reset()

    assert np.array_equal(obs1["grid"], obs2["grid"])
    assert env.step_no == 0


# -- basic step / submit -----------------------------------------------------

def test_submit_action_terminates_immediately(subtask):
    env = make_env()
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([0, 0, 0]))

    assert done is True
    assert isinstance(reward, (int, float, np.integer, np.floating))


def test_episode_terminates_at_max_episode_len(subtask):
    """Without ever submitting (and without solving the task by accident),
    the episode still ends once max_episode_len steps have been taken - via
    `truncated`, not `done` (done means the task was actually solved;
    truncated means the step limit was hit - they used to be conflated,
    with done always just recomputed as the step-limit check)."""
    env = make_env(max_episode_len=3, feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    done = False
    truncated = False
    steps = 0
    for _ in range(10):  # safety bound well above max_episode_len
        _, _, done, truncated, _ = env.step(np.array([1, 0, 0]))  # rotate90, never submits
        steps += 1
        if done or truncated:
            break

    assert not done  # rotating never solves this task on its own
    assert truncated
    assert steps == 3


# -- calling a real transformation through World -----------------------------

def test_step_dispatches_a_real_transformation(subtask):
    """A non-submit action should route through World.step ->
    arc_transformators and come back with a well-formed observation, not
    just the submit shortcut."""
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([1, 0, 0]))

    assert obs["grid"].shape == env.grid.shape
    assert "change_of_grid" in info
    assert isinstance(reward, (int, float, np.integer, np.floating))


# -- state save/restore -------------------------------------------------------

def test_get_set_state_roundtrip(subtask):
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()
    env.step(np.array([1, 0, 0]))

    state = env.get_state()
    grid_before = env.grid.copy()
    step_no_before = env.step_no

    env.step(np.array([1, 0, 0]))
    env.set_state(state)

    assert np.array_equal(env.grid, grid_before)
    assert env.step_no == step_no_before


# -- full random episode: crash-or-not smoke test -----------------------------

def test_full_random_episode_does_not_crash(subtask):
    env = make_env(max_episode_len=8, feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()

    done = False
    for _ in range(env.max_episode_len + 2):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        if done:
            break

    assert done is True


# -- reward_approach: sweep the working ones, track the broken one -----------

@pytest.mark.parametrize("reward_approach", [1, 2, 3])
def test_reward_approach_submit_does_not_crash(subtask, reward_approach):
    env = make_env(reward_approach=reward_approach)
    env.set_subtask(subtask)
    env.reset()

    obs, reward, done, truncated, info = env.step(np.array([0, 0, 0]))

    assert done is True


def test_reward_approach_4_is_currently_broken(subtask):
    """Regression tracker, not desired behavior: reward_approach == 4
    reads self.max_reward_base, which is never set anywhere in
    ARCGridWorld. If this starts passing, the bug's been fixed - update or
    remove this test rather than leaving it pinned to the old behavior."""
    env = make_env(reward_approach=4)
    env.set_subtask(subtask)
    env.reset()

    with pytest.raises(AttributeError):
        env.step(np.array([0, 0, 0]))


# -- gym.make() integration path: create_ARC_env / create_vec_env -----------
#
# Everything above constructs ARCGridWorld directly, bypassing gym.make()'s
# wrapper stack (OrderEnforcing, PassiveEnvChecker) entirely - which is why
# these two bugs went unnoticed: they only trigger through rl.training's
# create_ARC_env/create_vec_env, the actual PPO-training entry point.

def test_reset_accepts_options_kwarg(subtask):
    """Regression test: every wrapper gym.make() adds calls
    reset(seed=..., options=...) unconditionally - ARCGridWorld.reset()
    used to only accept `seed`, so any env built via gym.make() crashed
    with TypeError on its very first reset()."""
    env = make_env()
    env.set_subtask(subtask)
    obs, info = env.reset(seed=0, options={})
    assert isinstance(obs, dict)


def test_create_vec_env_accepts_a_single_subtask_wrapped_in_a_list(subtask):
    """Regression test: create_vec_env(subtasks, ...) iterates over its
    first argument - every caller in rl/training.py used to pass a bare
    ARCSubtask (not iterable -> TypeError) instead of [subtask]. This also
    exercises create_ARC_env's gym.make() path, which used to crash
    separately: env.set_subtask(subtask) needs env.unwrapped.set_subtask(...)
    now, since gymnasium's wrapper __getattr__ no longer forwards custom
    methods like set_subtask to the wrapped env."""
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5,
                              feasible_actions=SUBMIT_AND_ROTATE)
    assert vec_env.num_envs == 1


def test_create_agent_builds_a_real_ppo_agent_from_default_config(subtask):
    """Regression test: create_agent used to read vec_env.shapes_match, an
    attribute nothing ever set (not create_vec_env, not ARCGridWorld) -
    AttributeError on every real training run. action_heads=5 (not the
    project default of 3) is used deliberately here: ARCCustomNetwork's
    action_heads=3 branch hardcodes indices assuming a 5-dimensional
    action space that no longer exists (ARCGridWorld's is always 3-
    dimensional) - a separate, still-open issue, out of scope for this
    test."""
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5, feasible_actions=SUBMIT_ONLY)
    try:
        agent = create_agent(rl_config={"model_type": "PPO"}, vec_env=vec_env,
                              model_config={"action_heads": 5})
        assert agent is not None
    finally:
        vec_env.close()


def test_create_agent_restores_a_checkpoint(subtask, tmp_path):
    """The pretrained branch called `agent.load(...)` before `agent` was
    bound to anything, so it raised UnboundLocalError naming a local
    variable - which says nothing about the path argument that got you
    there. load is a classmethod on the algorithm, and the env has to go
    with it or the restored model has no spaces to check against."""
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5, feasible_actions=SUBMIT_ONLY)
    try:
        original = create_agent(rl_config={"model_type": "PPO"}, vec_env=vec_env,
                                model_config={"action_heads": 5})
        checkpoint = tmp_path / "agent.zip"
        original.save(checkpoint)

        restored = create_agent(rl_config={"model_type": "PPO"}, vec_env=vec_env,
                                 path_to_pretrained=str(checkpoint))

        assert restored is not None
        assert restored.get_env() is not None
    finally:
        vec_env.close()


def test_create_agent_names_an_unsupported_model_type(subtask):
    """Every branch has to either return an agent or say why it can't - one
    that falls through returns an unbound local, and the error then names
    the variable rather than the argument that caused it."""
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5, feasible_actions=SUBMIT_ONLY)
    try:
        with pytest.raises(ValueError, match="DQN"):
            create_agent(rl_config={"model_type": "DQN"}, vec_env=vec_env)
    finally:
        vec_env.close()


# -- the observation matches the space it was declared under -----------------

def _multi_object_subtask() -> ARCSubtask:
    """Built by hand rather than taken from the fixture task, which resizes:
    the env starts from a zeroed grid of the *output* shape while its objects
    come from the input, so a resizing task puts object coordinates outside
    the grid and transformations index out of bounds - unrelated to what
    these tests check. Several separate objects, so there are pairs for the
    relation embeddings to be about.
    """
    inp = np.zeros((8, 8), dtype=int)
    inp[1:3, 1:3] = 3          # a 2x2 block
    inp[5, 1:5] = 4            # a horizontal line
    inp[1:4, 6] = 2            # a vertical line
    out = inp.copy()
    out[5, 1:5] = 3
    return ARCSubtask("shape_preserving", inp, out)


@pytest.mark.parametrize("elements", [
    ["objects_emb"],
    ["relations_emb"],
    ["objects_emb", "relations_emb"],
    ["objects_emb", "relations_emb", "target"],
])
def test_every_observation_falls_inside_the_declared_space(elements):
    """The point of declaring a space is that observations belong to it.
    Three things have to line up for that: the width (OBJECT_DIM and
    RELATION_DIM, not a constant written out beside them), the dtype (the
    embeddings are real-valued, and the grid's integer dtype rounds every
    fraction away), and the bounds (size_ratio is a ratio of areas, so no
    finite upper bound holds).
    """
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE, observation_space_elements=elements)
    env.set_subtask(subtask)

    obs, _ = env.reset(seed=0)

    for key, value in obs.items():
        assert env.observation_space[key].contains(value), (
            f"reset's {key!r} {np.shape(value)} {np.asarray(value).dtype} is outside "
            f"{env.observation_space[key]}"
        )


def test_the_observation_keeps_its_type_across_a_step():
    """reset() cast the embeddings to the grid's dtype and step() did not,
    so the policy saw ints on the first observation and floats on the
    next."""
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE,
                   observation_space_elements=["objects_emb", "relations_emb"])
    env.set_subtask(subtask)
    first, _ = env.reset(seed=0)

    after_step, _, _, _, _ = env.step(np.array([1, 0, 0]))

    for key in ("objects_emb", "relations_emb"):
        assert after_step[key].dtype == first[key].dtype
        assert after_step[key].shape == first[key].shape
        assert env.observation_space[key].contains(after_step[key])


def test_the_embeddings_keep_their_fractional_values():
    """What the integer cast destroyed: shape_similarity, normalized_distance
    and the rest are fractions, and rounding them leaves the policy reading a
    handful of 0s and 1s where a measurement was."""
    subtask = _multi_object_subtask()
    env = make_env(observation_space_elements=["objects_emb", "relations_emb"])
    env.set_subtask(subtask)

    obs, _ = env.reset(seed=0)

    embeddings = np.concatenate([obs["objects_emb"].ravel(), obs["relations_emb"].ravel()])
    assert np.any(embeddings % 1 != 0), "every value is a whole number - they were rounded"


def test_relations_are_shaped_by_the_object_count():
    """One object means no pairs, but the observation still has to be the
    shape its Box declares - an unshaped empty array belongs to no space."""
    subtask = _multi_object_subtask()
    env = make_env(observation_space_elements=["relations_emb"])
    env.set_subtask(subtask)
    obs, _ = env.reset(seed=0)

    n_objects = len(env.initial_objects)

    assert obs["relations_emb"].ndim == 2
    assert obs["relations_emb"].shape[0] == n_objects
    assert env.observation_space["relations_emb"].contains(obs["relations_emb"])
