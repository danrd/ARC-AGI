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


@pytest.mark.parametrize("seed", range(8))
def test_maximal_intersection_matches_the_form_it_replaced(seed):
    """The rewrite is algebra, not a new definition: both halves ranged over
    the same valid region, so matches - misses is 2*matches - |valid|. Held
    against the original expression on random grids, because a scoring
    function that is merely close would move every reward in the system by a
    little and be very hard to notice.
    """
    rng = np.random.default_rng(seed)
    env = make_env(pad_val=10)
    env.train_out = rng.integers(0, 11, (6, 6))
    grid = rng.integers(0, 11, (6, 6))
    target = np.asarray(env.train_out)

    original = (((grid == target) * (grid != 10) * (target != 10)).sum()
                - ((grid != target) * (grid != 10) * (target != 10)).sum())

    assert env.maximal_intersection(grid) == original


def test_the_target_validity_mask_follows_the_target():
    """The mask is derived from train_out, so assigning a new target has to
    rebuild it - left over from the previous subtask it would score every
    grid against the wrong region, silently and forever."""
    env = make_env(pad_val=10)
    env.train_out = np.array([[10, 10], [10, 10]])
    assert env.maximal_intersection(np.array([[1, 1], [1, 1]])) == 0

    env.train_out = np.array([[1, 1], [1, 1]])

    assert env.maximal_intersection(np.array([[1, 1], [1, 1]])) == 4


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
    # Object slots, not this subtask's object count: the action space is the
    # same for every subtask so one agent can be trained across several.
    assert env.action_space.nvec[1] == env.max_objects
    assert env.action_space.nvec[2] == env.max_objects


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


def test_reward_approach_2_pays_for_partial_progress(subtask):
    """The one approach whose point is a gradient short of the answer.

    Asked of _submit_reward across the range rather than waited for from a
    rollout, because nothing solves these tasks and the interesting values
    are exactly the ones a rollout never reaches. Approach 2 paid the same
    penalty at every intersection below the target, which left the fully
    solved submit as the only positive one in any approach - so a search
    maximising total reward has nothing to climb towards and never submits.
    """
    env = make_env(reward_approach=2)
    env.set_subtask(subtask)
    env.reset()
    milestones = sorted(env.milestones)

    nothing = env._submit_reward(env.max_int)
    partial = [env._submit_reward(m) for m in milestones[:-1]]
    solved = env._submit_reward(env.target_int)

    assert nothing < 0, "a submit that achieved nothing should still cost"
    assert all(p > 0 for p in partial), \
        f"every reached milestone should pay: {partial}"
    assert partial == sorted(partial), f"payment should grow with progress: {partial}"
    assert solved > max(partial), "solving should beat any partial result"


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
    # objects_emb alone: this is about create_agent's branching, and the
    # default observation space builds a 133M-parameter extractor twice over
    # (see MAX_OBJECTS - the relation block is quadratic in it).
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5, feasible_actions=SUBMIT_ONLY,
                              observation_space_elements=["objects_emb"])
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


def test_a_handed_down_agent_is_pointed_at_the_env_it_was_handed(subtask):
    """train_on_task passes one agent through every subtask, and each call
    builds its own vec_env. create_agent used to return the agent untouched,
    so it went on collecting rollouts in subtask 0's env while being
    evaluated on the new one - five accuracies that read as five trainings
    were one policy measured on five grids."""
    first = create_vec_env([subtask], n_envs=1, max_episode_len=5,
                            feasible_actions=SUBMIT_ONLY,
                            observation_space_elements=["objects_emb"])
    second = create_vec_env([subtask], n_envs=1, max_episode_len=5,
                             feasible_actions=SUBMIT_ONLY,
                             observation_space_elements=["objects_emb"])
    try:
        agent = create_agent(rl_config={"model_type": "PPO"}, vec_env=first,
                             model_config={"action_heads": 5})

        again = create_agent(rl_config={"model_type": "PPO"}, vec_env=second,
                             agent_init=agent)

        assert again is agent  # the same policy, carried forward
        assert again.get_env() is second
    finally:
        first.close()
        second.close()


def test_create_agent_names_an_unsupported_model_type(subtask):
    """Every branch has to either return an agent or say why it can't - one
    that falls through returns an unbound local, and the error then names
    the variable rather than the argument that caused it."""
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=5, feasible_actions=SUBMIT_ONLY,
                              observation_space_elements=["objects_emb"])
    try:
        with pytest.raises(ValueError, match="DQN"):
            create_agent(rl_config={"model_type": "DQN"}, vec_env=vec_env)
    finally:
        vec_env.close()


# -- the observation matches the space it was declared under -----------------

def _multi_object_subtask(n_objects: int = 3) -> ARCSubtask:
    """Built by hand rather than taken from the fixture task, which resizes:
    the env starts from a zeroed grid of the *output* shape while its objects
    come from the input, so a resizing task puts object coordinates outside
    the grid and transformations index out of bounds - unrelated to what
    these tests check. Several separated objects, so there are pairs for the
    relation embeddings to be about.
    """
    inp = np.zeros((8, 8), dtype=int)
    spots = [(1, 1), (1, 4), (1, 6), (4, 1), (4, 4), (6, 6), (6, 1)]
    for k, (i, j) in enumerate(spots[:n_objects]):
        inp[i, j] = k % 9 + 1
    out = inp.copy()
    out[spots[0]] = 9
    return ARCSubtask(f"shape_preserving_{n_objects}", inp, out)


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


def test_the_embedding_blocks_are_sized_by_the_slot_count_not_the_task():
    """The whole point of max_objects: two subtasks with different object
    counts have to produce the same shapes, or one agent cannot see rollouts
    from both."""
    small = _multi_object_subtask()
    large = _multi_object_subtask(n_objects=6)
    shapes = []
    for subtask in (small, large):
        env = make_env(observation_space_elements=["objects_emb", "relations_emb"])
        env.set_subtask(subtask)
        obs, _ = env.reset(seed=0)
        shapes.append({k: obs[k].shape for k in ("objects_emb", "relations_emb")})
        assert obs["objects_emb"].shape[0] == env.max_objects
        assert obs["relations_emb"].shape[0] == env.max_objects

    assert len(env.initial_objects) > 3  # the two subtasks really do differ
    assert shapes[0] == shapes[1]


def test_padding_rows_are_zero_and_real_objects_are_not():
    """How a consumer tells the two apart without a separate mask - the
    convention ARCGNNExtractor already reads."""
    subtask = _multi_object_subtask()
    env = make_env(observation_space_elements=["objects_emb"])
    env.set_subtask(subtask)
    obs, _ = env.reset(seed=0)

    n_objects = len(env.initial_objects)
    occupied = obs["objects_emb"][:n_objects]
    padding = obs["objects_emb"][n_objects:]

    assert not padding.any()
    assert all(row.any() for row in occupied)


def test_an_action_naming_an_empty_slot_does_nothing_rather_than_raising():
    """The action space has max_objects slots whatever the subtask holds, so
    a sampled index can name a slot no object occupies. That has to be an
    action that changes nothing - not an IndexError, and not quietly
    redirected to some other object."""
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE,
                   observation_space_elements=["objects_emb", "relations_emb"])
    env.set_subtask(subtask)
    before, _ = env.reset(seed=0)
    empty_slot = env.max_objects - 1
    assert empty_slot >= len(env.initial_objects)

    obs, reward, done, truncated, info = env.step(np.array([1, empty_slot, 0]))

    assert np.array_equal(obs["grid"], before["grid"])
    assert reward < 0  # scored as an ineffective action
    assert env.observation_space["relations_emb"].contains(obs["relations_emb"])


# -- reward normalisation: the scale a real task puts the penalty on --------
#
# `round(reward / self.max_reward, 2)` used to sit at both normalisation
# sites. max_reward scales with the distance to the target, so on a real
# task it is in the thousands and the penalty for a useless action -
# -1/max_reward - rounded to exactly 0.00. Measured over 2000 random steps
# per task: with the rounding, 93% to 99% of steps paid exactly zero on
# three of four tasks; without it, 0.1% to 0.7%. MCTS never noticed
# (playouts pick actions without consulting reward, and removing the
# rounding changed nothing over 24 tasks), but PPO learns from nothing
# else. The two tests below are on a real ARC subtask on purpose: the
# synthetic ones above are close enough to their target that two decimals
# still carry the penalty, which is exactly why this went unseen.

def test_a_useless_step_is_paid_for_on_a_task_whose_reward_scale_is_large(subtask):
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()
    assert env.max_reward > 200, \
        f"needs a task where the penalty is below two decimals: {env.max_reward}"
    empty_slot = env.max_objects - 1

    _, reward, _, _, _ = env.step(np.array([1, empty_slot, empty_slot]))

    assert reward == pytest.approx(-env.action_penalty / env.max_reward)
    assert reward != 0.0


def test_the_simulated_step_pays_the_same_as_the_real_one(subtask):
    """MCTS scores nodes through simulate_action, which normalises at its
    own site - the two have to agree or a rollout values an action the env
    does not."""
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset()
    empty_slot = env.max_objects - 1
    action = np.array([1, empty_slot, empty_slot])

    _, _, _, simulated, _ = env.simulate_action(
        action, env.objects, env.grid, env.max_int, None)
    _, stepped, _, _, _ = env.step(action)

    assert simulated == pytest.approx(stepped)
    assert simulated != 0.0


def test_a_vec_env_spans_subtasks_with_different_object_counts():
    """What the fixed shapes buy: training on a mix of subtasks instead of
    one at a time. This raised `could not broadcast input array from shape
    (5,96) into shape (2,24)` - DummyVecEnv takes the first env's space and
    every other env's observation has to fit it."""
    subtasks = [_multi_object_subtask(), _multi_object_subtask(n_objects=6)]
    vec_env = create_vec_env(subtasks, n_envs=1, max_episode_len=5,
                              feasible_actions=SUBMIT_ONLY,
                              observation_space_elements=["objects_emb", "relations_emb"])
    try:
        obs = vec_env.reset()

        assert vec_env.num_envs == 2
        assert obs["relations_emb"].shape[0] == 2  # one row of observations per env
        assert obs["objects_emb"].shape[0] == 2
    finally:
        vec_env.close()


# -- what the default reward asks the agent to do ---------------------------
#
# Characterisation, not approval. Under the shipped default a submit that
# solved nothing pays 0 and never less, while acting costs -0.01 to -0.03 a
# step on the measured tasks - so giving up on the first step is the best
# return this reward offers, and PPO finds it: of six 30k runs, four ended
# submitting on 100% of steps, transform-head entropy 0.007-0.023 out of
# ~4.0, and the share of steps closing any distance fell from 17-20% to
# zero. ent_coef does not hold against that.
#
# approach 2, which charges for an empty submit and pays partial credit,
# does stop the collapse - submits to 0.0% of steps, entropy holding at
# 2.6-2.8 - and was the default briefly for that reason. By the outcome, the
# fraction of the distance the trained policy closes, it was no better on
# any of three tasks and worse on two (-1.513 against -1.392; -0.625 against
# -0.250, held-out -0.750 against 0.000), so it was reverted. These pin what
# the default actually does, so that replacing it is a decision someone
# makes rather than a line that drifts.

def test_the_default_reward_pays_nothing_for_giving_up(subtask):
    """And so leaves submitting immediately at least as good as acting,
    which is the shape of the collapse measured above - not a property to
    preserve, a property to know about."""
    from data.configs.rl_configs import rl_config

    env = make_env(reward_approach=rl_config["reward_approach"])
    env.set_subtask(subtask)
    env.reset()

    assert env._submit_reward(env.base_int) == 0


def test_the_default_reward_says_nothing_about_getting_part_way(subtask):
    """Every submit short of the solved one pays the same, so the reward
    carries no signal that half way beats nowhere. The step channel is
    where all of the gradient lives."""
    from data.configs.rl_configs import rl_config

    env = make_env(reward_approach=rl_config["reward_approach"])
    env.set_subtask(subtask)
    env.reset()

    milestones = list(env.milestones)
    short_of_solved = [env._submit_reward(m) for m in milestones[:-1]]

    assert len(set(short_of_solved)) == 1, (
        f"partial results are no longer indistinguishable: {short_of_solved}")
    assert env._submit_reward(milestones[-1]) > short_of_solved[0]


def test_both_config_sources_agree_on_the_reward():
    """rl.rl_module.RlConfig restates data.configs.rl_configs.rl_config for
    pydantic's sake, and the two are only useful while they say the same."""
    from data.configs.rl_configs import rl_config
    from rl.rl_module import RlConfig

    assert RlConfig().reward_approach == rl_config["reward_approach"]


# -- what the grid starts as, and whether the input is shown ----------------

@pytest.mark.parametrize("pattern,from_input,shown", [
    ("start", True, False),
    ("separate", False, True),
    ("start_separate", True, True),
    (False, False, False),
])
def test_input_pattern_decides_the_start_and_the_key_independently(
        pattern, from_input, shown):
    """'separate' beat 'start' by the widest margin of eight arms, and the
    comparison cannot say why: it moves the starting grid to zeros *and*
    adds the input to the observation at once. Four spellings, two
    questions, so the next sweep can hold one still and move the other.
    """
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE, input_pattern=pattern)
    env.set_subtask(subtask)

    obs, _ = env.reset(seed=0)

    starts_as_input = np.array_equal(env.grid, subtask.train_inp)
    assert starts_as_input is from_input, (
        f"input_pattern={pattern!r}: grid starts as the input? "
        f"{starts_as_input}, wanted {from_input}")
    assert ("input_pattern" in obs) is shown
    assert ("input_pattern" in env.observation_space.spaces) is shown
    if shown:
        assert np.array_equal(obs["input_pattern"], subtask.train_inp)


def test_the_shown_input_stays_the_input_after_the_grid_moves():
    """Under 'start_separate' the two start out equal, so the key is only
    worth its width if it keeps holding the original once the grid is
    edited - otherwise it is a second copy of the grid."""
    # An L, not the single cells _multi_object_subtask draws: a one-cell
    # object rotates onto itself and the step would leave the grid alone,
    # which is exactly the case this test cannot tell from a bug.
    inp = np.zeros((8, 8), dtype=int)
    inp[1, 1] = inp[1, 2] = inp[2, 1] = 3
    inp[5, 5] = inp[5, 6] = 4
    out = inp.copy()
    out[0, 0] = 9
    subtask = ARCSubtask("L_shaped", inp, out)
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE,
                   input_pattern="start_separate")
    env.set_subtask(subtask)
    env.reset(seed=0)

    obs, *_ = env.step(np.array([1, 0, 0]))

    assert np.array_equal(obs["input_pattern"], subtask.train_inp)
    assert not np.array_equal(obs["grid"], obs["input_pattern"])


def test_a_zeroed_start_claims_objects_its_grid_does_not_hold():
    """Characterisation, not a wish: this is what a zeroed start does now.

    set_subtask reads initial_objects off `subtask.train_inp`, whatever the
    episode starts on. Start from zeros and the policy is handed a full
    object block, and an action space indexing it, for objects that are
    nowhere in the grid it is editing. A zeroed start is for a setup that
    builds objects up by recolouring cells; this env has no such setup, and
    until it does, the state below is the reason 'separate' is not a
    candidate.
    """
    subtask = _multi_object_subtask(n_objects=3)
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE, input_pattern="separate")
    env.set_subtask(subtask)
    env.reset(seed=0)

    assert np.count_nonzero(env.grid) == 0
    assert len(env.initial_objects) == 3, "objects come from the input"
    assert env.visible_object_count() == 3, (
        "and the action space addresses all three against an empty grid")


def test_the_starting_grid_is_integer_whatever_it_starts_as():
    """np.zeros is float64, and only observed_grid cast it - so the
    observation matched its declared int64 space while self.grid, which
    maximal_intersection and GridSummary both read, did not."""
    subtask = _multi_object_subtask()
    for pattern in ("start", "separate", "start_separate", False):
        env = make_env(feasible_actions=SUBMIT_AND_ROTATE, input_pattern=pattern)
        env.set_subtask(subtask)
        env.reset(seed=0)
        assert np.asarray(env.grid).dtype == env.grid_dtype, (
            f"input_pattern={pattern!r} left the grid "
            f"{np.asarray(env.grid).dtype}")


def test_the_accuracy_denominator_moves_with_the_starting_grid():
    """Why an arm that changes the starting grid cannot be compared to one
    that does not: rl.training.distance_to_close is
    target_int - base_int, base_int is the intersection at reset, and reset
    depends on input_pattern. Measured over six shape-preserving training
    tasks the span is 4/32/84/86/24/40 from the input against
    16/40/112/98/20/496 from zeros - the same task, a denominator up to 12
    times larger.
    """
    subtask = _multi_object_subtask()
    spans = {}
    for pattern in ("start", "separate"):
        env = make_env(feasible_actions=SUBMIT_AND_ROTATE, input_pattern=pattern)
        env.set_subtask(subtask)
        env.reset(seed=0)
        spans[pattern] = env.target_int - env.base_int

    assert spans["start"] != spans["separate"], (
        f"the spans agree here ({spans}), so this task no longer shows what "
        "the test is about - pick one whose input is not near its output")


def test_start_separate_keeps_the_ruler_the_start_arm_uses():
    """The point of the arm: it moves the observation and nothing else, so
    its accuracies are on the same scale as base's and the paired
    difference means what it says."""
    subtask = _multi_object_subtask()
    spans = {}
    for pattern in ("start", "start_separate"):
        env = make_env(feasible_actions=SUBMIT_AND_ROTATE, input_pattern=pattern)
        env.set_subtask(subtask)
        env.reset(seed=0)
        spans[pattern] = (env.base_int, env.target_int)

    assert spans["start"] == spans["start_separate"], spans


# -- an explicit list of the triples worth choosing from -------------------

def _two_Ls() -> ARCSubtask:
    """Two three-cell Ls on a grid that starts as the input.

    Single cells rotate and flip onto themselves, and ARCGridWorld's own
    default input_pattern is False, which starts the grid at zeros: between
    them, a test written on _multi_object_subtask and make_env's defaults
    has no action that changes anything, and passes whatever the env does.
    """
    inp = np.zeros((9, 9), dtype=int)
    inp[1, 1] = inp[1, 2] = inp[2, 1] = 3
    inp[5, 5] = inp[5, 6] = inp[6, 5] = 4
    out = inp.copy()
    out[0, 0] = 9
    return ARCSubtask("two_Ls", inp, out)


def _live_triples(env, subtask):
    """Every (transform, object, object) that moves this grid from reset."""
    env.set_subtask(subtask)
    env.reset(seed=0)
    start = np.array(env.grid).copy()
    state = env.get_state()
    objects = env.visible_object_count()
    live = []
    for transform in range(1, len(env.actions_dict)):
        for first in range(objects):
            for second in range(objects):
                env.set_state(state)
                env.step(np.array([transform, first, second]))
                if not np.array_equal(np.array(env.grid), start):
                    live.append((transform, first, second))
    env.set_state(state)
    return live


def test_a_whitelist_makes_every_index_name_a_triple():
    """The point of the list: the policy picks an index into it, and the
    env turns that into the (transform, object, object) it stands for."""
    subtask = _multi_object_subtask(n_objects=3)
    whitelist = [(0, 0, 0), (1, 2, 1), (1, 0, 2)]
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE, action_whitelist=whitelist)
    env.set_subtask(subtask)
    env.reset(seed=0)

    assert list(env.action_space.nvec) == [3, 1, 1]
    for index, triple in enumerate(whitelist):
        assert tuple(env.resolved_action(np.array([index, 0, 0]))) == triple


def test_a_whitelisted_env_still_submits():
    """A list without submit in it is an episode that can only time out, so
    the translation has to happen before the submit check, not after."""
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE,
                   action_whitelist=[(1, 0, 0), (0, 0, 0)])
    env.set_subtask(subtask)
    env.reset(seed=0)

    _, _, done, _, _ = env.step(np.array([1, 0, 0]))  # index 1 is submit

    assert done is True


def test_no_whitelist_leaves_the_action_space_alone():
    subtask = _multi_object_subtask()
    env = make_env(feasible_actions=SUBMIT_AND_ROTATE)
    env.set_subtask(subtask)
    env.reset(seed=0)

    assert list(env.action_space.nvec) == [2, env.max_objects, env.max_objects]
    assert tuple(env.resolved_action(np.array([1, 2, 3]))) == (1, 2, 3)


def test_a_live_whitelist_removes_the_no_ops_and_keeps_the_rest():
    """What the ablation rests on: an env restricted to the triples that
    move the grid has to still move it on every choice, and has to still
    offer all of them."""
    # Multi-cell objects, because rotating or flipping a single cell maps it
    # onto itself: _multi_object_subtask draws single cells and has no live
    # action under these two transforms at all.
    inp = np.zeros((9, 9), dtype=int)
    inp[1, 1] = inp[1, 2] = inp[2, 1] = 3
    inp[5, 5] = inp[5, 6] = inp[6, 5] = 4
    out = inp.copy()
    out[0, 0] = 9
    subtask = ARCSubtask("two_Ls", inp, out)
    # input_pattern='start', because ARCGridWorld's own default is False and
    # that starts the grid at zeros - an empty grid no transform can move,
    # which is a no-op rate of 100% for reasons that have nothing to do with
    # what this test is about.
    actions = {0: "submit", 1: "rotate90", 2: "flip_h"}
    probe = make_env(feasible_actions=actions, input_pattern="start")
    live = _live_triples(probe, subtask)
    assert live, "the probe found nothing live, so this test proves nothing"

    env = make_env(feasible_actions=actions, input_pattern="start",
                   action_whitelist=[(0, 0, 0)] + live)
    env.set_subtask(subtask)
    env.reset(seed=0)
    start = np.array(env.grid).copy()
    state = env.get_state()

    moved = 0
    for index in range(1, len(live) + 1):
        env.set_state(state)
        env.step(np.array([index, 0, 0]))
        if not np.array_equal(np.array(env.grid), start):
            moved += 1

    assert moved == len(live), f"{len(live) - moved} of {len(live)} did nothing"


def test_a_restored_state_probes_the_same_way_whatever_came_before():
    """The property the whitelist probe needs and the old get_state did not
    have: restoring a state and trying an action has to give the same answer
    regardless of which actions were tried before it. World mutates the
    GridObjects in place, so a restore that put the grid back and left the
    objects moved answered differently depending on the order.
    """
    subtask = _two_Ls()
    actions = {0: "submit", 1: "rotate90", 2: "flip_h"}
    env = make_env(feasible_actions=actions, input_pattern="start")
    env.set_subtask(subtask)
    env.reset(seed=0)
    state = env.get_state()
    probe = (2, 1, 1)

    env.set_state(state)
    env.step(np.array(probe))
    alone = np.array(env.grid).copy()

    env.set_state(state)
    for other in [(1, 0, 0), (2, 1, 1), (1, 1, 1)]:
        env.step(np.array(other))
    assert not np.array_equal(np.array(env.grid), state["grid"]), (
        "nothing moved in between, so order cannot have mattered"
    )
    env.set_state(state)
    env.step(np.array(probe))

    assert np.array_equal(np.array(env.grid), alone), (
        "the same action from the same restored state produced two grids")


def test_a_restored_state_restores_the_objects_and_the_baseline():
    subtask = _two_Ls()
    env = make_env(feasible_actions={0: "submit", 1: "rotate90", 2: "flip_h"},
                   input_pattern="start",
                   observation_space_elements=["objects_emb", "relations_emb"])
    env.set_subtask(subtask)
    env.reset(seed=0)
    state = env.get_state()
    # coords, which is what World.apply_transform moves. An earlier version
    # of this read obj.cells, which GridObject does not have, so hasattr
    # made every entry None and the comparison below held for any env at all.
    before = [np.asarray(obj.coords).copy() for obj in env.objects]
    max_int_before = env.max_int

    for action in [(1, 0, 0), (2, 1, 1), (1, 1, 1)]:
        env.step(np.array(action))
    assert not np.array_equal(np.array(env.grid), state["grid"]), (
        "no action moved anything, so the restore below proves nothing")
    env.set_state(state)

    assert env.max_int == max_int_before
    after = [np.asarray(obj.coords).copy() for obj in env.objects]
    assert len(after) == len(before)
    for one, two in zip(before, after):
        assert np.array_equal(one, two), f"{one} became {two}"
