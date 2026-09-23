"""The network half of coordinate addressing.

Under coordinates an action is (action, i1, j1, i2, j2), and the four
coordinate heads choose a row or a column by scoring each one from its own
embedding - CoordinateRows builds those, ARCCombinedExtractor appends them
to the features, ARCCustomNetwork cuts them back off and hands them to the
heads. The answer (delta_target) goes to the critic only, whatever the
config says. Pinned here: that each row's embedding is of that row, that
padding is never chosen, that a mismatched action space is refused, and
that the actor never reads the answer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.arc_task import ARCSubtask
from rl.features import ARCCombinedExtractor, CoordinateRows
from rl.policy import PointerHead
from rl.training import ANSWER_KEYS, create_agent, create_vec_env, critic_only

ACTIONS = {0: "submit", 1: "red_fill", 2: "red_line", 3: "red_triangle"}


def observation(grids, shapes=None, delta=None):
    """A batch the way the env hands it over: grids padded with 10."""
    grid = torch.tensor(np.stack(grids), dtype=torch.int64)
    obs = {"grid": grid}
    if shapes is not None:
        obs["grid_shape"] = torch.tensor(shapes, dtype=torch.int64)
    if delta is not None:
        obs["delta_input"] = torch.tensor(np.stack(delta), dtype=torch.float32)
    return obs


class TestTheRowsAndColumns:
    def test_a_padded_cell_is_no_colour(self):
        rows = CoordinateRows(3, 3, [], dim=4)
        grid = np.full((3, 3), 10)
        grid[0, 0] = 0
        planes = rows.planes(observation([grid]))
        assert planes[0, :, 0, 0].tolist() == [1.0] + [0.0] * 9
        assert planes[0, :, 2, 2].sum() == 0

    def test_the_masks_are_the_true_size(self):
        rows = CoordinateRows(4, 5, [], dim=4)
        row_mask, col_mask = rows.masks(observation([np.zeros((4, 5))], shapes=[[2, 3]]))
        assert row_mask.tolist() == [[1, 1, 0, 0]]
        assert col_mask.tolist() == [[1, 1, 1, 0, 0]]

    def test_a_row_embedding_reads_that_row_and_no_other(self):
        """Change one cell: its row and its column move, every other row
        and column stays - what lets a head score "the row where the
        painting is" instead of a row number."""
        torch.manual_seed(0)
        module = CoordinateRows(4, 5, ["delta_input"], dim=8)
        before = np.zeros((4, 5), dtype=int)
        after = before.copy()
        after[2, 3] = 4
        zeros = [np.zeros((4, 5))]
        r0, _m, c0, _n = module(observation([before], delta=zeros))
        r1, _m, c1, _n = module(observation([after], delta=zeros))
        moved_rows = [i for i in range(4) if not torch.allclose(r0[0, i], r1[0, i])]
        moved_cols = [j for j in range(5) if not torch.allclose(c0[0, j], c1[0, j])]
        assert moved_rows == [2]
        assert moved_cols == [3]

    def test_the_deltas_are_read(self):
        torch.manual_seed(0)
        module = CoordinateRows(3, 3, ["delta_input"], dim=8)
        grid = [np.zeros((3, 3), dtype=int)]
        delta = np.zeros((3, 3))
        delta[1, 1] = 1
        r0, *_ = module(observation(grid, delta=[np.zeros((3, 3))]))
        r1, *_ = module(observation(grid, delta=[delta]))
        assert not torch.allclose(r0[0, 1], r1[0, 1])
        assert torch.allclose(r0[0, 0], r1[0, 0])

    def test_the_tail_is_rows_then_columns_each_with_its_mask(self):
        module = CoordinateRows(2, 3, [], dim=4)
        obs = observation([np.zeros((2, 3))], shapes=[[1, 2]])
        tail = module.tail(obs)
        assert tail.shape == (1, module.width) == (1, (2 + 3) * 5)
        rows, row_mask, cols, col_mask = module(obs)
        blocks = tail.view(5, 5)
        assert torch.allclose(blocks[:2, :4], rows[0])
        assert blocks[:2, 4].tolist() == row_mask[0].tolist()
        assert torch.allclose(blocks[2:, :4], cols[0])
        assert blocks[2:, 4].tolist() == col_mask[0].tolist()

    def test_the_extractor_appends_them_last(self):
        space = spaces.Dict({
            "grid": spaces.Box(0, 10, shape=(4, 4), dtype=np.int64),
            "grid_shape": spaces.Box(0, 30, shape=(2,), dtype=np.int64),
            "delta_input": spaces.Box(-1, 1, shape=(4, 4), dtype=np.float32)})
        extractor = ARCCombinedExtractor(space, coordinate_dim=6)
        assert extractor.coordinate_rows.delta_keys == ("delta_input",)
        obs = observation([np.zeros((4, 4))], shapes=[[4, 4]], delta=[np.zeros((4, 4))])
        features = extractor(obs)
        assert features.shape[1] == extractor.features_dim
        width = extractor.coordinate_rows.width
        assert torch.allclose(features[:, -width:], extractor.coordinate_rows.tail(obs))

    def test_without_a_width_there_are_none(self):
        space = spaces.Dict({"grid": spaces.Box(0, 10, shape=(4, 4), dtype=np.int64)})
        assert ARCCombinedExtractor(space).coordinate_rows is None


def two_sizes():
    """A 5x5 and a 7x7 example: the action space spans 7x7, and on the
    small one rows and columns 5 and 6 are padding."""
    small = np.zeros((5, 5), dtype=int)
    small[1, 1] = 1
    large = np.zeros((7, 7), dtype=int)
    large[3, 3] = 1
    return [ARCSubtask("small", small, np.where(small == 0, 2, small)),
            ARCSubtask("large", large, np.where(large == 0, 2, large))]


def coordinate_agent(elements=("delta_input", "delta_target"), critic_only_keys=(),
                     subtasks=None):
    vec_env = create_vec_env(subtasks or two_sizes(), n_envs=1, max_episode_len=4,
                             feasible_actions=ACTIONS,
                             observation_space_elements=list(elements),
                             observation_grid_shape=(7, 7), repr_level=1,
                             input_pattern="start", addressing="coordinates",
                             coordinate_shape=(7, 7))
    agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                         {"n_steps": 16, "batch_size": 8, "verbose": 0,
                          "critic_only_keys": critic_only_keys, "coordinate_dim": 8})
    return agent, vec_env


@pytest.fixture(scope="module")
def built():
    agent, vec_env = coordinate_agent()
    yield agent, vec_env
    vec_env.close()


class TestTheCoordinateHeads:
    def test_one_linear_head_for_the_action_and_four_pointer_heads(self, built):
        agent, _ = built
        heads = list(agent.policy.mlp_extractor.policy_nets)
        assert isinstance(heads[0], torch.nn.Linear) and heads[0].out_features == len(ACTIONS)
        assert len(heads) == 5
        assert all(isinstance(head, PointerHead) for head in heads[1:])

    def test_padding_is_never_chosen(self, built):
        """Sampled, not argmax, and many times: a masked row has
        probability zero, not merely a low one."""
        agent, vec_env = built
        env = vec_env.envs[0].unwrapped
        env.set_subtask(two_sizes()[0])
        obs, _ = env.reset()
        batch = {key: np.repeat(np.asarray(value)[None], 256, axis=0)
                 for key, value in obs.items()}
        torch.manual_seed(0)
        actions, _ = agent.predict(batch, deterministic=False)
        assert actions[:, 1:].max() <= 4
        # And on the 7x7 one the far rows are reachable at all.
        env.set_subtask(two_sizes()[1])
        obs, _ = env.reset()
        batch = {key: np.repeat(np.asarray(value)[None], 256, axis=0)
                 for key, value in obs.items()}
        actions, _ = agent.predict(batch, deterministic=False)
        assert actions[:, 1:].max() > 4

    def test_rows_are_masked_as_rows_and_columns_as_columns(self):
        """A grid 5 rows high and 7 wide, in a 7x7 space: the row heads
        must stop at 4 and the column heads must not. On a square grid the
        two masks are the same and swapping them shows nothing."""
        wide = np.zeros((5, 7), dtype=int)
        wide[2, 5] = 1
        agent, vec_env = coordinate_agent(subtasks=[
            ARCSubtask("wide", wide, np.where(wide == 0, 2, wide)), two_sizes()[1]])
        try:
            env = vec_env.envs[0].unwrapped
            env.set_subtask(ARCSubtask("wide", wide, np.where(wide == 0, 2, wide)))
            obs, _ = env.reset()
            batch = {key: np.repeat(np.asarray(value)[None], 256, axis=0)
                     for key, value in obs.items()}
            torch.manual_seed(0)
            actions, _ = agent.predict(batch, deterministic=False)
            assert actions[:, [1, 3]].max() <= 4
            assert actions[:, [2, 4]].max() > 4
        finally:
            vec_env.close()

    def test_the_actor_never_reads_the_answer(self, built):
        """critic_only_keys was () here; delta_target is kept from the
        actor anyway, because the observation carries it."""
        agent, _ = built
        actor = agent.policy._actor_extractor()
        assert "delta_target" not in actor.extractors
        assert "delta_input" in actor.extractors
        assert actor.coordinate_rows.delta_keys == ("delta_input",)

    def test_an_action_space_the_grid_does_not_span_is_refused(self):
        from rl.policy import ARCCustomNetwork
        with pytest.raises(ValueError, match="must agree"):
            ARCCustomNetwork(feature_dim=10 + 2 * 7 * 9 + 0, action_dims=[4, 7, 6, 7, 6],
                             action_heads=5, coordinate_shape=(7, 7), coordinate_dim=8)


class TestTheAnswerIsTheCriticsAlone:
    def test_every_answer_key_the_observation_carries_is_added(self):
        space = spaces.Dict({key: spaces.Box(0, 1, shape=(1,)) for key in
                             ("grid", "delta_input", "delta_target", "target")})
        assert set(critic_only(space, ())) == set(ANSWER_KEYS)

    def test_what_is_configured_is_kept_and_nothing_doubled(self):
        space = spaces.Dict({key: spaces.Box(0, 1, shape=(1,)) for key in
                             ("grid", "objects_emb", "target")})
        assert critic_only(space, ("objects_emb", "target")) == ("objects_emb", "target")

    def test_an_answer_the_observation_lacks_is_not_invented(self):
        space = spaces.Dict({"grid": spaces.Box(0, 1, shape=(1,))})
        assert critic_only(space, ()) == ()
