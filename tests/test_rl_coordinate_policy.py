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
from torch.distributions import Categorical

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


def small_network(heads="autoregressive", actions=2, rows=3, cols=3, dim=4, latent=16):
    from rl.policy import ARCCustomNetwork
    return ARCCustomNetwork(feature_dim=8 + (rows + cols) * (dim + 1),
                            action_dims=[actions, rows, cols, rows, cols],
                            net_arch={"pi": [latent], "vf": [latent]}, action_heads=5,
                            coordinate_shape=(rows, cols), coordinate_dim=dim,
                            coordinate_heads=heads)


def latent_for(network, batch=1, seed=0, rows=3, cols=3, dim=4, latent=16):
    from rl.policy import CoordinateLatent
    generator = torch.Generator().manual_seed(seed)
    return CoordinateLatent(torch.randn(batch, latent, generator=generator),
                            torch.randn(batch, rows, dim, generator=generator),
                            torch.ones(batch, rows),
                            torch.randn(batch, cols, dim, generator=generator),
                            torch.ones(batch, cols))


def every_action(actions=2, rows=3, cols=3):
    import itertools
    return torch.tensor(list(itertools.product(range(actions), range(rows), range(cols),
                                               range(rows), range(cols))))


class TestAutoregressiveCoordinateHeads:
    """Each coordinate chosen knowing the action and the cells before it -
    see AutoregressiveCoordinateDistribution."""

    def _distribution(self, network, latent):
        from rl.policy import AutoregressiveCoordinateDistribution
        return AutoregressiveCoordinateDistribution(network).proba_distribution(latent)

    def test_it_is_the_default(self, built):
        from rl.policy import AutoregressiveCoordinateDistribution
        agent, vec_env = built
        obs = vec_env.reset()
        obs_tensor, _ = agent.policy.obs_to_tensor(obs)
        assert isinstance(agent.policy.get_distribution(obs_tensor),
                          AutoregressiveCoordinateDistribution)

    def test_the_joint_is_a_distribution(self):
        """Summed over every one of the 162 actions, the probabilities the
        walk assigns come to one - the conditionals multiply into a joint,
        not into something that only looks like one."""
        network = small_network()
        everything = every_action()
        latent = latent_for(network)
        repeated = type(latent)(*(t.expand(len(everything), *t.shape[1:]) for t in latent))
        total = self._distribution(network, repeated).log_prob(everything).exp().sum()
        assert total.item() == pytest.approx(1.0, abs=1e-5)

    def test_a_later_choice_depends_on_an_earlier_one(self):
        network = small_network()
        latent = latent_for(network)
        distribution = self._distribution(network, latent)
        _, first = distribution._walk(given=torch.tensor([[0, 0, 0, 0, 0]]))
        _, other_action = distribution._walk(given=torch.tensor([[1, 0, 0, 0, 0]]))
        _, other_row = distribution._walk(given=torch.tensor([[0, 2, 0, 0, 0]]))
        _, other_column = distribution._walk(given=torch.tensor([[0, 0, 2, 0, 0]]))
        # i1 reads the action; j1 the row chosen for i1, the action held;
        # i2 the column chosen for j1.
        assert not torch.allclose(first[1].logits, other_action[1].logits)
        assert not torch.allclose(first[2].logits, other_row[2].logits)
        assert not torch.allclose(first[3].logits, other_column[3].logits)

    def test_a_sample_is_scored_as_a_fresh_walk_would_score_it(self):
        """sample() keeps its walk so log_prob costs nothing; PPO later
        scores the same actions from the buffer on a new distribution,
        and the two have to agree or the ratio starts off wrong."""
        network = small_network()
        latent = latent_for(network, batch=64)
        torch.manual_seed(0)
        distribution = self._distribution(network, latent)
        actions = distribution.sample()
        kept = distribution.log_prob(actions)
        fresh = self._distribution(network, latent).log_prob(actions.clone().float())
        assert torch.allclose(kept, fresh, atol=1e-6)

    def test_the_mode_is_the_greedy_walk(self):
        network = small_network()
        distribution = self._distribution(network, latent_for(network))
        mode = distribution.mode()
        _, dists = distribution._walk(given=mode)
        assert all(int(mode[0, i]) == int(d.probs.argmax()) for i, d in enumerate(dists))

    @pytest.mark.parametrize("heads,crossed_at_most,crossed_at_least",
                             [("autoregressive", 0.05, 0.0), ("independent", 1.0, 0.3)])
    def test_two_boxes_are_wanted_without_the_one_spanning_both(
            self, heads, crossed_at_most, crossed_at_least):
        """The reason for all this. Fitted by maximum likelihood to two
        strokes, (0,0)-(1,1) and (1,1)-(2,2), independent heads can only
        learn each coordinate's marginal, and half of what they then draw
        mixes the two - (0,0)-(2,2), (1,1)-(1,1) and the rest. The
        autoregressive heads learn the pair."""
        torch.manual_seed(0)
        network = small_network(heads)
        latent = latent_for(network, batch=1)
        wanted = torch.tensor([[1, 0, 0, 1, 1], [1, 1, 1, 2, 2]])
        batch = type(latent)(*(t.expand(2, *t.shape[1:]) for t in latent))
        optimiser = torch.optim.Adam(network.parameters(), lr=0.05)
        for _ in range(300):
            optimiser.zero_grad()
            loss = -self._log_prob(network, batch, wanted, heads).mean()
            loss.backward()
            optimiser.step()
        everything = every_action()
        spread = type(latent)(*(t.expand(len(everything), *t.shape[1:]) for t in latent))
        probs = self._log_prob(network, spread, everything, heads).exp()
        is_wanted = (everything[:, None, :] == wanted[None]).all(-1).any(-1)
        crossed = probs[~is_wanted].sum().item()
        assert crossed_at_least <= crossed <= crossed_at_most

    def _log_prob(self, network, latent, actions, heads):
        if heads == "autoregressive":
            return self._distribution(network, latent).log_prob(actions)
        # The independent heads as forward_actor runs them, on the same
        # context, so both arms start from the same latent.
        action_head, *cell_heads = network.policy_nets
        grids = [(latent.rows, latent.row_mask), (latent.cols, latent.col_mask)] * 2
        logits = [action_head(latent.context)] + [
            head(latent.context, rows, mask) for head, (rows, mask) in zip(cell_heads, grids)]
        return sum(Categorical(logits=scores).log_prob(actions[:, i])
                   for i, scores in enumerate(logits))

    def test_independent_heads_are_still_there_when_asked_for(self):
        from stable_baselines3.common.distributions import MultiCategoricalDistribution
        agent, vec_env = coordinate_agent_with({"coordinate_heads": "independent"})
        try:
            obs_tensor, _ = agent.policy.obs_to_tensor(vec_env.reset())
            assert isinstance(agent.policy.get_distribution(obs_tensor),
                              MultiCategoricalDistribution)
        finally:
            vec_env.close()

    def test_ppo_trains_through_it(self):
        agent, vec_env = coordinate_agent_with({})
        try:
            agent.learn(32)
        finally:
            vec_env.close()

    def test_an_unknown_setting_is_refused(self):
        with pytest.raises(ValueError, match="coordinate_heads"):
            small_network("sideways")


def coordinate_agent_with(extra):
    vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4,
                             feasible_actions=ACTIONS,
                             observation_space_elements=["delta_input", "delta_target"],
                             observation_grid_shape=(7, 7), repr_level=1,
                             input_pattern="start", addressing="coordinates",
                             coordinate_shape=(7, 7))
    agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                         {"n_steps": 16, "batch_size": 8, "verbose": 0,
                          "coordinate_dim": 8, **extra})
    return agent, vec_env
