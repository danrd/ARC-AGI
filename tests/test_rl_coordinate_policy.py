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
        repeated = type(latent)(*(None if t is None else t.expand(len(everything), *t.shape[1:]) for t in latent))
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
        batch = type(latent)(*(None if t is None else t.expand(2, *t.shape[1:]) for t in latent))
        optimiser = torch.optim.Adam(network.parameters(), lr=0.05)
        for _ in range(300):
            optimiser.zero_grad()
            loss = -self._log_prob(network, batch, wanted, heads).mean()
            loss.backward()
            optimiser.step()
        everything = every_action()
        spread = type(latent)(*(None if t is None else t.expand(len(everything), *t.shape[1:]) for t in latent))
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


class TestNoSubmit:
    """Submit held back (coordinate_submit_from above 0): the heads do not choose it until SubmitUnlock lifts the
    mask - see coordinate_submit_from in rl_configs for the measurement."""

    def _distribution(self, network, latent):
        from rl.policy import AutoregressiveCoordinateDistribution
        return AutoregressiveCoordinateDistribution(network).proba_distribution(latent)

    def test_a_masked_type_is_never_drawn_and_the_rest_is_still_a_distribution(self):
        network = small_network(actions=3)
        network.masked_action = 0
        everything = every_action(actions=3)
        latent = latent_for(network)
        repeated = type(latent)(*(None if t is None else t.expand(len(everything), *t.shape[1:]) for t in latent))
        probs = self._distribution(network, repeated).log_prob(everything).exp()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs[everything[:, 0] == 0].sum().item() < 1e-6
        big = type(latent)(*(None if t is None else t.expand(500, *t.shape[1:]) for t in latent))
        assert (self._distribution(network, big).sample()[:, 0] != 0).all()

    def test_the_independent_heads_mask_it_too(self):
        network = small_network(heads="independent", actions=3)
        network.masked_action = 0
        logits = network.type_logits(torch.zeros(4, 3))
        assert (logits[:, 0] < -1e8).all() and (logits[:, 1:] == 0).all()

    def test_the_independent_heads_hand_out_masked_type_logits(self):
        network = small_network(heads="independent", actions=3)
        network.masked_action = 0
        features = torch.randn(2, 8 + 6 * 5)
        type_logits = network.forward_actor(features)[0]
        assert (type_logits[:, 0] < -1e8).all() and (type_logits[:, 1:] > -1e8).all()

    def test_nothing_is_masked_unless_asked(self):
        assert small_network().masked_action is None
        assert torch.equal(small_network().type_logits(torch.ones(2, 2)), torch.ones(2, 2))

    def test_the_setting_reaches_the_heads_with_the_env_s_own_index_of_submit(self):
        vocabulary = {0: "red_fill", 1: "red_line", 2: "submit"}
        for setting, wanted in ((1.0, 2), (0.5, 2), (0.0, None)):
            vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=vocabulary,
                                     observation_space_elements=["delta_input", "delta_target"],
                                     observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                                     addressing="coordinates", coordinate_shape=(7, 7))
            try:
                agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                                     {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                                      "coordinate_submit_from": setting})
                assert agent.policy.mlp_extractor.masked_action == wanted
                if wanted is not None:
                    obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
                    actions = [agent.policy.get_distribution(obs).sample()[0, 0].item() for _ in range(100)]
                    assert wanted not in actions
                    agent.learn(32)
            finally:
                vec_env.close()

    def test_object_heads_are_not_touched(self):
        from rl.training import create_agent as make
        grid = np.zeros((3, 3), dtype=int)
        grid[1, 1] = 2
        vec_env = create_vec_env([ARCSubtask("s", grid, grid * 0)], n_envs=1, max_episode_len=4,
                                 feasible_actions={0: "submit", 1: "black_recolor"},
                                 observation_space_elements=["objects_emb"], repr_level=1,
                                 input_pattern="start", max_objects=2)
        try:
            agent = make({"model_type": "PPO", "addressing": "objects"}, vec_env,
                         {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_submit_from": 1.0})
            assert agent.policy.mlp_extractor.masked_action is None
        finally:
            vec_env.close()


class TestSubmitInTheSecondPhase:
    """coordinate_submit_from between 0 and 1: submit is masked for that share of the steps, then offered."""

    VOCABULARY = {0: "red_fill", 1: "red_line", 2: "submit"}

    def make(self, share):
        vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=self.VOCABULARY,
                                 observation_space_elements=["delta_input", "delta_target"],
                                 observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                                 addressing="coordinates", coordinate_shape=(7, 7))
        agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                             {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                              "coordinate_submit_from": share})
        return agent, vec_env

    def test_the_mask_is_lifted_once_its_share_of_the_steps_has_passed(self):
        from rl.callbacks import SubmitUnlock
        agent, vec_env = self.make(0.5)
        try:
            # Two pairs, two envs: a rollout of 16 steps is 32 timesteps, the unit `at` counts in.
            unlock = SubmitUnlock(at=40)
            agent.learn(16, callback=SubmitUnlock(at=40))
            assert agent.policy.masked_action == 2 and agent.policy.mlp_extractor.masked_action == 2
            agent.learn(64, callback=unlock)
            assert agent.policy.masked_action is None and agent.policy.mlp_extractor.masked_action is None
            assert unlock.unlocked_at is not None and 40 <= unlock.unlocked_at - unlock.start <= 41
            obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
            draws = [agent.policy.get_distribution(obs).sample()[0, 0].item() for _ in range(300)]
            assert 2 in draws
        finally:
            vec_env.close()

    def test_training_adds_the_unlock_at_the_share_of_its_steps(self, monkeypatch):
        import rl.training as training
        from rl.callbacks import SubmitUnlock
        built = {}

        def fake_create_agent(rl_config, vec_env, model_config=None, path_to_pretrained=None, agent_init=None):
            agent, _ = self.make(model_config["coordinate_submit_from"])
            built["agent"] = agent
            agent.learn = lambda total, callback=None: built.update(total=total, callback=callback)
            return agent

        monkeypatch.setattr(training, "create_agent", fake_create_agent)
        monkeypatch.setattr(training, "evaluate_ARC_policy", lambda *a, **k: (0.0, 0.0, None))
        config = {"n_envs": 1, "max_episode_len": 4, "repr_level": 1, "right_placement_reward": 5.0,
                  "action_penalty": 1.0, "repetitive_actions_penalty": 1.0, "seed": 0, "font_color": 0,
                  "padding": False, "input_pattern": "start", "milestones_rewards": (1, 2, 3, 4), "pad_val": 10,
                  "reward_approach": 3, "feasible_actions": self.VOCABULARY,
                  "observation_space_elements": ["delta_input", "delta_target"], "observation_grid_shape": (7, 7),
                  "addressing": "coordinates", "coordinate_shape": (7, 7), "n_eval_episodes": 1,
                  "total_steps": 1000, "evaluations": 1, "log_path": "/tmp/submit_unlock_test/"}
        for share, wanted in ((0.25, 250), (0.0, None), (1.0, None)):
            training.train_on_subtasks(two_sizes(), config, {"coordinate_submit_from": share})
            unlocks = [c for c in getattr(built["callback"], "callbacks", [built["callback"]])
                       if isinstance(c, SubmitUnlock)]
            assert [u.at for u in unlocks] == ([wanted] if wanted is not None else [])

    def test_a_share_outside_zero_to_one_is_refused(self):
        with pytest.raises(ValueError, match="coordinate_submit_from"):
            self.make(1.5)


class TestCornersAsCells:
    """coordinate_heads 'cells': each corner one cell of the grid, scored from that cell's embedding."""

    def network(self, actions=2, rows=3, cols=3, dim=4, latent=16):
        from rl.policy import ARCCustomNetwork
        return ARCCustomNetwork(feature_dim=8 + (rows + cols + rows * cols) * (dim + 1),
                                action_dims=[actions, rows, cols, rows, cols],
                                net_arch={"pi": [latent], "vf": [latent]}, action_heads=5,
                                coordinate_shape=(rows, cols), coordinate_dim=dim, coordinate_heads="cells")

    def latent(self, batch=1, seed=0, rows=3, cols=3, dim=4, latent=16, cell_mask=None):
        from rl.policy import CoordinateLatent
        g = torch.Generator().manual_seed(seed)
        return CoordinateLatent(torch.randn(batch, latent, generator=g), torch.randn(batch, rows, dim, generator=g),
                                torch.ones(batch, rows), torch.randn(batch, cols, dim, generator=g),
                                torch.ones(batch, cols), torch.randn(batch, rows * cols, dim, generator=g),
                                torch.ones(batch, rows * cols) if cell_mask is None else cell_mask)

    def distribution(self, network, latent):
        from rl.policy import AutoregressiveCoordinateDistribution
        return AutoregressiveCoordinateDistribution(network).proba_distribution(latent)

    def test_the_joint_over_every_action_is_a_distribution(self):
        network, everything = self.network(), every_action()
        latent = self.latent()
        repeated = type(latent)(*(None if t is None else t.expand(len(everything), *t.shape[1:]) for t in latent))
        probs = self.distribution(network, repeated).log_prob(everything).exp()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_a_drawn_action_is_scored_as_drawn(self):
        network = self.network()
        latent = self.latent(batch=64)
        dist = self.distribution(network, latent)
        actions = dist.sample()
        drawn = dist.log_prob(actions)
        again = self.distribution(network, latent).log_prob(actions.clone())
        assert torch.allclose(drawn, again, atol=1e-6)

    def test_the_second_corner_knows_the_first(self):
        network = self.network()
        with torch.no_grad():
            network.cell_feedback[0].weight.normal_()
        latent = self.latent()
        # The same second corner, cell 8, after two different first corners: its conditional differs.
        after_a = self.distribution(network, latent)._walk(given=torch.tensor([[0, 0, 0, 2, 2]]))[1][2]
        after_b = self.distribution(network, latent)._walk(given=torch.tensor([[0, 1, 1, 2, 2]]))[1][2]
        assert not torch.allclose(after_a.log_prob(torch.tensor([8])), after_b.log_prob(torch.tensor([8])))

    def test_a_padded_cell_is_never_a_corner(self):
        network = self.network()
        mask = torch.ones(200, 9)
        mask[:, [2, 5, 6, 7, 8]] = 0          # only the top-left 2x2 is grid
        actions = self.distribution(network, self.latent(batch=200, cell_mask=mask)).sample()
        assert (actions[:, 1:] <= 1).all()

    def test_an_agent_with_cell_corners_trains_and_stays_on_the_grid(self):
        vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=ACTIONS,
                                 observation_space_elements=["delta_input", "delta_target"],
                                 observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                                 addressing="coordinates", coordinate_shape=(7, 7))
        try:
            agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                                  "coordinate_heads": "cells", "spatial_channels": 8})
            assert agent.policy.features_extractor.coordinate_rows.cells
            agent.learn(32)
            obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
            actions = torch.stack([agent.policy.get_distribution(obs).sample()[0] for _ in range(100)])
            assert (actions[:, 1:] <= 4).all()   # the first env's pair is 5x5 in a 7x7 space
        finally:
            vec_env.close()

    def test_a_cell_is_read_from_the_map_at_that_cell(self):
        from rl.features import CoordinateRows
        reader = CoordinateRows(3, 3, (), dim=4, spatial_channels=2, cells=True)
        cell_map = torch.zeros(1, 2, 3, 3)
        cell_map[0, :, 1, 2] = 5.0
        observation = {"grid": torch.zeros(1, 3, 3)}
        cells, mask = reader.cell_embeddings(observation, cell_map)
        moved = torch.zeros(1, 2, 3, 3)
        moved[0, :, 2, 0] = 5.0
        cells_moved, _ = reader.cell_embeddings(observation, moved)
        assert cells.shape == (1, 9, 4) and mask.shape == (1, 9)
        assert not torch.allclose(cells[0, 5], cells_moved[0, 5])      # (1, 2) changed with the map there
        assert torch.allclose(cells[0, 0], cells_moved[0, 0])          # (0, 0) did not
        assert reader.width == (3 + 3 + 9) * 5


class TestStrokeColoursByRole:
    """coordinate_heads 'cell_colours': shape, corners as cells, then the colour scored from what the grid shows of it."""

    VOCABULARY = {0: "submit", 1: "black_fill", 2: "red_fill", 3: "blue_fill", 4: "red_line"}

    def network(self, rows=3, cols=3, dim=4, latent=16):
        from rl.action_structure import StrokeStructure
        from rl.policy import ARCCustomNetwork
        return ARCCustomNetwork(feature_dim=8 + (rows + cols + rows * cols) * (dim + 1) + rows * cols * 10,
                                action_dims=[5, rows, cols, rows, cols],
                                net_arch={"pi": [latent], "vf": [latent]}, action_heads=5,
                                coordinate_shape=(rows, cols), coordinate_dim=dim, coordinate_heads="cell_colours",
                                stroke_structure=StrokeStructure(self.VOCABULARY))

    def latent(self, grid, batch=1, seed=0, rows=3, cols=3, dim=4, latent=16):
        from rl.policy import CoordinateLatent
        g = torch.Generator().manual_seed(seed)
        colours = torch.nn.functional.one_hot(torch.as_tensor(grid).flatten(), 10).float()
        return CoordinateLatent(torch.randn(batch, latent, generator=g), torch.randn(batch, rows, dim, generator=g),
                                torch.ones(batch, rows), torch.randn(batch, cols, dim, generator=g),
                                torch.ones(batch, cols), torch.randn(batch, rows * cols, dim, generator=g),
                                torch.ones(batch, rows * cols), colours.expand(batch, -1, -1))

    def distribution(self, network, latent):
        from rl.policy import AutoregressiveCoordinateDistribution
        return AutoregressiveCoordinateDistribution(network).proba_distribution(latent)

    GRID = [[2, 0, 0], [0, 1, 0], [0, 0, 2]]

    def test_the_structure_takes_the_names_apart(self):
        from rl.action_structure import StrokeStructure
        structure = StrokeStructure(self.VOCABULARY)
        assert structure.shapes == ["submit", "fill", "line"] and structure.colours == [0, 1, 2]
        assert structure.table.tolist() == [[0, -1, -1], [1, 3, 2], [-1, -1, 4]]
        with pytest.raises(ValueError, match="not a stroke"):
            StrokeStructure({0: "submit", 1: "red_contour_connection_blue"})

    def test_the_joint_over_every_action_is_a_distribution(self):
        import itertools
        network = self.network()
        everything = torch.tensor(list(itertools.product(range(5), range(3), range(3), range(3), range(3))))
        latent = self.latent(self.GRID)
        repeated = type(latent)(*(None if t is None else t.expand(len(everything), *t.shape[1:]) for t in latent))
        probs = self.distribution(network, repeated).log_prob(everything).exp()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_swapping_two_colours_on_the_grid_swaps_their_scores(self):
        """Nothing in a colour's score is the colour itself: red and blue traded on the grid trade their logits."""
        network = self.network()
        swapped = [[{1: 2, 2: 1}.get(v, v) for v in row] for row in self.GRID]
        one, two = torch.tensor([0]), torch.tensor([4])          # corners (0, 0) and (1, 1)
        fill = torch.tensor([1])
        a = network.colour_logits(torch.zeros(1, 16), self.latent(self.GRID), one, two, fill)
        b = network.colour_logits(torch.zeros(1, 16), self.latent(swapped), one, two, fill)
        assert torch.allclose(a[:, [0, 1, 2]], b[:, [0, 2, 1]], atol=1e-6)

    def test_a_colour_the_shape_does_not_come_in_is_never_drawn_and_submit_costs_nothing(self):
        network = self.network()
        latent = self.latent(self.GRID, batch=400)
        actions = self.distribution(network, latent).sample()
        assert set(actions[:, 0].tolist()) <= set(self.VOCABULARY)
        submits = torch.tensor([[0, 0, 0, 0, 0]])
        dists = self.distribution(network, self.latent(self.GRID))._walk(given=submits)[1]
        assert dists[3].log_prob(torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)

    def test_submit_held_back_is_never_drawn(self):
        network = self.network()
        network.masked_action = 0
        actions = self.distribution(network, self.latent(self.GRID, batch=300)).sample()
        assert (actions[:, 0] != 0).all()

    def test_an_agent_with_them_trains_and_draws_actions_of_its_vocabulary(self):
        vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=self.VOCABULARY,
                                 observation_space_elements=["delta_input", "delta_target"],
                                 observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                                 addressing="coordinates", coordinate_shape=(7, 7))
        try:
            agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                                  "coordinate_heads": "cell_colours", "spatial_channels": 8})
            assert agent.policy.features_extractor.coordinate_rows.cell_colours
            agent.learn(32)
            obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
            actions = torch.stack([agent.policy.get_distribution(obs).sample()[0] for _ in range(50)])
            assert set(actions[:, 0].tolist()) <= set(self.VOCABULARY) and (actions[:, 1:] <= 4).all()
        finally:
            vec_env.close()

    def test_the_cells_carry_their_colours(self):
        reader = CoordinateRows(2, 2, (), dim=4, cells=True, cell_colours=True)
        tail = reader.tail({"grid": torch.tensor([[[3, 0], [0, 7]]])})
        colours = tail[:, -40:].view(1, 4, 10)
        assert colours[0, 0, 3] == 1 and colours[0, 3, 7] == 1 and colours[0, 1, 0] == 1
        assert reader.width == (2 + 2 + 4) * 5 + 40
        with pytest.raises(ValueError, match="need cells"):
            CoordinateRows(2, 2, (), dim=4, cell_colours=True)


class TestCellsReadFromTheirFeatures:
    """'cell_features' in the observation: a cell of the coordinate heads is read from its features
    (symbolic.cell_features) in place of its row and column."""

    def features(self, grid, shape=(3, 3)):
        from symbolic.cell_features import encoded
        return torch.tensor(encoded(np.asarray(grid), shape)).unsqueeze(0)

    def test_two_cells_alike_in_everything_but_place_are_read_alike(self):
        reader = CoordinateRows(3, 3, (), dim=4, cells=True, cell_features=True)
        grid = torch.zeros(1, 3, 3, dtype=torch.int64)
        features = self.features(np.zeros((3, 3), dtype=int))
        features[0, 2, 2] = features[0, 0, 0]                # (2, 2) described exactly as (0, 0)
        cells, _ = reader.cell_embeddings({"grid": grid, "cell_features": features})
        assert torch.allclose(cells[0, 0], cells[0, 8])
        by_position, _ = CoordinateRows(3, 3, (), dim=4, cells=True).cell_embeddings({"grid": grid})
        assert not torch.allclose(by_position[0, 0], by_position[0, 8])    # what c_cell reads: where they are

    def test_a_cell_reads_differently_when_its_features_do(self):
        reader = CoordinateRows(3, 3, (), dim=4, cells=True, cell_features=True)
        with torch.no_grad():
            for layer in reader.cell_encoder:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.normal_()
        grid = torch.zeros(1, 3, 3, dtype=torch.int64)
        plain = self.features(np.zeros((3, 3), dtype=int))
        marked = np.zeros((3, 3), dtype=int)
        marked[0, 2] = 4                                      # changes what (0, 0) sees to its right, not its colour
        cells, _ = reader.cell_embeddings({"grid": grid, "cell_features": plain})
        other, _ = reader.cell_embeddings({"grid": grid, "cell_features": self.features(marked)})
        assert not torch.allclose(cells[0, 0], other[0, 0])

    def test_the_features_are_one_hot_colours_and_scaled_counts(self):
        from symbolic.cell_features import schema
        reader = CoordinateRows(3, 3, (), dim=4, cells=True, cell_features=True)
        names, colours = schema()
        described = reader.described_cells(self.features(np.zeros((3, 3), dtype=int)))
        assert described.shape == (1, 11 * len(colours) + len(names) - len(colours), 3, 3)
        assert torch.allclose(described[0, :11 * len(colours)].sum(0), torch.full((3, 3), float(len(colours))))
        assert described.min() >= 0 and described.max() <= 1
        grid = np.zeros((3, 3), dtype=int)
        grid[0, 1] = 2
        described = reader.described_cells(self.features(grid))
        k = [name for name in names if name in colours].index("colour")
        assert described[0, 11 * k + 3, 0, 1] == 1 and described[0, 11 * k + 1, 0, 0] == 1   # red, black

    def test_features_without_cells_are_refused(self):
        with pytest.raises(ValueError, match="need cells"):
            CoordinateRows(3, 3, (), dim=4, cell_features=True)

    def test_an_agent_reading_cell_features_trains_and_its_env_computes_them_once(self, monkeypatch):
        import symbolic.cell_features as cf
        calls, seen = [], []
        original = cf.encoded
        monkeypatch.setattr(cf, "encoded", lambda *a, **k: calls.append(1) or original(*a, **k))
        described = CoordinateRows.described_cells
        monkeypatch.setattr(CoordinateRows, "described_cells",
                            lambda self, features: seen.append(features.detach().clone()) or described(self, features))
        vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=ACTIONS,
                                 observation_space_elements=["delta_input", "delta_target", "cell_features"],
                                 observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                                 addressing="coordinates", coordinate_shape=(7, 7))
        try:
            agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                                  "coordinate_heads": "cells", "spatial_channels": 8})
            assert agent.policy.features_extractor.coordinate_rows.cell_features
            agent.learn(32)
            obs = vec_env.reset()
            assert obs["cell_features"].shape[1:3] == (7, 7)
            assert len(calls) <= 2                              # once per example's input, not once per step
            # What the network reads is what the env computed: not divided by 255 as an image, not transposed.
            assert seen and all(f.shape[1:3] == (7, 7) for f in seen)
            assert torch.equal(seen[-1][0].to(torch.int64), torch.tensor(obs["cell_features"][0]).to(torch.int64))
            assert max(f.max().item() for f in seen) > 1
        finally:
            vec_env.close()


def test_an_agent_reads_the_core_features_from_their_own_key():
    vec_env = create_vec_env(two_sizes(), n_envs=1, max_episode_len=4, feasible_actions=ACTIONS,
                             observation_space_elements=["delta_input", "delta_target", "cell_features_core"],
                             observation_grid_shape=(7, 7), repr_level=1, input_pattern="start",
                             addressing="coordinates", coordinate_shape=(7, 7))
    try:
        agent = create_agent({"model_type": "PPO", "addressing": "coordinates"}, vec_env,
                             {"n_steps": 16, "batch_size": 8, "verbose": 0, "coordinate_dim": 8,
                              "coordinate_heads": "cells", "spatial_channels": 8})
        rows = agent.policy.features_extractor.coordinate_rows
        assert rows.cell_features == "cell_features_core" and rows.feature_is_colour.numel() == 25
        agent.learn(32)
        assert vec_env.reset()["cell_features_core"].shape[1:] == (7, 7, 25)
    finally:
        vec_env.close()
