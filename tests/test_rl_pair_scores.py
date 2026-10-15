"""The relation between the first object and each candidate, as a term in the second object's logit
(rl.features.PairScores, carried to rl.policy.FactoredObjectDistribution), and attention aggregation in
RelationMessages."""
import itertools

import numpy as np
import pytest
import torch

from rl.action_structure import ActionStructure
from rl.features import PairScores, RelationMessages, pairwise_relations
from symbolic.summaries import RELATION_DIM
from tests.test_rl_factored_heads import LATENT, NAMES, POINTER, SLOTS, distribution, latent

OBJECT_DIM = 20


def relations(batch=1, slots=SLOTS, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, slots, (slots - 1) * RELATION_DIM, generator=g)


def network(pair_width):
    from rl.policy import ARCCustomNetwork
    from tests.test_rl_factored_heads import OBJECT_DIM as RAW

    factored = SLOTS * RAW + 10
    return ARCCustomNetwork(feature_dim=8 + SLOTS * (POINTER + 1) + factored + pair_width,
                            action_dims=[len(NAMES), SLOTS, SLOTS],
                            net_arch={"pi": [LATENT], "vf": [LATENT]}, action_heads=3,
                            pointer_slots=SLOTS, pointer_dim=POINTER,
                            action_structure=ActionStructure(NAMES), factored_width=factored,
                            direction_keys="relative", pair_width=pair_width)


class TestPairScores:
    def test_it_starts_at_zero(self):
        assert PairScores()(relations()).abs().max().item() == 0.0

    def test_entry_ij_reads_the_relation_of_that_pair(self):
        scores = PairScores()
        torch.nn.init.normal_(scores.score[-1].weight)
        rel = relations()
        before = scores(rel)
        # slot 0's block for slot 2 is the second block of its row (slot 0 itself is skipped)
        changed = rel.clone()
        changed[0, 0, RELATION_DIM:2 * RELATION_DIM] += 3.0
        after = scores(changed)
        assert ((after - before).abs() > 1e-6)[0].nonzero().tolist() == [[0, 2]]

    def test_it_is_the_pair_unpacked_not_a_position(self):
        scores = PairScores()
        torch.nn.init.normal_(scores.score[-1].weight)
        rel = relations()
        pairs = pairwise_relations(rel)
        assert torch.allclose(scores(rel)[0, 1, 2], scores.score(pairs[0, 1, 2]).squeeze())


class TestTheSecondObjectReadsThePair:
    def _with(self, pairs, net=None):
        net = net or network(SLOTS * SLOTS)
        lat = latent()
        return distribution(net, type(lat)(*lat[:5], pairs))

    def test_a_pair_term_moves_the_second_choice_by_exactly_that_term(self):
        net = network(SLOTS * SLOTS)
        pairs = torch.zeros(1, SLOTS, SLOTS)
        pairs[0, 0, 2] = 2.0
        plain, boosted = self._with(torch.zeros(1, SLOTS, SLOTS), net), self._with(pairs, net)
        action = list(NAMES.values()).index("swap")          # two objects: first 0, candidates 1 and 2
        one, two = (torch.tensor([[action, 0, s]]) for s in (1, 2))
        d_plain = plain.log_prob(two) - plain.log_prob(one)
        d_boost = boosted.log_prob(two) - boosted.log_prob(one)
        assert (d_boost - d_plain).item() == pytest.approx(2.0, abs=1e-5)

    def test_the_term_of_another_first_object_does_not_leak_in(self):
        pairs = torch.zeros(1, SLOTS, SLOTS)
        pairs[0, 1, 2] = 5.0                       # first is 1: irrelevant when the first chosen is 0
        action = list(NAMES.values()).index("swap")
        net = network(SLOTS * SLOTS)
        boosted, zero = self._with(pairs, net), self._with(torch.zeros(1, SLOTS, SLOTS), net)
        given = torch.tensor([[action, 0, 2]])
        assert boosted.log_prob(given).item() == pytest.approx(zero.log_prob(given).item(), abs=1e-5)

    def test_the_joint_still_sums_to_one(self):
        pairs = torch.randn(1, SLOTS, SLOTS)
        everything = torch.tensor(list(itertools.product(range(len(NAMES)), range(SLOTS), range(SLOTS))))
        lat = latent()
        pairs_n = pairs.expand(len(everything), -1, -1)
        big = type(lat)(*(t.expand(len(everything), *t.shape[1:]) for t in lat[:5]), pairs_n)
        net = network(SLOTS * SLOTS)
        probs = distribution(net, big).log_prob(everything).exp()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_the_pair_tail_is_cut_from_the_end_and_not_read_as_objects(self):
        net = network(SLOTS * SLOTS)
        width = 8 + SLOTS * (POINTER + 1) + net.factored_width + net.pair_width
        features = torch.randn(2, width)
        pairs = net.split_pair_tail(features)
        assert pairs.shape == (2, SLOTS, SLOTS)
        assert torch.equal(pairs.flatten(1), features[:, -net.pair_width:])
        objects, shares = net.split_factored_tail(features)
        assert torch.equal(shares, features[:, -net.pair_width - 10:-net.pair_width])
        rows, mask = net.split_pointer_tail(features)
        start = 8
        assert torch.equal(rows[:, 0], features[:, start:start + POINTER])
        assert net.without_pointer_tail(features).shape[1] == 8

    def test_without_pair_scores_nothing_is_cut(self):
        net = network(0)
        assert net.split_pair_tail(torch.randn(1, 40)) is None


class TestTheExtractor:
    def _space(self):
        from gymnasium import spaces
        return spaces.Dict({
            "grid": spaces.Box(0, 10, (6, 6), np.int64),
            "objects_emb": spaces.Box(-10, 10, (SLOTS, 40), np.float32),
            "relations_emb": spaces.Box(-10, 10, (SLOTS, (SLOTS - 1) * RELATION_DIM), np.float32),
        })

    def test_it_refuses_pair_scores_without_the_factored_tail(self):
        from rl.features import ARCCombinedExtractor
        with pytest.raises(ValueError, match="factored_tail"):
            ARCCombinedExtractor(self._space(), relation_mode="messages", pair_scores=True)

    def test_it_ends_its_features_in_the_pair_scores(self):
        from rl.features import ARCCombinedExtractor
        extractor = ARCCombinedExtractor(self._space(), relation_mode="messages", factored_tail=True,
                                         pair_scores=True)
        assert extractor.pair_width == SLOTS * SLOTS
        observation = {key: torch.rand(2, *space.shape) for key, space in self._space().spaces.items()}
        observation["grid"] = torch.randint(0, 10, (2, 6, 6))
        torch.nn.init.normal_(extractor.pair_scores.score[-1].weight)
        features = extractor.eval()(observation)
        assert features.shape[1] == extractor.features_dim
        expected = extractor.pair_scores(observation["relations_emb"]).flatten(1)
        assert torch.allclose(features[:, -SLOTS * SLOTS:], expected, atol=1e-6)


class TestTheAgent:
    @pytest.fixture
    def built(self):
        from rl.arc_task import ARCSubtask
        from rl.training import create_agent, create_vec_env

        grid = np.zeros((6, 6), dtype=int)
        grid[0, 0], grid[2, 3], grid[5, 1] = 1, 2, 3
        vec_env = create_vec_env([ARCSubtask("s", grid, grid * 0)], n_envs=1, max_episode_len=4,
                                 feasible_actions={0: "submit", 1: "red_recolor", 2: "swap"},
                                 observation_space_elements=["objects_emb", "relations_emb"],
                                 repr_level=1, input_pattern="start", max_objects=4)
        config = {"n_steps": 16, "batch_size": 8, "verbose": 0, "object_heads": "factored",
                  "relation_mode": "messages", "relation_pointer": True}
        agent = create_agent({"model_type": "PPO", "addressing": "objects"}, vec_env, config)
        yield agent
        vec_env.close()

    def test_the_policy_carries_the_pair_scores_and_trains(self, built):
        assert built.policy.mlp_extractor.pair_width == 16
        built.learn(32)

    def test_the_relation_changes_the_second_object_s_distribution_once_trained_to_matter(self, built):
        policy = built.policy
        extractor = policy.features_extractor if not policy.critic_only_keys else policy.pi_features_extractor
        torch.nn.init.normal_(extractor.pair_scores.score[-1].weight)
        obs, _ = policy.obs_to_tensor(built.env.reset())
        base = policy.get_distribution(obs)
        altered = {key: value.clone() for key, value in obs.items()}
        altered["relations_emb"] = altered["relations_emb"] + 5.0
        other = policy.get_distribution(altered)
        action = torch.tensor([[2, 0, 1]])
        assert not torch.allclose(base.log_prob(action), other.log_prob(action))


class TestAttentionAggregation:
    def _messages(self, **kwargs):
        torch.manual_seed(0)
        return RelationMessages(object_dim=OBJECT_DIM, dropout=0.0, aggregation="attn", **kwargs).eval()

    def test_an_object_with_no_partners_hears_nothing_and_not_nan(self):
        net = self._messages()
        rows = torch.randn(1, 3, OBJECT_DIM)
        out = net(rows, relations(), torch.tensor([[1.0, 0.0, 0.0]]))
        assert torch.isfinite(out).all()
        pooled = net._pooled(rows, pairwise_relations(relations()),
                             torch.zeros(1, 3, 3, 1))
        assert pooled.abs().max().item() == 0.0

    def test_a_padded_partner_gets_no_weight(self):
        net = self._messages()
        rows = torch.randn(1, 3, OBJECT_DIM)
        rel = relations()
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        before = net(rows, rel, mask)
        changed = rel.clone()
        changed[0, 0, RELATION_DIM:] += 9.0        # slot 0 against slot 2, which is padding
        after = net(rows, changed, mask)
        assert torch.allclose(before[0, 0], after[0, 0], atol=1e-6)

    def test_it_weighs_partners_unlike_a_mean(self):
        attn = self._messages()
        torch.nn.init.normal_(attn.attention.weight, std=3.0)
        mean = RelationMessages(object_dim=OBJECT_DIM, dropout=0.0).eval()
        mean.load_state_dict({k: v for k, v in attn.state_dict().items() if not k.startswith("attention")})
        rows, rel, mask = torch.randn(1, 3, OBJECT_DIM), relations(), torch.ones(1, 3)
        assert not torch.allclose(attn(rows, rel, mask), mean(rows, rel, mask), atol=1e-4)

    def test_only_attn_builds_the_scoring_layer(self):
        assert RelationMessages(object_dim=OBJECT_DIM).attention is None
        with pytest.raises(ValueError, match="attn"):
            RelationMessages(object_dim=OBJECT_DIM, aggregation="median")
