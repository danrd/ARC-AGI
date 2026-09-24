"""Object actions chosen by their parts.

rl.action_structure takes the flat vocabulary's names apart; the factored
heads (rl.policy.FactoredObjectDistribution) choose the parts one after
another - type, object, object, colour, second colour, direction - with the
colour and direction scored from the observation. Pinned here: the parts
round-trip to the names, the joint over every (action, object, object) is
a distribution and puts nothing on what the env would do nothing with, a
sample is scored as PPO later scores it, and - the reason for all of it - a
name no training example used can be chosen.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.action_structure import (NO_COLOUR, NO_DIRECTION, ActionStructure, objects_taken,
                                 parse_name)
from symbolic.objects_analysis import OBJECT_DIM, OBJECT_SCHEMA

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestTakingNamesApart:
    def test_colours_in_order_and_the_direction(self):
        assert parse_name("red_emission_with_blue_object_recolor_N") == \
            ("emission_with_object_recolor", [2, 1], "N")
        assert parse_name("red_contour_connection_blue") == ("contour_connection", [2, 1], None)
        assert parse_name("shift_object_SW") == ("shift_object", [], "SW")
        assert parse_name("background_shortest_path_left") == \
            ("background_shortest_path_left", [], None)

    def test_every_generated_name_round_trips(self):
        from data.configs.env_configs import ALL_DIRECTIONS
        from rl.search_hints import build_vocabulary

        names = build_vocabulary(["black", "red", "blue", "sky"], ALL_DIRECTIONS)
        structure = ActionStructure(names)
        for index, parts in enumerate(structure.components):
            assert structure.flat[tuple(parts)] == index

    def test_the_valid_tables_are_what_the_names_allow(self):
        names = {0: "submit", 1: "red_recolor", 2: "blue_recolor", 3: "red_emission_N",
                 4: "red_emission_S", 5: "gravity"}
        s = ActionStructure(names)
        emission = s.types.index("emission")
        assert s.valid_colour[emission].nonzero()[0].tolist() == [2]
        assert s.valid_second_colour[emission, 2].nonzero()[0].tolist() == [NO_COLOUR]
        assert sorted(s.valid_direction[emission, 2, NO_COLOUR].nonzero()[0].tolist()) == [0, 3]
        gravity = s.types.index("gravity")
        assert s.valid_direction[gravity, NO_COLOUR, NO_COLOUR, NO_DIRECTION]

    def test_how_many_objects_a_type_takes(self):
        assert objects_taken("submit") == 0
        assert objects_taken("recolor") == 1
        assert objects_taken("emission_with_collision_stop") == 1
        assert objects_taken("swap") == 2
        assert objects_taken("contour_connection") == 2

    def test_a_vocabulary_with_gaps_is_refused(self):
        with pytest.raises(ValueError, match="gaps"):
            ActionStructure({0: "submit", 2: "red_recolor"})


SLOTS, POINTER, LATENT = 3, 8, 16
NAMES = {0: "submit", 1: "red_recolor", 2: "blue_recolor", 3: "swap",
         4: "red_emission_N", 5: "red_emission_E", 6: "blue_emission_N"}


def network(names=NAMES, slots=SLOTS, direction_keys="relative"):
    from rl.policy import ARCCustomNetwork

    factored = slots * OBJECT_DIM + 10
    return ARCCustomNetwork(feature_dim=8 + slots * (POINTER + 1) + factored,
                            action_dims=[len(names), slots, slots],
                            net_arch={"pi": [LATENT], "vf": [LATENT]}, action_heads=3,
                            pointer_slots=slots, pointer_dim=POINTER,
                            action_structure=ActionStructure(names), factored_width=factored,
                            direction_keys=direction_keys)


def latent(batch=1, seed=0, visible=SLOTS, objects=None, shares=None):
    from rl.policy import ObjectLatent

    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch, SLOTS)
    mask[:, :visible] = 1
    return ObjectLatent(torch.randn(batch, LATENT, generator=g),
                        torch.randn(batch, SLOTS, POINTER, generator=g), mask,
                        objects if objects is not None else torch.rand(batch, SLOTS, OBJECT_DIM,
                                                                       generator=g),
                        shares if shares is not None else torch.rand(batch, 10, generator=g))


def distribution(net, lat):
    from rl.policy import FactoredObjectDistribution
    return FactoredObjectDistribution(net).proba_distribution(lat)


def repeat(lat, n):
    return type(lat)(*(t.expand(n, *t.shape[1:]) for t in lat))


class TestTheJoint:
    def _everything(self):
        return torch.tensor(list(itertools.product(range(len(NAMES)), range(SLOTS),
                                                   range(SLOTS))))

    def test_it_sums_to_one(self):
        net, everything = network(), self._everything()
        probs = distribution(net, repeat(latent(), len(everything))).log_prob(everything).exp()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_nothing_on_what_the_env_would_do_nothing_with(self):
        """A single-object action on two slots, a two-object one on one
        slot twice, submit on anything but (0, 0), a slot past the objects
        this grid holds."""
        net, everything = network(), self._everything()
        probs = distribution(net, repeat(latent(visible=2), len(everything))
                             ).log_prob(everything).exp()
        for (action, first, second), p in zip(everything.tolist(), probs.tolist()):
            takes = objects_taken(parse_name(NAMES[action])[0])
            dead = ((takes == 0 and (first, second) != (0, 0))
                    or (takes == 1 and first != second)
                    or (takes == 2 and first == second)
                    or (takes > 0 and max(first, second) >= 2))
            if dead:
                assert p < 1e-6, (NAMES[action], first, second, p)

    def test_a_sample_is_scored_as_a_fresh_walk_scores_it(self):
        net, lat = network(), latent(batch=64, seed=1)
        torch.manual_seed(0)
        d = distribution(net, lat)
        actions = d.sample()
        kept = d.log_prob(actions)
        fresh = distribution(net, lat).log_prob(actions.clone().float())
        assert torch.allclose(kept, fresh, atol=1e-6)

    def test_the_mode_is_the_greedy_walk(self):
        net = network()
        d = distribution(net, latent(seed=2))
        mode = d.mode()
        _, dists = d._walk(given=mode)
        parts = net.action_components[mode[:, 0]]
        chosen = [parts[0, 0], mode[0, 1], mode[0, 2], parts[0, 1], parts[0, 2], parts[0, 3]]
        assert all(int(c) == int(dist.probs.argmax()) for c, dist in zip(chosen, dists))


def _set(row, name, value):
    position = 0
    for field, _group, arity in OBJECT_SCHEMA:
        if field == name:
            row[position] = value
            return
        position += arity


def triangle(major, minor, heavy):
    """An object's raw row: mostly `major` with a minority of `minor`, its
    mass pushed towards `heavy` inside its box."""
    row = torch.zeros(OBJECT_DIM)
    row[major], row[minor] = 0.89, 0.11
    for name, value in (("min_i", 0.2), ("max_i", 0.6), ("min_j", 0.2), ("max_j", 0.6)):
        _set(row, name, value)
    di, dj = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}[heavy]
    _set(row, "i_center", 0.4 + 0.1 * di)
    _set(row, "j_center", 0.4 + 0.1 * dj)
    return row


class TestANameNoExampleUsed:
    """25d487eb in miniature: emit from a triangle, in the colour of its
    dot, away from its heavy end. Three examples, three names; the fourth
    case needs a name none of them used - sky is nobody's dot colour in
    training, and no training example emitted sky at all. A flat head has
    no logit that was ever trained for it."""

    COLOURS = {"blue": 1, "red": 2, "green": 3, "yellow": 4, "sky": 8}
    OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def _vocabulary(self):
        names = ["submit"] + [f"{c}_emission_{d}" for c in self.COLOURS for d in "NSEW"] \
            + [f"{c}_recolor" for c in self.COLOURS]
        return {i: n for i, n in enumerate(names)}

    def _case(self, major, minor, direction):
        objects = torch.zeros(1, SLOTS, OBJECT_DIM)
        objects[0, 0] = triangle(self.COLOURS[major], self.COLOURS[minor],
                                 self.OPPOSITE[direction])
        shares = torch.zeros(1, 10)
        shares[0, self.COLOURS[major]], shares[0, self.COLOURS[minor]] = 0.1, 0.01
        return objects, shares

    def _trained(self, direction_keys):
        torch.manual_seed(0)
        names = self._vocabulary()
        index = {n: i for i, n in names.items()}
        net = network(names, direction_keys=direction_keys)
        train = [("green", "blue", "E"), ("red", "green", "N"), ("blue", "red", "S")]
        cases = [self._case(*c) for c in train]
        lat = latent(batch=1, visible=1)
        batch = type(lat)(lat.context.expand(3, -1), lat.rows.expand(3, -1, -1),
                          lat.row_mask.expand(3, -1), torch.cat([c[0] for c in cases]),
                          torch.cat([c[1] for c in cases]))
        wanted = torch.tensor([[index[f"{minor}_emission_{d}"], 0, 0] for _, minor, d in train])
        optimiser = torch.optim.Adam(net.parameters(), lr=0.02)
        for _ in range(400):
            optimiser.zero_grad()
            (-distribution(net, batch).log_prob(wanted).mean()).backward()
            optimiser.step()

        def probability(major, minor, direction):
            objects, shares = self._case(major, minor, direction)
            test = type(lat)(lat.context, lat.rows, lat.row_mask, objects, shares)
            action = torch.tensor([[index[f"{minor}_emission_{direction}"], 0, 0]])
            return distribution(net, test).log_prob(action).exp().item()
        return probability

    def test_a_colour_no_example_used(self):
        """sky: nobody's dot in training, emitted by no example."""
        assert self._trained("relative")("yellow", "sky", "N") > 0.5

    def test_a_direction_no_example_used(self):
        """West: never emitted in training, only ever a wrong answer."""
        assert self._trained("relative")("yellow", "sky", "W") > 0.5

    def test_which_absolute_directions_cannot_reach(self):
        """What 'both' costs: the direction's own embedding learns that west
        was always wrong. Pinned so the trade in direction_keys stays a
        measured one."""
        probability = self._trained("both")
        assert probability("yellow", "sky", "N") > 0.5
        assert probability("yellow", "sky", "W") < 0.1


class TestAnAgentBuiltWithThem:
    @pytest.fixture(scope="class")
    def built(self):
        from rl.arc_task import ARCSubtask
        from rl.training import create_agent, create_vec_env

        challenge = json.loads((REPO_ROOT / "data/datasets/ARC/training_challenges.json")
                               .read_text())["25d487eb"]
        subtasks = [ARCSubtask(f"s{i}", np.array(p["input"]), np.array(p["output"]))
                    for i, p in enumerate(challenge["train"])]
        vocabulary = {0: "submit", 1: "blue_emission_E", 2: "green_emission_N",
                      3: "red_emission_S", 4: "red_recolor", 5: "swap"}
        vec_env = create_vec_env(subtasks, n_envs=1, max_episode_len=4,
                                 feasible_actions=vocabulary,
                                 observation_space_elements=["objects_emb"],
                                 observation_grid_shape=(15, 15), repr_level=1,
                                 input_pattern="start", max_objects=4)
        agent = create_agent({"model_type": "PPO", "addressing": "objects"}, vec_env,
                             {"n_steps": 16, "batch_size": 8, "verbose": 0,
                              "object_heads": "factored"})
        yield agent, vec_env
        vec_env.close()

    def test_its_distribution_is_the_factored_one(self, built):
        from rl.policy import FactoredObjectDistribution

        agent, vec_env = built
        obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
        assert isinstance(agent.policy.get_distribution(obs), FactoredObjectDistribution)

    def test_it_reads_the_env_s_own_names(self, built):
        agent, vec_env = built
        structure = agent.policy.action_structure
        assert len(structure) == len(vec_env.envs[0].unwrapped.actions_dict)

    def test_ppo_trains_through_it(self, built):
        agent, _ = built
        agent.learn(32)

    def test_flat_is_still_there(self):
        from stable_baselines3.common.distributions import MultiCategoricalDistribution

        from rl.arc_task import ARCSubtask
        from rl.training import create_agent, create_vec_env

        grid = np.zeros((3, 3), dtype=int)
        grid[1, 1] = 2
        vec_env = create_vec_env([ARCSubtask("s", grid, grid * 0)], n_envs=1, max_episode_len=4,
                                 feasible_actions={0: "submit", 1: "black_recolor"},
                                 observation_space_elements=["objects_emb"], repr_level=1,
                                 input_pattern="start", max_objects=2)
        try:
            agent = create_agent({"model_type": "PPO"}, vec_env,
                                 {"n_steps": 16, "batch_size": 8, "verbose": 0,
                                  "object_heads": "flat"})
            obs, _ = agent.policy.obs_to_tensor(vec_env.reset())
            assert isinstance(agent.policy.get_distribution(obs), MultiCategoricalDistribution)
        finally:
            vec_env.close()

    def test_an_unknown_setting_is_refused(self):
        from rl.policy import ARCCustomActorCriticPolicy

        with pytest.raises(ValueError, match="object_heads"):
            ARCCustomActorCriticPolicy.__init__(object.__new__(ARCCustomActorCriticPolicy),
                                                None, None, None, object_heads="sideways")


class TestEachChoiceKnowsTheOnesBefore:
    def test_the_second_object_depends_on_the_first(self):
        """For a two-object action, with four slots: whichever the first
        is, slots 2 and 3 stay open for the second, and the odds between
        them move with the first choice - the first object's row is added
        to the context the second head reads."""
        from rl.policy import ObjectLatent

        slots = 4
        net = network(slots=slots)
        g = torch.Generator().manual_seed(0)
        lat = ObjectLatent(torch.randn(1, LATENT, generator=g),
                           torch.randn(1, slots, POINTER, generator=g), torch.ones(1, slots),
                           torch.rand(1, slots, OBJECT_DIM, generator=g),
                           torch.rand(1, 10, generator=g))
        swap = [i for i, n in NAMES.items() if n == "swap"][0]
        d = distribution(net, lat)
        odds = []
        for first in (0, 1):
            _, dists = d._walk(given=torch.tensor([[swap, first, 2]]))
            logits = dists[2].logits[0]
            odds.append((logits[3] - logits[2]).item())
        assert abs(odds[0] - odds[1]) > 1e-4
