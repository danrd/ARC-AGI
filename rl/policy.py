import torch
import torch.nn as nn
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from stable_baselines3.common.distributions import Distribution, MultiCategoricalDistribution
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from rl.action_structure import (DIRECTION_STEPS, N_COLOURS, N_DIRECTIONS, ActionStructure)
from data.configs.env_configs import ALL_DIRECTIONS
from rl.features import ARCCombinedExtractor, ARCGNNExtractor, ARCSeparateExtractor
from symbolic.objects_analysis import OBJECT_SCHEMA

class PointerHead(nn.Module):
    """Logits over object slots, each scored from that slot's own row.

    The head it replaces is nn.Linear(latent, max_objects) over the shared
    latent, and the shared latent comes from a mean over the objects. A mean
    is permutation invariant, so the logits are too: swapping the contents
    of two slots leaves every one of them unchanged, measured at 6.6e-07
    against a feature scale of 0.27 where replacing the objects outright
    moves the features by 0.427. The only thing distinguishing slot i from
    slot j there is the learned row W[i], a constant of the network rather
    than a function of the observation - so the policy can learn a prior
    over slot numbers and nothing else, and a slot number does not mean the
    same thing on the next grid.

    Scoring each slot from its own row makes the logits equivariant
    instead: permute the objects and the logits permute with them. "Act on
    the largest" becomes a score monotone in a field the row already
    carries, which is one layer, rather than something no arrangement of
    weights can express.

    Dot-product scoring, in the style of attention and of the pointer
    networks it is named after (Vinyals, Fortunato & Jaitly, 2015), which
    read out a distribution over input positions rather than over a fixed
    vocabulary.
    """

    def __init__(self, context_dim: int, object_dim: int, hidden: int = 64):
        super().__init__()
        self.query = nn.Linear(context_dim, hidden)
        self.key = nn.Linear(object_dim, hidden)
        self.scale = hidden ** 0.5

    def forward(self, context: torch.Tensor, rows: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        query = self.query(context).unsqueeze(1)
        keys = self.key(rows)
        logits = (query * keys).sum(dim=-1) / self.scale
        # A large finite penalty rather than -inf: a row whose every slot is
        # masked would make softmax(-inf, ...) NaN and take the whole batch
        # with it, which is the trap ObjectSetProcessor documents on its own
        # attention mask. object_slots() floors the count at two so it
        # should not arise, and should-not-arise is what NaN waits for.
        return logits.masked_fill(~mask.bool(), -1e9)


class CoordinateLatent(NamedTuple):
    """What forward_actor hands the distribution under autoregressive
    coordinate heads: not logits, which cannot be computed before the
    earlier choices are made, but what computing them needs."""
    context: torch.Tensor
    rows: torch.Tensor
    row_mask: torch.Tensor
    cols: torch.Tensor
    col_mask: torch.Tensor


class AutoregressiveCoordinateDistribution(Distribution):
    """(action, i1, j1, i2, j2), each chosen knowing the ones before it.

    MultiCategoricalDistribution draws the five independently given the
    state, and a joint over cells cannot be written that way: a policy
    torn between the box (0,0)-(2,2) and the box (5,5)-(7,7) puts its mass
    on both corners of each, and drawn independently those make
    (0,0)-(7,7) as often as either box - a stroke over both, which it
    wanted neither of. Here each choice is drawn from a head whose context
    carries the earlier ones: the action's embedding first, then the
    embedding of the row chosen for i1, the column for j1, the row for i2.
    The joint is exactly the product of those conditionals, so its log
    probability is their sum.

    Sampling walks the heads in order. Scoring given actions - PPO's
    evaluate_actions, on actions from the buffer - runs the same walk with
    the given choices in place of draws. The entropy is the sum of the
    conditionals' entropies along the path walked last, whose expectation
    is the joint's entropy; the joint's own is a sum over every path and
    not computable here.
    """

    def __init__(self, network: "ARCCustomNetwork"):
        super().__init__()
        self.network = network
        self.latent: Optional[CoordinateLatent] = None
        #: (actions, per-dimension Categoricals) of the last walk, so the
        #: log_prob of what sample() just drew costs no second pass.
        self._last = None

    def proba_distribution_net(self, *args, **kwargs):
        raise NotImplementedError("built by ARCCustomNetwork, not from a latent width")

    def proba_distribution(self, latent: CoordinateLatent) -> "AutoregressiveCoordinateDistribution":
        self.latent = latent
        self._last = None
        return self

    def _walk(self, given: Optional[torch.Tensor] = None, deterministic: bool = False):
        network, latent = self.network, self.latent
        action_head, *cell_heads = network.policy_nets
        batch = torch.arange(latent.context.shape[0], device=latent.context.device)
        grids = [(latent.rows, latent.row_mask), (latent.cols, latent.col_mask)] * 2

        def pick(distribution, index):
            if given is not None:
                return given[:, index].long()
            if deterministic:
                return distribution.probs.argmax(dim=-1)
            return distribution.sample()

        context = latent.context
        distribution = Categorical(logits=action_head(context))
        chosen = [pick(distribution, 0)]
        dists = [distribution]
        context = context + network.action_embedding(chosen[0])
        for index, (head, (embeddings, mask)) in enumerate(zip(cell_heads, grids)):
            distribution = Categorical(logits=head(context, embeddings, mask))
            choice = pick(distribution, index + 1)
            chosen.append(choice)
            dists.append(distribution)
            if index < len(network.cell_feedback):
                context = context + network.cell_feedback[index](embeddings[batch, choice])
        return torch.stack(chosen, dim=1), dists

    def sample(self) -> torch.Tensor:
        self._last = self._walk()
        return self._last[0]

    def mode(self) -> torch.Tensor:
        self._last = self._walk(deterministic=True)
        return self._last[0]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._last is None or self._last[0] is not actions:
            self._last = (actions, self._walk(given=actions)[1])
        _actions, dists = self._last
        return sum(d.log_prob(actions[:, i].long()) for i, d in enumerate(dists))

    def entropy(self) -> torch.Tensor:
        if self._last is None:
            self.sample()
        return sum(d.entropy() for d in self._last[1])

    def actions_from_params(self, latent: CoordinateLatent, deterministic: bool = False):
        self.proba_distribution(latent)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, latent: CoordinateLatent):
        actions = self.actions_from_params(latent)
        return actions, self.log_prob(actions)


class ObjectLatent(NamedTuple):
    """What forward_actor hands the distribution under factored object
    heads: the context, the rows the object heads score, and the raw
    material the colour and direction heads build their keys from."""
    context: torch.Tensor
    rows: torch.Tensor
    row_mask: torch.Tensor
    objects: torch.Tensor
    colour_shares: torch.Tensor


def _field(name: str) -> int:
    """Where one scalar field of the object embedding sits."""
    position = 0
    for field, _group, arity in OBJECT_SCHEMA:
        if field == name:
            return position
        position += arity
    raise KeyError(name)


_POSITION_FIELDS = [_field(name) for name in
                    ("i_center", "j_center", "min_i", "min_j", "max_i", "max_j")]
_STEPS = torch.tensor([DIRECTION_STEPS[d] for d in ALL_DIRECTIONS], dtype=torch.float32)


def colour_features(shares, objects, first, second):
    """(batch, N_COLOURS, 5): per colour, its share of the grid, its share
    of each chosen object, and whether it is a minority in each - present
    but under half. The last row, "no colour", is zeros.

    Relative to the observation rather than a colour's identity: "paint in
    the colour of the dot on the chosen object" is a score over these, the
    same on a grid where the dot is blue and one where it is sky.
    """
    batch = torch.arange(objects.shape[0], device=objects.device)
    in_first = objects[batch, first, :10]
    in_second = objects[batch, second, :10]
    minority = lambda share: ((share > 0) & (share < 0.5)).float()  # noqa: E731
    features = torch.stack([shares, in_first, in_second, minority(in_first),
                            minority(in_second)], dim=-1)
    return torch.cat([features, features.new_zeros(features.shape[0], 1, 5)], dim=1)


def direction_features(objects, chosen):
    """(batch, N_DIRECTIONS, 4): per direction, where the chosen object's
    mass sits along it within its box, how much of the grid lies beyond the
    box that way, and the direction's own step. The last row, "no
    direction", is zeros.

    "Emit away from the heavy end" - a triangle's base - is a score over
    the first of these on every grid, pointing whichever way the triangle
    does.
    """
    batch = torch.arange(objects.shape[0], device=objects.device)
    ic, jc, min_i, min_j, max_i, max_j = objects[batch, chosen][:, _POSITION_FIELDS].unbind(-1)
    mass_i = (ic - (min_i + max_i) / 2) / (max_i - min_i + 1e-3)
    mass_j = (jc - (min_j + max_j) / 2) / (max_j - min_j + 1e-3)
    steps = _STEPS.to(objects.device)
    di, dj = steps[:, 0], steps[:, 1]
    mass = mass_i[:, None] * di + mass_j[:, None] * dj
    room_i = torch.where(di > 0, 1 - max_i[:, None], torch.where(di < 0, min_i[:, None],
                                                                   torch.ones_like(mass)))
    room_j = torch.where(dj > 0, 1 - max_j[:, None], torch.where(dj < 0, min_j[:, None],
                                                                   torch.ones_like(mass)))
    features = torch.stack([mass, torch.minimum(room_i, room_j),
                            di.expand_as(mass), dj.expand_as(mass)], dim=-1)
    return torch.cat([features, features.new_zeros(features.shape[0], 1, 4)], dim=1)


class FactoredObjectDistribution(Distribution):
    """(action, object, object) chosen as parts: the type, the first
    object, the second, the colour, the second colour, the direction - each
    knowing the ones before it, each from a head shared by every name that
    has that part (see rl.action_structure).

    The flat vocabulary gave every name its own logit, so a held-out pair
    needing a name no training pair used could not be answered: on
    25d487eb the pairs need blue emission east, green north, red south, and
    the test sky north. Here the type is one choice, the colour another -
    scored from what the observation says about each colour and the chosen
    objects - and the direction a third, scored from where the chosen
    object's mass sits. None of the three has to have been seen together.

    A part the type does not have is not chosen: submit takes no objects,
    a single-object transform takes its first object twice (World runs one
    object when the two slots agree), a two-object one two different ones,
    and a colour or direction the type's names lack is masked. A forced
    choice has probability one, so it adds nothing to the log probability
    or the entropy.
    """

    def __init__(self, network: "ARCCustomNetwork"):
        super().__init__()
        self.network = network
        self.latent: Optional[ObjectLatent] = None
        self._last = None

    def proba_distribution_net(self, *args, **kwargs):
        raise NotImplementedError("built by ARCCustomNetwork, not from a latent width")

    def proba_distribution(self, latent: ObjectLatent) -> "FactoredObjectDistribution":
        self.latent = latent
        self._last = None
        return self

    def _walk(self, given: Optional[torch.Tensor] = None, deterministic: bool = False):
        net, latent = self.network, self.latent
        heads = net.factored_heads
        type_head, first_head, second_head = net.policy_nets
        batch_size, slots = latent.row_mask.shape
        batch = torch.arange(batch_size, device=latent.context.device)
        slot_ids = torch.arange(slots, device=latent.context.device)
        if given is not None:
            parts = net.action_components[given[:, 0].long()]
            wanted = [parts[:, 0], given[:, 1].long(), given[:, 2].long(),
                      parts[:, 1], parts[:, 2], parts[:, 3]]

        def pick(logits, allowed, index):
            distribution = Categorical(logits=logits.masked_fill(~allowed, -1e9))
            if given is not None:
                return distribution, wanted[index]
            if deterministic:
                return distribution, distribution.probs.argmax(dim=-1)
            return distribution, distribution.sample()

        context = latent.context
        dists = []
        dist, action_type = pick(type_head(context), torch.ones(
            batch_size, len(net.action_structure.types), dtype=torch.bool,
            device=context.device), 0)
        dists.append(dist)
        context = context + heads["type_feedback"](action_type)
        takes = net.objects_taken[action_type]

        visible = latent.row_mask.bool()
        slot_zero = slot_ids.unsqueeze(0) == 0
        allowed = torch.where((takes == 0).unsqueeze(1), slot_zero, visible)
        allowed = torch.where(allowed.any(dim=1, keepdim=True), allowed, slot_zero)
        dist, first = pick(first_head(context, latent.rows, torch.ones_like(visible)), allowed, 1)
        dists.append(dist)
        context = context + heads["first_feedback"](latent.rows[batch, first])

        same = slot_ids.unsqueeze(0) == first.unsqueeze(1)
        allowed = torch.where((takes == 0).unsqueeze(1), slot_zero,
                              torch.where((takes == 1).unsqueeze(1), same, visible & ~same))
        allowed = torch.where(allowed.any(dim=1, keepdim=True), allowed, same)
        dist, second = pick(second_head(context, latent.rows, torch.ones_like(visible)), allowed, 2)
        dists.append(dist)
        context = context + heads["second_feedback"](latent.rows[batch, second])

        colours = colour_features(latent.colour_shares, latent.objects, first, second)
        # Relative keys scored by a query that reads the context, plus a
        # bias per colour that reads nothing - see ARCCustomNetwork's
        # direction_keys for why the absolute part cannot sit in the keys.
        keys = heads["colour_key"](colours)
        scale = keys.shape[-1] ** 0.5
        logits = ((heads["colour_query"](context).unsqueeze(1) * keys).sum(-1) / scale
                  + net.colour_bias)
        dist, colour = pick(logits, net.valid_colour[action_type], 3)
        dists.append(dist)
        context = context + heads["colour_feedback"](colour)

        logits = ((heads["second_colour_query"](context).unsqueeze(1) * keys).sum(-1) / scale
                  + net.second_colour_bias)
        dist, second_colour = pick(logits, net.valid_second_colour[action_type, colour], 4)
        dists.append(dist)
        context = context + heads["second_colour_feedback"](second_colour)

        # Mass alone in the keys: nothing that says which way is which, and
        # nothing about where the object stands. Which way is which enters as
        # a bias that does not see the observation, when direction_keys asks
        # for it. The room beyond the object was a key too, and it is the
        # kind of cue that fits training by accident: on 25ff71a9 every pair
        # is a shift down of an object at the top, where down is also the
        # side with the most room - the heads learned "towards the room" and
        # on the test, whose room lies east, shifted east on every seed.
        keys = heads["direction_key"](direction_features(latent.objects, first)[..., :1])
        logits = (heads["direction_query"](context).unsqueeze(1) * keys).sum(-1) / scale
        if net.direction_keys == "both":
            logits = logits + net.direction_bias
        dist, direction = pick(logits, net.valid_direction[action_type, colour, second_colour], 5)
        dists.append(dist)

        flat = net.action_index[action_type, colour, second_colour, direction]
        return torch.stack([flat, first, second], dim=1), dists

    def sample(self) -> torch.Tensor:
        self._last = self._walk()
        return self._last[0]

    def mode(self) -> torch.Tensor:
        self._last = self._walk(deterministic=True)
        return self._last[0]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._last is None or self._last[0] is not actions:
            self._last = (actions, self._walk(given=actions)[1])
        dists = self._last[1]
        parts = self.network.action_components[actions[:, 0].long()]
        chosen = [parts[:, 0], actions[:, 1].long(), actions[:, 2].long(),
                  parts[:, 1], parts[:, 2], parts[:, 3]]
        return sum(d.log_prob(value) for d, value in zip(dists, chosen))

    def entropy(self) -> torch.Tensor:
        if self._last is None:
            self.sample()
        return sum(d.entropy() for d in self._last[1])

    def actions_from_params(self, latent: ObjectLatent, deterministic: bool = False):
        self.proba_distribution(latent)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, latent: ObjectLatent):
        actions = self.actions_from_params(latent)
        return actions, self.log_prob(actions)


class ARCCustomNetwork(nn.Module):
    """Custom network for policy and value function.

    Receives as input the features extracted by the features extractor.

    Args:
        feature_dim: Dimension of the features extracted by the features
            extractor (e.g. features from a CNN).
        action_dims: List of dimensions for each action space.
        use_sde: Whether to use state dependent exploration.
        net_arch: Network architecture for policy and value networks.
        action_heads: Number of action distribution heads (1, 2, 3, or 5).
    """

    def __init__(
        self,
        feature_dim: int,
        action_dims: list,
        use_sde: bool = False,
        net_arch: dict = {'pi': [64], 'vf': [64]},
        action_heads: int = 1,
        feature_dim_vf: Optional[int] = None,
        pointer_slots: Optional[int] = None,
        pointer_dim: Optional[int] = None,
        coordinate_shape: Optional[Tuple[int, int]] = None,
        coordinate_dim: Optional[int] = None,
        coordinate_heads: str = "autoregressive",
        action_structure: Optional[ActionStructure] = None,
        factored_width: int = 0,
        direction_keys: str = "both",
    ):
        super().__init__()
        if direction_keys not in ("relative", "both"):
            raise ValueError(f"direction_keys={direction_keys!r}: expected 'relative' or 'both'")
        #: What the factored heads score a direction from. The keys are
        #: relative either way - where the chosen object's mass sits along
        #: the direction and how much room lies that way - scored by a query
        #: that reads the context. 'both' adds a bias per direction that
        #: reads nothing: one absolute preference for the whole task.
        #:
        #: Where the absolute part sits is the whole question. As a
        #: direction's embedding or step in the keys, the context-dependent
        #: query could use it to give each training pair its own direction,
        #: and it did: on 25d487eb (east, north, south in the three pairs)
        #: training reached 1.0 on the pairs by remembering them, and the
        #: test got east - while with the absolute part made small at start,
        #: 25ff71a9 (every pair a shift down) was fitted by a relative cue
        #: that missed the test. Which one won was a matter of
        #: initialisation. A bias that sees no observation cannot remember
        #: pairs: it answers "down" only when down answers every pair, and
        #: otherwise the relative keys have to carry it. Colours get the
        #: same treatment, always on (colour_bias).
        self.direction_keys = direction_keys
        #: Set, the object heads choose an action by its parts
        #: (FactoredObjectDistribution) rather than one logit per name; the
        #: features then end in the raw object rows and the grid's colour
        #: shares (ARCCombinedExtractor.factored_tail).
        self.action_structure = action_structure
        self.factored_width = factored_width if action_structure is not None else 0
        if coordinate_heads not in ("autoregressive", "independent"):
            raise ValueError(f"coordinate_heads={coordinate_heads!r}: expected "
                             "'autoregressive' or 'independent'")
        #: Whether each coordinate is chosen knowing the action and the
        #: cells chosen before it (AutoregressiveCoordinateDistribution) or
        #: all five independently given the state, as the object heads are.
        self.coordinate_heads = coordinate_heads
        #: Under coordinate addressing the features end in one embedding per
        #: row and one per column of the grid (see ARCCombinedExtractor's
        #: coordinate_rows), and the four coordinate heads score those
        #: rather than reading a fixed Linear over the shared latent.
        self.coordinate_shape = (tuple(coordinate_shape)
                                 if coordinate_shape is not None else None)
        self.coordinate_dim = coordinate_dim
        self.coordinate_width = (0 if coordinate_shape is None
                                 else sum(coordinate_shape) * (coordinate_dim + 1))
        #: When the features carry per-object rows (see
        #: ARCCombinedExtractor.pointer_tail), the object heads score each
        #: slot from its own row. Without them they stay as they were.
        self.pointer_slots = pointer_slots
        self.pointer_dim = pointer_dim
        #: How much of the feature vector is the pointer tail. Everything
        #: before it is what the shared and value networks read.
        self.pointer_width = (0 if pointer_slots is None
                              else pointer_slots * (pointer_dim + 1))
        self.action_dims = action_dims
        self.action_heads = action_heads
        self.n_action_dims = len(action_dims)
        #: Everything past what the shared and value networks read: the
        #: object rows, the coordinate rows and columns, the factored heads'
        #: raw material - in that order, at the very end of the features.
        self.tail_width = self.pointer_width + self.coordinate_width + self.factored_width
        # The critic can be fed a wider observation than the actor - see
        # ARCCustomActorCriticPolicy's critic_only_keys - and then the two
        # halves start from different widths.
        feature_dim_vf = feature_dim if feature_dim_vf is None else feature_dim_vf
        # The tail is read by the pointer heads and by nothing else. Left in
        # the shared network's input it would make the context - and so the
        # query - depend on which slot holds which object, and the logits
        # would stop being an exact permutation of each other when the
        # objects are permuted: measured that way first, swapping two slots
        # moved every logit rather than swapping two of them. The value is a
        # property of the state and not of the ordering either, so the
        # critic drops it as well.
        feature_dim = feature_dim - self.tail_width
        feature_dim_vf = feature_dim_vf - self.tail_width

        # Get network architecture
        policy = net_arch['pi']
        self.latent_dim_pi = policy[-1]
        value = net_arch['vf']
        self.latent_dim_vf = value[-1]

        # Shared network
        shared_net = [nn.Linear(feature_dim, policy[0]), nn.ReLU()]
        for i in range(len(policy)-1):
            shared_net.append(nn.Linear(policy[i], policy[i+1]))
            shared_net.append(nn.ReLU())

        # Value network
        value_net = [nn.Linear(feature_dim_vf, value[0]), nn.ReLU()]
        for i in range(len(value)-1):
            value_net.append(nn.Linear(value[i], value[i+1]))
            value_net.append(nn.ReLU())

        # Policy network
        self.shared_net = nn.Sequential(*shared_net[:-1])  # Remove the last ReLU

        # Value network
        self.value_net = nn.Sequential(*value_net)

        # Create policy networks based on action_heads
        self.policy_nets = nn.ModuleList()

        if action_structure is not None:
            self._build_factored_heads(action_dims, action_structure)
        elif action_heads == 1:
            # One head for all 5 dimensions combined
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, sum(action_dims)))
        elif action_heads == 2:
            # First head for action type (first dimension)
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[0]))
            # Second head for the rest dimensions
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, sum(action_dims[1:])))
        elif action_heads == 3:
            # One head per dimension of ARCGridWorld's action space: the
            # transform, the first object, the second. This branch used to
            # index action_dims[3] and [4], written for a five-dimensional
            # space (type + two coordinate pairs) that nothing builds any
            # more - so every run configured with three heads, which is
            # what the shipped config asks for, died with IndexError before
            # its first step.
            if self.n_action_dims != 3:
                raise ValueError(
                    f"action_heads=3 means one head per dimension of a "
                    f"three-dimensional action space (transform, object, "
                    f"object); this space has {self.n_action_dims}")
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[0]))
            for dim in action_dims[1:]:
                if self.pointer_slots is None:
                    self.policy_nets.append(nn.Linear(self.latent_dim_pi, dim))
                    continue
                if dim != self.pointer_slots:
                    raise ValueError(
                        f"an object dimension of {dim} against "
                        f"{self.pointer_slots} slots in the features: the "
                        "action space and the observation disagree about how "
                        "many objects there are")
                self.policy_nets.append(
                    PointerHead(self.latent_dim_pi, self.pointer_dim))
        elif action_heads == 5 and self.coordinate_shape is not None:
            # (action, i1, j1, i2, j2): the action from the shared latent,
            # and each coordinate by scoring the rows - or the columns - of
            # the grid from their own embeddings. Four heads, not two
            # shared: the first cell and the second play different parts
            # (a triangle's right angle sits on the first's row), and one
            # head for both would have to score a row the same way for each.
            if self.n_action_dims != 5:
                raise ValueError(
                    f"coordinate heads need the five-dimensional action space "
                    f"(action, i1, j1, i2, j2); this one has {self.n_action_dims}")
            rows, cols = self.coordinate_shape
            if list(action_dims[1:]) != [rows, cols, rows, cols]:
                raise ValueError(
                    f"the action space spans {list(action_dims[1:])} and the "
                    f"observed grid is {rows}x{cols}: the coordinate heads "
                    "score the grid's rows and columns, so the two must agree")
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[0]))
            for _ in range(4):
                self.policy_nets.append(PointerHead(self.latent_dim_pi, coordinate_dim))
            if coordinate_heads == "autoregressive":
                # What each choice adds to the context the next head reads:
                # the action as a learned vector, then the embedding of the
                # row or column chosen, projected into the latent - the
                # row for i1, the column for j1, the row for i2. j2 is last
                # and feeds nothing.
                self.action_embedding = nn.Embedding(action_dims[0], self.latent_dim_pi)
                self.cell_feedback = nn.ModuleList(
                    nn.Linear(coordinate_dim, self.latent_dim_pi) for _ in range(3))
        elif action_heads == 5:
            # Separate head for each dimension
            for dim in action_dims:
                self.policy_nets.append(nn.Linear(self.latent_dim_pi, dim))
        else:
            raise ValueError(f"Unsupported number of action heads: {action_heads}")

    def _build_factored_heads(self, action_dims, structure: ActionStructure):
        """The heads FactoredObjectDistribution walks: a Linear over action
        types and the two pointer heads in policy_nets, as the flat heads
        have them, and around them what carries each choice into the next
        and what scores colours and directions."""
        if self.n_action_dims != 3 or self.pointer_slots is None:
            raise ValueError("factored heads choose (type, object, object) and score "
                             "the objects from their rows: they need the three-part "
                             "object action space and a pointer tail")
        if action_dims[0] != len(structure):
            raise ValueError(f"the action space has {action_dims[0]} actions and the "
                             f"vocabulary the heads were built from {len(structure)}")
        if not self.factored_width:
            raise ValueError("factored heads read the raw objects and colour shares "
                             "at the end of the features; this extractor has none")
        latent, hidden = self.latent_dim_pi, 32
        self.policy_nets.append(nn.Linear(latent, len(structure.types)))
        self.policy_nets.append(PointerHead(latent, self.pointer_dim))
        self.policy_nets.append(PointerHead(latent, self.pointer_dim))
        self.factored_heads = nn.ModuleDict({
            "type_feedback": nn.Embedding(len(structure.types), latent),
            "first_feedback": nn.Linear(self.pointer_dim, latent),
            "second_feedback": nn.Linear(self.pointer_dim, latent),
            "colour_key": nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
            "colour_query": nn.Linear(latent, hidden),
            "second_colour_query": nn.Linear(latent, hidden),
            "colour_feedback": nn.Embedding(N_COLOURS, latent),
            "second_colour_feedback": nn.Embedding(N_COLOURS, latent),
            "direction_key": nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden)),
            "direction_query": nn.Linear(latent, hidden),
        })
        #: The absolute half of the colour and direction choices: a number
        #: per colour (per direction) that reads no observation, so it is one
        #: preference for the whole task - "paint red", "shift down" - and
        #: cannot be a different answer per pair. Zero, so each part still
        #: starts uniform.
        self.colour_bias = nn.Parameter(torch.zeros(N_COLOURS))
        self.second_colour_bias = nn.Parameter(torch.zeros(N_COLOURS))
        self.direction_bias = nn.Parameter(torch.zeros(N_DIRECTIONS))
        # Buffers, so they follow the network to its device.
        self.register_buffer("action_components", torch.as_tensor(structure.components))
        self.register_buffer("action_index", torch.as_tensor(structure.flat))
        self.register_buffer("valid_colour", torch.as_tensor(structure.valid_colour))
        self.register_buffer("valid_second_colour",
                             torch.as_tensor(structure.valid_second_colour))
        self.register_buffer("valid_direction", torch.as_tensor(structure.valid_direction))
        self.register_buffer("objects_taken", torch.as_tensor(structure.objects))

    def start_near_uniform(self):
        """Start the chained heads choosing their parts about uniformly.

        stable-baselines3 initialises the mlp_extractor orthogonally at a
        gain of sqrt(2) and gives only its own action_net the small 0.01
        that keeps a fresh policy's choices even; nn.Embedding is left at
        PyTorch's N(0, 1). The flat heads live with that - one Linear over
        the latent - but the chained ones add embeddings to the context and
        score by dot products of two projections, and there it compounds.
        Measured on a fresh factored policy for 25d487eb: the type embedding
        had norm 16 against a context of 14, the second pair's colour came
        out at probability ~0 and its direction at 0.0007, so the action
        that solves the pair had probability 1e-7 against 1e-2 under the
        flat heads - never tried, and that pair never learned (0.44 over
        the training pairs where flat reached 1.0).

        So: embeddings at a small scale, and the layers that produce a
        head's logits - the type Linear, every query - at the 0.01 gain
        stable-baselines3 gives action_net. The keys and the feedback keep
        their size, so what the heads can come to express is unchanged.

        The autoregressive coordinate heads chain the same way and were
        checked: a fresh one's choices come out within 0.01 nats of uniform
        without this, so they are left as they are.
        """
        def small(module):
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            if self.action_structure is not None:
                heads = self.factored_heads
                for name in ("type_feedback", "colour_feedback",
                             "second_colour_feedback", "colour_query",
                             "second_colour_query", "direction_query"):
                    small(heads[name])
                small(self.policy_nets[0])
                for pointer in self.policy_nets[1:]:
                    small(pointer.query)

    def split_factored_tail(self, features: torch.Tensor):
        """(objects, colour_shares) off the very end of the features - the
        layout ARCCombinedExtractor.factored_tail writes."""
        tail = features[:, features.shape[1] - self.factored_width:]
        objects = tail[:, :-10].view(features.shape[0], self.pointer_slots, -1)
        return objects, tail[:, -10:]

    def forward(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns:
            List of latent_policy outputs (one for each action head), and
            latent_value.
        """
        return self.forward_actor(features), self.forward_critic(features)

    def split_pointer_tail(self, features: torch.Tensor):
        """(rows, mask) off the end of the feature vector, or (None, None).

        The shared network still sees the whole vector, tail included -
        this only reads the same numbers a second time, in the shape they
        were written in.
        """
        if self.pointer_slots is None:
            return None, None
        width = self.pointer_dim + 1
        end = features.shape[1] - self.coordinate_width - self.factored_width
        tail = features[:, end - self.pointer_slots * width:end]
        tail = tail.view(-1, self.pointer_slots, width)
        return tail[..., :-1], tail[..., -1]

    def split_coordinate_tail(self, features: torch.Tensor):
        """(rows, row_mask, cols, col_mask) off the very end of the
        feature vector, or None - the layout CoordinateRows.tail writes."""
        if self.coordinate_shape is None:
            return None
        rows, cols = self.coordinate_shape
        width = self.coordinate_dim + 1
        end = features.shape[1] - self.factored_width
        tail = features[:, end - self.coordinate_width:end]
        row_block = tail[:, :rows * width].view(-1, rows, width)
        col_block = tail[:, rows * width:].view(-1, cols, width)
        return (row_block[..., :-1], row_block[..., -1],
                col_block[..., :-1], col_block[..., -1])

    def without_pointer_tail(self, features: torch.Tensor) -> torch.Tensor:
        """The features the shared and value networks read - everything the
        object and coordinate tails are not."""
        if not self.tail_width:
            return features
        return features[:, :-self.tail_width]

    def forward_actor(self, features: torch.Tensor) -> List[torch.Tensor]:
        shared_features = self.shared_net(self.without_pointer_tail(features))
        coordinates = self.split_coordinate_tail(features)
        if coordinates is not None:
            rows, row_mask, cols, col_mask = coordinates
            if self.coordinate_heads == "autoregressive":
                # No logits yet: each head's context depends on what the
                # heads before it chose. The distribution walks them.
                return CoordinateLatent(shared_features, rows, row_mask, cols, col_mask)
            action_head, *cell_heads = self.policy_nets
            # i1 and i2 score rows, j1 and j2 columns.
            grids = [(rows, row_mask), (cols, col_mask)] * 2
            return [action_head(shared_features)] + [
                head(shared_features, embeddings, mask)
                for head, (embeddings, mask) in zip(cell_heads, grids)]
        rows, mask = self.split_pointer_tail(features)
        if self.action_structure is not None:
            objects, shares = self.split_factored_tail(features)
            return ObjectLatent(shared_features, rows, mask, objects, shares)
        return [net(shared_features, rows, mask)
                if isinstance(net, PointerHead) else net(shared_features)
                for net in self.policy_nets]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.without_pointer_tail(features))

class ARCCustomActorCriticPolicy(ActorCriticPolicy):
    """Actor and critic over ARCGridWorld's dict observation.

    `critic_only_keys` names observations the critic may read and the actor
    may not. The critic runs only during training - it exists to turn
    returns into advantages, and nothing calls it at inference - so it is
    free to read what will not exist at test time, while the actor, which is
    all that runs on a held-out pair, never sees it. 'target' is the case
    this was built for: the wanted output is known for every training
    example and unknown for the test one.

    stable-baselines3's own share_features_extractor=False does not do this.
    It builds two extractors over the *same* observation (see
    ActorCriticPolicy.extract_features), so both halves still see every key;
    routing different keys to each is what the code below adds.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        features_extractor_class=ARCCombinedExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        action_heads: int = 1,
        critic_only_keys: Tuple[str, ...] = (),
        coordinate_heads: str = "autoregressive",
        object_heads: str = "flat",
        action_names: Optional[Dict[int, str]] = None,
        direction_keys: str = "both",
        *args,
        **kwargs,
    ):
        # Save action_heads before passing to parent class
        self.action_heads = action_heads
        self.coordinate_heads = coordinate_heads
        if object_heads not in ("flat", "factored"):
            raise ValueError(f"object_heads={object_heads!r}: expected 'flat' or 'factored'")
        if object_heads == "factored" and action_names is None:
            raise ValueError("factored object heads need the vocabulary's names to take "
                             "apart - pass action_names")
        #: The vocabulary as parts when the object heads choose by them
        #: (FactoredObjectDistribution), None for one logit per name.
        self.action_structure = (ActionStructure(action_names) if object_heads == "factored"
                                 else None)
        self.direction_keys = direction_keys
        self.critic_only_keys = tuple(critic_only_keys)
        if self.critic_only_keys:
            # Two extractors, or there is nothing to route between.
            kwargs["share_features_extractor"] = False

        # Disable orthogonal initialization if needed
        kwargs["ortho_init"] = kwargs.get("ortho_init", True)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )
        # After the parent's _build, whose orthogonal initialisation is the
        # thing being corrected - see ARCCustomNetwork.start_near_uniform.
        self.mlp_extractor.start_near_uniform()

    def _actor_observation_space(self) -> spaces.Space:
        """What the actor is allowed to see."""
        if not self.critic_only_keys:
            return self.observation_space
        return spaces.Dict({key: space
                            for key, space in self.observation_space.spaces.items()
                            if key not in self.critic_only_keys})

    def _actor_observation(self, obs):
        if not self.critic_only_keys:
            return obs
        return {key: value for key, value in obs.items()
                if key not in self.critic_only_keys}

    def _build_mlp_extractor(self) -> None:
        # Runs inside ActorCriticPolicy._build, after both extractors exist
        # and before value_net and the optimiser - so replacing the actor's
        # extractor here still leaves it in self.parameters().
        if self.critic_only_keys:
            self.pi_features_extractor = self.features_extractor_class(
                self._actor_observation_space(),
                **(self.features_extractor_kwargs or {}))
            self.features_dim = self.pi_features_extractor.features_dim
        action_dims = self.action_space.nvec.tolist()
        self.mlp_extractor = ARCCustomNetwork(
            self.features_dim,
            action_dims=action_dims,
            net_arch=self.net_arch,
            action_heads=self.action_heads,
            feature_dim_vf=self.vf_features_extractor.features_dim,
            # From the actor's extractor, which is the one whose features
            # forward_actor slices - with critic_only_keys the two are
            # different objects built over different observations.
            pointer_slots=getattr(self._actor_extractor(), "pointer_slots", None),
            pointer_dim=getattr(self._actor_extractor(), "pointer_dim", None),
            coordinate_shape=self._coordinate_shape(),
            coordinate_dim=self._coordinate_dim(),
            coordinate_heads=self.coordinate_heads,
            action_structure=self.action_structure,
            factored_width=getattr(self._actor_extractor(), "factored_width", 0),
            direction_keys=self.direction_keys,
        )

    def _coordinate_shape(self):
        rows = getattr(self._actor_extractor(), "coordinate_rows", None)
        return None if rows is None else (rows.rows, rows.cols)

    def _coordinate_dim(self):
        rows = getattr(self._actor_extractor(), "coordinate_rows", None)
        return None if rows is None else rows.dim

    def _actor_extractor(self):
        return (self.pi_features_extractor if self.critic_only_keys
                else self.features_extractor)

    def _get_action_dist_from_latent(self, latent_pi: List[torch.Tensor]):
        """Create action distributions based on the number of action heads."""
        if isinstance(latent_pi, ObjectLatent):
            return FactoredObjectDistribution(self.mlp_extractor).proba_distribution(latent_pi)
        if isinstance(latent_pi, CoordinateLatent):
            return AutoregressiveCoordinateDistribution(self.mlp_extractor).proba_distribution(
                latent_pi)
        action_dims = self.action_space.nvec.tolist()
        logits = torch.hstack(latent_pi)
        distribution = MultiCategoricalDistribution(action_dims)
        distribution.proba_distribution(logits)
        return distribution

    def extract_features(self, obs, features_extractor=None):
        """Preprocess the observation if needed and extract features.

        Returns one tensor when the extractor is shared and (actor, critic)
        when it is not, matching ActorCriticPolicy - forward() and
        evaluate_actions() below unpack accordingly. The actor is handed an
        observation with critic_only_keys removed rather than one it is
        merely expected to ignore, so the split holds whatever an extractor
        does with keys it was not built for.
        """
        if self.share_features_extractor:
            extractor = features_extractor or self.features_extractor
            return extractor(obs)
        return (self.pi_features_extractor(self._actor_observation(obs)),
                self.vf_features_extractor(obs))

    def forward(self, obs, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both the actor and critic networks.

        Args:
            obs: Observation.
            deterministic: Whether to sample or use deterministic actions.

        Returns:
            Action, value, and log probability of the action.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)

        # Get latent representations
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # print(f'Value net: {self.value_net}')
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        # Get actions and log probabilities
        distribution = self._get_action_dist_from_latent(latent_pi)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        # Reshape actions to match action space
        actions = actions.reshape((-1, len(self.action_space.nvec)))
        # print(f'Values at the end: {values}')
        return actions, values, log_prob

class ARCGNNPolicy(ARCCustomActorCriticPolicy):
    """Policy using GNN approach"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # pop (not get): the key has to leave kwargs, or it'd also be passed
        # through **kwargs below and collide with the explicit argument.
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ARCGNNExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

class ARCSeparatePolicy(ARCCustomActorCriticPolicy):
    """Policy using enhanced separate processing approach"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ARCSeparateExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )
