import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from symbolic.objects_analysis import (
    COLOR,
    OBJECT_DIM,
    POSITION,
    SIZE,
    TOPOLOGY,
    group_indices,
)
from symbolic.summaries import (
    RELATION_DIM,
    SHAPE_REL,
    SIMILARITY_REL,
    SPATIAL_REL,
    relation_group_indices,
)

def unpadded_grid_features(extractor, grids, shapes=None, prepare=None):
    """Run `extractor` over the real grids, with observation padding off.

    stable-baselines3 needs one array shape per observation key across a
    whole vector of envs, so a task showing its rule at several grid sizes
    has to hand over padded grids (see ARCGridWorld.observed_grid). The
    padding is an artefact of that buffer and nothing the policy should
    see: fed through as-is it is a block of a colour that does not exist,
    covering more of the array than the grid does on the small examples.

    So it is cropped back off here. Samples are grouped by their true
    shape - `shapes` is the observation's grid_shape entry - and each group
    goes through the extractor at its own size, which is exact rather than
    approximately right, and works whatever `extractor` is. A vector holds
    one shape per env, so the loop runs a handful of times at most.

    `shapes=None` means the observation was never padded: one call, as
    before.
    """
    prepare = prepare or (lambda grid: grid.unsqueeze(1))
    if shapes is None:
        return extractor(prepare(grids))
    sizes = shapes.to(torch.int64)
    features = None
    for size in torch.unique(sizes, dim=0):
        rows, cols = int(size[0]), int(size[1])
        picked = (sizes == size).all(dim=1).nonzero(as_tuple=True)[0]
        computed = extractor(prepare(grids[picked][:, :rows, :cols]))
        if features is None:
            features = computed.new_zeros((grids.shape[0], computed.shape[1]))
        features[picked] = computed
    return features


# =============================================================================
# APPROACH 1: GRAPH NEURAL NETWORK (GNN) APPROACH
# =============================================================================

class ObjectRelationGNN(nn.Module):
    """Graph Neural Network for processing objects and their relations.
    Objects are nodes, relations are edges.
    """

    def __init__(self, object_dim=OBJECT_DIM, relation_dim=RELATION_DIM,
                 hidden_dim=128, output_dim=256, num_layers=3):
        super().__init__()

        self.object_dim = object_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initial object embedding projection
        self.object_projection = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge feature processing for relations
        self.edge_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1, edge_dim=hidden_dim)
                )
            elif i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.1, edge_dim=hidden_dim)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1, edge_dim=hidden_dim)
                )

        # Graph-level aggregation
        self.graph_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, batch_graphs):
        """batch_graphs: PyTorch Geometric Batch object containing multiple graphs
        """
        x, edge_index, edge_attr, batch = batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr, batch_graphs.batch

        # Initial node embedding
        x = self.object_projection(x)

        # Process edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Apply GAT layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat_layer(x, edge_index, edge_attr)
            x = layer_norm(x + x_new) if i > 0 else layer_norm(x_new)  # Skip connection except first layer
            x = F.relu(x)

        # Graph-level aggregation
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)

        # Final projection
        output = self.graph_aggregator(graph_repr)

        return output

class GraphDataConstructor:
    """ Constructs PyTorch Geometric Data objects from object and relation embeddings."""

    def __init__(self):
        pass

    def create_graph_from_embeddings(self, object_embeddings, relation_embeddings, object_pairs):
        """Create a graph from object and relation embeddings

        Args:
            object_embeddings: tensor of shape (num_objects, object_dim)
            relation_embeddings: tensor of shape (num_relations, relation_dim)
            object_pairs: list of tuples indicating which objects are connected

        Returns:
            PyTorch Geometric Data object
        """
        # Node features are object embeddings
        x = object_embeddings

        # Create edge index and edge attributes
        if len(object_pairs) > 0:
            edge_index = torch.tensor(object_pairs, dtype=torch.long).t().contiguous()
            edge_attr = relation_embeddings
        else:
            # If no relations, create empty edge structure
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, relation_embeddings.shape[1]))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def batch_graphs(self, graph_list):
        """Batch multiple graphs together"""
        return Batch.from_data_list(graph_list)

def pairwise_relations(relations: torch.Tensor) -> torch.Tensor:
    """`(B, n, (n-1)*R)` as `(B, n, n, R)`, with zeros down the diagonal.

    The env lays a row out as "the others, in slot order, with this slot
    left out", so the block holding slot j inside row i is at
    `j - (j > i)`. Unpacking it is what lets a relation be read by the pair
    it belongs to rather than by its position in a flattened vector -
    which is the whole difference between weights shared across pairs and
    a weight per (pair, feature) slot.

    Done with one gather rather than the pair of Python loops in
    graph_inputs: this runs inside the forward pass of every step, where
    that one runs per sample and per pair.
    """
    batch, slots, width = relations.shape
    if slots < 2:
        return relations.new_zeros(batch, slots, slots, width)
    dim = width // (slots - 1)
    blocks = relations.view(batch, slots, slots - 1, dim)

    rows = torch.arange(slots, device=relations.device).unsqueeze(1)
    columns = torch.arange(slots, device=relations.device).unsqueeze(0)
    # Clamped at both ends because of the diagonal, which is not a pair and
    # gets cleared below: for the last slot the formula points one past the
    # final block, and gather raises rather than ignoring an index whose
    # value is about to be multiplied by zero.
    block_of = (columns - (columns > rows).long()).clamp(min=0, max=slots - 2)
    index = block_of.view(1, slots, slots, 1).expand(batch, slots, slots, dim)
    unpacked = blocks.gather(2, index)

    # The diagonal is not a pair. block_of puts something there (slot i
    # reads its own first block) and it has to be cleared, or every object
    # gets a message from itself that means whatever its first partner did.
    keep = (rows != columns).view(1, slots, slots, 1)
    return unpacked * keep


class RelationMessages(nn.Module):
    """One round of message passing over the objects, along the relations.

    The alternative already in this file is `Flatten -> Linear(d, 2d)` over
    the whole relation matrix, whose width is itself quadratic in the slot
    count - so its parameter count grows as the fourth power: 7.4M at 8
    slots, 18.8M at 10, 132.9M at 16, against 155k for the object branch
    beside it. That is not the price of relations, it is the price of
    reading them positionally: after the flatten, "the distance between
    slot 3 and slot 7" is index 1234 and gets its own column of weights,
    unrelated to the column holding the same feature for another pair, and
    the indices shift when the object count changes.

    Here the same little network is applied to every pair, so the
    parameters are constant in the slot count and what is learned about one
    pair transfers to the rest. Each object then takes the mean of the
    messages from its partners - masked, so padded slots send nothing and
    an object with no partners gets zeros rather than a division by zero.

    The messages are kept beside the object rows and merged by a learned
    projection rather than concatenated into the object vector: an object's
    own description and what it stands in relation to are different kinds
    of thing, and the merge is where the network decides how much of the
    second to let into the first.
    """

    def __init__(self, relation_dim: int = RELATION_DIM, hidden: int = 32,
                 object_dim: int = 128, dropout: float = 0.1,
                 endpoints: bool = False, aggregation: str = "mean",
                 rounds: int = 1, endpoint_dim: int = None):
        """The axes a relation-architecture sweep varies. The defaults are
        what the first measurement used, so it stays reproducible.

        `endpoints` is the one that changes what this is. With it off a
        message is f(e_ij) - a function of the relation alone, which makes
        this an aggregation over edges rather than message passing. A
        graph network's message is f(h_i, h_j, e_ij): what one object
        tells another depends on both of them as well as on the relation
        between them. "The object to my left is the same shape as me"
        cannot be said by the relation vector by itself.

        `aggregation` is how an object combines what its partners sent.
        A mean says "what are my relations like on average", a max says
        "is any of them like this", a sum also counts how many there are -
        which is the one that can distinguish two partners from five, and
        the one whose scale grows with the object count.

        `rounds` is how far news travels. One round tells an object about
        its partners; two tell it about its partners' partners, which is
        what a rule spanning three objects would need.

        `endpoint_dim` is what makes `endpoints` a fair test of itself.
        Concatenated raw, the two object rows are 256 numbers at unit scale
        against the relation's 24 at a mean magnitude of 0.344 - measured
        on a built module, changing the relations then moves a message by
        0.0138 where changing the objects moves it by 0.1518, eleven times
        more, and the same relation change moves it 2.9x *less* than it
        does without endpoints at all. So adding the objects mostly
        subtracts the relation. Setting endpoint_dim projects each row to
        that width and normalises it first, which is the version worth
        measuring; None keeps the raw concatenation that was.
        """
        super().__init__()
        if aggregation not in ("mean", "max", "sum"):
            raise ValueError(f"aggregation must be mean, max or sum, "
                             f"got {aggregation!r}")
        if rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {rounds}")
        self.endpoints = endpoints
        self.aggregation = aggregation
        self.rounds = rounds
        self.endpoint_projection = None
        if endpoints and endpoint_dim:
            # LayerNorm as well as the projection: the width is half the
            # imbalance and the scale is the other half. bias=False is what
            # makes the pair of them scale-free - a bias does not scale
            # with its input, so LayerNorm(10Wx + b) is not LayerNorm(Wx +
            # b), and rows that drifted louder would drown the relation
            # again through the back door.
            self.endpoint_projection = nn.Sequential(
                nn.Linear(object_dim, endpoint_dim, bias=False),
                nn.LayerNorm(endpoint_dim),
            )
        endpoint_width = (endpoint_dim or object_dim) if endpoints else 0
        message_in = relation_dim + 2 * endpoint_width
        # One set of weights re-used across rounds rather than one per
        # round: the parameter count stays put, and a second round is then
        # a claim about distance rather than about capacity.
        self.message = nn.Sequential(
            nn.Linear(message_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.merge = nn.Sequential(
            nn.Linear(object_dim + hidden, object_dim),
            nn.ReLU(),
            nn.LayerNorm(object_dim),
        )
        self.hidden = hidden

    def forward(self, rows: torch.Tensor, relations: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """rows `(B, n, object_dim)`, relations `(B, n, (n-1)*R)`, mask
        `(B, n)` marking the slots that hold an object."""
        pairs = pairwise_relations(relations)
        slots = rows.shape[1]
        # A message counts when its sender holds an object and the pair is
        # not an object with itself. Without the sender mask a padded slot
        # contributes self.message(0), which is not zero once the second
        # layer has a bias.
        senders = mask.unsqueeze(1).unsqueeze(-1).to(pairs.dtype)
        not_self = (~torch.eye(slots, dtype=torch.bool, device=rows.device)
                    ).view(1, slots, slots, 1).to(pairs.dtype)
        weights = senders * not_self

        for _ in range(self.rounds):
            rows = self.merge(torch.cat(
                [rows, self._pooled(rows, pairs, weights)], dim=-1))
        return rows

    def _pooled(self, rows, pairs, weights):
        """What each object hears from its partners, this round."""
        if self.endpoints:
            slots = rows.shape[1]
            ends = rows if self.endpoint_projection is None \
                else self.endpoint_projection(rows)
            receiver = ends.unsqueeze(2).expand(-1, -1, slots, -1)
            sender = ends.unsqueeze(1).expand(-1, slots, -1, -1)
            pairs = torch.cat([receiver, sender, pairs], dim=-1)
        messages = self.message(pairs) * weights

        if self.aggregation == "sum":
            return messages.sum(2)
        if self.aggregation == "max":
            # Masked-out messages are already zero, and a max over zeros is
            # zero - which is what an object with no partners should hear,
            # and is not the -inf a padded slot would otherwise contribute.
            return messages.max(2).values
        return messages.sum(2) / weights.sum(2).clamp(min=1.0)


class RelationBias(nn.Module):
    """Relations as an additive bias on the attention between objects.

    The other way of spending them, and the cheaper one. RelationMessages
    computes a vector per pair and merges it into the object's row;
    this computes a *number* per pair and head, and adds it to the
    attention logit for that pair - so a relation does not enter what an
    object is, only who it looks at. nn.MultiheadAttention takes exactly
    that: a float attn_mask is added to the scores rather than used as a
    gate, which is what makes this a few lines instead of a hand-written
    attention.

    Cheaper in parameters (one output per head against a hidden layer per
    pair) and in what it claims: the object rows keep their own meaning and
    the relations only re-weight the mixing. Whether that is enough is the
    measurement - message passing was worth +0.131 against the control
    with the deltas alongside it, and nothing on its own.

    Requires the self-attention it biases: with self_attention off there is
    no attention to bias, and this quietly becomes a no-op. The extractor
    refuses that combination rather than measuring it.
    """

    def __init__(self, relation_dim: int = RELATION_DIM, num_heads: int = 4):
        super().__init__()
        # No bias term: an all-zero relation vector - which is what a
        # padded pair and the diagonal both carry - has to score zero, or
        # every non-pair gets a constant nudge that the attention then
        # spends its softmax on.
        self.score = nn.Linear(relation_dim, num_heads, bias=False)
        self.num_heads = num_heads

    def forward(self, relations: torch.Tensor) -> torch.Tensor:
        """relations `(B, n, (n-1)*R)` -> `(B * num_heads, n, n)`, laid out
        the way MultiheadAttention wants a per-head float mask."""
        pairs = pairwise_relations(relations)
        scores = self.score(pairs)
        batch, slots, _, heads = scores.shape
        return scores.permute(0, 3, 1, 2).reshape(batch * heads, slots, slots)


def graph_inputs(object_embeddings, relation_embeddings):
    """One observation's objects and relations as (nodes, edges, edge_attr).

    The env packs relations as `(max_objects, (max_objects - 1) *
    RELATION_DIM)`: row i holds object i's vector against each *other*
    object, laid end to end in object order with i itself skipped. The
    previous version of this read that matrix as if it were one row per
    pair, sliced it by pair count, and passed rows of
    `(max_objects - 1) * RELATION_DIM` values as edge features of width
    RELATION_DIM. It also built edge indices out of the padded object
    slots, which do not index the node tensor once padding is dropped.

    Both directions are kept when both are recorded: i's vector against j
    and j's against i are different rows and mean different things. A pair
    with an all-zero vector is left unconnected - that is what "no relation
    recorded" looks like coming out of the padding.
    """
    filled = (object_embeddings.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
    if len(filled) == 0:
        return (torch.zeros((1, OBJECT_DIM), dtype=object_embeddings.dtype),
                [], torch.empty((0, RELATION_DIM), dtype=object_embeddings.dtype))

    nodes = object_embeddings[filled]
    edges, attributes = [], []
    for source, source_slot in enumerate(filled.tolist()):
        row = relation_embeddings[source_slot]
        for target, target_slot in enumerate(filled.tolist()):
            if source_slot == target_slot:
                continue
            # The block for `target_slot` within `source_slot`'s row: the
            # others in slot order, with the row's own slot left out.
            block = target_slot - (1 if target_slot > source_slot else 0)
            if (block + 1) * RELATION_DIM > row.shape[0]:
                continue
            vector = row[block * RELATION_DIM:(block + 1) * RELATION_DIM]
            if not bool((vector != 0).any()):
                continue
            edges.append([source, target])
            attributes.append(vector)
    if not edges:
        return nodes, [], torch.empty((0, RELATION_DIM), dtype=nodes.dtype)
    return nodes, edges, torch.stack(attributes)


class ARCGNNExtractor(BaseFeaturesExtractor):
    """Feature extractor using Graph Neural Networks for object-relation processing."""

    def __init__(self, observation_space: spaces.Dict, extr_arch=None,
                 gnn_output_dim=256, grid_cnn_features=128):
        super().__init__(observation_space, features_dim=1)
        needs_relations(observation_space, type(self).__name__)

        self.gnn_output_dim = gnn_output_dim
        self.grid_cnn_features = grid_cnn_features

        # The same grid encoder the other extractors get, from the same
        # config key: this took no extr_arch at all, so create_agent - which
        # always passes one - could not build this class, and no run ever
        # reached its first step.
        self.grid_extractor = build_grid_arch(extr_arch)
        self.grid_cnn_output_size = grid_arch_width(self.grid_extractor)

        # Grid feature projection
        self.grid_projection = nn.Sequential(
            nn.Linear(self.grid_cnn_output_size, grid_cnn_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # GNN for object-relation processing
        self.gnn = ObjectRelationGNN(
            object_dim=OBJECT_DIM,
            relation_dim=RELATION_DIM,
            output_dim=gnn_output_dim
        )

        # Graph constructor
        self.graph_constructor = GraphDataConstructor()

        # Final feature combination
        self.feature_combiner = nn.Sequential(
            nn.Linear(grid_cnn_features + gnn_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self._features_dim = 256

    def forward(self, observations):
        batch_size = observations['grid'].shape[0]

        # Process grid
        # one_hot_grid, not the raw grid: a colour is a name, not a
        # quantity - see one_hot_grid. Cropped to the true grid first, since
        # the pad value is not a colour.
        grid_features = unpadded_grid_features(self.grid_extractor, observations['grid'],
                                               observations.get('grid_shape'),
                                               one_hot_grid)
        grid_features = self.grid_projection(grid_features)

        # Process object-relation graphs
        gnn_features = []

        for i in range(batch_size):
            nodes, edges, edge_attr = graph_inputs(
                observations['objects_emb'][i], observations['relations_emb'][i])
            gnn_features.append(self.graph_constructor.create_graph_from_embeddings(
                nodes, edge_attr, edges))

        # Batch graphs and process through GNN
        batch_graphs = self.graph_constructor.batch_graphs(gnn_features)
        gnn_output = self.gnn(batch_graphs)

        # Combine features
        combined_features = torch.cat([grid_features, gnn_output], dim=1)
        final_features = self.feature_combiner(combined_features)

        return final_features

# =============================================================================
# APPROACH 2: SEPARATE PROCESSING APPROACH
# =============================================================================

class ObjectProcessor(nn.Module):
    """Enhanced object processor with better feature grouping and attention
    """

    def __init__(self, object_dim=OBJECT_DIM, hidden_dim=128, output_dim=64,
                 dropout=0.1, grouped=True, cross_attention=True,
                 use_position=True):
        """`dropout`, `grouped`, `cross_attention` and `use_position` are the
        axes an object-branch sweep varies; the defaults are what this was.

        dropout reaches five places between here and ObjectSetProcessor -
        0.1 in each group processor, 0.1 in the cross-attention, 0.2 in the
        fusion and 0.1 in the self-attention across objects. It regularises
        a long run, and a policy here gets around a hundred updates, so
        whether it is doing anything but adding noise is a question rather
        than a setting.

        use_position=False keeps the split OBJECT_SCHEMA already declares -
        INTERNAL_GROUPS against EXTERNAL_GROUPS - which nothing had ever
        used. Most ARC transformations are translation invariant, so
        absolute position is exactly the field a policy can fit and not
        transfer.
        """
        super().__init__()
        self.grouped = grouped
        self.use_cross_attention = cross_attention

        # Which slots of the object vector feed each head, taken from the
        # schema that defines the vector rather than restated as ranges: the
        # groups a consumer cares about need not sit next to each other, and
        # a hard-coded range silently reads different fields once the schema
        # changes. Registered as buffers so they follow the module's device.
        spatial_groups = (SIZE, POSITION) if use_position else (SIZE,)
        self.register_buffer("color_index", torch.tensor(group_indices(COLOR), dtype=torch.long))
        self.register_buffer("spatial_index", torch.tensor(group_indices(*spatial_groups), dtype=torch.long))
        self.register_buffer("shape_index", torch.tensor(group_indices(TOPOLOGY), dtype=torch.long))

        self.color_dim = len(group_indices(COLOR))
        self.spatial_dim = len(group_indices(*spatial_groups))
        self.shape_dim = len(group_indices(TOPOLOGY))

        def group_net(width):
            return nn.Sequential(nn.Linear(width, 32), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(32, 16))

        if grouped:
            # Specialized processors for each feature group
            self.color_processor = group_net(self.color_dim)
            self.spatial_processor = group_net(self.spatial_dim)
            self.shape_processor = group_net(self.shape_dim)
            fused_width = 48
            if cross_attention:
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=16, num_heads=4, dropout=dropout, batch_first=True)
        else:
            # One network over the whole vector: the grouping is a claim
            # that colour, extent and topology want separate treatment, and
            # it is only worth its three heads if it beats not making it.
            self.plain_processor = nn.Sequential(
                nn.Linear(self.color_dim + self.spatial_dim + self.shape_dim, 64),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 48))
            fused_width = 48

        # Feature fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(fused_width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(min(dropout * 2, 0.9)),
            nn.Linear(hidden_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """x: tensor of shape (batch_size, max_objects, OBJECT_DIM)"""
        # Split into feature groups
        color_features = x.index_select(-1, self.color_index)
        spatial_features = x.index_select(-1, self.spatial_index)
        shape_features = x.index_select(-1, self.shape_index)

        if not self.grouped:
            combined_features = self.plain_processor(
                torch.cat([color_features, spatial_features, shape_features], dim=-1))
        else:
            # Process each group
            color_emb = self.color_processor(color_features)    # (batch, max_objects, 16)
            spatial_emb = self.spatial_processor(spatial_features)  # (batch, max_objects, 16)
            shape_emb = self.shape_processor(shape_features)    # (batch, max_objects, 16)

            if self.use_cross_attention:
                # Apply cross-attention between spatial and shape features
                spatial_attended, _ = self.cross_attention(spatial_emb, shape_emb, shape_emb)
                spatial_emb = spatial_emb + spatial_attended

            # Concatenate all features
            combined_features = torch.cat([color_emb, spatial_emb, shape_emb], dim=-1)

        # Final processing
        output = self.fusion_net(combined_features)
        output = self.layer_norm(output)

        return output

class RelationProcessor(nn.Module):
    """Enhanced relation processor with semantic grouping
    """

    def __init__(self, relation_dim=RELATION_DIM, hidden_dim=64, output_dim=32):
        super().__init__()

        # Slots feeding each head, taken from RELATION_SCHEMA rather than
        # restated as ranges - see ObjectProcessor for the reasoning.
        self.register_buffer("similarity_index",
                             torch.tensor(relation_group_indices(SIMILARITY_REL), dtype=torch.long))
        self.register_buffer("shape_rel_index",
                             torch.tensor(relation_group_indices(SHAPE_REL), dtype=torch.long))
        self.register_buffer("spatial_rel_index",
                             torch.tensor(relation_group_indices(SPATIAL_REL), dtype=torch.long))

        self.similarity_dim = len(relation_group_indices(SIMILARITY_REL))
        self.shape_rel_dim = len(relation_group_indices(SHAPE_REL))
        self.spatial_rel_dim = len(relation_group_indices(SPATIAL_REL))

        # One head per relation group, each sized from its own slice of the
        # schema. The name of a head, the width it is built at and the
        # features fed to it in forward() all have to name the same group -
        # they read as three independent choices and are one.
        self.similarity_processor = nn.Sequential(
            nn.Linear(self.similarity_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.spatial_processor = nn.Sequential(
            nn.Linear(self.spatial_rel_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Wider than the other two: shape relations are the ones a match has
        # to be judged on rather than read off, and dropout because they are
        # the noisiest (shape_similarity and match_score are both estimates).
        self.shape_processor = nn.Sequential(
            nn.Linear(self.shape_rel_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        # Relation fusion. 8 (similarity) + 8 (spatial) + 16 (shape).
        self.fusion_net = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """x: tensor of shape (batch_size, max_relations, RELATION_DIM)"""
        batch_size, max_relations, _ = x.shape

        # Split into feature groups
        similarity_features = x.index_select(-1, self.similarity_index)
        shape_rel_features = x.index_select(-1, self.shape_rel_index)
        spatial_rel_features = x.index_select(-1, self.spatial_rel_index)

        # Process each group
        similarity_emb = self.similarity_processor(similarity_features)
        shape_rel_emb = self.shape_processor(shape_rel_features)
        spatial_rel_emb = self.spatial_processor(spatial_rel_features)

        # Combine features
        combined_features = torch.cat([similarity_emb, shape_rel_emb, spatial_rel_emb], dim=-1)

        # Final processing
        output = self.fusion_net(combined_features)
        output = self.layer_norm(output)

        return output

class ARCSeparateExtractor(BaseFeaturesExtractor):
    """Enhanced separate processing approach with improved object and relation handling
    """

    def __init__(self, observation_space: spaces.Dict, extr_arch=None,
                 object_output_dim=64, relation_output_dim=64, grid_cnn_features=128):
        super().__init__(observation_space, features_dim=1)
        needs_relations(observation_space, type(self).__name__)

        # The same grid encoder the other extractors get, from the same
        # config key - see ARCGNNExtractor.
        self.grid_extractor = build_grid_arch(extr_arch)
        self.grid_cnn_output_size = grid_arch_width(self.grid_extractor)

        self.grid_projection = nn.Sequential(
            nn.Linear(self.grid_cnn_output_size, grid_cnn_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Enhanced object processor
        self.object_processor = ObjectProcessor(
            object_dim=OBJECT_DIM,
            output_dim=object_output_dim
        )

        # Enhanced relation processor
        self.relation_processor = RelationProcessor(
            relation_dim=RELATION_DIM,
            output_dim=relation_output_dim
        )

        # Object aggregation with attention
        self.object_aggregator = nn.MultiheadAttention(
            embed_dim=object_output_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Relation aggregation
        self.relation_aggregator = nn.Sequential(
            nn.Linear(relation_output_dim, relation_output_dim),
            nn.ReLU()
        )

        # Final feature combination
        total_features = grid_cnn_features + object_output_dim + relation_output_dim
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)
        )

        self._features_dim = 256

    def forward(self, observations):
        batch_size = observations['grid'].shape[0]

        # Process grid
        # one_hot_grid, not the raw grid: a colour is a name, not a
        # quantity - see one_hot_grid. Cropped to the true grid first, since
        # the pad value is not a colour.
        grid_features = unpadded_grid_features(self.grid_extractor, observations['grid'],
                                               observations.get('grid_shape'),
                                               one_hot_grid)
        grid_features = self.grid_projection(grid_features)

        # Process objects
        obj_embeddings = self.object_processor(observations['objects_emb'])

        # Aggregate objects with attention-based pooling
        obj_mask = (observations['objects_emb'].sum(dim=-1) != 0)  # Valid object mask
        if obj_mask.any():
            # The guard above is per batch, and the NaN is per row: one
            # all-background grid among several masks every key of its own
            # row, and softmax over nothing but -inf is NaN. Such a row
            # attends over its first slot and the masked mean below
            # multiplies the result away - see ObjectSetProcessor.forward.
            attends = obj_mask.clone()
            attends[~obj_mask.any(dim=1), 0] = True
            obj_attended, _ = self.object_aggregator(obj_embeddings, obj_embeddings, obj_embeddings,
                                                   key_padding_mask=~attends)
            # Masked mean pooling
            obj_mask_expanded = obj_mask.unsqueeze(-1).float()
            obj_features = (obj_attended * obj_mask_expanded).sum(dim=1) / obj_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            obj_features = torch.zeros(batch_size, obj_embeddings.shape[-1], device=obj_embeddings.device)

        # Process relations
        rel_embeddings = self.relation_processor(observations['relations_emb'])

        # Aggregate relations with mean pooling
        rel_mask = (observations['relations_emb'].sum(dim=-1) != 0)
        if rel_mask.any():
            rel_mask_expanded = rel_mask.unsqueeze(-1).float()
            rel_features = (rel_embeddings * rel_mask_expanded).sum(dim=1) / rel_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            rel_features = torch.zeros(batch_size, rel_embeddings.shape[-1], device=rel_embeddings.device)

        rel_features = self.relation_aggregator(rel_features)

        # Combine all features
        combined_features = torch.cat([grid_features, obj_features, rel_features], dim=1)
        final_features = self.feature_combiner(combined_features)

        return final_features

# =============================================================================
# APPROACH 3: Combined approach
# =============================================================================
#: Observation keys shaped like a grid, each encoded by its own copy of the
#: grid architecture. 'grid' is the one being worked on, 'input_pattern' the
#: example's input, 'target' the wanted output, and the two deltas are
#: per-cell comparisons against the input and the target - see
#: ARCGridWorld._add_deltas for why a delta is carried instead of the grid
#: it compares against.
GRID_KEYS = ('grid', 'input_pattern', 'target')

#: Grid-shaped, but not grids: a delta holds one bit per cell rather than a
#: colour, and it is read by DeltaReadout rather than by the colour encoder.
DELTA_KEYS = ('delta_input', 'delta_target')


def delta_plane(plane):
    """A delta as the single channel it is.

    One bit per cell, so the ten colour planes a grid needs would leave
    eight of them dead. DeltaReadout takes this.
    """
    return plane.to(torch.float32).unsqueeze(1)


class DeltaReadout(nn.Module):
    """A delta plane read as a map, not summarised as a quantity.

    The colour encoder ends in AdaptiveAvgPool2d((1,1)), and a mean over a
    delta is the fraction of cells that differ - which is what max_int
    already is, arrived at more expensively. Worse, it divides by the grid
    area: measured on a 20x20 plane through the shipped convolutions, one
    differing cell reads 0.00039 above an empty plane where a max reads
    0.04787, a hundred and twenty times larger, and it is precisely the
    endgame - a handful of wrong cells left - where the mean vanishes.

    A max does not vanish, but it saturates: one differing cell reads
    0.04787 and five read 0.05313, so it answers "is anything wrong" and
    cannot count. Neither says where: over the whole plane, one cell at the
    top left and one at the bottom right differ only through the boundary
    effect of padding.

    So all three readings are taken, because they are three different
    questions:

        mass  the mean - how much is still wrong
        peak  the max  - whether anything is
        ci,cj the centre of mass - where it is

    The centre of mass treats each channel as a distribution over cells and
    takes its expected row and column, normalised to [0, 1] so the answer
    means the same at any grid size. That is the canonical readout of an
    attention map, which is what a delta is. An empty plane has nothing to
    locate and its centre reads 0 alongside a peak of 0, which is
    distinguishable from a real difference at the top-left corner because
    that one has a peak.

    Four numbers per channel rather than one, and the three that matter are
    the three the pooled encoder could not express.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        # bias=False is what makes the centre of mass mean anything. With a
        # bias, a convolution fires on empty cells too - ReLU(0*w + b) is
        # positive wherever b is - so every channel has a constant
        # background over the whole plane, the distribution below is that
        # background plus a speck, and its expected position is the middle
        # of the grid whatever the delta holds. Measured before the fix: one
        # cell at (1,1) and one at (18,18) both read a centre of
        # (0.524, 0.475). Without a bias the convolution of a zero
        # neighbourhood is exactly zero, so the support of each channel is
        # the difference and its surroundings, and its centre is where the
        # difference is.
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.channels = channels
        self.out_dim = channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.convs(x)
        rows, cols = features.shape[-2:]
        mass = features.mean(dim=(2, 3))
        peak = features.amax(dim=(2, 3))
        flat = features.flatten(2)
        # clamp, not an epsilon added to the total: a channel that fired
        # nowhere has no centre, and dividing by the clamp leaves its
        # weights at zero rather than spreading them evenly over the grid,
        # which would read as "the difference is in the middle".
        weights = flat / flat.sum(dim=2, keepdim=True).clamp(min=1e-6)
        down = torch.arange(rows, device=x.device, dtype=features.dtype)
        across = torch.arange(cols, device=x.device, dtype=features.dtype)
        down = (down / max(rows - 1, 1)).view(1, 1, rows, 1).expand(1, 1, rows, cols)
        across = (across / max(cols - 1, 1)).view(1, 1, 1, cols).expand(1, 1, rows, cols)
        centre_i = (weights * down.flatten(2)).sum(dim=2)
        centre_j = (weights * across.flatten(2)).sum(dim=2)
        return torch.cat([mass, peak, centre_i, centre_j], dim=1)


def one_hot_grid(grid):
    """A grid of colour numbers as one plane per colour.


    Colours are names, not quantities: fed as a single channel, colour 9 is
    nine times colour 1 to a convolution, and the difference between two
    colours becomes a distance. ARCCombinedExtractor has always one-hot
    encoded; ARCGNNExtractor and ARCSeparateExtractor passed the raw grid
    through a one-channel convolution instead, which is a different model of
    what a colour is - and made the three incomparable as architectures.
    """
    planes = torch.nn.functional.one_hot(grid.to(torch.int64), num_classes=10)
    return planes.float().permute(0, 3, 1, 2)


def needs_relations(observation_space, name):
    """Refuse an observation without relations, and say what to set.

    ARCGNNExtractor builds its graph out of them and ARCSeparateExtractor
    processes them in its own branch, so neither can run on an observation
    that carries objects alone - which is what rl_config ships
    (observation_space_elements = ["objects_emb"]). Without this the failure
    is a KeyError from inside forward, naming a dict key rather than the
    setting that decides it.
    """
    if "relations_emb" in getattr(observation_space, "spaces", {}):
        return
    raise ValueError(
        f"{name} reads relation embeddings and this observation has none. "
        "Add 'relations_emb' to rl_config's observation_space_elements, or "
        "use ARCCombinedExtractor, which works without them.")


def build_grid_arch(extr_arch=None):
    """One grid encoder, from a module to copy, a factory, or the default.

    `extr_arch` describes an architecture; a module handed over is copied so
    that two extractors built from one config do not share weights - see
    ARCCombinedExtractor.
    """
    if callable(extr_arch) and not isinstance(extr_arch, nn.Module):
        return extr_arch()
    if extr_arch is not None:
        return copy.deepcopy(extr_arch)
    return default_grid_arch()


def grid_arch_width(arch):
    """How many features a grid encoder emits, by asking it.

    Read off arch[2].out_channels before, which is the second convolution
    of the one architecture shipped and an IndexError or a wrong number for
    anything else a config might name.
    """
    with torch.no_grad():
        return arch(torch.zeros(1, 10, 30, 30)).shape[1]


def default_grid_arch():
    """The grid encoder used when a config does not supply one.

    Pools to 3x3 rather than to 1x1, and the same as
    data.configs.rl_configs.lin, which carries the measurement: a global
    mean returns how much of each texture the grid holds and nothing about
    where, and scored zero on the held-out pair in all six of its runs
    where every encoder keeping some notion of where scored above zero
    somewhere.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 3)),
        nn.Flatten(),
    )


class CoordinateRows(nn.Module):
    """One embedding per row of the grid and one per column - what the
    coordinate heads score when they choose a row or a column.

    The pooled grid encoder answers what the grid holds and, at its 3x3
    resolution, roughly where; a head choosing among thirty rows needs to
    know what is in each of them. So every row is read whole - its cells'
    colours and, when the observation carries them, its cells' deltas - and
    encoded on its own, with its position, into one vector. Columns the
    same way. A head then scores each row from its own vector, the way
    PointerHead scores objects, and "the row where the painting stopped"
    becomes a score over a field that row carries rather than a weight tied
    to a row number that means a different place on the next grid.

    Rows past the grid's true size - the observation is padded to the
    largest grid of the task - are masked, and the heads never pick them.
    """

    def __init__(self, rows: int, cols: int, delta_keys, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.delta_keys = tuple(delta_keys)
        channels = 10 + len(self.delta_keys)
        self.row_encoder = nn.Sequential(
            nn.Linear(cols * channels + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.col_encoder = nn.Sequential(
            nn.Linear(rows * channels + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.dim = dim

    def planes(self, observation):
        """(batch, channels, rows, cols): ten colour planes and one per
        delta, zero wherever the observation is padding."""
        grid = observation["grid"].to(torch.int64)
        # The padding value is not a colour: one class past the ten and
        # then dropped, so a padded cell is all zeros.
        colours = torch.nn.functional.one_hot(grid.clamp(0, 10), num_classes=11)[..., :10]
        stacked = [colours.permute(0, 3, 1, 2).float()]
        for key in self.delta_keys:
            stacked.append(observation[key].float().unsqueeze(1))
        return torch.cat(stacked, dim=1)

    def masks(self, observation):
        """(batch, rows), (batch, cols): which rows and columns are grid."""
        batch = observation["grid"].shape[0]
        device = observation["grid"].device
        shape = observation.get("grid_shape")
        if shape is None:
            return (torch.ones(batch, self.rows, device=device),
                    torch.ones(batch, self.cols, device=device))
        shape = shape.to(torch.int64)
        rows = torch.arange(self.rows, device=device).unsqueeze(0) < shape[:, :1]
        cols = torch.arange(self.cols, device=device).unsqueeze(0) < shape[:, 1:2]
        return rows.float(), cols.float()

    def forward(self, observation):
        """(row_embeddings, row_mask, col_embeddings, col_mask)."""
        planes = self.planes(observation)
        batch = planes.shape[0]
        row_mask, col_mask = self.masks(observation)
        where_row = torch.linspace(0, 1, self.rows, device=planes.device)
        where_col = torch.linspace(0, 1, self.cols, device=planes.device)
        rows = planes.permute(0, 2, 1, 3).reshape(batch, self.rows, -1)
        rows = torch.cat([rows, where_row.view(1, -1, 1).expand(batch, -1, 1)], dim=2)
        cols = planes.permute(0, 3, 1, 2).reshape(batch, self.cols, -1)
        cols = torch.cat([cols, where_col.view(1, -1, 1).expand(batch, -1, 1)], dim=2)
        return (self.row_encoder(rows), row_mask,
                self.col_encoder(cols), col_mask)

    @property
    def width(self) -> int:
        """How much of the feature vector the rows and columns take."""
        return (self.rows + self.cols) * (self.dim + 1)

    def tail(self, observation):
        """The rows and columns as one flat block, each followed by its
        mask bit - the layout ARCCustomNetwork.split_coordinate_tail reads."""
        rows, row_mask, cols, col_mask = self(observation)
        return torch.cat([
            torch.cat([rows, row_mask.unsqueeze(-1)], dim=2).flatten(1),
            torch.cat([cols, col_mask.unsqueeze(-1)], dim=2).flatten(1)], dim=1)


class ARCCombinedExtractor(BaseFeaturesExtractor):
    """The grid-shaped observations get one encoder each, of the shape
    `extr_arch` describes; the object and relation embeddings get their own.

    `extr_arch` is a description, not a network to adopt. The shipped config
    holds a single module built once at import (data.configs.rl_configs.
    lin_arch), so every agent constructed in one process was handed the very
    same nn.Conv2d weights: an agent started where the previous one's
    training had left the encoder, and two agents alive at once had their
    optimisers writing to one set of parameters. Copying makes each
    extractor own what it trains, which is what a config naming an
    architecture means.
    """

    def __init__(self, observation_space: spaces.Dict, extr_arch=None,
                 pointer_dim: int = 32, object_arch=None,
                 relation_mode: str = "flat", relation_arch=None,
                 coordinate_dim: int = 0, factored_tail: bool = False):
        """`relation_mode` decides how 'relations_emb' enters, when the
        observation carries it at all:

          'flat'      the whole matrix through Flatten and two Linears -
                      what this shipped, and quadratic in width, so
                      quartic in the slot count
          'messages'  one round of message passing with weights shared
                      across pairs, merged into the object rows - see
                      RelationMessages
          'bias'      a number per pair added to the attention logits
                      between objects, so relations decide who looks at
                      whom rather than what an object is - see RelationBias

        'messages' reaches the pointer head, because it changes the rows
        the head scores; 'flat' cannot, because it produces one vector for
        the whole observation. That is a difference in kind and not only
        in size: an action names an object, and only the first of these
        can say anything about a particular one.
        """
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        self.relation_mode = relation_mode
        self.relation_messages = None
        #: How wide the per-object rows carried for a pointer head are, and
        #: how many slots there are - None when the observation holds no
        #: objects, or when pointer_dim is 0. ARCCustomNetwork reads both
        #: off the extractor and falls back to the Linear object heads when
        #: there are no rows, which is what makes "the same observation
        #: without a pointer head" a control this can be measured against.
        self.pointer_dim = pointer_dim
        self.pointer_slots = None
        self.build_grid_arch = lambda: build_grid_arch(extr_arch)
        self.extr_arch = self.build_grid_arch()
        for key, subspace in observation_space.spaces.items():
            if key == "objects_emb":
                # print(f'objects_emb subspace.shape:{subspace.shape}')
                extractor, output_dim = create_object_extractor(
                    subspace.shape, object_arch)
                extractors[key] = extractor
                total_concat_size += output_dim
                # Room for the per-object rows a pointer head scores, on
                # top of the pooled vector every existing reader takes -
                # see pointer_tail.
                if pointer_dim:
                    self.pointer_slots = subspace.shape[0]
                    total_concat_size += self.pointer_slots * (pointer_dim + 1)
                # print(f'objects_emb_concat_size: {output_dim}')
            elif key == "relations_emb":
                if relation_mode == "messages":
                    # Nothing of its own in the concatenated vector: the
                    # messages are merged into the object rows, which the
                    # object branch already pools and the pointer tail
                    # already carries. Built after the loop, where the
                    # object branch's width is known.
                    continue
                if relation_mode == "bias":
                    # Same as 'messages': nothing of its own in the
                    # concatenated vector. It re-weights the attention the
                    # object branch already runs.
                    continue
                if relation_mode != "flat":
                    raise ValueError(
                        f"relation_mode must be 'flat', 'messages' or "
                        f"'bias', got {relation_mode!r}")
                dim = subspace.shape[0] * subspace.shape[1]
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim), nn.ReLU())
                total_concat_size += dim
            elif key in GRID_KEYS:
                # 'grid' is the grid being worked on; 'input_pattern' is the
                # example's input kept alongside it (rl_config's
                # input_pattern='separate'), 'target' the wanted output.
                # All three are grids and all three are encoded as one, each
                # through its own weights - the config used to declare these
                # observations and the extractor used to reject them, so
                # every run asking for either died with 'Unknown feature'
                # before its first step.
                extractors[key] = self.extr_arch if key == 'grid' else self.build_grid_arch()
                total_concat_size += grid_arch_width(extractors[key])
            elif key in DELTA_KEYS:
                # Grid-shaped and read differently: see DeltaReadout for why
                # a mean over a delta is max_int computed the long way round.
                readout = DeltaReadout()
                extractors[key] = readout
                total_concat_size += readout.out_dim
            elif key in ('action_space', 'grid_shape'):
                # Neither is a feature to embed. The action space's own
                # .nvec (varies per task) is in every observation so the
                # policy can see it; grid_shape is there to undo the
                # observation padding, and forward() below uses it for
                # exactly that rather than encoding it.
                continue
            else:
                raise ValueError(f'Unknown feature: {key}')

        self.extractors = nn.ModuleDict(extractors)
        self.relation_bias = None
        has_relations = ("relations_emb" in observation_space.spaces
                         and "objects_emb" in extractors)
        if relation_mode == "messages" and has_relations:
            self.relation_messages = RelationMessages(
                object_dim=extractors["objects_emb"].processor.hidden_dim,
                **(relation_arch or {}))
        if relation_mode == "bias" and has_relations:
            processor = extractors["objects_emb"].processor
            if processor.self_attention is None:
                # There is no attention to bias, so this would build
                # cleanly, train, and measure the object branch under
                # another name.
                raise ValueError(
                    "relation_mode='bias' needs the object branch's "
                    "self-attention, which object_arch turned off")
            self.relation_bias = RelationBias(
                num_heads=processor.self_attention.num_heads)
        if self.pointer_slots is not None:
            self.pointer_projection = nn.Linear(
                extractors["objects_emb"].processor.hidden_dim, pointer_dim)
        #: The rows and columns the coordinate heads score, or None. Built
        #: over the observation's own grid shape - which coordinate
        #: addressing pads to the action space's - from the colours and
        #: whichever deltas this observation carries: the actor's has
        #: delta_input, the critic's delta_target as well.
        self.coordinate_rows = None
        if coordinate_dim:
            if "grid" not in observation_space.spaces:
                raise ValueError(
                    "coordinate heads score the rows and columns of the "
                    "grid, and this observation has no grid")
            rows, cols = observation_space.spaces["grid"].shape
            self.coordinate_rows = CoordinateRows(
                rows, cols,
                [key for key in DELTA_KEYS if key in observation_space.spaces],
                dim=coordinate_dim)
            total_concat_size += self.coordinate_rows.width
        #: What the factored object heads read besides the rows: each
        #: slot's raw embedding - its colour shares and where its mass sits
        #: in its box, which the rows have been through attention and a
        #: projection to lose - and the grid's own colour shares. See
        #: rl.policy.FactoredObjectDistribution. Last of all, after the
        #: pointer and coordinate tails.
        self.factored_width = 0
        if factored_tail:
            if "objects_emb" not in observation_space.spaces:
                raise ValueError("factored object heads read the objects, and this "
                                 "observation has none")
            slots, object_dim = observation_space.spaces["objects_emb"].shape
            self.factored_width = slots * object_dim + 10
            total_concat_size += self.factored_width
        # print(f'total_concat_size: {total_concat_size}')
        self._features_dim = total_concat_size

    def forward(self, observation) -> torch.Tensor:
        encoded_tensor_list = []
        object_slot = None
        if self.relation_bias is not None:
            # Before the loop, not after it: the object branch runs inside
            # the loop and reads this during its attention. Set afterwards
            # it survives to the *next* observation and biases that one
            # instead - which still moves the features, so a test that only
            # checked "relations change the output" passed.
            self.extractors["objects_emb"].processor.attn_bias = \
                self.relation_bias(observation["relations_emb"])
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(f'observation key {key} has shape: {observation[key].shape}')
            if key in DELTA_KEYS:
                # Cropped back to the real grid for the same reason a grid
                # is: the observation padding is zeros here, and a centre of
                # mass taken over the padded plane would be pulled towards
                # the top-left corner by the region the grid does not
                # occupy.
                res = unpadded_grid_features(extractor, observation[key],
                                             observation.get('grid_shape'),
                                             delta_plane)
            elif key in GRID_KEYS:
                def prepare(grid):
                    x = torch.nn.functional.one_hot(torch.tensor(grid, dtype=torch.int64), num_classes=10)  # Shape: (Batch, H, W, 10)
                    x = x.float()  # Convert to float
                    return x.permute(0, 3, 1, 2)  # Change to (Batch, 10, H, W)
                # Cropped to the real grid first when the observation was
                # padded to a common shape - the pad value is not a colour
                # and one_hot would not know what to do with it.
                res = unpadded_grid_features(extractor, observation[key],
                                             observation.get('grid_shape'), prepare)
            else:
                res = extractor(observation[key].unsqueeze(1))
                # print(f'output for key {key} has shape: {res.shape}')
            if key == "objects_emb":
                object_slot = len(encoded_tensor_list)
            encoded_tensor_list.append(res)
        if self.relation_messages is not None and object_slot is not None:
            encoded_tensor_list[object_slot] = self.pass_messages(observation)
        if self.pointer_slots is not None:
            encoded_tensor_list.append(self.pointer_tail())
        # Last, after the object tail: ARCCustomNetwork cuts the two off the
        # end in that order.
        if self.coordinate_rows is not None:
            encoded_tensor_list.append(self.coordinate_rows.tail(observation))
        if self.factored_width:
            encoded_tensor_list.append(self.factored_tail(observation))
        return torch.cat(encoded_tensor_list, dim=1)

    def factored_tail(self, observation) -> torch.Tensor:
        """The raw object rows, flattened, and the share of the grid each
        of the ten colours covers - counted over the true grid, so the
        observation's padding is not a colour."""
        objects = observation["objects_emb"].float().flatten(1)
        grid = observation["grid"].to(torch.int64)
        valid = grid < 10
        shape = observation.get("grid_shape")
        if shape is not None:
            shape = shape.to(torch.int64)
            rows = torch.arange(grid.shape[1], device=grid.device).view(1, -1, 1)
            cols = torch.arange(grid.shape[2], device=grid.device).view(1, 1, -1)
            valid = valid & (rows < shape[:, :1, None]) & (cols < shape[:, 1:2, None])
        colours = torch.nn.functional.one_hot(grid.clamp(0, 9), num_classes=10).float()
        counts = (colours * valid.unsqueeze(-1).float()).sum(dim=(1, 2))
        shares = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.cat([objects, shares], dim=1)

    def pass_messages(self, observation) -> torch.Tensor:
        """Let each object see its relations, and re-pool what comes out.

        Both halves matter. The rows are written back because the pointer
        head scores them, and that is the only way a relation can reach the
        choice of *which* object to act on. The pooled vector is recomputed
        from the updated rows through the object branch's own aggregation,
        because otherwise the summary handed to the rest of the network
        would still be the one taken before any message arrived - the
        relations would reach the head and nothing else.
        """
        extractor = self.extractors["objects_emb"]
        processor = extractor.processor
        updated = self.relation_messages(processor.per_object,
                                         observation["relations_emb"],
                                         processor.per_object_mask)
        # On the extractor, which is what pointer_tail reads: OptimalObjectExtractor
        # copies per_object out of its processor after the forward, and the
        # copy is the one every later reader sees. Writing to the processor
        # as well looked tidier and was untestable - nothing reads it.
        extractor.per_object = updated

        mask = processor.per_object_mask.unsqueeze(-1).to(updated.dtype)
        aggregated = (updated * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return processor.layer_norm2(processor.aggregation(aggregated))

    def pointer_tail(self) -> torch.Tensor:
        """The per-object rows, flattened, each with a mask column.

        An action names an object slot, and the logit for that slot comes
        from the pooled vector - which ObjectSetProcessor's mean makes
        permutation invariant. Swapping the contents of two slots therefore
        leaves every logit unchanged: measured at 6.6e-07 against a feature
        scale of 0.27, where replacing the objects outright moves the
        features by 0.427. The observation knows which objects are present
        and cannot say which slot holds which, while the action addresses
        exactly that. This tail is the channel between the two - a head
        scoring slot i from row i is equivariant where the pooled vector is
        invariant.

        The mask rides along as a column rather than being recovered from
        the rows. A padded object is an all-zero row in the observation, but
        these rows have been through attention and a LayerNorm and are not
        zero any more, so a head deriving the mask from them would point at
        slots holding nothing.

        Projected to pointer_dim first: ObjectSetProcessor works at
        hidden_dim 128, and 16 slots of that would add 2048 numbers to a
        feature vector whose other branches are 16 to 144 wide.
        """
        processor = self.extractors["objects_emb"]
        rows = self.pointer_projection(processor.per_object)
        mask = processor.per_object_mask.unsqueeze(-1).to(rows.dtype)
        return torch.cat([rows, mask], dim=-1).flatten(1)

class ObjectSetProcessor(nn.Module):
    """Processes variable number of objects using attention mechanism for
    permutation invariance and better object interaction modeling.

    forward() returns the pooled vector, and `per_object` on the way out
    holds the embeddings it was pooled from. The mean below is where the
    policy stops being able to say "this object": an action names a slot,
    and the logit for slot i is produced by ARCCustomNetwork from a vector
    in which slot i's embedding has already been averaged with every other
    one. Nothing connects "the object in slot i is small and red" to "slot
    i scores high", so the only thing an object head can learn is a prior
    over slot numbers - which is a property of the example it trained on,
    not of the rule, and does not survive a grid whose slot 2 holds
    something else.

    That is the same failure the action-whitelist ablation measured from
    the other side: lists built from the training examples helped on them
    and moved the held-out pair by exactly zero, on every task and every
    criterion, because a list of slot indices carries coordinates rather
    than a rule. Keeping the per-object embeddings is what a head scoring
    each slot from its own embedding would need.
    """

    def __init__(self, embedding_dim, hidden_dim=128, num_heads=4,
                 dropout=0.1, self_attention=True, grouped=True,
                 cross_attention=True, use_position=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_self_attention = self_attention

        # Object embedding processor
        self.object_processor = ObjectProcessor(
            hidden_dim=hidden_dim*2, output_dim=hidden_dim, dropout=dropout,
            grouped=grouped, cross_attention=cross_attention,
            use_position=use_position)

        # Self-attention for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        ) if self_attention else None

        # Final aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        #: An additive bias for the attention below, set by the caller for
        #: the next forward and cleared by it. A side channel like
        #: per_object above, and for the same reason: every caller and
        #: stable-baselines3 itself expect one tensor in and one out.
        self.attn_bias = None

    def forward(self, x, mask=None):
        """x: tensor of shape (batch_size, max_objects, 32)
        mask: optional mask for variable number of objects
        """
        batch_size, max_objects, _ = x.shape
        # print(self.object_processor)
        # Process individual objects
        object_embeddings = self.object_processor(x)  # (batch, max_objects, hidden_dim)

        # Self-attention for object interactions
        if mask is not None:
            # A row with no objects at all - an all-background grid - would
            # mask every key, and a softmax over nothing but -inf is NaN,
            # which spreads to the whole batch through the shared layers and
            # comes out as "Expected parameter logits ... found invalid
            # values". So such a row attends over its first slot, and the
            # masked mean below multiplies the result away again: the row's
            # contribution is zeros, as it should be, rather than NaN.
            attends = mask.clone()
            attends[~mask.any(dim=1), 0] = True
            # Convert mask to attention mask format
            attn_mask = attends.unsqueeze(1).expand(-1, max_objects, -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
        else:
            attends, attn_mask = None, None
        # print(f'forward in ObjectSetProcessor: batch_size:{batch_size} max_objects:{max_objects}' )
        # print(f'forward in ObjectSetProcessor: object_embeddings.shape:{object_embeddings.shape}' )
        # print(self.self_attention)
        if self.use_self_attention:
            attended, _ = self.self_attention(
                object_embeddings, object_embeddings, object_embeddings,
                key_padding_mask=~attends if attends is not None else None,
                attn_mask=self.attn_bias,
            )
            # Cleared here rather than by the caller: a bias left over from
            # the previous observation would be added to this one's
            # attention, and with the batch sizes stable-baselines3 uses it
            # would be the right shape to do so silently.
            self.attn_bias = None

            # Residual connection
            object_embeddings = self.layer_norm1(object_embeddings + attended)

        # Aggregate objects (mean pooling with mask consideration)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            masked_embeddings = object_embeddings * mask_expanded
            aggregated = masked_embeddings.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            aggregated = object_embeddings.mean(dim=1)

        # Final processing
        output = self.aggregation(aggregated)
        output = self.layer_norm2(output)

        # Kept, not returned: every caller and stable-baselines3 itself
        # expect one tensor out of a features extractor, so the pooled
        # vector stays the return value and the per-object embeddings ride
        # alongside. Detached from nothing - they are part of the graph, so
        # a head reading them trains the same encoder.
        self.per_object = object_embeddings
        self.per_object_mask = mask

        return output

# Usage example for your extractor
class OptimalObjectExtractor(nn.Module):
    def __init__(self, input_shape, output_dim=None, object_arch=None):
        super().__init__()

        # Assuming input_shape is (max_objects, 32)
        max_objects, feature_dim = input_shape
        self.max_objects = max_objects
        self.feature_dim = feature_dim

        # Calculate reasonable output dimension
        if output_dim is None:
            output_dim = max(64, max_objects * 8)  # Adaptive based on object count

        self.processor = ObjectSetProcessor(
            embedding_dim=output_dim,
            hidden_dim=128,
            num_heads=4,
            **(object_arch or {})
        )

    def forward(self, x):
        """x: flattened object embeddings of shape (batch_size, max_objects * 32)
        """
        batch_size = x.shape[0]

        # Reshape to (batch_size, max_objects, 32)
        x_reshaped = x.view(batch_size, self.max_objects, self.feature_dim)

        # Create mask for valid objects (assuming invalid objects are all zeros)
        mask = (x_reshaped.sum(dim=-1) != 0)  # (batch_size, max_objects)

        pooled = self.processor(x_reshaped, mask)
        self.per_object = self.processor.per_object
        self.per_object_mask = self.processor.per_object_mask
        return pooled

# Integration with your existing code
def create_object_extractor(subspace_shape, object_arch=None):
    """Replace your current objects_emb extractor with this

    `object_arch` is the object branch's own architecture, the way
    `extr_arch` is the grid's: a dict of keyword arguments for
    ObjectSetProcessor - dropout, self_attention, grouped, cross_attention,
    use_position. None is the shipped shape.
    """
    max_objects, feature_dim = subspace_shape
    output_dim = max(64, max_objects * 8)

    return (OptimalObjectExtractor(subspace_shape, output_dim, object_arch),
            output_dim)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
