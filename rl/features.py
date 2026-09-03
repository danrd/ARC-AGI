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

    def __init__(self, observation_space: spaces.Dict, gnn_output_dim=256, grid_cnn_features=128):
        super().__init__(observation_space, features_dim=1)

        self.gnn_output_dim = gnn_output_dim
        self.grid_cnn_features = grid_cnn_features

        # Grid CNN extractor
        self.grid_extractor = nn.Sequential(
                              nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: [batch, 16, 1, 1]
                              nn.Flatten()                   # Output shape: [batch, 16]
                            )

        # Calculate grid CNN output size
        with torch.no_grad():
            sample_grid = torch.randn(1, 1, 30, 30)  # Assuming max 30x30 grid
            grid_output = self.grid_extractor(sample_grid)
            self.grid_cnn_output_size = grid_output.shape[1]

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
        grid_features = self.grid_extractor(observations['grid'].unsqueeze(1))
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

    def __init__(self, object_dim=OBJECT_DIM, hidden_dim=128, output_dim=64):
        super().__init__()

        # Which slots of the object vector feed each head, taken from the
        # schema that defines the vector rather than restated as ranges: the
        # groups a consumer cares about need not sit next to each other, and
        # a hard-coded range silently reads different fields once the schema
        # changes. Registered as buffers so they follow the module's device.
        self.register_buffer("color_index", torch.tensor(group_indices(COLOR), dtype=torch.long))
        self.register_buffer("spatial_index", torch.tensor(group_indices(SIZE, POSITION), dtype=torch.long))
        self.register_buffer("shape_index", torch.tensor(group_indices(TOPOLOGY), dtype=torch.long))

        self.color_dim = len(group_indices(COLOR))
        self.spatial_dim = len(group_indices(SIZE, POSITION))
        self.shape_dim = len(group_indices(TOPOLOGY))

        # Specialized processors for each feature group
        self.color_processor = nn.Sequential(
            nn.Linear(self.color_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        self.spatial_processor = nn.Sequential(
            nn.Linear(self.spatial_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        self.shape_processor = nn.Sequential(
            nn.Linear(self.shape_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )

        # Cross-attention between feature groups
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=16, num_heads=4, dropout=0.1, batch_first=True
        )

        # Feature fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(48, hidden_dim),  # 16 * 3 feature groups
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """x: tensor of shape (batch_size, max_objects, OBJECT_DIM)"""
        batch_size, max_objects, _ = x.shape

        # Split into feature groups
        color_features = x.index_select(-1, self.color_index)
        spatial_features = x.index_select(-1, self.spatial_index)
        shape_features = x.index_select(-1, self.shape_index)

        # Process each group
        color_emb = self.color_processor(color_features)    # (batch, max_objects, 16)
        spatial_emb = self.spatial_processor(spatial_features)  # (batch, max_objects, 16)
        shape_emb = self.shape_processor(shape_features)    # (batch, max_objects, 16)

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

    def __init__(self, observation_space: spaces.Dict,
                 object_output_dim=64, relation_output_dim=64, grid_cnn_features=128):
        super().__init__(observation_space, features_dim=1)

        # Grid CNN (reuse from original)
        self.grid_extractor = nn.Sequential(
                              nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: [batch, 16, 1, 1]
                              nn.Flatten()                   # Output shape: [batch, 16]
                            )

        # Calculate grid CNN output size
        with torch.no_grad():
            sample_grid = torch.randn(1, 1, 30, 30)
            grid_output = self.grid_extractor(sample_grid)
            self.grid_cnn_output_size = grid_output.shape[1]

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
        grid_features = self.grid_extractor(observations['grid'].unsqueeze(1))
        grid_features = self.grid_projection(grid_features)

        # Process objects
        obj_embeddings = self.object_processor(observations['objects_emb'])

        # Aggregate objects with attention-based pooling
        obj_mask = (observations['objects_emb'].sum(dim=-1) != 0)  # Valid object mask
        if obj_mask.any():
            obj_attended, _ = self.object_aggregator(obj_embeddings, obj_embeddings, obj_embeddings,
                                                   key_padding_mask=~obj_mask)
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
class ARCCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, extr_arch=None):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        if extr_arch:
            self.extr_arch =  extr_arch
        else:
            self.extr_arch = nn.Sequential(
                                          nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten()
                                        )
        for key, subspace in observation_space.spaces.items():
            if key == "objects_emb":
                # print(f'objects_emb subspace.shape:{subspace.shape}')
                extractor, output_dim = create_object_extractor(subspace.shape)
                extractors[key] = extractor
                total_concat_size += output_dim
                # print(f'objects_emb_concat_size: {output_dim}')
            elif key == "relations_emb":
                dim = subspace.shape[0] * subspace.shape[1]
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim), nn.ReLU())
                total_concat_size += dim
            elif key == 'grid':
                extractors[key] = self.extr_arch
                total_concat_size += self.extr_arch[2].out_channels
                # print(f'cnn_concat_size: {total_concat_size}')
            elif key == 'action_space':
                # The action space's own .nvec (varies per task) - ARCGridWorld
                # includes it in every observation so the policy can see it,
                # but it's not a feature to embed: forward() below never
                # reads it, same as ARCGNNExtractor/ARCSeparateExtractor.
                continue
            else:
                raise ValueError(f'Unknown feature: {key}')

        self.extractors = nn.ModuleDict(extractors)
        # print(f'total_concat_size: {total_concat_size}')
        self._features_dim = total_concat_size

    def forward(self, observation) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(f'observation key {key} has shape: {observation[key].shape}')
            if key == 'grid':
                x = torch.nn.functional.one_hot(torch.tensor(observation[key], dtype=torch.int64), num_classes=10)  # Shape: (Batch, H, W, 10)
                x = x.float()  # Convert to float
                x = x.permute(0, 3, 1, 2)  # Change to (Batch, 10, H, W)
                res = extractor(x)
            else:
                res = extractor(observation[key].unsqueeze(1))
                # print(f'output for key {key} has shape: {res.shape}')
            encoded_tensor_list.append(res)
        return torch.cat(encoded_tensor_list, dim=1)

class ObjectSetProcessor(nn.Module):
    """Processes variable number of objects using attention mechanism for
    permutation invariance and better object interaction modeling.
    """

    def __init__(self, embedding_dim, hidden_dim=128, num_heads=4):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Object embedding processor
        self.object_processor = ObjectProcessor(hidden_dim=hidden_dim*2, output_dim=hidden_dim)

        # Self-attention for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Final aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

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
            # Convert mask to attention mask format
            attn_mask = mask.unsqueeze(1).expand(-1, max_objects, -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
        else:
            attn_mask = None
        # print(f'forward in ObjectSetProcessor: batch_size:{batch_size} max_objects:{max_objects}' )
        # print(f'forward in ObjectSetProcessor: object_embeddings.shape:{object_embeddings.shape}' )
        # print(self.self_attention)
        attended, _ = self.self_attention(
            object_embeddings, object_embeddings, object_embeddings,
            key_padding_mask=~mask if mask is not None else None
        )

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

        return output

# Usage example for your extractor
class OptimalObjectExtractor(nn.Module):
    def __init__(self, input_shape, output_dim=None):
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
            num_heads=4
        )

    def forward(self, x):
        """x: flattened object embeddings of shape (batch_size, max_objects * 32)
        """
        batch_size = x.shape[0]

        # Reshape to (batch_size, max_objects, 32)
        x_reshaped = x.view(batch_size, self.max_objects, self.feature_dim)

        # Create mask for valid objects (assuming invalid objects are all zeros)
        mask = (x_reshaped.sum(dim=-1) != 0)  # (batch_size, max_objects)

        return self.processor(x_reshaped, mask)

# Integration with your existing code
def create_object_extractor(subspace_shape):
    """Replace your current objects_emb extractor with this
    """
    max_objects, feature_dim = subspace_shape
    output_dim = max(64, max_objects * 8)

    return OptimalObjectExtractor(subspace_shape, output_dim), output_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
