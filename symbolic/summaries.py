import typing
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from copy import copy
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from itertools import product
from scipy.spatial.distance import euclidean
from scipy import stats as st
from rl.arc_task import ARCSubtask
from symbolic.objects_analysis import GridObject, matching_transforms
from symbolic.utils import find_upper_left_corner, coords_transform, count_unique_cells, dict_to_list, check_subset_condition
from symbolic.patterns import generate_patterns, find_connected_components_with_color, find_connected_components_excluding_colors

colors_mapping = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown', 10: 'white'
}
inverse_colors_mapping = {v:k for k, v in colors_mapping.items()}

# ---------------------------------------------------------------------------
# Relation embedding schema - the relation-side counterpart to
# objects_analysis.OBJECT_SCHEMA, and likewise the one place the vector's
# width and layout are defined. _create_embedding fills these slots in
# exactly this order, and rl.features asks for the dimensions and group
# indices instead of restating them.
# ---------------------------------------------------------------------------

SIMILARITY_REL = "similarity"
SHAPE_REL = "shape"
SPATIAL_REL = "spatial"

#: (feature name, group), in vector order.
RELATION_SCHEMA = (
    ('same_color',           SIMILARITY_REL),
    ('same_size',            SIMILARITY_REL),
    ('same_vert_size',       SIMILARITY_REL),
    ('same_hor_size',        SIMILARITY_REL),
    ('shape_similarity',     SHAPE_REL),
    ('match_score',          SHAPE_REL),
    ('translation_symmetry', SHAPE_REL),
    ('horizontal_symmetry',  SHAPE_REL),
    ('vertical_symmetry',    SHAPE_REL),
    ('rotation',             SHAPE_REL),
    ('in_contour',           SPATIAL_REL),
    ('touches',              SPATIAL_REL),
    ('size_ratio',           SIMILARITY_REL),
    ('in_line',              SPATIAL_REL),
    ('in_diagonal',          SPATIAL_REL),
    ('x_aligned_with',       SPATIAL_REL),
    ('y_aligned_with',       SPATIAL_REL),
    ('aligned_top',          SPATIAL_REL),
    ('aligned_bottom',       SPATIAL_REL),
    ('aligned_left',         SPATIAL_REL),
    ('aligned_right',        SPATIAL_REL),
    ('normalized_distance',  SPATIAL_REL),
    ('x_offset',             SPATIAL_REL),
    ('y_offset',             SPATIAL_REL),
)

RELATION_FEATURE_NAMES = tuple(name for name, _group in RELATION_SCHEMA)
RELATION_DIM = len(RELATION_SCHEMA)
RELATION_SCHEMA_VERSION = 3


def relation_group_indices(*groups: str) -> tuple:
    """Vector positions belonging to the named relation groups, in vector
    order - see objects_analysis.group_indices for why indices rather than
    ranges."""
    wanted = set(groups)
    unknown = wanted - {group for _name, group in RELATION_SCHEMA}
    if unknown:
        raise ValueError(f"Unknown relation embedding group(s): {sorted(unknown)}")
    return tuple(i for i, (_name, group) in enumerate(RELATION_SCHEMA) if group in wanted)

# Immutable dataclasses for representation levels
@dataclass(frozen=True)
class ObjectsSummary:
    """Immutable dataclass for objects summary statistics."""
    objects_properties: Dict[str, int] = field(default_factory=dict)
    size2shape: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    shape2size: Dict[str, int] = field(default_factory=dict)
    mean_size: float = 0.0
    mode_size: float = 0.0
    mean_hor_size: float = 0.0
    mean_vert_size: float = 0.0
    hor_size2shape: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    shape2hor_size: Dict[str, int] = field(default_factory=dict)
    vert_size2shape: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    shape2vert_size: Dict[str, int] = field(default_factory=dict)
    shapes: Dict[str, int] = field(default_factory=dict)
    shape_colors: Dict[str, int] = field(default_factory=dict)
    colors: Dict[str, int] = field(default_factory=dict)
    shape_hor_size_description: Dict[int, str] = field(default_factory=dict)
    shape_vert_size_description: Dict[int, str] = field(default_factory=dict)
    shape_size_description: Dict[int, str] = field(default_factory=dict)
    shape_color_description: Dict[str, str] = field(default_factory=dict)
    color_description: Dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RelationStatistics:
    """Immutable dataclass for relation statistics."""
    same_color: int = 0
    same_shape: int = 0
    same_size: int = 0
    in_contour: int = 0
    touches: int = 0
    in_line: int = 0
    x_y_aligned_with: int = 0
    x_aligned_with: int = 0
    y_aligned_with: int = 0
    aligned_top: int = 0
    aligned_bottom: int = 0
    aligned_left: int = 0
    aligned_right: int = 0

@dataclass(frozen=True)
class ObjectTriples:
    """Immutable dataclass for storing triples where this object is the head."""
    label: str
    triples: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class Triples:
    """Immutable dataclass for all object triples in the level."""
    object_triples: Tuple[ObjectTriples, ...] = field(default_factory=tuple)

    def get_triples_for_object(self, obj_label: str) -> Tuple[Tuple[str, str, str], ...]:
        """Get triples where specified object is the head."""
        for obj_triples in self.object_triples:
            if obj_triples.label == obj_label:
                return obj_triples.triples
        return tuple()

    def get_triples_between_objects(self, obj1_label: str, obj2_label: str) -> Tuple[Tuple[str, str, str], ...]:
        """Get triples between two specific objects."""
        result = []
        for obj_triples in self.object_triples:
            if obj_triples.label == obj1_label:
                for triple in obj_triples.triples:
                    if triple[2] == obj2_label:  # tail of triple matches obj2
                        result.append(triple)
        return tuple(result)

@dataclass(frozen=True)
class Cell2Obj:
    """Immutable dataclass for cell to object mapping."""
    cell_mappings: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def get_object_at_cell(self, coord: Tuple[int, int]) -> Optional[int]:
        """Get object index at specified cell coordinate."""
        return self.cell_mappings.get(coord)

@dataclass(frozen=True)
class ObjectDistances:
    """Immutable dataclass for storing distances between objects."""
    distances: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def get_distance(self, obj1_label: str, obj2_label: str) -> float:
        """Get distance between two objects."""
        key1 = (obj1_label, obj2_label)
        key2 = (obj2_label, obj1_label)
        return self.distances.get(key1, self.distances.get(key2, 0.0))

@dataclass(frozen=True)
class RelationEmbeddings:
    """Immutable dataclass for storing relation embeddings."""
    embeddings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_embedding(self, obj1_label: str, obj2_label: str) -> Optional[Any]:
        """Get embedding between two objects."""
        return self.embeddings.get(obj1_label, {}).get(obj2_label)

    def get_embeddings_for_object(self, obj_label: str) -> Dict[str, Any]:
        """Get all embeddings where specified object is the first object."""
        return self.embeddings.get(obj_label, {})

@dataclass(frozen=True)
class RepresentationLevel:
    """Immutable dataclass for a complete representation level."""
    objects: Tuple[Any, ...] = field(default_factory=tuple)
    objects_summary: ObjectsSummary = field(default_factory=ObjectsSummary)
    triples: Triples = field(default_factory=Triples)
    relation_statistics: RelationStatistics = field(default_factory=RelationStatistics)
    cell2obj: Optional[Cell2Obj] = None
    distances: ObjectDistances = field(default_factory=ObjectDistances)
    relation_embeddings: Optional[RelationEmbeddings] = None

@dataclass(frozen=True)
class SubtaskSummary:
    """Immutable dataclass for subtask summary."""
    subtask: ARCSubtask = None
    subtask_label: str = None
    # Annotated as Any rather than GridSummary only because that class is
    # defined further down this module; without an annotation at all these
    # two are plain class attributes instead of fields, and create() below
    # then fails on its own keyword arguments.
    inp_grid_summary: Any = None
    out_grid_summary: Any = None
    grids_x_ratio: float = 1.0
    grids_y_ratio: float = 1.0

    @classmethod
    def create(cls, subtask: ARCSubtask, train: bool = True, levels=(2,)) -> 'SubtaskSummary':
        """Factory method to create SubtaskSummary.

        `levels` has to cover whatever level prepare_features() is asked
        for - GridSummary only builds the representations it is told to.
        """
        levels = list(levels)
        inp_grid_summary = GridSummary(
            grid=subtask.train_inp,
            shape=subtask.train_inp_shape,
            levels=levels,
        )

        if train:
            out_grid_summary = GridSummary(
                grid=subtask.train_out,
                shape=subtask.train_out_shape,
                levels=levels,
            )
            grids_x_ratio = subtask.train_inp_shape[0] / subtask.train_out_shape[0]
            grids_y_ratio = subtask.train_inp_shape[1] / subtask.train_out_shape[1]
        else:
            out_grid_summary = None
            grids_x_ratio = 1.0
            grids_y_ratio = 1.0

        return cls(
            subtask=subtask,
            subtask_label=subtask.label,
            inp_grid_summary=inp_grid_summary,
            out_grid_summary=out_grid_summary,
            grids_x_ratio=grids_x_ratio,
            grids_y_ratio=grids_y_ratio
        )

    def prepare_features(self, level: int = 2) -> Dict[str, float]:
        """Prepare features by comparing input and output grid summaries."""
        if self.out_grid_summary is None:
            raise ValueError("Cannot prepare features without output grid summary")
        if level not in self.inp_grid_summary.repr_levels:
            raise ValueError(
                f"Representation level {level} was not built - pass levels=[{level}] to create()"
            )

        inp_level_2 = self.inp_grid_summary.repr_levels[level]
        out_level_2 = self.out_grid_summary.repr_levels[level]

        # Calculate x_change_ratio and y_change_ratio
        inp_shape = self.inp_grid_summary.shape
        out_shape = self.out_grid_summary.shape
        x_change_ratio = out_shape[1] / inp_shape[1] if inp_shape[1] > 0 else 1.0
        y_change_ratio = out_shape[0] / inp_shape[0] if inp_shape[0] > 0 else 1.0

        # Calculate share of non-font cells using color statistics
        def calculate_share_non_font(grid_summary, level=level):
            total_cells = grid_summary.shape[0] * grid_summary.shape[1]
            if total_cells == 0:
                return 0

            # Sum all shape colors (these are non-font by definition)
            total_non_font = sum(grid_summary.repr_levels[level].objects_summary.shape_colors.values())
            return total_non_font / total_cells

        share_non_font_inp = calculate_share_non_font(self.inp_grid_summary)
        share_non_font_out = calculate_share_non_font(self.out_grid_summary)

        # Calculate number of objects
        total_objects_inp = len(inp_level_2.objects)
        total_objects_out = len(out_level_2.objects)

        unique_elements_inp = np.unique(self.inp_grid_summary.grid)
        unique_elements_out = np.unique(self.out_grid_summary.grid)

        # Calculate mean object compactness
        def calculate_mean_compactness(level):
            """Calculate mean compactness of objects in the level."""
            if not level.objects:
                return 0.0
            total_compactness = 0.0
            compactness_count = 0
            for obj in level.objects:
                if hasattr(obj, 'compactness'):
                    total_compactness += obj.compactness
                    compactness_count += 1
            return total_compactness / compactness_count if compactness_count > 0 else 0.0

        mean_compactness_inp = calculate_mean_compactness(inp_level_2)
        mean_compactness_out = calculate_mean_compactness(out_level_2)

        # Calculate mean distance between objects
        def calculate_mean_distance(level):
            """Calculate mean distance between all object pairs."""
            if len(level.objects) < 2:
                return 0.0

            total_distance = 0.0
            count = 0

            # Get object labels or indices
            object_labels = []
            if hasattr(level.objects[0], 'label'):
                object_labels = [obj.label for obj in level.objects]
            else:
                # Use indices if no label
                object_labels = list(range(len(level.objects)))

            # Calculate distances for all unique pairs
            for i in range(len(object_labels)):
                for j in range(i + 1, len(object_labels)):
                    distance = level.distances.get_distance(
                        str(object_labels[i]),
                        str(object_labels[j])
                    )
                    total_distance += distance
                    count += 1

            return total_distance / count if count > 0 else 0.0

        mean_distance_inp = calculate_mean_distance(inp_level_2)
        mean_distance_out = calculate_mean_distance(out_level_2)

        # Extract objects summary and relation statistics
        obj_summary_1 = inp_level_2.objects_summary
        rel_summary_1 = inp_level_2.relation_statistics
        obj_summary_2 = out_level_2.objects_summary
        rel_summary_2 = out_level_2.relation_statistics

        # Create input feature dict
        inp_features = {}
        inp_features.update(self._extract_object_features(obj_summary_1))
        inp_features.update(self._extract_relation_features(rel_summary_1))

        # Create output feature dict
        out_features = {}
        out_features.update(self._extract_object_features(obj_summary_2))
        out_features.update(self._extract_relation_features(rel_summary_2))

        # Calculate differences
        features = {}
        for k in inp_features.keys():
            if k in out_features:
                value = out_features[k] - inp_features[k]
                if isinstance(value, float):
                    features[k] = round(value, 3)  # округление разностей
                else:
                    features[k] = value

        # Add ratio features
        features['grids_x_ratio'] = round(self.grids_x_ratio, 3)
        features['grids_y_ratio'] = round(self.grids_y_ratio, 3)
        features['x_change_ratio'] = round(x_change_ratio, 3)
        features['y_change_ratio'] = round(y_change_ratio, 3)
        features['share_non_font_diff'] = round(share_non_font_out - share_non_font_inp, 3)
        features['total_objects_diff'] = int(total_objects_out - total_objects_inp)
        features['unique_elements_diff'] = max(len(set(unique_elements_out).difference(unique_elements_inp)), len(set(unique_elements_inp).difference(unique_elements_out)))
        features['mean_compactness_diff'] = round(mean_compactness_out - mean_compactness_inp, 3)
        features['mean_distance_diff'] = round(mean_distance_out - mean_distance_inp, 3)

        return features


    def _extract_object_features(self, obj_summary: ObjectsSummary) -> Dict[str, float]:
        """Extract numerical features from ObjectsSummary."""
        features = {}

        # Basic statistics
        # float() rather than bare round(): these come off numpy reductions,
        # and a numpy scalar survives round() unchanged - carrying a type
        # that will not serialise into a checkpoint or a log.
        features['mean_size'] = round(float(obj_summary.mean_size), 3)
        features['mode_size'] = round(float(obj_summary.mode_size), 3)
        features['mean_hor_size'] = round(float(obj_summary.mean_hor_size), 3)
        features['mean_vert_size'] = round(float(obj_summary.mean_vert_size), 3)

        # Shape counts
        # for shape, count in obj_summary.shapes.items():
        #     features[f'shape_{shape}_count'] = float(count)

        # Color counts from shape_colors
        for color, count in obj_summary.shape_colors.items():
            if color != 'multicolor':  # Handle multicolor separately if needed
                features[f'color_{color}_count'] = int(count)

        # Additional color counts
        for color, count in obj_summary.colors.items():
            features[f'total_color_{color}_count'] = int(count)

        # Add objects_properties features
        if hasattr(obj_summary, 'objects_properties'):
            for prop, value in obj_summary.objects_properties.items():
                features[f'{prop}'] = int(value)

        return features

    def _extract_relation_features(self, rel_summary: RelationStatistics) -> Dict[str, float]:
        """Extract numerical features from RelationStatistics."""
        return {
            'same_color': int(rel_summary.same_color),
            'same_shape': int(rel_summary.same_shape),
            'same_size': int(rel_summary.same_size),
            'in_contour': int(rel_summary.in_contour),
            'in_line': int(rel_summary.in_line),
            'x_y_aligned_with': int(rel_summary.x_y_aligned_with),
            'x_aligned_with': int(rel_summary.x_aligned_with),
            'y_aligned_with': int(rel_summary.y_aligned_with)
        }


class GridSummary():
    """Class for creating summary for a given grid."""
    def __init__(self, grid:np.array, shape:tuple, font_color:float=0, levels:List[int]=[1], shape_types=None):
        self.grid = grid
        self.shape = shape
        self.shape_types = ('line' ,'rectangle', 'diagonal', 'l_shape', 't_shape', 's_shape', 'tv_shape',
                            'hs_shape', 'cross', 'flower', 'markup', 'partition_lines', 'cell', 'complex') if shape_types is None else shape_types
        self.font_color = font_color
        self.grid_corners = self.define_grid_corners()
        self.font_segments = find_connected_components_with_color(self.grid, target_color=self.font_color)
        self.relations_for_stats = ("same_color", "same_shape", "same_size", "rotation", "horizontal_symmetry", "vertical_symmetry",
                                    "in_contour", "touches",
                                    "in_line", "x_y_aligned_with", "x_aligned_with", "y_aligned_with",
                                    "aligned_top", "aligned_bottom", "aligned_left", "aligned_right")
        self.levels = levels
        self.repr_levels = self.set_repr_levels()

    def define_grid_corners(self):
        grid_size = self.shape
        ul = find_upper_left_corner(grid_size)
        bl = (ul[0]+grid_size[0]-1, ul[1])
        ur = (ul[0], ul[1]+grid_size[1]-1)
        br = (ul[0]+grid_size[0]-1, ul[1]+grid_size[1]-1)
        return (ul, bl, ur, br)

    def set_repr_levels(self) -> Dict[int, RepresentationLevel]:
        """Create representation levels for grid objects based on their properties."""
        repr_levels = {}
        for level in self.levels:
            if level == 5:
                repr_levels[level] = self.process_cell_level()
            else:
                if level == 1:
                    objects = self.retrieve_connected_components_hetero(self.grid)
                elif level == 2:
                    objects = self.retrieve_connected_components_homo(self.grid)
                elif level in [3, 4]:
                    objects =  self.retrieve_shapes(self.shape_types)
                repr_levels[level] = self.process_repr_level(objects, level)
        return repr_levels

    def process_repr_level(self, init_objects, level) -> RepresentationLevel:
        """Process a representation level and return immutable RepresentationLevel."""
        level_objects = self.filter_objects(init_objects, level)
        level_objects_tuple = tuple(dict_to_list(level_objects))
        level_objects_summary = self.create_objects_summary(level_objects)
        level_triples, level_relation_statistics, distances = self.set_relations(level_objects)
        cell2obj = self.grid_markup(list(level_objects_tuple))
        relation_embeddings = self._create_relation_embeddings_for_objects(level_objects_tuple, level_triples, distances)
        return RepresentationLevel(
            objects=level_objects_tuple,
            objects_summary=level_objects_summary,
            triples=level_triples,
            relation_statistics=level_relation_statistics,
            cell2obj=cell2obj,
            distances=distances,
            relation_embeddings=relation_embeddings
        )

    def process_cell_level(self) -> RepresentationLevel:
        """Process cell level representation (level 5)."""
        cell_objects = defaultdict(list)
        ul = find_upper_left_corner(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                grid_i = ul[0] + i
                grid_j = ul[1] + j
                cell_color = self.grid[grid_i, grid_j]
                if cell_color != 0 and cell_color != self.font_color:
                    label = f'cell_{i}_{j}'
                    cell = (grid_i, grid_j)
                    obj = GridObject('cell', [cell], [cell_color], label, self.shape, self.font_color, self.grid)
                    cell_objects['cell'].append(obj)
        level_objects_tuple = tuple(dict_to_list(cell_objects))
        level_objects_summary = self.create_objects_summary(cell_objects)
        level_triples, level_relation_statistics, distances = self.set_relations(cell_objects)
        relation_embeddings = self._create_relation_embeddings_for_objects(level_objects_tuple, level_triples, distances)
        return RepresentationLevel(
            objects=level_objects_tuple,
            objects_summary=level_objects_summary,
            triples=level_triples,
            relation_statistics=level_relation_statistics,
            distances=distances,
            relation_embeddings=relation_embeddings
        )

    def retrieve_shapes(self, shape_types:tuple)->typing.Dict[str, List[GridObject]]:
        """Retrieve all possible objects from the grid and return corresponding GridObject instances."""
        patterns = generate_patterns(self.shape, shape_types)
        objects = defaultdict(list)
        candidate = False
        used_coordinates = []
        grid = copy(self.grid)
        for k, v in patterns.items():
            shape_patterns = v
            for idx, pattern_list in enumerate(shape_patterns):
                for pattern in pattern_list:
                    i, j = coords_transform(pattern)
                    retrieval = set(grid[i, j])
                    if len(retrieval) > 1:
                        break
                    else:
                        color = retrieval.pop()
                        if color != self.font_color and (k!='diagonal' or count_unique_cells(k, pattern, used_coordinates)) > 0:
                            label = f'{k}_{idx}'
                            obj = GridObject(k, pattern, [color], label, self.shape, self.font_color, grid)
                            used_coordinates.extend(pattern)
                            candidate = True
                if candidate:
                    objects[k].append(copy(obj))
                    candidate = False
        used_coordinates = set(used_coordinates)
        ul = find_upper_left_corner(self.shape)
        all_coordinates = set(product(range(ul[0], ul[0]+self.shape[0]), range(ul[1], ul[1]+self.shape[1])))
        cells_coordinates = list(all_coordinates.difference(used_coordinates))
        for idx, cell in enumerate(cells_coordinates):
            color = grid[cell]
            if color != self.font_color:
                label = f'cell_{idx}'
                obj = GridObject('cell', [cell], [grid[cell]], label, self.shape, self.font_color, grid)
                objects['cell'].append(obj)
        return objects

    def retrieve_connected_components_hetero(self, grid:np.array):
        """Retrieve all hetero colored connected components and return corresponding GridObject instances."""
        components = {}
        objects = defaultdict(list)
        comp_idx = 0
        heter_components = find_connected_components_excluding_colors(grid, pad_val=10, font_color=self.font_color)
        for comp in heter_components:
            color_numbers = list(set([grid[coord] for coord in comp]))
            label = f'complex_{comp_idx}'
            obj = GridObject('complex', comp, color_numbers, label, self.shape, self.font_color, grid)
            components[label] = obj
            comp_idx += 1
        objects['complex'] = list(components.values())
        return objects

    def retrieve_connected_components_homo(self, grid:np.array):
        """Retrieve all homo colored connected components and return corresponding GridObject instances."""
        components = {}
        objects = defaultdict(list)
        comp_idx = 0
        colors = [x for x in range(10) if x != self.font_color]
        for i in colors:
            col = i
            homo_components = find_connected_components_with_color(grid, target_color=col)
            for comp in homo_components:
                label = f'complex_{comp_idx}'
                obj = GridObject('complex', comp, [col], label, self.shape, self.font_color, grid)
                components[label] = obj
                comp_idx += 1
        objects['complex'] = list(components.values())
        return objects

    def grid_markup(self, level_objects: List[GridObject]) -> Cell2Obj:
        """Create cell to object mapping."""
        cell_mappings = {}
        for idx, obj in enumerate(level_objects):
            for coord in obj.coords:
                cell_mappings[coord] = idx
        return Cell2Obj(cell_mappings=cell_mappings)

    def _calculate_hu_moments_similarity(self, obj1, obj2):
        """Calculate shape similarity based on Hu moments."""
        if not (hasattr(obj1, 'hu_moments') and hasattr(obj2, 'hu_moments') and
                obj1.hu_moments is not None and obj2.hu_moments is not None):
            return 0.0

        # Calculate Euclidean distance between Hu moments
        distance = euclidean(obj1.hu_moments, obj2.hu_moments)
        # Convert to similarity (0 = identical, higher values = more different)
        # Use exponential decay to convert distance to similarity [0,1]
        similarity = np.exp(-distance)
        return similarity

    @staticmethod
    def filter_objects(objects, repr_level:int=1):
        """Apply ObjectsFilter class for filtering out possibly unimportant objects."""
        return ObjectsFilter(objects, repr_level).filter_objects()

    @staticmethod
    def calculate_distance(obj1, obj2):
        """Chebyshev distance between the objects' bounding boxes.

        Per axis the separation is zero whenever the two ranges overlap,
        and the two axes combine by max rather than min: on an 8-connected
        grid that is the number of moves between them, so a diagonal
        neighbour is one step away like an orthogonal one. Overlapping or
        interleaved boxes give 0, touching gives 1.

        Taking the min instead would report zero for every pair that merely
        shares a row or column, however far apart they sit - which is what
        this measure exists to distinguish.
        """
        gap_i = max(0, max(obj1.min_i, obj2.min_i) - min(obj1.max_i, obj2.max_i))
        gap_j = max(0, max(obj1.min_j, obj2.min_j) - min(obj1.max_j, obj2.max_j))
        return max(gap_i, gap_j)

    def create_objects_summary(self, objects) -> ObjectsSummary:
        """Create a summary for grid objects to get aggregate information about their shapes, sizes, colors."""
        edges_positioning = set(['at_top_edge', 'at_left_edge', 'at_bottom_edge', 'at_right_edge'])
        size2shape = defaultdict(list)
        shape2size = {}
        hor_size2shape = defaultdict(list)
        shape2hor_size = {}
        vert_size2shape = defaultdict(list)
        shape2vert_size = {}
        shapes = {shape:0 for shape in self.shape_types}
        objects_properties = {'at_edge':0, 'has_holes':0, 'symmetric':0, 'total_holes':0}
        # Initialize shape_colors with multicolor support
        shape_colors = {colors_mapping[i]:0 for i in range(10)}
        shape_colors['multicolor'] = 0
        colors = {colors_mapping[i]:0 for i in range(10)}

        object_hor_sizes = []
        object_vert_sizes = []

        for k, v in objects.items():
            for idx, obj in enumerate(v):
                object_hor_sizes.append(obj.hor_size)
                object_vert_sizes.append(obj.vert_size)

                size2shape[obj.size].append(obj)
                hor_size2shape[obj.hor_size].append(obj)
                vert_size2shape[obj.vert_size].append(obj)
                shape2size[obj.label] = obj.size
                shape2hor_size[obj.label] = obj.hor_size
                shape2vert_size[obj.label] = obj.vert_size
                shapes[obj.shape] += 1

                if len(set(obj.positioning).intersection(edges_positioning)) > 0:
                    objects_properties['at_edge'] += 1
                if obj.symmetry != 'assymetry':
                    objects_properties['symmetric'] += 1

                has_outer_holes = hasattr(obj, 'outer_holes') and len(obj.outer_holes) > 0
                has_inner_holes = hasattr(obj, 'inner_holes') and len(obj.inner_holes) > 0
                if has_outer_holes or has_inner_holes:
                    objects_properties['has_holes'] += 1
                if has_outer_holes:
                    objects_properties['total_holes'] += len(obj.outer_holes)
                if has_inner_holes:
                    objects_properties['total_holes'] += len(obj.inner_holes)

                if obj.shape != 'complex':
                    # For simple shapes, use the main color
                    if obj.colors and len(obj.colors) > 0:
                        main_color = obj.colors[0]
                        shape_colors[main_color] += 1
                        # Count all cells of this object for the color
                        colors[main_color] += len(obj.coords)
                else:
                    # For complex shapes
                    if len(obj.colors) > 1:
                        shape_colors['multicolor'] += 1
                        # Count each color in multicolor object
                        color_counts = {}
                        for coord in obj.coords:
                            color_val = self.grid[coord]
                            color_name = colors_mapping[color_val]
                            color_counts[color_name] = color_counts.get(color_name, 0) + 1

                        # Add to colors dictionary
                        for color_name, count in color_counts.items():
                            colors[color_name] += count
                    else:
                        # Single color complex object
                        if obj.colors and len(obj.colors) > 0:
                            main_color = obj.colors[0]
                            shape_colors[main_color] += 1
                            colors[main_color] += len(obj.coords)


        # Size sorting and descriptions
        sorted_keys = sorted(list(size2shape.keys()), reverse=True)
        size2shape = {k:size2shape[k] for k in sorted_keys}
        shape2size_values = list(shape2size.values())
        n_sizes = len(shape2size_values)
        size2description = {shape2size_values[i]:f'{i+1} by size' for i in range(n_sizes)}

        # Horizontal size sorting and descriptions
        sorted_keys = sorted(list(hor_size2shape.keys()), reverse=True)
        hor_size2shape = {k:hor_size2shape[k] for k in sorted_keys}
        shape2hor_size_values = list(shape2hor_size.values())
        n_sizes = len(shape2hor_size_values)
        hor_size2description = {shape2hor_size_values[i]:f'{i+1} by horizontal size' for i in range(n_sizes)}

        # Vertical size sorting and descriptions
        sorted_keys = sorted(list(vert_size2shape.keys()), reverse=True)
        vert_size2shape = {k:vert_size2shape[k] for k in sorted_keys}
        shape2vert_size_values = list(shape2vert_size.values())
        n_sizes = len(shape2vert_size_values)
        vert_size2description = {shape2vert_size_values[i]:f'{i+1} by vertical size' for i in range(n_sizes)}

        # Color frequency descriptions
        shape_colors2freq_values = sorted([v for v in shape_colors.values() if v > 0], reverse=True)
        shape_colors_reversed = {v:k for k, v in shape_colors.items() if v > 0}
        shape_color2description = {shape_colors_reversed[shape_colors2freq_values[i]]:f'{i+1} by shape color freq' for i in range(len(shape_colors2freq_values))}

        color2freq_values = sorted([v for v in colors.values() if v > 0], reverse=True)
        colors_reversed = {v:k for k, v in colors.items() if v > 0}
        color2description = {colors_reversed[color2freq_values[i]]:f'{i+1} by color freq' for i in range(len(color2freq_values))}

        # Size statistics
        mean_size = np.mean(shape2size_values) if len(shape2size_values) > 0 else 0
        mode_size = st.mode(shape2size_values).mode if len(shape2size_values) > 0 else 0
        mean_hor_size = np.mean(object_hor_sizes) if len(object_hor_sizes) > 0 else 0
        mean_vert_size = np.mean(object_vert_sizes) if len(object_vert_sizes) > 0 else 0

        return ObjectsSummary(objects_properties=objects_properties, size2shape=dict(size2shape), shape2size=shape2size, mean_size=mean_size,
            mode_size=mode_size, mean_hor_size=mean_hor_size, mean_vert_size=mean_vert_size, hor_size2shape=dict(hor_size2shape),
            shape2hor_size=shape2hor_size, vert_size2shape=dict(vert_size2shape), shape2vert_size=shape2vert_size, shapes=shapes,
            shape_colors=shape_colors, colors=colors, shape_hor_size_description=hor_size2description, shape_vert_size_description=vert_size2description,
            shape_size_description=size2description, shape_color_description=shape_color2description, color_description=color2description
        )

    def set_relations(self, objects) -> Tuple[Triples, RelationStatistics, ObjectDistances]:
        """Iterate over objects to identify relations between them."""
        all_object_triples = []
        # Seed every known relation at zero so the statistics always carry
        # the full vocabulary. Counting the name tuple itself would instead
        # start each relation at one, and the offset then travels into the
        # summary text and the feature vector as if it were an observation.
        relation_statistics = Counter({relation: 0 for relation in self.relations_for_stats})
        distances_dict = {}
        all_objects = dict_to_list(objects)

        # Create mapping from label to triples for each object
        object_to_triples = defaultdict(list)

        for idx, obj1 in enumerate(all_objects):
            for obj2 in all_objects[idx+1:]:
                analyzer = RelationAnalyzer(obj1, obj2, self.shape)
                triples = analyzer.triples
                relation_counter = analyzer.relation_counter

                # Add triples for both objects
                for triple in triples[0]:  # obj1 as head
                    object_to_triples[obj1.label].append(triple)
                for triple in triples[1]:  # obj2 as head
                    object_to_triples[obj2.label].append(triple)

                relation_statistics.update(relation_counter)

                # Distance calculation
                distance = self.calculate_distance(obj1, obj2)
                distances_dict[(obj1.label, obj2.label)] = distance
                distances_dict[(obj2.label, obj1.label)] = distance

        # Create ObjectTriples for each object
        for obj in all_objects:
            obj_triples = ObjectTriples(
                label=obj.label,
                triples=tuple(object_to_triples[obj.label])
            )
            all_object_triples.append(obj_triples)

        # Create immutable data structures
        triples = Triples(object_triples=tuple(all_object_triples))
        relation_stats = RelationStatistics(
            same_color=relation_statistics.get('same_color', 0),
            same_shape=relation_statistics.get('same_shape', 0),
            same_size=relation_statistics.get('same_size', 0),
            in_contour=relation_statistics.get('in_contour', 0),
            touches=relation_statistics.get('touches', 0),
            in_line=relation_statistics.get('in_line', 0),
            x_y_aligned_with=relation_statistics.get('x_y_aligned_with', 0),
            x_aligned_with=relation_statistics.get('x_aligned_with', 0),
            y_aligned_with=relation_statistics.get('y_aligned_with', 0),
            aligned_top=relation_statistics.get('aligned_top', 0),
            aligned_bottom=relation_statistics.get('aligned_bottom', 0),
            aligned_left=relation_statistics.get('aligned_left', 0),
            aligned_right=relation_statistics.get('aligned_right', 0),
        )
        distances = ObjectDistances(distances=distances_dict)

        return triples, relation_stats, distances

    def _create_relation_embeddings_for_objects(self, objects_tuple: Tuple, triples: Triples, distances: ObjectDistances) -> RelationEmbeddings:
        """Create relation embeddings for a set of objects."""
        if len(objects_tuple) <= 1:
            return RelationEmbeddings(embeddings={})

        embeddings_dict = {}
        grid_size = max(self.shape)

        # Pre-compute relation flags for faster lookup
        relation_lookup = defaultdict(lambda: defaultdict(set))
        for obj_triples in triples.object_triples:
            for triple in obj_triples.triples:
                head, relation, tail = triple
                relation_lookup[head][tail].add(relation)

        for i, obj1 in enumerate(objects_tuple):
            embeddings_dict[obj1.label] = {}

            for j, obj2 in enumerate(objects_tuple):
                if i == j:
                    continue

                # Create embedding efficiently
                embedding = self._create_embedding(
                    obj1, obj2, relation_lookup, distances, grid_size, objects_tuple
                )
                embeddings_dict[obj1.label][obj2.label] = embedding

        return RelationEmbeddings(embeddings=embeddings_dict)

    def _create_embedding(self, obj1, obj2, relation_lookup, distances, grid_size, objects_tuple):
        """Relation embedding creation.

        Values are computed by name and then laid out in RELATION_SCHEMA's
        order, rather than written into a running index. The positional
        version only worked while this function and the schema were edited
        together in lockstep: inserting one feature into the middle of the
        schema silently shifted every feature after it, and the vector kept
        the same width, so nothing failed - the fields just quietly stopped
        meaning what they said. Building it this way makes the schema the
        single source of truth and turns a mismatch into a KeyError here.
        """
        obj_relations = relation_lookup[obj1.label][obj2.label]
        distance = distances.get_distance(obj1.label, obj2.label)
        has_centers = hasattr(obj1, 'center') and hasattr(obj2, 'center')

        values = {
            'same_color': 1.0 if obj1.colors == obj2.colors else 0.0,
            'same_size': 1.0 if obj1.size == obj2.size else 0.0,
            'same_vert_size': 1.0 if obj1.vert_size == obj2.vert_size else 0.0,
            'same_hor_size': 1.0 if obj1.hor_size == obj2.hor_size else 0.0,
            'size_ratio': RelationAnalyzer.size_ratio(obj1, obj2),
            'shape_similarity': calculate_shape_similarity(obj1, obj2),
            'match_score': calculate_match_score(self.grid, obj1, obj2, objects_tuple, self.font_color),
            'normalized_distance': min(distance / grid_size, 1.0) if grid_size > 0 else 0.0,
            'x_offset': ((obj2.center[1] - obj1.center[1]) / grid_size
                         if has_centers and grid_size > 0 else 0.0),
            'y_offset': ((obj2.center[0] - obj1.center[0]) / grid_size
                         if has_centers and grid_size > 0 else 0.0),
        }
        # Everything else is a flag already decided while the triples were
        # built - read it back rather than recomputing it here, so the
        # vector and the triples can never disagree about the same pair.
        for relation in ('translation_symmetry', 'horizontal_symmetry', 'vertical_symmetry',
                         'rotation', 'in_contour', 'touches', 'in_line', 'in_diagonal',
                         'x_aligned_with', 'y_aligned_with',
                         'aligned_top', 'aligned_bottom', 'aligned_left', 'aligned_right'):
            values[relation] = 1.0 if relation in obj_relations else 0.0

        return np.array([values[name] for name in RELATION_FEATURE_NAMES], dtype=np.float32)

    def _recreate_level_with_embeddings(self, level: RepresentationLevel) -> RepresentationLevel:
        """Recreate a level with relation embeddings."""
        relation_embeddings = self._create_relation_embeddings_for_objects(
            level.objects, level.triples, level.distances
        )

        return RepresentationLevel(
            objects=level.objects,
            objects_summary=level.objects_summary,
            triples=level.triples,
            relation_statistics=level.relation_statistics,
            cell2obj=level.cell2obj,
            distances=level.distances,
            relation_embeddings=relation_embeddings
        )

    def update_representation_level(self, level: int, changed_object) -> RepresentationLevel:
        """Create a new representation level with updated relations for a changed object."""
        current_level = self.repr_levels[level]
        all_objects = list(current_level.objects)

        # Find and replace the changed object
        object_found = False
        for i, obj in enumerate(all_objects):
            if obj.label == changed_object.label:
                all_objects[i] = changed_object
                object_found = True
                break

        if not object_found:
            all_objects.append(changed_object)

        # Recreate objects dict for processing
        objects_dict = defaultdict(list)
        for obj in all_objects:
            objects_dict[obj.shape].append(obj)

        # Recreate all components with embeddings
        objects_tuple = tuple(all_objects)
        objects_summary = self.create_objects_summary(objects_dict)
        triples, relation_statistics, distances = self.set_relations(objects_dict)
        cell2obj = self.grid_markup(all_objects)
        relation_embeddings = self._create_relation_embeddings_for_objects(objects_tuple, triples, distances)

        new_level = RepresentationLevel(
            objects=objects_tuple,
            objects_summary=objects_summary,
            triples=triples,
            relation_statistics=relation_statistics,
            cell2obj=cell2obj,
            distances=distances,
            relation_embeddings=relation_embeddings
        )

        # Update the representation level in the class
        self.repr_levels[level] = new_level
        return new_level

    def get_relation_embeddings_as_numpy(self, level=1):
        """Return all relation embeddings for the specified level in numpy array format."""
        current_level = self.repr_levels[level]

        if current_level.relation_embeddings is None:
            # Recreate level with embeddings
            self.repr_levels[level] = self._recreate_level_with_embeddings(current_level)
            current_level = self.repr_levels[level]

        relation_embeddings = current_level.relation_embeddings.embeddings
        all_objects = current_level.objects
        n_objects = len(all_objects)

        if n_objects <= 1:
            return np.array([])

        # Get embedding dimensions from first valid embedding
        sample_length = 0
        for obj1_label, embeddings_dict in relation_embeddings.items():
            for obj2_label, embedding in embeddings_dict.items():
                if isinstance(embedding, np.ndarray) and len(embedding) > 0:
                    sample_length = len(embedding)
                    break
            if sample_length > 0:
                break

        if sample_length == 0:
            return np.array([])

        # Pre-allocate result array
        result = np.zeros((n_objects, (n_objects-1) * sample_length), dtype=np.float32)

        for i, obj in enumerate(all_objects):
            obj_label = obj.label
            col_idx = 0

            for j, other_obj in enumerate(all_objects):
                if i == j:
                    continue

                other_label = other_obj.label
                embedding = relation_embeddings.get(obj_label, {}).get(other_label)

                if isinstance(embedding, np.ndarray) and len(embedding) == sample_length:
                    result[i, col_idx:col_idx+sample_length] = embedding

                col_idx += sample_length

        return result

class ObjectsFilter():
    """Class for filtering out potentialy unimportant objects."""
    def __init__(self, objects:typing.Dict[str, List[GridObject]], repr_level:int):
        self.objects = objects
        self.repr_level = repr_level

    @staticmethod
    def merge_shapes(objects: typing.Dict[str, GridObject],
                    container_shapes: typing.List[str],
                    contained_shapes: typing.List[str]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some container shape.

        Args:
            objects: Dictionary of shape objects
            container_shapes: List of shape types that can contain other shapes
            contained_shapes: List of shape types that can be contained by container shapes
        """
        deletion_list = []
        filtered_objects = defaultdict(list) | {k: v for k, v in objects.items() if k not in contained_shapes}
        other_objects = {k: v for k, v in objects.items() if k in contained_shapes}

        # Iterate over each container shape
        for container_type in container_shapes:
            if container_type not in filtered_objects:
                continue

            for container_shape in filtered_objects[container_type]:
                container_set = set(container_shape.coords)

                # Iterate over each contained shape type
                for shape_type, shapes in other_objects.items():
                    for idx, shape in enumerate(shapes):
                        if check_subset_condition(container_set, shape.coords):
                            deletion_list.append((shape_type, idx))
                            # Add smaller object to parts list of container shape
                            container_shape.sub_objects[shape.shape].append(shape)

        deletion_list = list(set(deletion_list))

        # Keep only objects not from deletion_list
        for shape_type, shapes in other_objects.items():
            for idx, shape in enumerate(shapes):
                if (shape_type, idx) not in deletion_list:
                    filtered_objects[shape_type].append(shape)

        return filtered_objects

    @staticmethod
    def merge_rectangles(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out smaller rectangles each of which is subset of some larger rectangle."""
        rects = defaultdict(list)
        deletion_list = []
        for obj in objects['rectangle']: # sort rectangles based on size
            key = obj.size
            rects[key].append((obj, set(obj.coords)))
        keys = list(rects.keys()) # iterate over each possible size from larger to smaller
        keys.sort(reverse=True)
        for key_larger in keys[:-1]:
            for key_smaller in keys[1:]:
                for obj_larger in rects[key_larger]:
                    for idx, obj_smaller in enumerate(rects[key_smaller]):
                        if obj_smaller[0].size != obj_larger[0].size and obj_smaller[1].issubset(obj_larger[1]): # if smaller rectangle in larger rectangle - delete smaller
                            deletion_list.append((key_smaller, idx)) # save position of smaller rectangle
                            # Add smaller object to parts list of larger object
                            obj_larger[0].sub_objects[obj_smaller[0].shape].append(obj_smaller[0])
        deletion_list = list(set(deletion_list))
        sorted_rects = [] # save rectangles that are not from deletion_list
        used_shapes = [] # save coordinates to exclude rectangles with the same coordinates
        for k, v in rects.items():
            for idx, rect in enumerate(v):
                if (k, idx) not in deletion_list and sorted(rect[0].coords) not in used_shapes:
                    sorted_rects.append(rect[0])
                    used_shapes.append(sorted(rect[0].coords))
        objects['rectangle'] = sorted_rects
        return objects

    @staticmethod
    def merge_lines(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out smaller lines each of which is subset of some larger line."""
        lines = defaultdict(list)
        deletion_list = []
        for obj in objects['line']: # sort lines based on size
            key = obj.size
            lines[key].append((obj, set(obj.coords)))
        keys = list(lines.keys())# iterate over each possible size from larger to smaller
        keys.sort(reverse=True)
        for key_larger in keys[:-1]:
            for key_smaller in keys[1:]:
                for obj_larger in lines[key_larger]:
                    for idx, obj_smaller in enumerate(lines[key_smaller]):
                        if obj_smaller[0].size != obj_larger[0].size and obj_smaller[1].issubset(obj_larger[1]): # if smaller line in larger line - delete smaller
                            deletion_list.append((key_smaller, idx)) # save position of smaller line
                            # Add smaller object to parts list of larger object
                            obj_larger[0].sub_objects[obj_smaller[0].shape].append(obj_smaller[0])
        deletion_list = list(set(deletion_list))
        sorted_lines = [] # save rectangles that are not from deletion_list
        for k, v in lines.items():
            for idx, line in enumerate(v):
                if (k, idx) not in deletion_list and line[0].coords not in sorted_lines:
                    sorted_lines.append(line[0])
            objects['line'] = sorted_lines
        return objects

    @staticmethod
    def merge_in_rectangles(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some rectangle."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['rectangle', 'cell', 'complex'],
            contained_shapes=['rectangle', 'cell', 'complex']
        )

    @staticmethod
    def merge_in_t_shapes(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some t_shape."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['t_shape'],
            contained_shapes=['l_shape', 'line']
        )

    @staticmethod
    def merge_in_s_shapes(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some s_shape."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['s_shape'],
            contained_shapes=['l_shape', 'line']
        )

    @staticmethod
    def merge_in_hs_shapes(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some hs_shape."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['hs_shape'],
            contained_shapes=['l_shape', 'line']
        )

    @staticmethod
    def merge_in_tv_shapes(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some tv_shape."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['tv_shape'],
            contained_shapes=['l_shape', 'line', 'flower', 'hs_shape']
        )

    @staticmethod
    def merge_in_crosses(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some cross shape."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['cross'],
            contained_shapes=['l_shape', 'line', 't_shape', 'flower']
        )

    @staticmethod
    def merge_in_markup(objects: typing.Dict[str, GridObject]) -> typing.Dict[str, GridObject]:
        """Filter out objects intersecting with markup."""
        return ObjectsFilter.merge_shapes(
            objects,
            container_shapes=['markup_matrix'],
            contained_shapes=['cross', 'l_shape', 'tv_shape', 'line', 't_shape']
        )

    def filter_objects(self):
        """Apply all filtration approaches for the objects."""
        if self.repr_level == 3:
            objects_after_rectangle_merging = self.merge_rectangles(self.objects)
            objects_after_lines_merging = self.merge_lines(objects_after_rectangle_merging)
            objects_after_merging_in_rectangles = self.merge_in_rectangles(objects_after_lines_merging)
            objects_after_merging_in_t_shapes = self.merge_in_t_shapes(objects_after_merging_in_rectangles)
            objects_after_merging_in_s_shapes = self.merge_in_s_shapes(objects_after_merging_in_t_shapes)
            objects_after_merging_in_hs_shapes = self.merge_in_hs_shapes(objects_after_merging_in_s_shapes)
            objects_after_merging_in_tv_shapes = self.merge_in_tv_shapes(objects_after_merging_in_hs_shapes)
            objects_after_merging_crosses = self.merge_in_crosses(objects_after_merging_in_tv_shapes)
            objects_after_merging_in_markup = self.merge_in_markup(objects_after_merging_crosses)
            return objects_after_merging_in_markup
        else:
            return self.objects

class RelationAnalyzer():
    """Class for setting relations between objects on a grid."""
    def __init__(self, obj1:GridObject=None, obj2:GridObject=None, shape:tuple=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.shape = shape
        self.triples, self.relation_counter = self.set_relations()

    @staticmethod
    def rotation_symmetry(obj1:GridObject, obj2:GridObject):
        """Identify rotation relations between objects.

        Gated on the objects' cached congruence_key (see GridObject / this
        module's calculate_shape_similarity): two masks that aren't
        congruent under *any* rotation or reflection can't be related by
        one of the specific ones checked below either, so that O(1) key
        comparison skips the O(size) transform work entirely for the
        overwhelming majority of pairs, which share no shape at all.
        matching_transforms (not relating_transform) is used because the
        three relation names below are independent flags, not one label -
        a mask symmetric under more than one operation can legitimately
        earn more than one of them, same as the previous per-transform
        checks did.
        """
        rotations = []
        if len(obj1.coords) > 1 and len(obj2.coords) > 1: # exclude cells
            if len(obj1.coords) == len(obj2.coords):
                key1 = getattr(obj1, "congruence_key", None)
                key2 = getattr(obj2, "congruence_key", None)
                if key1 is not None and key1 == key2:
                    matches = matching_transforms(obj1.obj_mask, obj2.obj_mask)
                    if matches & {'rot90', 'rot180', 'rot270'}:
                        rotations.append('rotation')
                    if 'horizontal_flip' in matches:
                        rotations.append('horizontal_symmetry')
                    if 'vertical_flip' in matches:
                        rotations.append('vertical_symmetry')
        return rotations

    @staticmethod
    def translation_symmetry(coords1:List[tuple], coords2:List[tuple], shape:tuple):
        """Identify if each coordinate of object_1 equals each coordinate of object_2 after some shifting."""
        if len(coords1) == len(coords2):
            ul = find_upper_left_corner(shape)
            coords1_shifted = np.array([(tup[0] - ul[0], tup[1] - ul[1]) for tup in coords1])
            coords2_shifted = np.array([(tup[0] - ul[0], tup[1] - ul[1]) for tup in coords2])
            offsets = coords1_shifted - coords2_shifted
            unique_offsets = np.unique(offsets, axis=0)
            if unique_offsets.shape[0] == 1:
                return tuple(unique_offsets[0])
        return (0, 0)

    @staticmethod
    def _bbox_inside(inner: GridObject, outer: GridObject) -> bool:
        """Is `inner`'s bounding box strictly inside `outer`'s?

        Necessary for containment but nowhere near sufficient: two objects
        can sit side by side inside one sprawling L, and this says yes.
        Measured over real tasks, it holds for 3.59% of object pairs while
        only 1.76% are actually contained - so on its own it is right 49%
        of the time. It never misses a real containment though, which is
        what makes it a sound prefilter for the flood fill below.
        """
        return (inner.max_i < outer.max_i and inner.max_j < outer.max_j
                and inner.min_i > outer.min_i and inner.min_j > outer.min_j)

    @staticmethod
    def _is_enclosed_by(inner: GridObject, outer: GridObject, grid_shape: tuple) -> bool:
        """Can `inner` reach the edge of the grid without crossing `outer`?

        This is what containment means - `outer` forms a barrier around
        `inner` - and it is the only test that distinguishes a dot inside a
        ring from a dot merely standing in a bigger object's bounding box.
        Flood fills inward from the border, treating `outer`'s own cells as
        walls; anything the fill can't reach is enclosed.
        """
        height, width = grid_shape
        walls = set(map(tuple, outer.coords))
        target = set(map(tuple, inner.coords))

        reached = set()
        frontier = deque()
        for i in range(height):
            for j in (0, width - 1):
                if (i, j) not in walls and (i, j) not in reached:
                    reached.add((i, j))
                    frontier.append((i, j))
        for j in range(width):
            for i in (0, height - 1):
                if (i, j) not in walls and (i, j) not in reached:
                    reached.add((i, j))
                    frontier.append((i, j))

        while frontier:
            i, j = frontier.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (i + di, j + dj)
                if (0 <= neighbour[0] < height and 0 <= neighbour[1] < width
                        and neighbour not in reached and neighbour not in walls):
                    reached.add(neighbour)
                    frontier.append(neighbour)

        return not (target & reached)

    @classmethod
    def in_contour(cls, obj1: GridObject, obj2: GridObject, grid_shape: Optional[tuple] = None):
        """Which of the two objects the other one encloses, or None.

        The cheap bounding-box test decides whether it's worth running the
        flood fill at all - it fires for 3.59% of pairs and never misses a
        real containment - and the fill then confirms it. Skipping that
        second step (as this used to) called half its own claims wrong: a
        bounding box inside a bounding box is two objects' extents
        overlapping, not one surrounding the other.

        `grid_shape` is what the fill needs and older callers didn't pass;
        without it the bounding-box answer is returned unconfirmed, which
        is the previous behaviour rather than a silent failure.
        """
        if cls._bbox_inside(obj2, obj1):
            inner, outer, verdict = obj2, obj1, 'object_2'
        elif cls._bbox_inside(obj1, obj2):
            inner, outer, verdict = obj1, obj2, 'object_1'
        else:
            return None

        if grid_shape is None:
            return verdict
        return verdict if cls._is_enclosed_by(inner, outer, grid_shape) else None

    @staticmethod
    def touches(obj1: GridObject, obj2: GridObject) -> bool:
        """Do the two objects sit against each other, sharing an edge?

        Distinct from every distance relation already here: those measure
        how far apart two centres are, which says nothing about whether the
        objects meet. Two long bars can touch along their whole length and
        still have distant centres. Adjacency is what makes a pair read as
        one assembled thing rather than two things near each other, and
        nothing in the vocabulary expressed it - it holds for 13.17% of
        object pairs in real tasks.

        Edge adjacency only, matching the 4-connectivity the object
        extraction itself uses: two objects meeting at a single corner are
        not part of one structure by the same rule that kept them separate
        objects in the first place.
        """
        cells2 = set(map(tuple, obj2.coords))
        for i, j in obj1.coords:
            if ((i + 1, j) in cells2 or (i - 1, j) in cells2
                    or (i, j + 1) in cells2 or (i, j - 1) in cells2):
                return True
        return False

    @staticmethod
    def size_ratio(obj1: GridObject, obj2: GridObject) -> float:
        """How many times bigger the larger object is, as a number.

        `same_size` already answers the ratio-of-1 case; everything else it
        collapses into "not equal", which loses the scalings ARC tasks are
        full of - measured on real pairs, 33.5% sit at exactly 1x but
        another 28.3% land on a whole 2x-5x. Reported largest-over-smallest
        so it doesn't depend on which object came first.
        """
        larger, smaller = max(obj1.size, obj2.size), min(obj1.size, obj2.size)
        return larger / smaller if smaller else 0.0

    def in_diagonal(self, obj1: GridObject, obj2: GridObject):
        """Identify if object_1 and object_2 can be connected by diagonal."""
        center1 = obj1.center
        center2 = obj2.center

        # Diagonal: absolute difference in rows equals absolute difference in columns
        # This covers all 4 diagonal directions (NE, NW, SE, SW)
        return abs(center1[0] - center2[0]) == abs(center1[1] - center2[1]) and center1 != center2

    def in_line(self, obj1: GridObject, obj2: GridObject):
        """Identify if object_1 and object_2 can be connected by line."""
        # Get connection cells
        center1 = obj1.center
        center2 = obj2.center

        # Line: same row OR same column (but not the same point)
        return (center1[0] == center2[0] or center1[1] == center2[1]) and center1 != center2

    @staticmethod
    def x_alignment(obj1:GridObject, obj2:GridObject):
        """Identify if object_1 and object_2 are aligned in relation to x axis."""
        return obj1.max_i == obj2.max_i and obj1.min_i == obj2.min_i

    @staticmethod
    def y_alignment(obj1:GridObject, obj2:GridObject):
        """Identify if object_1 and object_2 are aligned in relation to y axis."""
        return obj1.max_j == obj2.max_j and obj1.min_j == obj2.min_j

    @staticmethod
    def edge_alignments(obj1: GridObject, obj2: GridObject) -> Tuple[str, ...]:
        """Which single edges the two objects share a line with.

        x_alignment/y_alignment demand that *both* edges of an axis
        coincide - the objects occupy exactly the same span of rows or
        columns. That is a much rarer thing than it sounds: a one-cell
        object and a three-cell one can never satisfy it however they are
        placed. Measured on real tasks, strict alignment holds for 7.63% of
        pairs while another 14.81% share one edge and were reported as
        nothing at all - the larger half of the signal was invisible.

        Sharing a single edge is its own structural fact, and which edge
        it is matters: objects hanging from a common top line and objects
        standing on a common bottom line are arranged differently.

        These fire independently of the strict relations rather than being
        suppressed by them. Two strictly aligned objects genuinely do share
        a top edge; withholding that would make this mean "shares a top
        edge but not a bottom one", which is a stranger thing to reason
        about than the plain statement. x_aligned_with remains the
        conjunction.

        Centre alignment is deliberately absent - in_line already reports
        two objects whose centres share a row or a column.
        """
        aligned = []
        if obj1.min_i == obj2.min_i:
            aligned.append("aligned_top")
        if obj1.max_i == obj2.max_i:
            aligned.append("aligned_bottom")
        if obj1.min_j == obj2.min_j:
            aligned.append("aligned_left")
        if obj1.max_j == obj2.max_j:
            aligned.append("aligned_right")
        return tuple(aligned)

    def set_relations(self):
        """Set all considered relations."""
        assert self.obj1 is not None and self.obj2 is not None and self.shape is not None, "Obj1, Obj2 and grid shape should be specified"
        triples1 = []
        triples2 = []
        relation_statistics = Counter()

        if self.obj1.colors == self.obj2.colors:
            triples2.append((self.obj2.label, "same_color", self.obj1.label))
            triples1.append((self.obj1.label, "same_color", self.obj2.label))
            relation_statistics["same_color"] += 1

        if (self.obj1.obj_mask.shape == self.obj2.obj_mask.shape) and (self.obj1.obj_mask==self.obj2.obj_mask).all():
            triples2.append((self.obj2.label, "same_shape", self.obj1.label))
            triples1.append((self.obj1.label, "same_shape", self.obj2.label))
            relation_statistics["same_shape"] += 1

        if self.obj1.size == self.obj2.size:
            triples2.append((self.obj2.label, "same_size", self.obj1.label))
            triples1.append((self.obj1.label, "same_size", self.obj2.label))
            relation_statistics["same_size"] += 1

        rotations = self.rotation_symmetry(self.obj1, self.obj2)
        if rotations != []:
            for rotation in rotations:
                triples1.append((self.obj1.label, rotation, self.obj2.label))
                triples2.append((self.obj2.label, rotation, self.obj1.label))

        (i_offset, j_offset) = self.translation_symmetry(self.obj1.coords, self.obj2.coords, self.shape)
        if i_offset != 0 and j_offset != 0:
            triples1.append((self.obj1.label, "translation_symmetry", self.obj2.label))
            triples2.append((self.obj2.label, "translation_symmetry", self.obj1.label))

        in_contour = self.in_contour(self.obj1, self.obj2, self.shape)
        if in_contour == "object_2":
            triples2.append((self.obj2.label, "in_contour", self.obj1.label))
            triples1.append((self.obj1.label, "has_in_contour", self.obj2.label))
            relation_statistics["in_contour"] += 1

        if in_contour == "object_1":
            triples1.append((self.obj1.label, "in_contour", self.obj2.label))
            triples2.append((self.obj2.label, "has_in_contour", self.obj1.label))
            # Counted here too: this branch used to write the triples and
            # skip the tally, so containment was only ever counted when the
            # contained object happened to be the second of the pair - which
            # of the two that is depends on iteration order, not on the grid.
            relation_statistics["in_contour"] += 1

        if self.touches(self.obj1, self.obj2):
            triples1.append((self.obj1.label, "touches", self.obj2.label))
            triples2.append((self.obj2.label, "touches", self.obj1.label))
            relation_statistics["touches"] += 1

        # Check for in_line relation with connection cells
        is_in_line = self.in_line(self.obj1, self.obj2)
        if is_in_line:
            triples1.append((self.obj1.label, "in_line", self.obj2.label))
            triples2.append((self.obj2.label, "in_line", self.obj1.label))
            relation_statistics["in_line"] += 1

        # Check for in_diagonal relation with connection cells
        is_diagonal = self.in_diagonal(self.obj1, self.obj2)
        if is_diagonal:
            triples1.append((self.obj1.label, "in_diagonal", self.obj2.label))
            triples2.append((self.obj2.label, "in_diagonal", self.obj1.label))
            relation_statistics["in_diagonal"] += 1

        for alignment in self.edge_alignments(self.obj1, self.obj2):
            triples1.append((self.obj1.label, alignment, self.obj2.label))
            triples2.append((self.obj2.label, alignment, self.obj1.label))
            relation_statistics[alignment] += 1

        x_alignment = self.x_alignment(self.obj1, self.obj2)
        y_alignment = self.y_alignment(self.obj1, self.obj2)
        if x_alignment and y_alignment:
            triples1.append((self.obj1.label, "x_y_aligned_with", self.obj2.label))
            triples2.append((self.obj2.label, "x_y_aligned_with", self.obj1.label))
            relation_statistics["x_y_aligned_with"] += 1

        else:
            if self.x_alignment(self.obj1, self.obj2):
                triples1.append((self.obj1.label, "x_aligned_with", self.obj2.label))
                triples2.append((self.obj2.label, "x_aligned_with", self.obj1.label))
                relation_statistics["x_aligned_with"] += 1


            if self.y_alignment(self.obj1, self.obj2):
                triples1.append((self.obj1.label, "y_aligned_with", self.obj2.label))
                triples2.append((self.obj2.label, "y_aligned_with", self.obj1.label))
                relation_statistics["y_aligned_with"] += 1

        return (triples1, triples2), relation_statistics


def calculate_shape_similarity(obj1, obj2):
    """Shape similarity: 1.0 when the two objects are the same shape up to
    rotation/reflection, else an IoU-based overlap ratio as a softer signal.

    The congruence check comes first and is the cheap path (an O(1) key
    comparison, once each key is cached on construction - see GridObject's
    congruence_key) - it also catches every rotated/reflected duplicate that
    the overlap ratio below cannot: that ratio only ever crops at two fixed
    alignments (top-left, center) without trying any rotation, so a shape
    and its 90-degree turn scored near 0 even though they're identical.
    """
    key1 = getattr(obj1, "congruence_key", None)
    key2 = getattr(obj2, "congruence_key", None)
    if key1 is not None and key1 == key2:
        return 1.0

    # Quick size filter
    size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
    if size_ratio < 0.5:
        return 0.0

    mask1 = obj1.obj_mask
    mask2 = obj2.obj_mask

    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # Use smaller dimensions
    h = min(h1, h2)
    w = min(w1, w2)

    def compute_iou_for_crop(start1, start2):
        """Helper function to compute IoU for specific crop positions."""
        # Crop both masks
        m1_crop = mask1[start1[0]:start1[0]+h, start1[1]:start1[1]+w]
        m2_crop = mask2[start2[0]:start2[0]+h, start2[1]:start2[1]+w]

        # Compute intersection and union
        intersection = np.count_nonzero(m1_crop & m2_crop)
        union = np.count_nonzero(m1_crop | m2_crop)

        # IoU (Jaccard similarity)
        return intersection / union if union > 0 else 0.0

    # Upper-left corner cropping
    ul_similarity = compute_iou_for_crop((0, 0), (0, 0))

    # Center cropping
    center_similarity = compute_iou_for_crop(
        ((h1-h)//2, (w1-w)//2),
        ((h2-h)//2, (w2-w)//2)
    )

    # Return maximum similarity from both cropping methods
    return max(ul_similarity, center_similarity)



def has_holes(obj):
    """Check if object has any holes."""
    return (hasattr(obj, 'inner_holes') and obj.inner_holes) or \
           (hasattr(obj, 'outer_holes') and obj.outer_holes)

def get_all_holes(obj):
    """Get all holes from an object."""
    holes = []
    if hasattr(obj, 'inner_holes') and obj.inner_holes:
        holes.extend(obj.inner_holes)
    if hasattr(obj, 'outer_holes') and obj.outer_holes:
        holes.extend(obj.outer_holes)
    return holes

def calculate_size_compatibility(hole, filler):
    """Calculate how compatible the sizes are for hole filling.
    """
    # Filler should be slightly smaller than hole for perfect fit
    size_ratio = filler.size / hole.size if hole.size > 0 else 0

    if size_ratio > 1.0:  # Filler too big
        return 0.0
    elif size_ratio > 0.8:  # Almost perfect fit
        return 1.0
    elif size_ratio > 0.5:  # Reasonable fit
        return 0.7
    elif size_ratio > 0.3:  # Poor fit but possible
        return 0.3
    else:  # Too small
        return 0.1

def calculate_shape_compatibility(hole, filler):
    """Calculate shape compatibility for hole filling, considering both original
    and 90° rotated orientations of the filler.
    """
    # Calculate hole aspect ratio
    hole_aspect = hole.hor_size / max(1, hole.vert_size)

    # Calculate filler aspect ratios for both original and rotated orientations
    filler_aspect_original = filler.hor_size / max(1, filler.vert_size)
    filler_aspect_rotated = filler.vert_size / max(1, filler.hor_size)

    # Calculate compatibility scores for both orientations
    def calculate_aspect_score(hole_asp, filler_asp):
        aspect_ratio = max(hole_asp, filler_asp) / min(hole_asp, filler_asp)

        if aspect_ratio < 1.5:  # Similar aspect ratio
            return 1.0
        elif aspect_ratio < 2.5:  # Moderately different
            return 0.5
        else:  # Very different
            return 0.2

    # Get scores for both orientations
    score_original = calculate_aspect_score(hole_aspect, filler_aspect_original)
    score_rotated = calculate_aspect_score(hole_aspect, filler_aspect_rotated)

    # Return the best score (filler can be rotated for better fit)
    return max(score_original, score_rotated)

def find_promising_hole_matches(obj1, obj2):
    """Find if objects can be matched based on hole filling.
    Returns list of promising configurations.
    """
    promising_configs = []

    # Determine which object has holes and which doesn't
    obj1_has_holes = has_holes(obj1)
    obj2_has_holes = has_holes(obj2)

    if not obj1_has_holes and not obj2_has_holes:
        return []  # No holes to fill

    if obj1_has_holes and not obj2_has_holes:
        # obj1 has holes, obj2 can potentially fill them
        hole_filler, hole_owner = obj2, obj1
    elif not obj1_has_holes and obj2_has_holes:
        # obj2 has holes, obj1 can potentially fill them
        hole_filler, hole_owner = obj1, obj2
    else:
        if obj1.size > obj2.size:
           hole_filler, hole_owner = obj2, obj1
        elif obj1.size < obj2.size:
           hole_filler, hole_owner = obj1, obj2
        else:
            return []

    # Find best hole to fill
    best_hole, score = find_best_hole_for_filler(hole_owner, hole_filler)
    if not best_hole:
        return []

    # Calculate position to place filler in the hole
    target_position = calculate_optimal_placement(best_hole, hole_filler)

    promising_configs.append({
        'type': 'hole_filling',
        'hole_owner': hole_owner,
        'filler': hole_filler,
        'hole': best_hole,
        'target_position': target_position,
        'score': score,
        'rotation': 0  # Start with no rotation
    })

    return promising_configs

def find_best_hole_for_filler(hole_owner, filler):
    """Find the best hole in hole_owner for the filler to fill.
    """
    holes = get_all_holes(hole_owner)
    if not holes:
        return None

    best_hole = None
    best_score = 0.0

    for hole in holes:
        size_score = calculate_size_compatibility(hole, filler)
        shape_score = calculate_shape_compatibility(hole, filler)

        total_score = size_score * 0.4 + shape_score * 0.6

        if total_score > best_score:
            best_score = total_score
            best_hole = hole

    return (best_hole, best_score) if best_score > 0.3 else (None, 0)

def calculate_optimal_placement(hole, filler):
    """Calculate the best position to place filler in the hole.
    """
    # Simple strategy: align centers
    hole_center = hole.center
    filler_center = filler.center

    # Calculate offset to move filler to hole center
    offset_x = hole_center[0] - filler_center[0]
    offset_y = hole_center[1] - filler_center[1]

    return (filler.coords[0][0] + offset_x, filler.coords[0][1] + offset_y)

def calculate_match_score_hole_based(grid, obj1, obj2, all_objects, font_color):
    """New match score based on hole filling potential.
    """
    # Quick compatibility check
    if not has_holes(obj1) and not has_holes(obj2):
        return 0.0

    # Find promising hole-based matches
    promising_matches = find_promising_hole_matches(obj1, obj2)
    if not promising_matches:
        return 0.0

    # Return the best score
    best_score = max(match['score'] for match in promising_matches)
    return best_score

def calculate_match_score(grid, obj1, obj2, all_objects, font_color):
    return calculate_match_score_hole_based(grid, obj1, obj2, all_objects, font_color)
