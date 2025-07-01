import typing
from typing import List
import numpy as np
from copy import copy
from collections import defaultdict, Counter
from itertools import product
from scipy.spatial.distance import euclidean
from rl.ARC_task import ARCTask, ARCSubtask
from symbolic.objects_analysis import GridObject, ObjectsFilter, RelationAnalyzer
from symbolic.utils import find_upper_left_corner, coords_transform, count_unique_cells, dict_to_list, check_subset_condition
from symbolic.patterns import generate_patterns, find_connected_components_with_color, find_connected_components_excluding_colors
from llm.prompts import COLOR_MAPPING

colors_mapping = {
    0: 'black', 0.1: 'blue', 0.2: 'red', 0.3: 'green', 0.4: 'yellow', 
    0.5: 'gray', 0.6: 'magenta', 0.7: 'orange', 0.8: 'sky', 0.9: 'brown', 1: 'white'
}

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict, Counter
from copy import copy
import typing

# Immutable dataclasses for representation levels
@dataclass(frozen=True)
class ObjectsSummary:
    """Immutable dataclass for objects summary statistics."""
    size2shape: Dict[int, List[Any]] = field(default_factory=dict)
    shape2size: Dict[str, int] = field(default_factory=dict)
    mean_size: float = 0.0
    median_size: float = 0.0
    hor_size2shape: Dict[int, List[Any]] = field(default_factory=dict)
    shape2hor_size: Dict[str, int] = field(default_factory=dict)
    vert_size2shape: Dict[int, List[Any]] = field(default_factory=dict)
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
    in_line: int = 0
    x_y_aligned_with: int = 0
    x_aligned_with: int = 0
    y_aligned_with: int = 0

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
class TaskSummary():
    def __init__(self, task:ARCTask, patterns:dict):
        self.patterns = patterns
        self.train_subtasks = task.subtasks
        self.test_subtask = task.test_subtask
        self.train_subtasks_summaries = []
   
    def obtain_summaries(self):
        for subtask in self.train_subtasks:
            self.train_subtasks_summaries.append(SubtaskSummary(subtask, self.patterns))

class SubtaskSummary():
    def __init__(self, subtask:ARCSubtask, train:bool=True):
        self.subtask = subtask
        self.subtask_label = subtask.label
        self.inp_grid_summary = GridSummary(self.subtask.train_inp, self.subtask.train_inp_shape)
        if train:
            self.out_grid_summary = GridSummary(subtask.train_out, self.subtask.train_out_shape)
            self.grids_x_ratio = self.subtask.train_inp_shape[0] / self.subtask.train_out_shape[0]
            self.grids_y_ratio = self.subtask.train_inp_shape[1] / self.subtask.train_out_shape[1]       

    def prepare_features(self):
        obj_summary_1 = self.inp_grid_summary.objects_summary
        rel_summary_1 = self.inp_grid_summary.relations_summary
        obj_summary_2 = self.out_grid_summary.objects_summary
        rel_summary_2 = self.out_grid_summary.relations_summary
        inp = obj_summary_1 | obj_summary_1['shapes'] | obj_summary_1['colors'] | rel_summary_1  
        out = obj_summary_2 | obj_summary_2['shapes'] | obj_summary_2['colors'] | rel_summary_2 
        features = {k:(out[k]-inp[k]) for k in inp.keys() if k not in ['shape2size', 'size2shape', 'shapes', 'colors']}
        features['grids_x_ratio'] = self.grids_x_ratio
        features['grids_y_ratio'] = self.grids_y_ratio 
        return features
shape_templates = {'line': (0.07393889006401545, 0.04934261273778651, 0.0, 0.0, 0.0, 0.0, 0.0),
 'diagonal': (0.14707373398612458, 0.19500122240442375, 0.0, 0.0, 0.0, 0.0, 0.0),
 'rectangle': (0.016459418754238012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
 'cross': (0.03198908180540403, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
 'l_shape': (0.062418746747512514, 0.031239831446031256, 0.007812341058161014, 0.007812341058161014, 0.0009765621895592603, 0.00390623013190634, 0.0),
 't_shape': (0.04796316994220891, 0.0063999126200983, 0.005119955261226454, 0.0051199552612264535, -0.000655359906175024, 0.0020479971366932728, 0.0),
 's_shape': (0.062418746747512514, 0.015623728558408866, 0.009765314570983716, 0.009765314570983716, 0.0009155270879547009, 0.0029296791181256054, -0.0012207025186705602),
 'tv_shape': (0.023433209408330664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
 'hs_shape': (0.032058980893054694, 0.00041649310378085293, 5.94990181959871e-05, 5.94990181959871e-05, 1.7346652555742852e-07, -8.499859752109381e-06, 0.0),
 'flower': (0.06391276159528161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}

class GridSummary():
    """Class for creating summary for a given grid."""
    def __init__(self, grid:np.array, shape:tuple, font_color:float=0.0, levels:List[int]=[1], shape_types=None):
        self.grid = grid
        self.shape = shape
        self.shape_types = ('line' ,'rectangle', 'diagonal', 'l_shape', 't_shape', 's_shape', 'tv_shape', 
                            'hs_shape', 'cross', 'flower', 'markup_matrix','markup_line', 'cell', 'complex') if shape_types == None else shape_types
        self.font_color = font_color
        self.grid_corners = self.define_grid_corners()
        self.objects_dict = {}
        self.initial_objects = self.retrieve_objects(self.grid, self.shape, self.shape_types)
        self.connected_components = self.retrieve_connected_components(self.grid) 
        self.font_segments = find_connected_components_with_color(self.grid, target_color=self.font_color)
        self.relations_for_stats = ("same_color", "same_shape", "same_size", "rotation", "horizontal_symmetry", "vertical_symmetry",
                                    "in_line", "x_y_aligned_with", "x_aligned_with", "y_aligned_with")
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
        level1_init_objects = copy(self.initial_objects)
        init_objects = copy(self.initial_objects)
        init_objects['complex'] = [obj for obj in init_objects['complex'] if obj.color_homo]
        
        for level in self.levels:
            if level == 1:
                repr_levels[level] = self.process_repr_level(level1_init_objects, level)
            elif level == 5:
                repr_levels[level] = self.process_cell_level()
            else: 
                repr_levels[level] = self.process_repr_level(init_objects, level)
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
                    obj = GridObject('cell', [cell], [cell_color], label, self.shape)
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
        
    def retrieve_objects(self, grid:np.array, shape:tuple, shape_types:tuple)->typing.Dict[str, List[GridObject]]:
        """Retrieve all possible objects from the grid and return corresponding GridObject instances."""
        patterns = generate_patterns(shape, shape_types)
        objects = defaultdict(list)
        candidate = False
        used_coordinates = []
        
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
                        if color != 0 and (k != 'diagonal' or count_unique_cells(k, pattern, used_coordinates)) > 0:
                            label = f'{k}_{idx}'
                            obj = GridObject(k, pattern, [color], label, self.shape, self.font_color, grid)
                            self.objects_dict[label] = obj
                            used_coordinates.extend(pattern)
                            candidate = True
                if candidate:
                    objects[k].append(copy(obj))
                    candidate = False
                    
        used_coordinates = set(used_coordinates)
        ul = find_upper_left_corner(shape)
        all_coordinates = set(product(range(ul[0], ul[0]+shape[0]), range(ul[1], ul[1]+shape[1])))
        cells_coordinates = list(all_coordinates.difference(used_coordinates))
        
        for idx, cell in enumerate(cells_coordinates):
            color = grid[cell]
            if color != 0:
                label = f'cell_{idx}'
                obj = GridObject('cell', [cell], [grid[cell]], label, self.shape, self.font_color, grid)
                objects['cell'].append(obj)
        return objects

    def retrieve_connected_components(self, grid:np.array):
        """Retrieve all connected components and return corresponding GridObject instances."""
        components = {}
        comp_idx = 0
        heter_components = find_connected_components_excluding_colors(grid, pad_val=1, font_color=0.0)
        
        for comp in heter_components:
            if len(comp) > 1:
                color_numbers = list(set([grid[coord] for coord in comp]))
                label = f'complex_{comp_idx}'
                obj = GridObject('complex', comp, color_numbers, label, self.shape, self.font_color, grid)
                components[label] = obj
                comp_idx += 1
                
        for i in range(1, 10):
            col = i / 10
            homo_components = find_connected_components_with_color(grid, target_color=col)
            for comp in homo_components:
                if comp not in heter_components:
                    if len(comp) > 1:
                        label = f'complex_{comp_idx}'
                        obj = GridObject('complex', comp, [col], label, self.shape, self.font_color, grid)
                        components[label] = obj
                        comp_idx += 1 
                        
        self.initial_objects['complex'] = list(components.values())
        return components

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
        """Calculate distance between objects."""
        i_dist = min(abs(obj1.max_i - obj2.max_i), abs(obj1.max_i - obj2.min_i), 
                    abs(obj1.min_i - obj2.max_i), abs(obj1.min_i - obj2.min_i)) 
        j_dist = min(abs(obj1.max_j - obj2.max_j), abs(obj1.max_j - obj2.min_j), 
                    abs(obj1.min_j - obj2.max_j), abs(obj1.min_j - obj2.min_j)) 
        return min(i_dist, j_dist)

    def create_objects_summary(self, objects) -> ObjectsSummary:
        """Create a summary for grid objects to get aggregate information about their shapes, sizes, colors.""" 
        size2shape = defaultdict(list)
        shape2size = {}
        hor_size2shape = defaultdict(list)
        shape2hor_size = {}
        vert_size2shape = defaultdict(list)
        shape2vert_size = {}
        shapes = {shape:0 for shape in self.shape_types}
        
        # Initialize shape_colors with multicolor support
        shape_colors = {colors_mapping[i/10]:0 for i in range(10)}
        shape_colors['multicolor'] = 0
        
        colors = {colors_mapping[i/10]:0 for i in range(10)}
        
        # Initialize cluster-based shape tracking
        cluster_shapes = defaultdict(int)
        
        for k, v in objects.items():
            for idx, obj in enumerate(v):
                size2shape[obj.size].append(obj)
                hor_size2shape[obj.hor_size].append(obj)
                vert_size2shape[obj.vert_size].append(obj)
                shape2size[obj.label] = obj.size
                shape2hor_size[obj.label] = obj.hor_size
                shape2vert_size[obj.label] = obj.vert_size
                shapes[obj.shape] += 1
                
                if obj.shape != 'complex':
                    color = obj.colors[0]
                    shape_colors[color] += 1
                else:
                    # Handle multicolor objects
                    if len(obj.colors) > 1:
                        shape_colors['multicolor'] += 1
                    else:
                        color = obj.colors[0]
                        shape_colors[color] += 1
                    
                    if hasattr(obj, 'structure_analysis'):
                        obj.structure_analysis()
                        if hasattr(obj, 'objects_summary'):
                            obj_summary = obj.objects_summary
                            for c in list(colors_mapping.values())[:-1]:
                                if c in obj_summary.get('shape_colors', {}):
                                    shape_colors[c] += obj_summary['shape_colors'][c]
                                if c in obj_summary.get('colors', {}):
                                    colors[c] += obj_summary['colors'][c]

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
        median_size = np.median(shape2size_values) if len(shape2size_values) > 0 else 0
        
        return ObjectsSummary(
            size2shape=dict(size2shape),
            shape2size=shape2size,
            mean_size=mean_size,
            median_size=median_size,
            hor_size2shape=dict(hor_size2shape),
            shape2hor_size=shape2hor_size,
            vert_size2shape=dict(vert_size2shape),
            shape2vert_size=shape2vert_size,
            shapes=shapes,
            shape_colors=shape_colors,
            colors=colors,
            shape_hor_size_description=hor_size2description,
            shape_vert_size_description=vert_size2description,
            shape_size_description=size2description,
            shape_color_description=shape_color2description,
            color_description=color2description
        )
        
    def set_relations(self, objects) -> Tuple[Triples, RelationStatistics, ObjectDistances]:
        """Iterate over objects to identify relations between them.""" 
        all_object_triples = []
        relation_statistics = Counter(self.relations_for_stats)
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
            in_line=relation_statistics.get('in_line', 0),
            x_y_aligned_with=relation_statistics.get('x_y_aligned_with', 0),
            x_aligned_with=relation_statistics.get('x_aligned_with', 0),
            y_aligned_with=relation_statistics.get('y_aligned_with', 0)
        )
        distances = ObjectDistances(distances=distances_dict)
        
        return triples, relation_stats, distances

    def _create_relation_embeddings_for_objects(self, objects_tuple: Tuple, triples: Triples, distances: ObjectDistances) -> RelationEmbeddings:
        """Create relation embeddings for a set of objects - optimized version."""
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
                embedding = self._create_embedding_optimized(
                    obj1, obj2, relation_lookup, distances, grid_size, objects_tuple
                )
                embeddings_dict[obj1.label][obj2.label] = embedding
        
        return RelationEmbeddings(embeddings=embeddings_dict)
    
    def _create_embedding_optimized(self, obj1, obj2, relation_lookup, distances, grid_size, objects_tuple):
        """Optimized embedding creation."""
        relation_feature_names = (
            'same_color', 'same_size', 'same_vert_size', 'same_hor_size', 
            'shape_similarity', 'match_score', 'translation_symmetry',
            'horizontal_symmetry', "vertical_symmetry", 'rotation',
            'in_line', 'in_diagonal', 'x_aligned_with', 'y_aligned_with', 
            'normalized_distance', 'x_offset', 'y_offset'
        )
        embedding = np.zeros(len(relation_feature_names), dtype=np.float32)
        idx = 0
        
        # Basic comparisons
        embedding[idx] = 1.0 if obj1.colors == obj2.colors else 0.0
        idx += 1
        embedding[idx] = 1.0 if obj1.size == obj2.size else 0.0
        idx += 1
        embedding[idx] = 1.0 if obj1.vert_size == obj2.vert_size else 0.0
        idx += 1
        embedding[idx] = 1.0 if obj1.hor_size == obj2.hor_size else 0.0
        idx += 1
        # Shape similarity
        embedding[idx] = self._calculate_hu_moments_similarity(obj1, obj2)
        idx += 1

        
        # Match score (simplified for performance - can be computed on demand)
        embedding[idx] = calculate_match_score(self.grid, obj1, obj2, objects_tuple, self.font_color)  # Placeholder - compute only when needed
        idx += 1
        # Relation flags from lookup
        relations_to_check = ['translation_symmetry', 'horizontal_symmetry', 'vertical_symmetry', 
                              'rotation', 'in_line', 'in_diagonal', 'x_aligned_with', 'y_aligned_with']
        obj_relations = relation_lookup[obj1.label][obj2.label]
        
        for relation in relations_to_check:
            embedding[idx] = 1.0 if relation in obj_relations else 0.0
            idx += 1
        
        # Distance metrics
        distance = distances.get_distance(obj1.label, obj2.label)
        embedding[idx] = min(distance / grid_size, 1.0) if grid_size > 0 else 0.0
        idx += 1
        
        # Offsets
        if hasattr(obj1, 'center') and hasattr(obj2, 'center'):
            embedding[idx] = (obj2.center[1] - obj1.center[1]) / grid_size if grid_size > 0 else 0.0
            idx += 1
            embedding[idx] = (obj2.center[0] - obj1.center[0]) / grid_size if grid_size > 0 else 0.0
        else:
            embedding[idx] = 0.0
            idx += 1
            embedding[idx] = 0.0
        
        return embedding
        
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
    def merge_in_rectangles(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some rectangle."""
        deletion_list = []
        select_shapes = ['rectangle', 'cell', 'complex']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k not in select_shapes}
        for rect in filtered_objects['rectangle']: # iterate over each rectangle
            rect_set = set(rect.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                
                for idx, shape in enumerate(v):
                    if check_subset_condition(rect_set, shape.coords): # delete shape if it is inside rectangle
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of rectangle
                        rect.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects

    @staticmethod
    def merge_in_t_shapes(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some t_shape."""
        deletion_list = []
        select_shapes = ['l_shape', 'line']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k in select_shapes}
        for t_shape in filtered_objects['t_shape']: # iterate over each t_shape
            t_shape_set = set(t_shape.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(t_shape_set, shape.coords): # delete shape if it is inside t_shape
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of t_shape
                        t_shape.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects       
    
    @staticmethod
    def merge_in_s_shapes(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some s_shape."""
        deletion_list = []
        select_shapes = ['l_shape', 'line']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k in select_shapes}
        for s_shape in filtered_objects['s_shape']: # iterate over each s_shape
            s_shape_set = set(s_shape.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(s_shape_set, shape.coords): # delete shape if it is inside s_shape
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of s_shape
                        s_shape.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects  
    
    @staticmethod
    def merge_in_hs_shapes(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]: 
        """Filter out objects each of which is subset of some hs_shape."""
        deletion_list = []
        select_shapes = ['l_shape', 'line']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k in select_shapes}
        for hs_shape in filtered_objects['hs_shape']: # iterate over each hs_shape
            hs_shape_set = set(hs_shape.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(hs_shape_set, shape.coords): # delete shape if it is inside hs_shape
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of hs_shape
                        hs_shape.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects  
    
    @staticmethod
    def merge_in_tv_shapes(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]: 
        """Filter out objects each of which is subset of some tv_shape."""
        deletion_list = []
        select_shapes = ['l_shape', 'line', 'flower', 'hs_shape']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k in select_shapes}
        for tv_shape in filtered_objects['tv_shape']: # iterate over each tv_shape
            tv_shape_set = set(tv_shape.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(tv_shape_set, shape.coords): # delete shape if it is inside tv_shape
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of tv_shape
                        tv_shape.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects  
    
    @staticmethod
    def merge_in_crosses(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:  
        """Filter out objects each of which is subset of some cross shape."""
        deletion_list = []
        select_shapes = ['l_shape', 'line', 't_shape', 'flower']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k in select_shapes}
        for cross in filtered_objects['cross']: # iterate over each cross
            cross_set = set(cross.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(cross_set, shape.coords): # delete shape if it is inside cross
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of cross
                        cross.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_objects.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects  
    
    @staticmethod
    def merge_in_markup(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects instersecting with markup."""
        deletion_list = []
        select_shapes = ['cross', 'l_shape', 'tv_shape', 'line', 't_shape']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes}      
        other_shapes = {k:v for k, v in objects.items() if k in select_shapes} 
        for markup in filtered_objects['markup_matrix']: # iterate over each markup
            markup_set = set(markup.coords)
            for k, v in other_shapes.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(markup_set, shape.coords): # delete shape if it is inside markup
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of markup
                        markup.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_shapes.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects

    @staticmethod
    def merge_in_components(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects instersecting with connected components."""
        deletion_list = []
        select_shapes = ['cross', 'l_shape', 'tv_shape', 's_shape', 't_shape', 'hs_shape', 'rectangle', 'line', 'diagonal', 'flower', 'cell']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k not in select_shapes}      
        other_shapes = {k:v for k, v in objects.items() if k in select_shapes} 
        for component in filtered_objects['complex']: # iterate over each markup
            component_set = set(component.coords)
            for k, v in other_shapes.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(component_set, shape.coords): # delete shape if it is inside markup
                        deletion_list.append((k, idx))
                        # Add smaller object to parts list of markup
                        component.sub_objects[shape.shape].append(shape)
        deletion_list = list(set(deletion_list))
        for k, v in other_shapes.items(): # keep only objects not from deletion_list
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects
       
    def filter_objects(self):
        """Apply all filtration approaches for the objects."""
        objects_after_rectangle_merging = self.merge_rectangles(self.objects)
        objects_after_lines_merging = self.merge_lines(objects_after_rectangle_merging)
        if self.repr_level <= 3:
            objects_after_merging_in_rectangles = self.merge_in_rectangles(objects_after_lines_merging)
            objects_after_merging_in_t_shapes = self.merge_in_t_shapes(objects_after_merging_in_rectangles) 
            objects_after_merging_in_s_shapes = self.merge_in_s_shapes(objects_after_merging_in_t_shapes) 
            objects_after_merging_in_hs_shapes = self.merge_in_hs_shapes(objects_after_merging_in_s_shapes)  
            objects_after_merging_in_tv_shapes = self.merge_in_tv_shapes(objects_after_merging_in_hs_shapes)
            objects_after_merging_crosses = self.merge_in_crosses(objects_after_merging_in_tv_shapes)  
            objects_after_merging_in_markup = self.merge_in_markup(objects_after_merging_crosses)
            if self.repr_level <= 2:
                objects_after_merging_in_components = self.merge_in_components(objects_after_merging_in_markup) 
                return objects_after_merging_in_components
            else:
                return objects_after_merging_in_markup
        return objects_after_lines_merging

class RelationAnalyzer():
    """Class for setting relations between objects on a grid."""
    def __init__(self, object_1:GridObject=None, object_2:GridObject=None, shape:tuple=None):
        self.object_1 = object_1
        self.object_2 = object_2
        self.shape = shape
        self.triples, self.relation_counter = self.set_relations()
    
    @staticmethod
    def rotation_symmetry(coords_1:List[tuple], coords_2:List[tuple], shape:tuple):
        """Identify rotation relations between objects."""
        rotations = []
        if len(coords_1) > 1 and len(coords_2) > 1: # exclude cells
            if len(coords_1) == len(coords_2):
                ul = find_upper_left_corner(shape)
                coords_1_shifted = [(tup[0]-ul[0], tup[1]-ul[1]) for tup in coords_1]
                coords_2_shifted = [(tup[0]-ul[0], tup[1]-ul[1]) for tup in coords_2]
                grid_1 = np.zeros((max(shape), max(shape)))
                i_1, j_1 = coords_transform(coords_1_shifted)
                grid_1[i_1, j_1] = 1
                grid_2 = np.zeros((max(shape), max(shape)))
                i_2, j_2 = coords_transform(coords_2_shifted)
                grid_2[i_2, j_2] = 1
                if (grid_1 == np.rot90(grid_2, k=1)).all():
                    rotations.append('rotation_90')
                if (grid_1 == np.rot90(grid_2, k=2)).all():
                    rotations.append('rotation_180')
                if (grid_1 == np.rot90(grid_2, k=3)).all():
                    rotations.append('rotation_270')
                if (grid_1 == np.flipud(grid_2)).all():
                    rotations.append('horizontal_symmetry')
                if (grid_1 == np.fliplr(grid_2)).all():
                    if 'horizontal_symmetry' in rotations:
                        rotations.pop()
                        rotations.append('horizontal_and_vertical_symmetry')  
                    else:
                        rotations.append('vertical_symmetry')               
        return rotations  
    
    @staticmethod
    def translation_symmetry(coords_1:List[tuple], coords_2:List[tuple], shape:tuple):
        """Identify if each coordinate of object_1 equals each coordinate of object_2 after some shifting."""   
        if len(coords_1) == len(coords_2):
            ul = find_upper_left_corner(shape)
            coords_1_shifted = np.array([(tup[0] - ul[0], tup[1] - ul[1]) for tup in coords_1])
            coords_2_shifted = np.array([(tup[0] - ul[0], tup[1] - ul[1]) for tup in coords_2])
            offsets = coords_1_shifted - coords_2_shifted
            unique_offsets = np.unique(offsets, axis=0)
            if unique_offsets.shape[0] == 1:
                return tuple(unique_offsets[0])
        return (0, 0)

    @staticmethod
    def in_contour(object_1:GridObject, object_2:GridObject):
        """Identify if all coordinates of object_1 are surrounded by coordinates of object_2 or in the reverse order."""
        in_contour = None
        if object_2.max_i < object_1.max_i and object_2.max_j < object_1.max_j and object_2.min_i > object_1.min_i and object_2.min_j > object_1.min_j:
            in_contour = 'object_2'
        if object_1.max_i < object_2.max_i and object_1.max_j < object_2.max_j and object_1.min_i > object_2.min_i and object_1.min_j > object_2.min_j:
            in_contour = 'object_1'        
        return in_contour   

    @staticmethod
    def find_connection_cells(object_1: GridObject, object_2: GridObject):
        """Find all cells from both objects that are at the minimum distance from each other."""
        min_distance = float('inf')
        
        # First pass: find the minimum distance
        for coord1 in object_1.coords:
            for coord2 in object_2.coords:
                # Calculate Manhattan distance
                distance = abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
                if distance < min_distance:
                    min_distance = distance
        
        # Second pass: collect all cell pairs at minimum distance
        connection_cells_1 = []
        connection_cells_2 = []
        connection_pairs = []
        
        for coord1 in object_1.coords:
            for coord2 in object_2.coords:
                distance = abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
                if distance == min_distance:
                    if coord1 not in connection_cells_1:
                        connection_cells_1.append(coord1)
                    if coord2 not in connection_cells_2:
                        connection_cells_2.append(coord2)
                    connection_pairs.append((coord1, coord2))
        
        return connection_cells_1, connection_cells_2, connection_pairs, min_distance
    
    def in_diagonal(self, object_1: GridObject, object_2: GridObject):
        """Identify if object_1 and object_2 can be connected by diagonal."""
        # Get connection cells
        conn_cells_1, conn_cells_2, conn_pairs, distance = self.find_connection_cells(object_1, object_2)
        
        # Check if any connection forms a diagonal (not in same row or column)
        diagonal_pairs = []
        for pair in conn_pairs:
            coord1, coord2 = pair
            if coord1[0] != coord2[0] and coord1[1] != coord2[1]:
                diagonal_pairs.append(pair)
        
        if diagonal_pairs:
            diagonal_cells_1 = [pair[0] for pair in diagonal_pairs]
            diagonal_cells_2 = [pair[1] for pair in diagonal_pairs]
            return True, diagonal_cells_1, diagonal_cells_2
        
        return False, None, None

    def in_line(self, object_1: GridObject, object_2: GridObject):
        """Identify if object_1 and object_2 can be connected by line."""
        # Get connection cells
        conn_cells_1, conn_cells_2, conn_pairs, distance = self.find_connection_cells(object_1, object_2)
        
        # Check if any connection forms a line (same row or column)
        line_pairs = []
        for pair in conn_pairs:
            coord1, coord2 = pair
            if coord1[0] == coord2[0] or coord1[1] == coord2[1]:
                line_pairs.append(pair)
        
        if line_pairs:
            line_cells_1 = [pair[0] for pair in line_pairs]
            line_cells_2 = [pair[1] for pair in line_pairs]
            return True, line_cells_1, line_cells_2
        
        return False, None, None

    @staticmethod
    def x_alignment(object_1:GridObject, object_2:GridObject):
        """Identify if object_1 and object_2 are aligned in relation to x axis."""
        return object_1.max_i == object_2.max_i and object_1.min_i == object_2.min_i 

    @staticmethod
    def y_alignment(object_1:GridObject, object_2:GridObject):
        """Identify if object_1 and object_2 are aligned in relation to y axis."""
        return object_1.max_j == object_2.max_j and object_1.min_j == object_2.min_j     
            
    def set_relations(self):
        """Set all considered relations."""
        assert self.object_1!=None and self.object_2!=None and self.shape!=None, f"Object_1, Object_2 and grid shape should be specified"       
        triples1 = []
        triples2 = []
        relation_statistics = Counter()

        if self.object_1.colors == self.object_2.colors:
            triples2.append((self.object_2.label, f"same_color", self.object_1.label))
            triples1.append((self.object_1.label, f"same_color", self.object_2.label))
            relation_statistics[f"same_color"] += 1 
            
        if self.object_1.shape == self.object_2.shape:
            triples2.append((self.object_2.label, f"same_shape", self.object_1.label))
            triples1.append((self.object_1.label, f"same_shape", self.object_2.label))
            relation_statistics[f"same_shape"] += 1  
            
        if self.object_1.size == self.object_2.size:
            triples2.append((self.object_2.label, f"same_size", self.object_1.label))
            triples1.append((self.object_1.label, f"same_size", self.object_2.label))
            relation_statistics[f"same_size"] += 1  
            
        rotations = self.rotation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if rotations != []:
            for rotation in rotations:
                triples1.append((self.object_1.label, rotation, self.object_2.label))
                triples2.append((self.object_2.label, rotation, self.object_1.label))         
        
        (i_offset, j_offset) = self.translation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if i_offset != 0 and j_offset != 0:
            triples1.append((self.object_1.label, f"translation_symmetry", self.object_2.label))
            triples2.append((self.object_2.label, f"translation_symmetry", self.object_1.label))          

        in_contour = self.in_contour(self.object_1, self.object_2)
        if in_contour == "object_2":
            triples2.append((self.object_2.label, f"in_contour", self.object_1.label))
            triples1.append((self.object_1.label, f"has_in_contour", self.object_2.label))
            relation_statistics[f"in_contour"] += 1 
        
        if in_contour == "object_1":
            triples1.append((self.object_1.label, f"in_contour", self.object_2.label))
            triples2.append((self.object_2.label, f"has_in_contour", self.object_1.label))
          
        # Check for in_line relation with connection cells
        is_in_line, line_cells_1, line_cells_2 = self.in_line(self.object_1, self.object_2)
        if is_in_line:
            triples1.append((self.object_1.label, f"in_line", self.object_2.label))
            triples2.append((self.object_2.label, f"in_line", self.object_1.label))  
            relation_statistics[f"in_line"] += 1
    
        # Check for in_diagonal relation with connection cells
        is_diagonal, diag_cells_1, diag_cells_2 = self.in_diagonal(self.object_1, self.object_2)
        if is_diagonal:
            triples1.append((self.object_1.label, f"in_diagonal", self.object_2.label))
            triples2.append((self.object_2.label, f"in_diagonal", self.object_1.label))  
            relation_statistics[f"in_diagonal"] += 1

        x_alignment = self.x_alignment(self.object_1, self.object_2)
        y_alignment = self.y_alignment(self.object_1, self.object_2)
        if x_alignment and y_alignment:
            triples1.append((self.object_1.label, f"x_y_aligned_with", self.object_2.label))
            triples2.append((self.object_2.label, f"x_y_aligned_with", self.object_1.label))
            relation_statistics[f"x_y_aligned_with"] += 1  

        else:    
            if self.x_alignment(self.object_1, self.object_2):
                triples1.append((self.object_1.label, f"x_aligned_with", self.object_2.label))
                triples2.append((self.object_2.label, f"x_aligned_with", self.object_1.label))
                relation_statistics[f"x_aligned_with"] += 1 

            
            if self.y_alignment(self.object_1, self.object_2):
                triples1.append((self.object_1.label, f"y_aligned_with", self.object_2.label))
                triples2.append((self.object_2.label, f"y_aligned_with", self.object_1.label))
                relation_statistics[f"y_aligned_with"] += 1 

        return (triples1, triples2), relation_statistics
    


colors_mapping = {
    0: 'black', 0.1: 'blue', 0.2: 'red', 0.3: 'green', 0.4: 'yellow', 
    0.5: 'gray', 0.6: 'magenta', 0.7: 'orange', 0.8: 'sky', 0.9: 'brown', 1: 'white'
}

inverse_colors_mapping = {v:k for k, v in colors_mapping.items()}

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from collections import defaultdict, Counter

@dataclass(frozen=True)
class RelationStatistics:
    """Statistics about spatial relationships between objects."""
    same_color: int
    same_shape: int
    same_size: int
    in_contour: int
    in_line: int
    x_y_aligned_with: int
    x_aligned_with: int
    y_aligned_with: int

@dataclass(frozen=True)
class ObjectChange:
    """Represents a change to a single object."""
    obj_label: str
    change_type: str  # 'added', 'removed', 'modified', 'moved'
    old_coords: Optional[Tuple[Tuple[int, int], ...]] = None
    new_coords: Optional[Tuple[Tuple[int, int], ...]] = None
    old_color: Optional[str] = None
    new_color: Optional[str] = None
    movement_vector: Optional[Tuple[int, int]] = None
    size_change: Optional[int] = None

@dataclass(frozen=True)
class GridChangeSummary:
    """Concise summary of changes between input and output grids."""
    # Grid-level changes
    grid_size_change: Tuple[int, int]  # (rows_delta, cols_delta)
    
    # Object-level changes
    objects_added: int
    objects_removed: int
    objects_moved: int
    objects_recolored: int
    objects_resized: int
    
    # Color changes
    colors_added: Set[str]
    colors_removed: Set[str]
    dominant_color_change: Optional[Tuple[str, str]]  # (from, to)
    
    # Spatial patterns
    common_movement_vector: Optional[Tuple[int, int]]
    common_scaling_factor: Optional[float]
    
    # Shape patterns
    shape_transformations: Dict[str, str]  # old_shape -> new_shape
    
    # Detailed changes (for complex cases)
    object_changes: Tuple[ObjectChange, ...]
    
    # High-level operation summary
    operation_type: str  # 'recolor', 'move', 'copy', 'scale', 'rotate', 'fill', 'complex'
    operation_description: str  # Human-readable description

    median_size_change: Optional[float]
    colors_change: Dict[str, int]  # color -> count_delta
    shape_colors_change: Dict[str, int]  # color -> count_delta
    relation_stats_change: Optional[RelationStatistics]

class GridChangeAnalyzer:
    """Analyzes changes between input and output grids."""
    
    def __init__(self, input_summary: GridSummary, output_summary: GridSummary):
        self.input_summary = input_summary
        self.output_summary = output_summary
        self.input_grid = input_summary.grid
        self.output_grid = output_summary.grid
        
    def analyze_changes(self, level: int = 1) -> GridChangeSummary:
        """Generate comprehensive change summary."""
        input_objects = self.input_summary.repr_levels[level].objects
        output_objects = self.output_summary.repr_levels[level].objects
        
        # Grid size changes
        grid_size_change = self._calculate_grid_size_change()
        
        # Object matching and change detection
        object_changes, change_counts = self._analyze_object_changes(input_objects, output_objects)
        
        # Color analysis
        color_changes = self._analyze_color_changes()
        
        # Pattern detection
        movement_vector = self._detect_common_movement(object_changes)
        scaling_factor = self._detect_common_scaling(object_changes)
        shape_transformations = self._detect_shape_transformations(object_changes)
        
        # Summary statistics changes
        median_size_change = self._calculate_median_size_change(level)
        colors_change = self._calculate_colors_change(level)
        shape_colors_change = self._calculate_shape_colors_change(level)
        relation_stats_change = self._calculate_relation_stats_change(level)
        
        # High-level operation detection
        operation_type, operation_description = self._detect_operation_type(
            object_changes, change_counts, color_changes, movement_vector
        )
        
        return GridChangeSummary(
            grid_size_change=grid_size_change,
            objects_added=change_counts['added'],
            objects_removed=change_counts['removed'],
            objects_moved=change_counts['moved'],
            objects_recolored=change_counts['recolored'],
            objects_resized=change_counts['resized'],
            colors_added=color_changes['added'],
            colors_removed=color_changes['removed'],
            dominant_color_change=color_changes['dominant_change'],
            common_movement_vector=movement_vector,
            common_scaling_factor=scaling_factor,
            shape_transformations=shape_transformations,
            object_changes=tuple(object_changes),
            operation_type=operation_type,
            operation_description=operation_description,
            median_size_change=median_size_change,
            colors_change=colors_change,
            shape_colors_change=shape_colors_change,
            relation_stats_change=relation_stats_change
        )


    def _calculate_median_size_change(self, level) -> Optional[float]:
        """Calculate change in median size between input and output."""
        if (hasattr(self.input_summary.repr_levels[level], 'objects_summary') and 
            hasattr(self.output_summary.repr_levels[level], 'objects_summary')):
            input_median = self.input_summary.repr_levels[level].objects_summary.median_size
            output_median = self.output_summary.repr_levels[level].objects_summary.median_size
            return output_median - input_median
        return None
    
    def _calculate_colors_change(self, level) -> Dict[str, int]:
        """Calculate change in color counts between input and output."""
        colors_change = {}
        if (hasattr(self.input_summary.repr_levels[level], 'objects_summary') and 
            hasattr(self.output_summary.repr_levels[level], 'objects_summary')):
            input_colors = self.input_summary.repr_levels[level].objects_summary.colors
            output_colors = self.output_summary.repr_levels[level].objects_summary.colors

            input_colors = {int(inverse_colors_mapping[k]*10):v for k,v in input_colors.items()} 
            output_colors = {int(inverse_colors_mapping[k]*10):v for k,v in output_colors.items()}       
            
            # Get all colors that appear in either input or output
            all_colors = set(input_colors.keys()) | set(output_colors.keys())
            
            for color in all_colors:
                input_count = input_colors.get(color, 0)
                output_count = output_colors.get(color, 0)
                delta = output_count - input_count
                if delta != 0:
                    colors_change[color] = delta
        
        return colors_change
    
    def _calculate_shape_colors_change(self, level) -> Dict[str, int]:
        """Calculate change in shape color counts between input and output."""
        shape_colors_change = {}
        if (hasattr(self.input_summary.repr_levels[level], 'objects_summary') and 
            hasattr(self.output_summary.repr_levels[level], 'objects_summary')):
            input_shape_colors = self.input_summary.repr_levels[level].objects_summary.shape_colors
            output_shape_colors = self.output_summary.repr_levels[level].objects_summary.shape_colors

            input_shape_colors = {int(inverse_colors_mapping[k]*10):v for k,v in input_shape_colors .items() if k!='multicolor'} | {'multicolor': input_shape_colors['multicolor']}
            output_shape_colors = {int(inverse_colors_mapping[k]*10):v for k,v in output_shape_colors.items() if k!='multicolor'} | {'multicolor': output_shape_colors ['multicolor']}
            # Get all colors that appear in either input or output
            all_colors = set(input_shape_colors.keys()) | set(output_shape_colors.keys())
            
            for color in all_colors:
                input_count = input_shape_colors.get(color, 0)
                output_count = output_shape_colors.get(color, 0)
                delta = output_count - input_count
                if delta != 0:
                    shape_colors_change[color] = delta
        
        return shape_colors_change
    
    def _calculate_relation_stats_change(self,level) -> Optional[RelationStatistics]:
        """Calculate change in relation statistics between input and output."""
        if (hasattr(self.input_summary.repr_levels[level], 'relation_statistics') and 
            hasattr(self.output_summary.repr_levels[level], 'relation_statistics')):
            input_stats = self.input_summary.repr_levels[level].relation_statistics
            output_stats = self.output_summary.repr_levels[level].relation_statistics
            
            return RelationStatistics(
                same_color=output_stats.same_color - input_stats.same_color,
                same_shape=output_stats.same_shape - input_stats.same_shape,
                same_size=output_stats.same_size - input_stats.same_size,
                in_contour=output_stats.in_contour - input_stats.in_contour,
                in_line=output_stats.in_line - input_stats.in_line,
                x_y_aligned_with=output_stats.x_y_aligned_with - input_stats.x_y_aligned_with,
                x_aligned_with=output_stats.x_aligned_with - input_stats.x_aligned_with,
                y_aligned_with=output_stats.y_aligned_with - input_stats.y_aligned_with
            )
        return None
    
    def _calculate_grid_size_change(self) -> Tuple[int, int]:
        """Calculate change in grid dimensions."""
        input_shape = self.input_summary.shape
        output_shape = self.output_summary.shape
        return (output_shape[0] - input_shape[0], output_shape[1] - input_shape[1])
    
    def _analyze_object_changes(self, input_objects, output_objects) -> Tuple[List[ObjectChange], Dict[str, int]]:
        """Analyze changes in objects between input and output."""
        object_changes = []
        change_counts = {'recolored':0, 'moved':0, 'resized':0, 'removed':0, 'added':0}
        
        # Create mappings for easier comparison
        input_by_coords = {obj.coords: obj for obj in input_objects}
        output_by_coords = {obj.coords: obj for obj in output_objects}
        
        # Track matched objects to identify additions/removals
        matched_input = set()
        matched_output = set()
        
        # Find exact matches (same coordinates)
        for coords, input_obj in input_by_coords.items():
            if coords in output_by_coords:
                output_obj = output_by_coords[coords]
                matched_input.add(input_obj.label)
                matched_output.add(output_obj.label)
                
                # Check for color changes
                if input_obj.colors != output_obj.colors:
                    change_counts['recolored'] += 1
                    object_changes.append(ObjectChange(
                        obj_label=input_obj.label,
                        change_type='modified',
                        old_coords=input_obj.coords,
                        new_coords=output_obj.coords,
                        old_color=input_obj.colors[0] if input_obj.colors else None,
                        new_color=output_obj.colors[0] if output_obj.colors else None
                    ))
        
        # Find moved objects (same shape/color, different position)
        unmatched_input = [obj for obj in input_objects if obj.label not in matched_input]
        unmatched_output = [obj for obj in output_objects if obj.label not in matched_output]
        
        for input_obj in unmatched_input:
            for output_obj in unmatched_output:
                if (input_obj.shape == output_obj.shape and 
                    input_obj.colors == output_obj.colors and
                    input_obj.size == output_obj.size):
                    
                    # Calculate movement vector
                    input_center = input_obj.center
                    output_center = output_obj.center
                    movement = (output_center[0] - input_center[0], 
                              output_center[1] - input_center[1])
                    
                    change_counts['moved'] += 1
                    object_changes.append(ObjectChange(
                        obj_label=input_obj.label,
                        change_type='moved',
                        old_coords=input_obj.coords,
                        new_coords=output_obj.coords,
                        movement_vector=movement
                    ))
                    
                    matched_input.add(input_obj.label)
                    matched_output.add(output_obj.label)
                    break
        
        # Find resized objects (same position/color, different size)
        remaining_input = [obj for obj in input_objects if obj.label not in matched_input]
        remaining_output = [obj for obj in output_objects if obj.label not in matched_output]
        
        for input_obj in remaining_input:
            for output_obj in remaining_output:
                if (input_obj.center == output_obj.center and
                    input_obj.colors == output_obj.colors and
                    input_obj.shape == output_obj.shape):
                    
                    size_change = output_obj.size - input_obj.size
                    change_counts['resized'] += 1
                    object_changes.append(ObjectChange(
                        obj_label=input_obj.label,
                        change_type='modified',
                        old_coords=input_obj.coords,
                        new_coords=output_obj.coords,
                        size_change=size_change
                    ))
                    
                    matched_input.add(input_obj.label)
                    matched_output.add(output_obj.label)
                    break
        
        # Remaining unmatched objects are additions/removals
        final_unmatched_input = [obj for obj in input_objects if obj.label not in matched_input]
        final_unmatched_output = [obj for obj in output_objects if obj.label not in matched_output]
        
        for obj in final_unmatched_input:
            change_counts['removed'] += 1
            object_changes.append(ObjectChange(
                obj_label=obj.label,
                change_type='removed',
                old_coords=obj.coords
            ))
        
        for obj in final_unmatched_output:
            change_counts['added'] += 1
            object_changes.append(ObjectChange(
                obj_label=obj.label,
                change_type='added',
                new_coords=obj.coords
            ))
        
        return object_changes, dict(change_counts)
    
    def _analyze_color_changes(self) -> Dict:
        """Analyze color changes in the grids."""
        input_colors = set(np.unique(self.input_grid))
        output_colors = set(np.unique(self.output_grid))
        
        # Convert to color names
        input_color_names = {str(int(c*10)) for c in input_colors}
        output_color_names = {str(int(c*10)) for c in output_colors}
        
        added_colors = output_color_names - input_color_names
        removed_colors = input_color_names - output_color_names
        
        # Find dominant color change
        input_counts = Counter([str(int(el*10)) for el in self.input_grid.flatten() if el != self.input_summary.font_color])
        output_counts = Counter([str(int(el*10)) for el in self.output_grid.flatten() if el != self.output_summary.font_color])
        
        dominant_change = None
        if len(input_counts) > 0 and len(output_counts) > 0:
            input_dominant = f'({str(int(input_counts.most_common(2)[0][0]*10))})'
            output_dominant = f'({str(int(input_counts.most_common(2)[0][0]*10))})'
            if input_dominant != output_dominant:
                dominant_change = (input_dominant, output_dominant)
        
        return {
            'added': added_colors,
            'removed': removed_colors,
            'dominant_change': dominant_change
        }
    
    def _detect_common_movement(self, object_changes: List[ObjectChange]) -> Optional[Tuple[int, int]]:
        """Detect if there's a common movement pattern."""
        movements = [change.movement_vector for change in object_changes 
                    if change.movement_vector is not None]
        
        if not movements:
            return None
        
        # Find most common movement
        movement_counts = Counter(movements)
        if len(movement_counts) > 0:
            most_common = movement_counts.most_common(1)[0]
            if most_common[1] >= len(movements) * 0.5:  # At least 50% of movements
                return most_common[0]
        
        return None
    
    def _detect_common_scaling(self, object_changes: List[ObjectChange]) -> Optional[float]:
        """Detect common scaling factor."""
        size_changes = [change.size_change for change in object_changes 
                       if change.size_change is not None and change.size_change != 0]
        
        if not size_changes:
            return None
        
        # Simple scaling detection - could be more sophisticated
        if len(set(size_changes)) == 1:
            return size_changes[0]
        
        return None
    
    def _detect_shape_transformations(self, object_changes: List[ObjectChange]) -> Dict[str, str]:
        """Detect shape transformation patterns."""
        transformations = {}
        # This would need access to shape information from objects
        # Simplified implementation
        return transformations
    
    def _detect_operation_type(self, object_changes: List[ObjectChange], 
                              change_counts: Dict[str, int],
                              color_changes: Dict,
                              movement_vector: Optional[Tuple[int, int]]) -> Tuple[str, str]:
        """Detect the high-level operation type and generate description."""
        
        total_changes = sum(change_counts.values())
        
        if total_changes == 0:
            return 'none', 'No changes detected'
        
        # Recoloring operation
        if change_counts.get('recolored', 0) > 0 and total_changes == change_counts['recolored']:
            if color_changes['dominant_change']:
                old_color, new_color = color_changes['dominant_change']
                return 'recolor', f'All objects recolored from {old_color} to {new_color}'
            return 'recolor', 'Objects recolored'
        
        # Movement operation
        if change_counts.get('moved', 0) > 0 and movement_vector:
            direction = self._describe_movement(movement_vector)
            return 'move', f'Objects moved {direction} by {movement_vector}'
        
        # Addition operation
        if change_counts.get('added', 0) > 0 and change_counts.get('removed', 0) == 0:
            return 'add', f'{change_counts["added"]} objects added'
        
        # Removal operation
        if change_counts.get('removed', 0) > 0 and change_counts.get('added', 0) == 0:
            return 'remove', f'{change_counts["removed"]} objects removed'
        
        # Copy operation (more additions than removals)
        if change_counts.get('added', 0) > change_counts.get('removed', 0):
            return 'copy', f'Objects copied ({change_counts["added"]} added, {change_counts["removed"]} removed)'
        
        # Scaling/resizing
        if change_counts.get('resized', 0) > 0:
            return 'scale', f'{change_counts["resized"]} objects resized'
        
        # Complex transformation
        return 'complex', f'Complex transformation: {total_changes} total changes'
    
    def _describe_movement(self, movement_vector: Tuple[int, int]) -> str:
        """Convert movement vector to human-readable direction."""
        di, dj = movement_vector
        
        if di == 0 and dj == 0:
            return 'no movement'
        elif di == 0:
            return 'right' if dj > 0 else 'left'
        elif dj == 0:
            return 'down' if di > 0 else 'up'
        else:
            vertical = 'down' if di > 0 else 'up'
            horizontal = 'right' if dj > 0 else 'left'
            return f'{vertical}-{horizontal}'

def summarize_training_pair(input_grid_summary: GridSummary, output_grid_summary: GridSummary, 
                            levels: List[int] = [1]) -> str:
    """Generate a concise text summary of changes for LLM consumption."""
    analyzer = GridChangeAnalyzer(input_grid_summary, output_grid_summary)
    summary = analyzer.analyze_changes(level=levels[0])
    
    lines = []
    lines.append(f"Operation: {summary.operation_description}")
    
    if summary.grid_size_change != (0, 0):
        lines.append(f"Grid size changed by {summary.grid_size_change}")
    
    if summary.objects_added > 0:
        lines.append(f"Added {summary.objects_added} objects")
    
    if summary.objects_removed > 0:
        lines.append(f"Removed {summary.objects_removed} objects")
    
    if summary.objects_moved > 0:
        if summary.common_movement_vector:
            direction = analyzer._describe_movement(summary.common_movement_vector)
            lines.append(f"Moved {summary.objects_moved} objects {direction}")
        else:
            lines.append(f"Moved {summary.objects_moved} objects")
    
    if summary.objects_recolored > 0:
        if summary.dominant_color_change:
            old_color, new_color = summary.dominant_color_change
            lines.append(f"Recolored {summary.objects_recolored} objects from {old_color} to {new_color}")
        else:
            lines.append(f"Recolored {summary.objects_recolored} objects")
    
    if summary.colors_added:
        lines.append(f"New colors: {', '.join(summary.colors_added)}")
    
    if summary.colors_removed:
        lines.append(f"Removed colors: {', '.join(summary.colors_removed)}")

    if summary.median_size_change is not None and summary.median_size_change != 0:
        lines.append(f"Median size changed by {summary.median_size_change}")
    
    if summary.colors_change:
        color_changes_desc = []
        for color, delta in summary.colors_change.items():
            if delta > 0:
                color_changes_desc.append(f"+{delta} color ({color})")
            else:
                color_changes_desc.append(f"+{delta} color ({color})")
        if color_changes_desc:
            lines.append(f"Color count changes: {', '.join(color_changes_desc)}")
    
    if summary.shape_colors_change:
        shape_color_changes_desc = []
        for color, delta in summary.shape_colors_change.items():
            if delta > 0:
                shape_color_changes_desc.append(f"+{delta} shapes with color ({color})")
            else:
                shape_color_changes_desc.append(f"+{delta} shapes with color ({color})")
        if shape_color_changes_desc:
            lines.append(f"Shape color changes: {', '.join(shape_color_changes_desc)}")
    
    if summary.relation_stats_change:
        relation_changes = []
        stats = summary.relation_stats_change
        if stats.same_color != 0:
            relation_changes.append(f"same_color: {stats.same_color:+d}")
        if stats.same_shape != 0:
            relation_changes.append(f"same_shape: {stats.same_shape:+d}")
        if stats.same_size != 0:
            relation_changes.append(f"same_size: {stats.same_size:+d}")
        if stats.in_contour != 0:
            relation_changes.append(f"in_contour: {stats.in_contour:+d}")
        if stats.in_line != 0:
            relation_changes.append(f"in_line: {stats.in_line:+d}")
        if stats.x_y_aligned_with != 0:
            relation_changes.append(f"x_y_aligned: {stats.x_y_aligned_with:+d}")
        if stats.x_aligned_with != 0:
            relation_changes.append(f"x_aligned: {stats.x_aligned_with:+d}")
        if stats.y_aligned_with != 0:
            relation_changes.append(f"y_aligned: {stats.y_aligned_with:+d}")
        
        if relation_changes:
            lines.append(f"Relation changes: {', '.join(relation_changes)}")
    
    return "; ".join(lines)

def get_rotations(coords: List[tuple]) -> List[List[tuple]]:
    """
    Generate all possible 90-degree rotations of a set of coordinates.
    
    Args:
        coords: List of (x, y) coordinate tuples
    
    Returns:
        List of lists of coordinate tuples, each representing a rotation
    """
    if not coords:
        return []
    
    # Sort coordinates for consistency
    coords = sorted(coords, key=lambda x: (x[1], x[0]))
    
    # Get reference point (top-left)
    ref_x, ref_y = coords[0]
    
    # Normalize coordinates relative to reference point
    normalized = [(x - ref_x, y - ref_y) for x, y in coords]
    
    # Find the size of the bounding box
    max_x = max(x for x, y in normalized)
    max_y = max(y for x, y in normalized)
    
    rotations = []
    # Original orientation
    rotations.append(coords.copy())
    
    # 90 degrees clockwise - (x, y) -> (y, -x + max_x)
    rot_90 = [(ref_x + y, ref_y + (max_x - x)) for x, y in normalized]
    rotations.append(sorted(rot_90, key=lambda x: (x[1], x[0])))
    
    # 180 degrees - (x, y) -> (-x + max_x, -y + max_y)
    rot_180 = [(ref_x + (max_x - x), ref_y + (max_y - y)) for x, y in normalized]
    rotations.append(sorted(rot_180, key=lambda x: (x[1], x[0])))
    
    # 270 degrees clockwise - (x, y) -> (-y + max_y, x)
    rot_270 = [(ref_x + (max_y - y), ref_y + x) for x, y in normalized]
    rotations.append(sorted(rot_270, key=lambda x: (x[1], x[0])))
    
    return rotations

def calculate_adjacency_positions(obj: 'GridObject') -> List[tuple]:
    """
    Calculate all possible positions adjacent to an object where another object could be placed.
    
    Args:
        obj: GridObject to find adjacency positions for
    
    Returns:
        List of (x, y) coordinate tuples representing adjacent positions
    """
    adjacent_positions = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for x, y in obj.coords:
        for dx, dy in directions:
            adjacent_pos = (x + dx, y + dy)
            if adjacent_pos not in obj.coords:
                adjacent_positions.add(adjacent_pos)
    return list(adjacent_positions)

def count_holes(obj: 'GridObject') -> int:
    """
    Count the total number of holes in a GridObject.
    
    Args:
        obj: GridObject to count holes for
    
    Returns:
        Total number of inner and outer holes
    """
    if obj.shape in ['inner_hole', 'outer_hole']:
        return 0
    return len(obj.inner_holes) + len(obj.outer_holes)

def check_intersection(coords1: List[tuple], coords2: List[tuple]) -> bool:
    """
    Check if two sets of coordinates intersect.
    
    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates
    
    Returns:
        True if there's an intersection, False otherwise
    """
    return bool(set(coords1).intersection(set(coords2)))

def evaluate_match_configuration(obj1: 'GridObject', obj2: 'GridObject', 
                                 position: tuple, rotation_idx: int,
                                 all_grid_objects: List['GridObject'], 
                                 grid_shape:tuple, font_color,
                                 grid) -> Dict:
    """
    Evaluate a potential match configuration between two objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        position: Position to place obj2 (reference point)
        rotation_idx: Index of rotation to use for obj2
        all_grid_objects: List of all objects on the grid to check for intersections
    
    Returns:
        Dictionary with evaluation metrics:
        - valid: Whether the configuration is valid
        - hole_reduction: Reduction in holes (higher is better)
        - compactness: Measure of how compact the resulting shape is
    """
    # Get the rotated coordinates for obj2
    rotation_coords = get_rotations(obj2.coords)[rotation_idx]
    
    # Calculate the offset based on position and reference point 
    offset_x = position[0] - rotation_coords[0][0]
    offset_y = position[1] - rotation_coords[0][1]
    
    # Apply offset to all coordinates in the rotation
    shifted_coords = [(x + offset_x, y + offset_y) for x, y in rotation_coords]
    
    # Check if the shifted obj2 intersects with obj1 coords
    if not check_intersection(obj1.coords, shifted_coords):
        # Check if shifted obj2 intersects with any other grid objects
        for other_obj in all_grid_objects:
            # Fix the object equality check - avoid direct comparison
            if id(other_obj) != id(obj1) and id(other_obj) != id(obj2):
                if check_intersection(other_obj.coords, shifted_coords):
                    return {"valid": False, "hole_reduction": 0, "compactness": 0}
        
        # Create a temporary merged object to evaluate
        merged_coords = tuple(set(obj1.coords + tuple(shifted_coords)))
        merged_obj = GridObject(
            shape="complex",
            coords=merged_coords,
            color=obj1.color_numbers+obj2.color_numbers,  # Assume we keep the color of obj1
            label=f"merged_{obj1.label}_{obj2.label}",
            grid_shape=grid_shape,  # Assume we keep the positioning of obj1
            font_color=font_color,
            grid=grid
        )
        
        # Calculate hole reduction
        original_holes = count_holes(obj1) + count_holes(obj2)
        merged_holes = count_holes(merged_obj)
        hole_reduction = original_holes - merged_holes
        
        # Calculate compactness (area of bounding box / number of cells)
        merged_area = merged_obj.hor_size * merged_obj.vert_size
        merged_cells = len(merged_coords)
        compactness = merged_cells / merged_area if merged_area > 0 else 0
        
        return {
            "valid": True,
            "hole_reduction": hole_reduction,
            "compactness": compactness,
            "shifted_coords": shifted_coords,
            "merged_obj": merged_obj
        }
    
    return {"valid": False, "hole_reduction": 0, "compactness": 0}

def find_best_object_match(obj1: 'GridObject', obj2: 'GridObject', 
                           all_grid_objects: List['GridObject'], grid_shape:tuple,
                           font_color, grid) -> Dict:
    """
    Find the best match configuration between two grid objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        all_grid_objects: List of all objects on the grid
    
    Returns:
        Dictionary with the best match configuration, or None if no valid match exists
    """
    best_match = None
    best_score = -float('inf')
    
    # Get all possible adjacent positions for obj1
    adjacent_positions = calculate_adjacency_positions(obj1)
    
    # Try all rotations of obj2
    for rotation_idx in range(4):
        # Try placing obj2 at each adjacent position of obj1
        for position in adjacent_positions:
            match_config = evaluate_match_configuration(
                obj1, obj2, position, rotation_idx, all_grid_objects, grid_shape, font_color, grid
            )
            
            if match_config["valid"]:
                # Calculate a score based on hole reduction and compactness
                # Weight hole reduction higher than compactness
                score = match_config["hole_reduction"] * 10 + match_config["compactness"]
                
                if score > best_score:
                    best_score = score
                    best_match = match_config
                    best_match["rotation_idx"] = rotation_idx
                    best_match["position"] = position
    
    return best_match

def merge_objects(grid:np.ndarray, obj1:GridObject, obj2:GridObject, match_config:Dict, font_color:float):
    """
    Merge two objects based on a match configuration.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        match_config: Match configuration from find_best_object_match
        grid: The current grid array
    
    Returns:
        Tuple of (merged_object, updated_grid)
    """
    # Create the merged object
    shifted_coords = match_config["shifted_coords"]

    height, width = grid.shape
    # Update the grid
    updated_grid = grid.copy()
    
    for x, y in obj2.coords:
        updated_grid[x, y] = font_color
    
    # Add the merged object to the grid
    for x, y in shifted_coords:
        if x < height and y < width:
            updated_grid[x, y] = obj2.color_numbers[0]  # Use the color of obj2

    updated_obj = copy(obj2)
    updated_obj.reinit_obj(shifted_coords, updated_grid)
    
    return updated_obj, updated_grid

def find_most_probable_merge(grid: np.ndarray, obj1: 'GridObject', obj2: 'GridObject', 
                           all_grid_objects: List['GridObject'], font_color) -> Dict:
    """
    Find the most probable merge configuration between two specific objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        all_grid_objects: List of all GridObjects on the grid (to check for intersections)
        grid: The current grid array
    
    Returns:
        Dictionary with the best match configuration, or None if no valid match exists
    """
    # Skip if either object is a hole
    if obj1.shape in ['inner_hole', 'outer_hole'] or obj2.shape in ['inner_hole', 'outer_hole']:
        return None
    
    # Use a filtered list of grid objects that excludes the two objects we're working with
    # to avoid the equality comparison issue
    filtered_grid_objects = [obj for obj in all_grid_objects if id(obj) != id(obj1) and id(obj) != id(obj2)]
    
    # Try matching in both directions
    match = find_best_object_match(obj1, obj2, [obj1, obj2] + filtered_grid_objects, grid.shape, font_color, grid)
    
    best_match = None
    best_score = -float('inf')
    
    if match and match["valid"]:
        match["obj1"] = obj1
        match["obj2"] = obj2
        match["score"] = match["hole_reduction"] * 10 + match["compactness"]
        if match["score"] > best_score:
            best_score = match["score"]
            best_match = match
    
    return best_score