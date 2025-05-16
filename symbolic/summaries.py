import typing
from typing import List
import numpy as np
from copy import copy
from collections import defaultdict, Counter
from itertools import product
from rl.ARC_task import ARCTask, ARCSubtask
from symbolic.objects_analysis import GridObject, ObjectsFilter, RelationAnalyzer
from symbolic.utils import find_upper_left_corner, coords_transform, count_unique_cells, dict_to_list
from symbolic.patterns import generate_patterns, find_connected_components_with_color, find_connected_components_excluding_colors
from llm.prompts import COLOR_MAPPING

colors_mapping = {
    0: 'black', 0.1: 'blue', 0.2: 'red', 0.3: 'green', 0.4: 'yellow', 
    0.5: 'gray', 0.6: 'magenta', 0.7: 'orange', 0.8: 'sky', 0.9: 'brown', 1: 'white'
}
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
class GridSummary():
    """Class for creating summary for a given grid."""
    def __init__(self, grid:np.array, shape:tuple, shape_types=None):
        self.grid = grid
        self.shape = shape
        self.shape_types = ['line' ,'rectangle', 'diagonal', 'l_shape', 't_shape', 's_shape', 'tv_shape', 
                            'hs_shape', 'cross', 'flower', 'markup_matrix','markup_line', 'cell', 'complex'] if shape_types == None else shape_types
        self.patterns = generate_patterns(self.shape)
        self.grid_corners = self.define_grid_corners()
        self.objects_dict = {}
        self.initial_objects = self.retrieve_objects(self.grid, self.patterns, self.shape)
        self.connected_components = self.retrieve_connected_components(self.grid) 
        self.font_segments = find_connected_components_with_color(self.grid, target_color=0.0)
        self.relations_for_stats = ["same_color", "same_shape", "same_size", "in_contour", 
                                    "in_line", "x_y_aligned_with", "x_aligned_with", "y_aligned_with"]
        self.repr_levels = self.set_repr_levels()

    def define_grid_corners(self):
        grid_size = self.shape
        ul = find_upper_left_corner(grid_size)
        bl = (ul[0]+grid_size[0]-1, ul[1])
        ur = (ul[0], ul[1]+grid_size[1]-1)
        br = (ul[0]+grid_size[0]-1, ul[1]+grid_size[1]-1)
        return (ul, bl, ur, br)

    def set_repr_levels(self):
        repr_levels = {}
        level1_init_objects = copy(self.initial_objects)
        init_objects = copy(self.initial_objects)
        init_objects['complex'] = [obj for obj in init_objects['complex'] if obj.color_homo]
        for level in range(1, 5):
            if level == 1:
                repr_levels[level] = self.process_repr_level(level1_init_objects, level)
            else: 
                repr_levels[level] = self.process_repr_level(init_objects, level)
        return repr_levels

    def process_repr_level(self, init_objects, level):
        level_objects = self.filter_objects(init_objects, level)
        level_objects_summary = self.create_objects_summary(level_objects)
        level_triples, level_relation_statistics = self.set_relations(level_objects)
        level_relations_summary = self.create_relations_summary(level_objects, level_relation_statistics)
        return {f'objects':dict_to_list(copy(level_objects)), f'objects_summary':copy(level_objects_summary), 
                f'triples':copy(level_triples), f'relation_statistics':copy(level_relation_statistics),
                f'relations_summary': copy(level_relations_summary)}
        
    def retrieve_objects(self, grid:np.array, patterns:typing.Dict['str', List[List[List[tuple]]]], shape:tuple)->typing.Dict[str, List[GridObject]]:
        """Retrieve all possible objects from the grid and return corresponding GridObject instances."""
        objects = defaultdict(list)
        candidate = False # flag indicating existance of candidate figure
        used_coordinates = [] # save occupied cells
        for k, v in patterns.items(): # iterating over shapes
            shape_patterns = v
            for idx, pattern_list in enumerate(shape_patterns):
                for pattern in pattern_list:
                    i, j = coords_transform(pattern) # transform list of tuples into two lists for i and j coordinates
                    retrieval = set(grid[i, j]) # extract cells colors with lists of coordinates and keep only unique 
                    if len(retrieval) > 1: # if colors more than 1 - not a candidate
                        break
                    else:
                        color = retrieval.pop()
                        if color != 0 and (k != 'diagonal' or count_unique_cells(k, pattern, used_coordinates)) > 0:
                            label = f'{k}_{idx}'
                            obj = GridObject(k, pattern, [color], label, self.shape) # otherwise create candidate object
                            self.objects_dict[label] = obj
                            used_coordinates.extend(pattern) # save occupied cells 
                            candidate = True
                if candidate:
                    objects[k].append(copy(obj))
                    candidate = False
        used_coordinates = set(used_coordinates) # keep only unique coordinates    
        ul = find_upper_left_corner(shape)
        all_coordinates = set(product(range(ul[0], ul[0]+shape[0]), range(ul[1], ul[1]+shape[1])))
        cells_coordinates = list(all_coordinates.difference(used_coordinates))
        for idx, cell in enumerate(cells_coordinates): # if some cell is not belong to some figure - create cell object
            color = grid[cell]
            if color != 0:
                label = f'cell_{idx}'
                obj = GridObject('cell', [cell], [grid[cell]], label, self.shape)
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
                obj = GridObject('complex', comp, color_numbers, label, self.shape) # otherwise create candidate object
                components[label] = obj
                comp_idx += 1
        for i in range(1, 10):
            col = i / 10
            homo_components = find_connected_components_with_color(grid, target_color=col)
            for comp in homo_components:
                if comp not in heter_components:
                    if len(comp) > 1:
                        label = f'complex_{comp_idx}'
                        obj = GridObject('complex', comp, [col], label, self.shape) # otherwise create candidate object
                        components[label] = obj
                        comp_idx += 1 
        self.initial_objects['complex'] = list(components.values())
        return components

    @staticmethod
    def filter_objects(objects, repr_level:int=1):
        """Apply ObjectsFilter class for filtering out possibly umimportant objects."""
        return ObjectsFilter(objects, repr_level).filter_objects()
    
    @staticmethod 
    def calculate_distance(obj_1, obj_2):
        """Calculate distance between objects."""
        i_dist = min(abs(obj_1.max_i - obj_2.max_i), abs(obj_1.max_i - obj_2.min_i), abs(obj_1.min_i - obj_2.max_i), abs(obj_1.min_i - obj_2.min_i)) 
        j_dist = min(abs(obj_1.max_j - obj_2.max_j), abs(obj_1.max_j - obj_2.min_j), abs(obj_1.min_j - obj_2.max_j), abs(obj_1.min_j - obj_2.min_j)) 
        return min(i_dist, j_dist)

    def create_objects_summary(self, objects):
        """Create a summary for grid objects to get aggregate information about their shapes, sizes, colors.""" 
        size2shape = defaultdict(list)
        shape2size = {}
        hor_size2shape = defaultdict(list)
        shape2hor_size = {}
        vert_size2shape = defaultdict(list)
        shape2vert_size = {}
        shapes = {shape:0 for shape in self.shape_types}
        shape_colors = {colors_mapping[i/10]:0 for i in range(10)}
        colors = {colors_mapping[i/10]:0 for i in range(10)}
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
                    obj.structure_analysis()
                    obj_summary = obj.objects_summary
                    shape_colors = {c:shape_colors[c]+obj_summary['shape_colors'][c] for c in list(colors_mapping.values())[:-1]}
                    colors = {c:colors[c]+obj_summary['colors'][c] for c in list(colors_mapping.values())[:-1]}
        # size
        sorted_keys = sorted(list(size2shape.keys()), reverse=True)
        size2shape = {k:size2shape[k] for k in sorted_keys}
        shape2size_values = list(shape2size.values())
        n_sizes = len(shape2size_values)
        size2description = {shape2size_values[i]:f'{i+1} by size' for i in range(n_sizes)}
        # horizontal size
        sorted_keys = sorted(list(hor_size2shape.keys()), reverse=True)
        hor_size2shape = {k:hor_size2shape[k] for k in sorted_keys}
        shape2hor_size_values = list(shape2hor_size.values())
        n_sizes = len(shape2hor_size_values)
        hor_size2description = {shape2hor_size_values[i]:f'{i+1} by horizontal size' for i in range(n_sizes)}
        # vertical size
        sorted_keys = sorted(list(vert_size2shape.keys()), reverse=True)
        vert_size2shape = {k:vert_size2shape[k] for k in sorted_keys}
        shape2vert_size_values = list(shape2vert_size.values())
        n_sizes = len(shape2vert_size_values)
        vert_size2description = {shape2vert_size_values[i]:f'{i+1} by vertical size' for i in range(n_sizes)}
        # shape colors
        shape_colors2freq_values = sorted(list(shape_colors.values()), reverse=True) 
        shape_colors_reversed = {v:k for k, v in shape_colors.items()}
        shape_color2description = {shape_colors_reversed[shape_colors2freq_values[i]]:f'{i+1} by shape color freq' for i in range(10)}
        # colors
        color2freq_values = sorted(list(colors.values()), reverse=True)
        colors_reversed = {v:k for k, v in colors.items()}
        color2description = {colors_reversed[color2freq_values[i]]:f'{i+1} by color freq' for i in range(10)} 
        # size stats
        mean_size = 0
        median_size = 0
        if len(shape2size_values) > 0:
            mean_size = np.mean(shape2size_values)
            median_size = np.median(shape2size_values)
        
        objects_summary = {'size2shape':size2shape, 'shape2size':shape2size, 'mean_size':mean_size, 
                           'median_size':median_size, 'hor_size2shape':hor_size2shape, 'shape2hor_size':shape2hor_size,
                           'vert_size2shape':vert_size2shape, 'shape2vert_size':shape2vert_size,                         
                           'shapes':shapes, 'shape_colors':shape_colors, 'colors':colors,
                           'shape_hor_size_description':hor_size2description, 'shape_vert_size_description':vert_size2description,
                           'shape_size_description':size2description, 'shape_color_description':shape_color2description,
                           'color_description':color2description,      
                           }
        return objects_summary
 
    def create_relations_summary(self, objects, relation_statistics):
        """Create a summary for relation between grid objects to get useful information for reasoning.""" 
        relations_summary = relation_statistics
        objects_properties = {'symmetry':0}
        for k, v in objects.items():
            for idx, obj in enumerate(v):
                if obj.symmetry != [] and obj.symmetry != 'assymetry':
                    objects_properties['symmetry'] += 1
        relations_summary.update(objects_properties)       
        return relations_summary
    
    def set_relations(self, objects):
        """Iterate over objects to identify relations between them.""" 
        all_triples = []
        relation_statistics = Counter(self.relations_for_stats)
        distances = defaultdict(lambda: 0)
        all_objects = dict_to_list(objects)
        for idx, obj_1 in enumerate(all_objects):
            for obj_2 in all_objects[idx+1:]: 
                if obj_2.label not in obj_1.relations.keys():
                    analyzer = RelationAnalyzer(obj_1, obj_2, self.shape)
                    triples = analyzer.triples
                    relation_counter = analyzer.relation_counter
                    all_triples.extend(triples)
                    relation_statistics.update(relation_counter)
                    # distance calculation
                    distance = self.calculate_distance(obj_1, obj_2)
                    obj_1.distances[obj_2.label] = distance
                    obj_2.distances[obj_1.label] = distance
                    distances[f'{obj_1.label}-{obj_2.label}'] = obj_1.distances[obj_2.label]
        return all_triples, relation_statistics