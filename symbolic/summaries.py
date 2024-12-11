import typing
from typing import List
import numpy as np
import copy
from collections import defaultdict, Counter
from itertools import product
from rl.ARC_task import ARCTask, ARCSubtask
from symbolic.objects_analysis import GridObject, ObjectCombiner, ObjectsFilter, RelationAnalyzer
from utils.plotting import plot_grid
from symbolic.utils import find_upper_left_corner, coords_transform, count_unique_cells, dict_to_list
from symbolic.patterns import generate_patterns
from llm.prompts import COLOR_MAPPING

class TaskSummary():
    """Class for inferencing information about a task."""
    def __init__(self, task:ARCTask, patterns:dict):
        self.patterns = patterns
        self.train_subtasks = task.subtasks[:-1]
        self.test_subtask = task.subtasks[-1]
        self.train_subtasks_summaries = []
        self.grid_shapes_ratios = self.compare_grid_shapes()
        self.need_resize = bool(1-all(self.grid_shapes_ratios))
    
    def compare_grid_shapes(self):
        """Compare the shapes of the grids in the train subtasks with the test subtask."""
        ratios = []
        for subtask in self.train_subtasks:
            ratios.append(subtask.train_inp_shape/subtask.train_out_shape)
        return ratios
   
    def obtain_summaries(self):
        """Obtain the summaries of the train subtasks."""
        for subtask in self.train_subtasks:
            self.train_subtasks_summaries.append(SubtaskSummary(subtask, self.patterns))

class SubtaskSummary():
    """Class for inferencing information about a subtask."""
    def __init__(self, subtask:ARCSubtask, train:bool=True, need_resize=None):
        self.subtask = subtask
        self.subtask_label = self.subtask_label
        self.need_resize = need_resize
        if train:
            self.out_grid_summary = GridSummary(subtask.train_out)
        self.grids_intersection = self.analyze_grids_intersection()
        self.inp_grid_summary = GridSummary(self.subtask.train_inp, self.subtask.train_inp_shape, self.subtask_label)

    def analyze_grids_intersection(self):
        pass

class GridSummary():
    """Class for creating summary for a given grid."""
    def __init__(self, grid:np.array, shape:tuple):
        self.grid = grid
        self.shape = shape
        self.patterns = generate_patterns(shape)
        self.grid_corners = self.define_grid_corners()
        self.objects_dict = {}
        self.initial_objects = self.retrieve_objects(self.grid, self.patterns, self.shape)
        self.objects = self.filter_objects()
        # self.complex_objects = self.construct_complex_objects() 
        self.objects_summary = self.create_objects_summary()
        self.relations_for_stats = ["same_color", "same_shape", "same_size", "in_contour", "in_line", "x_y_aligned_with", "x_aligned_with", "y_aligned_with"]
        self.triples, self.relation_statistics = self.set_relations()
        self.relations_summary = self.create_relations_summary()

    def define_grid_corners(self):
        grid_size = self.shape
        ul = find_upper_left_corner(grid_size)
        bl = (ul[0]+grid_size[0]-1, ul[1])
        ur = (ul[0], ul[1]+grid_size[1]-1)
        br = (ul[0]+grid_size[0]-1, ul[1]+grid_size[1]-1)
        return (ul, bl, ur, br)

    def define_object_positioning(self, object_coords:List[tuple]):
        """Identify if an object is located at spicific position on the grid."""
        positioning = []
        ul, bl, ur, br = self.grid_corners
        if ul in object_coords:
            positioning.append('in_upper_left_corner')
        if bl in object_coords:
            positioning.append('in_bottom_left_corner')  
        if ur in object_coords:
            positioning.append('in_upper_right_corner')             
        if br in object_coords:
            positioning.append('in_bottom_left_corner')
        list_i, list_j = coords_transform(object_coords)  
        if ul[0] in list_i:
            positioning.append('at_upper_edge')
        if ul[1] in list_j:
            positioning.append('at_left_edge')
        if br[0] in list_i:
            positioning.append('at_bottom_edge')
        if br[1] in list_j:
            positioning.append('at_right_edge')
        return positioning
        
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
                            object_positioning = self.define_object_positioning(pattern)
                            obj = GridObject(k, pattern, [color], label, object_positioning) # otherwise create candidate object
                            self.objects_dict[label] = obj
                            used_coordinates.extend(pattern) # save occupied cells 
                            candidate = True
                if candidate:
                    objects[k].append(copy.copy(obj))
                    candidate = False
        used_coordinates = set(used_coordinates) # keep only unique coordinates    
        ul = find_upper_left_corner(shape)
        all_coordinates = set(product(range(ul[0], ul[0]+shape[0]), range(ul[1], ul[1]+shape[1])))
        cells_coordinates = list(all_coordinates.difference(used_coordinates))
        for idx, cell in enumerate(cells_coordinates): # if some cell is not belong to some figure - create cell object
            color = grid[cell]
            if color != 0:
                label = f'cell_{idx}'
                obj = GridObject('cell', [cell], [grid[cell]], label)
                objects['cell'].append(obj)
        return objects

    def filter_objects(self):
        """Apply ObjectsFilter class for filtering out possibly umimportant objects."""
        return ObjectsFilter(self.initial_objects).filter_objects()
    
    def construct_complex_objects(self):
        """Try to merge some objects on the grid based on positional relations between them."""
        complex_objects = [] 
        cache = []
        all_objects = dict_to_list(self.objects)
        for idx, obj_1 in enumerate(all_objects):
            for obj_2 in all_objects[idx+1:]:
                complex_obj = ObjectCombiner(obj_1, obj_2).merge_attempt()
                if complex_obj is not None:
                    complex_objects.append(complex_obj)
                    cache.append(complex_obj)
        while cache != []:
            complex_obj = cache.pop(0)
            for obj in all_objects:
                new_complex_obj = ObjectCombiner(complex_obj, obj).merge_attempt()
                if new_complex_obj is not None:
                    complex_objects.append(new_complex_obj)
                    cache.append(new_complex_obj)
            all_objects.append(complex_obj)
        self.complex_objects = complex_objects
        self.objects['complex_objects'] = complex_objects
        return complex_objects 

    @staticmethod 
    def calculate_distance(obj_1, obj_2):
        """Calculate distance between objects."""
        i_dist = min(abs(obj_1.max_i - obj_2.max_i), abs(obj_1.max_i - obj_2.min_i), abs(obj_1.min_i - obj_2.max_i), abs(obj_1.min_i - obj_2.min_i)) 
        j_dist = min(abs(obj_1.max_j - obj_2.max_j), abs(obj_1.max_j - obj_2.min_j), abs(obj_1.min_j - obj_2.max_j), abs(obj_1.min_j - obj_2.min_j)) 
        return min(i_dist, j_dist)

    def create_objects_summary(self):
        """Create a summary for grid objects to get aggregate information about their shapes, sizes, colors.""" 
        size2shape = defaultdict(list)
        shape2size = {}
        shapes = {shape:0 for shape in (list(self.patterns.keys())+['cell'])}
        colors = {COLOR_MAPPING[i/10]:0 for i in range(10)}
        for k, v in self.objects.items():
            for idx, obj in enumerate(v):
                size2shape[obj.size].append(obj)
                shape2size[obj.label] = obj.size
                shapes[obj.shape] += 1
                if obj.shape != 'complex':
                    color = COLOR_MAPPING[obj.color_number[0]]
                    colors[color] += 1
        sorted_keys = sorted(list(size2shape.keys()), reverse=True)
        size2shape = {k:size2shape[k] for k in sorted_keys}
        shape2size_values = list(shape2size.values())
        mean_size = 0
        median_size = 0
        if len(shape2size_values)>0:
            mean_size = np.mean(shape2size_values)
            median_size = np.median(shape2size_values)
        objects_summary = {'size2shape':size2shape, 'shape2size':shape2size, 'mean_size':mean_size, 'median_size':median_size, 'shapes':shapes, 'colors':colors}
        return objects_summary

    def create_relations_summary(self):
        """Create a summary for relation between grid objects to get useful information for reasoning.""" 
        relations_summary = self.relation_statistics
        objects_properties = {'symmetry':0}
        for k, v in self.objects.items():
            for idx, obj in enumerate(v):
                if obj.symmetry != [] and obj.symmetry != 'assymetry':
                    objects_properties['symmetry'] += 1
        relations_summary.update(objects_properties)       
        return relations_summary
    
    def set_relations(self):
        """Iterate over objects to identify relations between them.""" 
        all_triples = []
        relation_statistics = Counter(self.relations_for_stats)
        distances = defaultdict(lambda: 0)
        all_objects = dict_to_list(self.objects)
        for idx, obj_1 in enumerate(all_objects):
            for obj_2 in all_objects[idx+1:]: 
                analyzer = RelationAnalyzer(obj_1, obj_2, self.shape)
                triples = analyzer.triples
                relation_counter = analyzer.relation_counter
                all_triples.extend(triples)
                relation_statistics.update(relation_counter)
                distances[f'{obj_1.label}-{obj_2.label}'] = self.calculate_distance(obj_1, obj_2)
        return all_triples, relation_statistics