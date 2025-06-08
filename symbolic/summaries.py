import typing
from typing import List
import numpy as np
from copy import copy
from collections import defaultdict, Counter
from itertools import product
from rl.ARC_task import ARCTask, ARCSubtask
from symbolic.objects_analysis import GridObject, ObjectsFilter, RelationAnalyzer
from symbolic.utils import find_upper_left_corner, coords_transform, count_unique_cells, dict_to_list, check_subset_condition
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
        self.relations_for_stats = ("same_color", "same_shape", "same_size", "in_contour", 
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

    def set_repr_levels(self):
        """Create representation levels for grid objects based on their properties."""
        repr_levels = {}
        level1_init_objects = copy(self.initial_objects)
        init_objects = copy(self.initial_objects)
        init_objects['complex'] = [obj for obj in init_objects['complex'] if obj.color_homo]
        for level in self.levels:  # Changed to include level 5
            if level == 1:
                repr_levels[level] = self.process_repr_level(level1_init_objects, level)
            elif level == 5:
                # Handle the cell level
                repr_levels[level] = self.process_cell_level()
            else: 
                repr_levels[level] = self.process_repr_level(init_objects, level)
        return repr_levels

    def process_repr_level(self, init_objects, level):
        level_objects = self.filter_objects(init_objects, level)
        level_objects_summary = self.create_objects_summary(level_objects)
        level_triples, level_relation_statistics = self.set_relations(level_objects)
        cell2obj = self.grid_markup(dict_to_list(level_objects))
        return {f'objects':tuple(dict_to_list(copy(level_objects))), f'objects_summary':copy(level_objects_summary), 
                f'triples':copy(level_triples), f'relation_statistics':copy(level_relation_statistics),
                f'cell2obj':copy(cell2obj)}

    def process_cell_level(self):
        """Process cell level representation (level 5)."""
        # Create a dictionary of cell objects
        cell_objects = defaultdict(list)
        
        # Iterate over grid to find non-font-colored cells
        ul = find_upper_left_corner(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                grid_i = ul[0] + i
                grid_j = ul[1] + j
                cell_color = self.grid[grid_i, grid_j]
                
                # Add non-font-colored cells (assuming font_color is 0.0 by default)
                if cell_color != 0 and cell_color != self.font_color:
                    label = f'cell_{i}_{j}'
                    cell = (grid_i, grid_j)
                    obj = GridObject('cell', [cell], [cell_color], label, self.shape)
                    cell_objects['cell'].append(obj)
        
        # Process the cell level objects the same way as other levels
        level_objects_summary = self.create_objects_summary(cell_objects)
        level_triples, level_relation_statistics = self.set_relations(cell_objects)
        
        return {
            'objects': dict_to_list(copy(cell_objects)), 
            'objects_summary': copy(level_objects_summary), 
            'triples': copy(level_triples), 
            'relation_statistics': copy(level_relation_statistics),
    }
        
    def retrieve_objects(self, grid:np.array, shape:tuple, shape_types:tuple)->typing.Dict[str, List[GridObject]]:
        """Retrieve all possible objects from the grid and return corresponding GridObject instances."""
        patterns = generate_patterns(shape, shape_types)
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

    def grid_markup(self, level_objects:List[GridObject]):
        cell2object = {}
        for idx, obj in enumerate(level_objects):
            for coord in obj.coords:
                cell2object[coord] = idx
        return cell2object

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
        
    def set_relations(self, objects):
        """Iterate over objects to identify relations between them.""" 
        all_triples = defaultdict(dict)
        relation_statistics = Counter(self.relations_for_stats)
        distances = defaultdict(lambda: 0)
        all_objects = dict_to_list(objects)
        for idx, obj_1 in enumerate(all_objects):
            for obj_2 in all_objects[idx+1:]: 
                # if obj_2.label not in obj_1.relations.keys():
                analyzer = RelationAnalyzer(obj_1, obj_2, self.shape)
                triples = analyzer.triples
                relation_counter = analyzer.relation_counter
                all_triples[obj_1.label][obj_2.label] = triples[0]
                all_triples[obj_2.label][obj_1.label] = triples[1]
                relation_statistics.update(relation_counter)
                # distance calculation
                # distance = self.calculate_distance(obj_1, obj_2)
                # obj_1.distances[obj_2.label] = distance
                # obj_2.distances[obj_1.label] = distance
                # else:
                #     all_triples[obj_1.label][obj_2.label] = [(obj_1.label, relation ,obj_2.label) for relation in obj_1.relations[obj_2.label]]
                #     all_triples[obj_2.label][obj_1.label] = [(obj_2.label, relation ,obj_1.label) for relation in obj_2.relations[obj_1.label]]
                #     relation_counter = {triple[1]:1 for triple in obj_1.relations[obj_2.label]}
                #     relation_statistics.update(relation_counter)
                # distances[f'{obj_1.label}-{obj_2.label}'] = obj_1.distances[obj_2.label]
        return all_triples, relation_statistics
    
    def create_embedding(self, obj_1, obj_2):
            """Create embedding vector for a pair of objects based on their relations."""
            # Initialize with zeros
            relation_feature_names = (
                'same_color', 'same_size', 'same_vert_size', 'same_hor_size', 
                'translation_symmetry', 'in_contour', 'has_in_contour',
                'in_line', 'in_diagonal', 'x_aligned_with', 'y_aligned_with', 'distance'
            )
            embedding = np.zeros(len(relation_feature_names))
            
            # Get the existing triple relations between objects
            obj_triples = {}
            if obj_1.label in self.repr_levels[1]['triples']:
                if obj_2.label in self.repr_levels[1]['triples'][obj_1.label]:
                    obj_triples = self.repr_levels[1]['triples'][obj_1.label][obj_2.label]
            
            # Set boolean values based on relations
            feature_idx = 0
            
            # Same color relation
            obj1_colors = getattr(obj_1, 'colors', ())
            obj2_colors = getattr(obj_2, 'colors', ())
            embedding[feature_idx] = 1 if obj1_colors == obj2_colors else 0
            feature_idx += 1
            
            # Same size relation
            obj1_size = getattr(obj_1, 'size', 0)
            obj2_size = getattr(obj_2, 'size', 0)
            embedding[feature_idx] = 1 if obj1_size == obj2_size else 0
            feature_idx += 1
            
            # Same vertical size relation
            obj1_vert = getattr(obj_1, 'vert_size', 0)
            obj2_vert = getattr(obj_2, 'vert_size', 0)
            embedding[feature_idx] = 1 if obj1_vert == obj2_vert else 0
            feature_idx += 1
            
            # Same horizontal size relation
            obj1_hor = getattr(obj_1, 'hor_size', 0)
            obj2_hor = getattr(obj_2, 'hor_size', 0)
            embedding[feature_idx] = 1 if obj1_hor == obj2_hor else 0
            feature_idx += 1
            
            # Check for specific relations in triples
            relation_checks = [
                "translation_symmetry", "in_contour", "has_in_contour", 
                "in_line", "in_diagonal", "x_aligned_with", "y_aligned_with"
            ]
            
            for relation in relation_checks:
                has_relation = any(triple[1] == relation for triple in obj_triples if len(triple) > 1)
                embedding[feature_idx] = 1 if has_relation else 0
                feature_idx += 1
            
            # Distance relation (normalized) with bounds checking
            grid_size = max(self.shape) if hasattr(self, 'shape') and self.shape else 1
            distance = getattr(obj_1, 'distances', {}).get(obj_2.label, 0)
            normalized_distance = min(distance / grid_size, 1.0) if grid_size > 0 else 0.0
            embedding[feature_idx] = normalized_distance
            
            return embedding
        
    def create_relation_embeddings(self, level=1):
        """Generate embeddings for all object pairs."""
        all_objects = self.repr_levels[level]['objects']
        n_objects = len(all_objects)
        relation_embeddings = defaultdict(dict)
        
        # Initialize embeddings matrix for all object pairs
        for i in range(n_objects):
            obj_1 = all_objects[i]
            for j in range(n_objects):
                if i == j:
                    continue  # Skip same object
                obj_2 = all_objects[j]
                relation_embeddings[obj_1.label][obj_2.label] = self.create_embedding(obj_1, obj_2)
        self.repr_levels[level]['relation_embeddings'] = relation_embeddings
    
    def update_relation_embeddings(self, changed_object, level=1):
        """Update relation embeddings when an object changes."""
        # First update all relations for the changed object
        self.update_relations_for_object(changed_object, level=level)
        
        # Then update embeddings for the changed object
        self.update_embeddings_for_changed_object(changed_object, level=level)
        
    def update_relations_for_object(self, changed_object, level=1):
        """Update relations for a specific changed object."""
        all_objects = self.repr_levels[level]['objects']
        
        # Clear existing relations for the changed object
        for obj in all_objects:
            if obj.label == changed_object.label:
                continue
        
        # Re-analyze relations with all other objects
        for obj in all_objects:
            if obj.label == changed_object.label:
                continue
            
            analyzer = RelationAnalyzer(changed_object, obj, self.shape)
            triples = analyzer.triples
            
            # Update triples in repr_levels
            if changed_object.label not in self.repr_levels[level]['triples']:
                self.repr_levels[level]['triples'][changed_object.label] = {}
            
            if obj.label not in self.repr_levels[1]['triples']:
                self.repr_levels[level]['triples'][obj.label] = {}
            
            self.repr_levels[level]['triples'][changed_object.label][obj.label] = triples[0]
            self.repr_levels[level]['triples'][obj.label][changed_object.label] = triples[1]
            
            # Update distance
            distance = self.calculate_distance(changed_object, obj)
            changed_object.distances[obj.label] = distance
            obj.distances[changed_object.label] = distance
    
    def update_embeddings_for_changed_object(self, changed_obj, level=1):
        """Update embeddings when one object has changed."""
        all_objects = self.repr_levels[level]['objects']
        # Update embeddings for all pairs involving the changed object
        for obj in all_objects:
            if obj.label == changed_obj.label:
                continue  # Skip the object itself
            
            # Update embedding in both directions
            self.repr_levels[level]['relation_embeddings'][changed_obj.label][obj.label] = self.create_embedding(changed_obj, obj)
            self.repr_levels[level]['relation_embeddings'][obj.label][changed_obj.label] = self.create_embedding(obj, changed_obj)

    def get_relation_embeddings_as_numpy(self, level=1):
        """
        Return all relation embeddings for the specified level in numpy array format with validation.
        """
        # Validate level exists
        if not hasattr(self, 'repr_levels') or level not in self.repr_levels:
            raise ValueError(f"Level {level} not found in repr_levels")
        
        # Get relation embeddings from the specified level
        if 'relation_embeddings' not in self.repr_levels[level]:
            self.create_relation_embeddings(level)
        
        relation_embeddings = self.repr_levels[level]['relation_embeddings']
        all_objects = self.repr_levels[level]['objects']
        n_objects = len(all_objects)
        
        # Handle edge cases
        if n_objects <= 1:
            return np.array([])
        
        # Determine the length of a single embedding
        sample_length = 0
        for obj1_label, embeddings_dict in relation_embeddings.items():
            for obj2_label, embedding in embeddings_dict.items():
                if hasattr(embedding, '__len__'):
                    sample_length = len(embedding)
                    break
            if sample_length > 0:
                break
        
        if sample_length == 0:
            return np.array([])
        
        # Initialize the numpy array to store all embeddings
        result = np.zeros((n_objects, (n_objects-1) * sample_length))
        
        # Fill the array with concatenated embeddings
        for i, obj in enumerate(all_objects):
            if not hasattr(obj, 'label'):
                continue
                
            obj_label = obj.label
            col_idx = 0
            
            # Concatenate embeddings for this object with all other objects
            for other_obj in all_objects:
                if not hasattr(other_obj, 'label') or obj_label == other_obj.label:
                    continue
                    
                other_label = other_obj.label
                
                if (obj_label in relation_embeddings and 
                    other_label in relation_embeddings[obj_label]):
                    # Get embedding for this object pair
                    embedding = relation_embeddings[obj_label][other_label]
                    if hasattr(embedding, '__len__') and len(embedding) == sample_length:
                        # Add to the result array at the correct position
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