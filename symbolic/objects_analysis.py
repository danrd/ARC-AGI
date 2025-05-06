import typing
import numpy as np
from typing import List
from collections import defaultdict
from utils.plotting import plot_grid
from symbolic.utils import check_subset_condition, find_upper_left_corner, coords_transform
from symbolic.patterns import find_connected_components_excluding_colors
from collections import Counter
from llm.prompts import COLOR_MAPPING

colors_mapping = {
    0: 'black', 0.1: 'blue', 0.2: 'red', 0.3: 'green', 0.4: 'yellow', 
    0.5: 'gray', 0.6: 'magenta', 0.7: 'orange', 0.8: 'sky', 0.9: 'brown', 1: 'white'
}
class GridObject():
    """Class for storing identified objects on a grid."""
    def __init__(self, shape:str, coords:List[tuple], color:List[float], label:str, positioning=[]):
        self.shape = shape
        self.coords = sorted(coords, key=lambda x: (x[1],x[0]))
        self.center = self.find_object_center()
        self.coords_offsets = [(coord[0]-self.coords[0][0], coord[1]-self.coords[0][1]) for coord in self.coords]
        self.size = len(coords)
        self.positioning = positioning
        self.edges = self.define_edges()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        self.color_numbers = color
        self.colors = [colors_mapping[color] for color in self.color_numbers]
        self.color_homo = True if len(self.color_numbers) else False
        self.label = label
        if self.shape not in ['inner_hole', 'outer_hole']:
            self.inner_contour = self.define_inner_contour()
            self.non_object_coords = list(set(self.inner_contour).difference(set(self.coords)))
            self.inner_holes, self.outer_holes = self.define_holes()
            self.relations = {}
            self.symmetry = self.check_symmetry() if self.shape != 'cell' else 'horizontal_and_vertical_symmetry'
            self.sub_objects = defaultdict(list)
        
    def __repr__(self):
        if self.shape != 'complex':
            return f'{self.colors[0]} {self.shape} with size {self.size}, horizontal size {self.hor_size} and vetrical size {self.vert_size} with positioning {self.positioning}'
        else:
            return f'Segment with {self.colors} colors with size {self.size}, horizontal size {self.hor_size} and vetrical size {self.vert_size} with positioning {self.positioning}'
    
    def __eq__(self, other_GridObject):
        isGridObject = isinstance(other_GridObject, self.__class__)
        if not isGridObject:
            return False
        else:
            cells_check = self.check_equality(grid_shape=self.shape, other_object=other_GridObject)
            return cells_check and self.color == other_GridObject.color

    def reinit_obj(self, new_coords:List[tuple]):
        """Updated object attributed based on new coordinates."""
        self.coords = new_coords
        self.coords_offsets = [(coord[0]-self.coords[0][0], coord[1]-self.coords[0][1]) for coord in self.coords]
        self.edges = self.define_edges()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        if self.shape not in ['inner_hole', 'outer_hole']:
            self.inner_contour = self.define_inner_contour()
            self.non_object_coords = list(set(self.inner_contour).difference(set(self.coords)))
            self.inner_holes, self.outer_holes = self.define_holes()
    
    def check_equality(self, other_object):
        coords_1_shifted = [(tup[0]-self.min_i, tup[1]-self.min_j) for tup in self.coords]
        coords_1_shifted.sort(key=lambda x: x[0])
        coords_2_shifted = [(tup[0]-other_object.min_i, tup[1]-self.othet_object.min_j) for tup in other_object]
        coords_2_shifted.sort(key=lambda x: x[0])
        return coords_1_shifted == coords_2_shifted
    
    def define_edges(self):
        i_coords = [cell[0] for cell in self.coords]
        j_coords = [cell[1] for cell in self.coords]
        return (min(i_coords), max(i_coords), min(j_coords), max(j_coords))
    
    def check_symmetry(self):
        """Identify symmetric propetries for the object."""
        shape = max(self.max_i, self.max_j) - min(self.min_i, self.min_j) + 1
        min_coord = min(self.min_i, self.min_j)
        upper_left_grid_corner = (min_coord, min_coord)
        grid = np.zeros((shape, shape))
        symmetries = []
        for coord in self.coords:
            grid[(coord[0]-upper_left_grid_corner[0], coord[1]-upper_left_grid_corner[0])] = 1
        if np.array_equal(np.flipud(grid), grid):
            symmetries.append('horizontal_symmetry')
        if np.array_equal(np.fliplr(grid), grid):
            symmetries.append('vertical_symmetry')
        if len(symmetries) == 0:
            return 'assymetry'
        elif len(symmetries) == 2:
            return 'horizontal_and_vertical_symmetry'
        else: 
            return symmetries[0]

    def create_object_triples(self):
        """Create triples based on objects properties."""
        triples = [] 
        for position in self.positioning:
            triples.append((self.label, 'located', position))
        triples.append((self.label, 'has_shape', self.shape))
        triples.append((self.label, 'has_color', self.color[0]))
        triples.append((self.label, 'has_size', self.size))
        triples.append((self.label, 'has', self.symmetry))
        return triples

    def define_inner_contour(self):
        """
        Returns the minimal bounding rectangle around the shape.
        """
        contour = []
        offset = (self.min_i, self.min_j)
        obj_mask = np.zeros((self.max_i+1-self.min_i, self.max_j+1-self.min_j))
        obj_structure = np.zeros((self.max_i+1-self.min_i, self.max_j+1-self.min_j))
        for i in range(self.min_i, self.max_i+1):
            for j in range(self.min_j, self.max_j+1):
                contour.append((i, j))
                if (i, j) in self.coords:
                    obj_mask[i-offset[0], j-offset[1]] = 1
                    obj_structure[i-offset[0], j-offset[1]] = self.coords.index((i, j)) + 1
        self.obj_mask = obj_mask
        self.obj_structure = obj_structure
        return contour 

    def define_holes(self):
        inner_holes = []
        outer_holes = []
        offset_i = self.min_i
        offset_j = self.min_j
        connected_components = find_connected_components_excluding_colors(self.obj_mask, font_color=1.0, pad_val=10)
        for idx, comp in enumerate(connected_components):
            hole = []
            hole_type = 'inner_hole'
            for coord in comp:
                if coord[0]+offset_i == self.min_i or coord[0]+offset_i == self.max_i or coord[1]+offset_j == self.min_j or coord[1]+offset_j == self.max_j:
                    hole_type = 'outer_hole'
                real_coord = (coord[0]+offset_i, coord[1]+offset_j)
                hole.append(real_coord)
            label = f'{hole_type}_hole_{idx}_of_{self.label}'
            hole_object = GridObject(hole_type, hole, [0], label, []) # otherwise create candidate object
            if hole_type == 'inner_hole':
                inner_holes.append(hole_object)
            else:
                outer_holes.append(hole_object)
        return inner_holes, outer_holes

    def find_object_center(self):
        """
        Find the center cell of an object represented as a list of coordinate tuples.
        
        Parameters:
        -----------
        object_cells : list of tuples
            List of (x, y) coordinates representing cells of the object
        
        Returns:
        --------
        tuple:
            (center_x, center_y) - the coordinates of the cell closest to the geometric center
            This will be one of the actual cells from the input list
        """
        object_cells = self.coords
        if not object_cells:
            raise ValueError("Object must contain at least one cell")
        
        # Calculate the geometric center (may be floating point)
        sum_x = sum(cell[0] for cell in object_cells)
        sum_y = sum(cell[1] for cell in object_cells)
        num_cells = len(object_cells)
        geometric_center_x = int(sum_x / num_cells)
        geometric_center_y = int(sum_y / num_cells)   
        return (geometric_center_x, geometric_center_y)
                
    
    def structure_analysis(self):
        """Get destributions describing object inner structure in terms of shapes and colors.""" 
        shape_types = ['line' ,'rectangle', 'diagonal', 'l_shape', 't_shape', 's_shape', 'tv_shape', 
                            'hs_shape', 'cross', 'flower', 'markup_matrix','markup_line', 'cell', 'complex']        
        size2shape = defaultdict(list)
        shape2size = {}
        hor_size2shape = defaultdict(list)
        shape2hor_size = {}
        vert_size2shape = defaultdict(list)
        shape2vert_size = {}
        shapes = {shape:0 for shape in shape_types}
        shape_colors = {colors_mapping[i/10]:0 for i in range(10)}
        colors = {colors_mapping[i/10]:0 for i in range(10)}
        for k, v in self.sub_objects.items():
            for idx, obj in enumerate(v):
                size2shape[obj.size].append(obj)
                hor_size2shape[obj.hor_size].append(obj)
                vert_size2shape[obj.vert_size].append(obj)
                shape2size[obj.label] = obj.size
                shape2hor_size[obj.label] = obj.hor_size
                shape2vert_size[obj.label] = obj.vert_size
                shapes[obj.shape] += 1
                color = obj.colors[0]
                shape_colors[color] += 1
                colors[color] += obj.size
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
        # aggregating
        objects_summary = {'size2shape':size2shape, 'shape2size':shape2size,
                           'hor_size2shape':hor_size2shape, 'shape2hor_size':shape2hor_size,
                           'vert_size2shape':vert_size2shape, 'shape2vert_size':shape2vert_size,                         
                           'shapes':shapes, 'shape_colors':shape_colors, 'colors':colors,
                           'shape_hor_size_description':hor_size2description, 'shape_vert_size_description':vert_size2description,
                           'shape_size_description':size2description, 'shape_color_description':shape_color2description,
                           'color_description':color2description
                          }
        self.objects_summary = objects_summary
    
    def plot(self):
        """Plot the object."""
        grid = np.zeros((30,30))
        for coord in self.coords:
            grid[coord] = self.color_number
        plot_grid(grid)

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
        triples = []  
        relation_statistics = Counter()

        if self.object_1.colors == self.object_2.colors:
            triples.append((self.object_2.label, f"same_color", self.object_1.label))
            triples.append((self.object_1.label, f"same_color", self.object_2.label))
            relation_statistics[f"same_color"] += 1 
            
        if self.object_1.shape == self.object_2.shape:
            triples.append((self.object_2.label, f"same_shape", self.object_1.label))
            triples.append((self.object_1.label, f"same_shape", self.object_2.label))
            relation_statistics[f"same_shape"] += 1  
            
        if self.object_1.size == self.object_2.size:
            triples.append((self.object_2.label, f"same_size", self.object_1.label))
            triples.append((self.object_1.label, f"same_size", self.object_2.label))
            relation_statistics[f"same_size"] += 1  
            
        rotations = self.rotation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if rotations != []:
            for rotation in rotations:
                triples.append((self.object_1.label, rotation, self.object_2.label))
                triples.append((self.object_2.label, rotation, self.object_1.label))
        
        (i_offset, j_offset) = self.translation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if i_offset != 0 and j_offset != 0:
            triples.append((self.object_1.label, f"translation_symmetry", self.object_2.label))
            triples.append((self.object_2.label, f"translation_symmetry", self.object_1.label))

        in_contour = self.in_contour(self.object_1, self.object_2)
        if in_contour == "object_2":
            triples.append((self.object_2.label, f"in_contour", self.object_1.label))
            triples.append((self.object_1.label, f"has_in_contour", self.object_2.label))
            relation_statistics[f"in_contour"] += 1 

        if in_contour == "object_1":
            triples.append((self.object_1.label, f"in_contour", self.object_2.label))
            triples.append((self.object_2.label, f"has_in_contour", self.object_1.label))

        # Check for in_line relation with connection cells
        is_in_line, line_cells_1, line_cells_2 = self.in_line(self.object_1, self.object_2)
        if is_in_line:
            triples.append((self.object_1.label, f"in_line", self.object_2.label))
            triples.append((self.object_2.label, f"in_line", self.object_1.label))  
            relation_statistics[f"in_line"] += 1
            
            # Store connection cells in object's relations dictionary
            self.object_1.relations[self.object_2.label] = [self.object_2.label, "in_line", line_cells_1]
            self.object_2.relations[self.object_1.label] = [self.object_1.label, "in_line", line_cells_2]
    
        # Check for in_diagonal relation with connection cells
        is_diagonal, diag_cells_1, diag_cells_2 = self.in_diagonal(self.object_1, self.object_2)
        if is_diagonal:
            triples.append((self.object_1.label, f"in_diagonal", self.object_2.label))
            triples.append((self.object_2.label, f"in_diagonal", self.object_1.label))  
            relation_statistics[f"in_diagonal"] += 1
            
            # Store connection cells in object's relations dictionary
            self.object_1.relations[self.object_2.label] = [self.object_2.label, "in_diagonal", diag_cells_1]
            self.object_2.relations[self.object_1.label] = [self.object_1.label, "in_diagonal", diag_cells_2]

        x_alignment = self.x_alignment(self.object_1, self.object_2)
        y_alignment = self.y_alignment(self.object_1, self.object_2)
        if x_alignment and y_alignment:
            triples.append((self.object_1.label, f"x_y_aligned_with", self.object_2.label))
            triples.append((self.object_2.label, f"x_y_aligned_with", self.object_1.label))
            relation_statistics[f"x_y_aligned_with"] += 1  
        else:    
            if self.x_alignment(self.object_1, self.object_2):
                triples.append((self.object_1.label, f"x_aligned_with", self.object_2.label))
                triples.append((self.object_2.label, f"x_aligned_with", self.object_1.label))
                relation_statistics[f"x_aligned_with"] += 1 
            
            if self.y_alignment(self.object_1, self.object_2):
                triples.append((self.object_1.label, f"y_aligned_with", self.object_2.label))
                triples.append((self.object_2.label, f"y_aligned_with", self.object_1.label))
                relation_statistics[f"y_aligned_with"] += 1 
        return triples, relation_statistics

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
        triples = []  
        relation_statistics = Counter()

        if self.object_1.colors == self.object_2.colors:
            triples.append((self.object_2.label, f"same_color", self.object_1.label))
            triples.append((self.object_1.label, f"same_color", self.object_2.label))
            relation_statistics[f"same_color"] += 1 
            
        if self.object_1.shape == self.object_2.shape:
            triples.append((self.object_2.label, f"same_shape", self.object_1.label))
            triples.append((self.object_1.label, f"same_shape", self.object_2.label))
            relation_statistics[f"same_shape"] += 1  
            
        if self.object_1.size == self.object_2.size:
            triples.append((self.object_2.label, f"same_size", self.object_1.label))
            triples.append((self.object_1.label, f"same_size", self.object_2.label))
            relation_statistics[f"same_size"] += 1  
            
        rotations = self.rotation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if rotations != []:
            for rotation in rotations:
                triples.append((self.object_1.label, rotation, self.object_2.label))
                triples.append((self.object_2.label, rotation, self.object_1.label))
        
        (i_offset, j_offset) = self.translation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if i_offset != 0 and j_offset != 0:
            triples.append((self.object_1.label, f"translation_symmetry", self.object_2.label))
            triples.append((self.object_2.label, f"translation_symmetry", self.object_1.label))

        in_contour = self.in_contour(self.object_1, self.object_2)
        if in_contour == "object_2":
            triples.append((self.object_2.label, f"in_contour", self.object_1.label))
            triples.append((self.object_1.label, f"has_in_contour", self.object_2.label))
            relation_statistics[f"in_contour"] += 1 

        if in_contour == "object_1":
            triples.append((self.object_1.label, f"in_contour", self.object_2.label))
            triples.append((self.object_2.label, f"has_in_contour", self.object_1.label))

        # Check for in_line relation with connection cells
        is_in_line, line_cells_1, line_cells_2 = self.in_line(self.object_1, self.object_2)
        if is_in_line:
            triples.append((self.object_1.label, f"in_line", self.object_2.label))
            triples.append((self.object_2.label, f"in_line", self.object_1.label))  
            relation_statistics[f"in_line"] += 1
            
            # Store connection cells in object's relations dictionary
            self.object_1.relations[self.object_2.label] = [self.object_2.label, "in_line", line_cells_1]
            self.object_2.relations[self.object_1.label] = [self.object_1.label, "in_line", line_cells_2]
    
        # Check for in_diagonal relation with connection cells
        is_diagonal, diag_cells_1, diag_cells_2 = self.in_diagonal(self.object_1, self.object_2)
        if is_diagonal:
            triples.append((self.object_1.label, f"in_diagonal", self.object_2.label))
            triples.append((self.object_2.label, f"in_diagonal", self.object_1.label))  
            relation_statistics[f"in_diagonal"] += 1
            
            # Store connection cells in object's relations dictionary
            self.object_1.relations[self.object_2.label] = [self.object_2.label, "in_diagonal", diag_cells_1]
            self.object_2.relations[self.object_1.label] = [self.object_1.label, "in_diagonal", diag_cells_2]

        x_alignment = self.x_alignment(self.object_1, self.object_2)
        y_alignment = self.y_alignment(self.object_1, self.object_2)
        if x_alignment and y_alignment:
            triples.append((self.object_1.label, f"x_y_aligned_with", self.object_2.label))
            triples.append((self.object_2.label, f"x_y_aligned_with", self.object_1.label))
            relation_statistics[f"x_y_aligned_with"] += 1  
        else:    
            if self.x_alignment(self.object_1, self.object_2):
                triples.append((self.object_1.label, f"x_aligned_with", self.object_2.label))
                triples.append((self.object_2.label, f"x_aligned_with", self.object_1.label))
                relation_statistics[f"x_aligned_with"] += 1 
            
            if self.y_alignment(self.object_1, self.object_2):
                triples.append((self.object_1.label, f"y_aligned_with", self.object_2.label))
                triples.append((self.object_2.label, f"y_aligned_with", self.object_1.label))
                relation_statistics[f"y_aligned_with"] += 1 
        return triples, relation_statistics

class ObjectCombiner():
    """Class for creating complex objects by merging base types of shapes."""
    def __init__(self, object_1:GridObject=None, object_2:GridObject=None):
        self.object_1 = object_1
        self.object_2 = object_2
        
    @staticmethod
    def intersection(coords_1, coords_2):
        intersection = False
        for coord_1 in coords_1:
            for coord_2 in coords_2:
                if coord_1[0] == coord_2[0] and coord_1[1] == coord_2[1]:
                    intersection = True
                    break
        return intersection    
    
    @staticmethod
    def hor_adjacency(coords_1, coords_2):
        hor_adjacency = False
        for coord_1 in coords_1:
            for coord_2 in coords_2:
                if coord_1[0] == coord_2[0] and (coord_1[1] == coord_2[1]+1 or coord_1[1] == coord_2[1]-1):
                    hor_adjacency = True
                    break
            if hor_adjacency:
                break
        return hor_adjacency
    
    @staticmethod
    def vert_adjacency(coords_1, coords_2):
        vert_adjacency = False
        for coord_1 in coords_1:
            for coord_2 in coords_2:
                if coord_1[1] == coord_2[1] and (coord_1[0] == coord_2[0]+1 or coord_1[0] == coord_2[0]-1):
                    vert_adjacency = True
                    break
            if vert_adjacency:
                break
        return vert_adjacency 

    @staticmethod
    def diag_adjacency(coords_1, coords_2):
        diag_adjacency = False
        for coord_1 in coords_1:
            for coord_2 in coords_2:
                if (coord_1[0] == coord_2[0]+1 and coord_1[1] == coord_2[1]+1) or (coord_1[0] == coord_2[0]-1 and coord_1[1] == coord_2[1]-1):
                    diag_adjacency = True
                    break
            if diag_adjacency:
                break
        return diag_adjacency 
    
    def merge_attempt(self):
        """Try to create a complex object based on identified relations between objects."""
        assert self.object_1!=None and self.object_2!=None, f"Object_1 and Object_2 should be specified"
        complex_object = None
        intersection = self.intersection(self.object_1.coords, self.object_2.coords)
        if intersection:
            return None 
        else:
            hor_adjacency = self.hor_adjacency(self.object_1.coords, self.object_2.coords)
            vert_adjacency = self.vert_adjacency(self.object_1.coords, self.object_2.coords)
            diag_adjacency = self.diag_adjacency(self.object_1.coords, self.object_2.coords)
          
        if hor_adjacency or vert_adjacency or diag_adjacency:
            complex_shape_coords = self.object_1.coords + self.object_2.coords
            complex_shape_label = f'complex_shape_{int(self.object_1.label.split("_")[-1])+int(self.object_2.label.split("_")[-1])}'
            complex_shape_label_colors = self.object_1.color + self.object_2.color
            complex_object = GridObject(shape='complex_shape', coords=complex_shape_coords, color=self.object_1.color_number, label=complex_shape_label)                                        
        return complex_object