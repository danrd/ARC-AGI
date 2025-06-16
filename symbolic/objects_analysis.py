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
    def __init__(self, shape:str, coords:List[tuple], color:List[float], label:str, grid_shape:tuple, font_color=None, grid=None):
        self.shape = shape
        self.grid_shape = grid_shape
        self.coords = tuple(sorted(coords, key=lambda x: (x[1],x[0])))
        self.size = len(coords)
        self.center = self.find_object_center()
        self.coords_offsets = tuple((coord[0]-self.coords[0][0], coord[1]-self.coords[0][1]) for coord in self.coords)
        self.positioning = self.define_positioning()
        self.edges = self.define_edges()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        self.font_color = font_color
        self.color_numbers = tuple(color)
        self.colors = tuple([colors_mapping[color] for color in self.color_numbers])
        self.color_homo = True if len(self.color_numbers) else False
        self.color_structure = grid[self.min_i:self.max_i+1, self.min_j:self.max_j+1].copy()
        self.color_shares = self.calculate_color_shares()
        self.label = label
        if self.shape not in ['inner_hole', 'outer_hole']:
            self.inner_contour = self.define_inner_contour()
            self.inner_part, self.contour = self.inner_contour_split()
            self.non_object_coords = tuple(set(self.inner_contour).difference(set(self.coords)))
            self.inner_holes, self.outer_holes = self.define_holes(grid)
            self.symmetry = self.check_symmetry() if self.shape != 'cell' else 'horizontal_and_vertical_symmetry'
            self.sub_objects = defaultdict(list)
            self.hu_moments = self.calculate_hu_moments()
            self.compactness = self.calculate_compactness()
            self.inner_holes_share = self.calculate_inner_holes_share()
        
    def __repr__(self):
        if self.shape != 'complex' and self.shape not in ['inner_hole', 'outer_hole']:
            representation = (
                f"{self.symmetry} {self.colors[0]} {self.shape} with size {self.size}, "
                f"with dimensions {self.hor_size}x{self.vert_size} (horizontal x vertical), "
                f"with center at {self.center}, with (min_i, max_i, min_j, max_j) = ({self.min_i}, {self.max_i}, {self.min_j}, {self.max_j}), "
                f"positioning {self.positioning}, {len(self.inner_holes)} inner holes "
                f"and {len(self.outer_holes)} outer holes"
            )        
        elif self.shape  in ['inner_hole', 'outer_hole']:
            representation = (
                f"{self.colors[0]} {self.shape} with size {self.size}, "
                f"with dimensions {self.hor_size}x{self.vert_size} (horizontal x vertical), "
                f"with center at {self.center},"
                f"with (min_i, max_i, min_j, max_j) = ({self.min_i}, {self.max_i}, {self.min_j}, {self.max_j}),"
            )  
        else:
            representation = (
                f"{self.symmetry} {self.colors[0]} segment with size {self.size}, "
                f"with dimensions {self.hor_size}x{self.vert_size} (horizontal x vertical), "
                f"with center at {self.center}, with (min_i, max_i, min_j, max_j) = ({self.min_i}, {self.max_i}, {self.min_j}, {self.max_j}), "
                f"positioning {self.positioning}, {len(self.inner_holes)} inner holes "
                f"and {len(self.outer_holes)} outer holes"
            )  
        return representation
    
    def __eq__(self, other_GridObject):
        isGridObject = isinstance(other_GridObject, self.__class__)
        if not isGridObject:
            return False
        else:
            cells_check = self.check_equality(grid_shape=self.shape, other_object=other_GridObject)
            return cells_check and self.color == other_GridObject.color

    def info(self):
        """Print all core attributes of the GridObject in a structured format."""
        print(f"===== {self.label} =====")
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size} cells")
        print(f"Dimensions: {self.hor_size}x{self.vert_size} (horizontal x vertical)")
        print(f"Center: {self.center}")
        print(f"Boundaries: ({self.min_i},{self.min_j}) to ({self.max_i},{self.max_j})")
        print(f"Colors: {self.colors}")
        print(f"Color homogeneous: {self.color_homo}")
        print(f"Positioning: {self.positioning}")
        
        if self.shape not in ['inner_hole', 'outer_hole']:
            print(f"Symmetry: {self.symmetry}")
            print(f"Inner holes: {len(self.inner_holes)}")
            print(f"Outer holes: {len(self.outer_holes)}")
            
            if any(self.sub_objects.values()):
                print("\nSub-objects:")
                for shape_type, objects in self.sub_objects.items():
                    if objects:
                        print(f"  {shape_type}: {len(objects)} objects")
        
        print(f"\nCoordinates: {self.coords[:5]}{' ...' if len(self.coords) > 20 else ''}")
        print("=" * (len(self.label) + 12))


    def calculate_color_shares(self):
        unique_elements, counts = np.unique(self.color_structure, return_counts=True)
        nomalized_counts = counts / sum(counts)
        return dict(zip(unique_elements, nomalized_counts))

    def calculate_hu_moments(self):
        """Calculate Hu moments for shape identification using discrete grid coordinates."""
        if len(self.coords) < 2:
            return tuple([0.0] * 7)
        
        # Convert coordinates to relative positions (centered)
        coords_array = np.array(self.coords)
        center_i, center_j = self.center
        
        # Calculate central moments up to order 3
        def central_moment(p: int, q: int) -> float:
            moment = 0.0
            for coord in self.coords:
                i, j = coord
                moment += ((i - center_i) ** p) * ((j - center_j) ** q)
            return moment
        
        # Calculate moments
        m00 = len(self.coords)  # Number of pixels
        m20 = central_moment(2, 0)
        m02 = central_moment(0, 2)
        m11 = central_moment(1, 1)
        m30 = central_moment(3, 0)
        m03 = central_moment(0, 3)
        m21 = central_moment(2, 1)
        m12 = central_moment(1, 2)
        
        # Normalize central moments
        if m00 == 0:
            return tuple([0.0] * 7)
        
        mu20 = m20 / m00
        mu02 = m02 / m00
        mu11 = m11 / m00
        mu30 = m30 / (m00 ** 1.5)
        mu03 = m03 / (m00 ** 1.5)
        mu21 = m21 / (m00 ** 1.5)
        mu12 = m12 / (m00 ** 1.5)
        
        # Calculate 7 Hu moments
        hu1 = mu20 + mu02
        hu2 = (mu20 - mu02) ** 2 + 4 * mu11 ** 2
        hu3 = (mu30 - 3 * mu12) ** 2 + (3 * mu21 - mu03) ** 2
        hu4 = (mu30 + mu12) ** 2 + (mu21 + mu03) ** 2
        hu5 = (mu30 - 3 * mu12) * (mu30 + mu12) * ((mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) + \
              (3 * mu21 - mu03) * (mu21 + mu03) * (3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)
        hu6 = (mu20 - mu02) * ((mu30 + mu12) ** 2 - (mu21 + mu03) ** 2) + \
              4 * mu11 * (mu30 + mu12) * (mu21 + mu03)
        hu7 = (3 * mu21 - mu03) * (mu30 + mu12) * ((mu30 + mu12) ** 2 - 3 * (mu21 + mu03) ** 2) - \
              (mu30 - 3 * mu12) * (mu21 + mu03) * (3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)
        
        # Apply log transform and normalize for small discrete grids
        hu_moments = [hu1, hu2, hu3, hu4, hu5, hu6, hu7]
        hu_normalized = []
        for hu in hu_moments:
            if abs(hu) < 1e-10:
                hu_normalized.append(0.0)
            else:
                # Scale down for small grids and apply tanh for bounded output
                scaled_hu = hu / (self.size ** 2)  # Scale by object size squared
                hu_normalized.append(np.tanh(scaled_hu))
        
        return tuple(hu_normalized)

    def calculate_compactness(self):
        """Calculate what share of rectangle contour is filled with non-font colors."""
        total_contour_cells = self.hor_size * self.vert_size
        filled_cells = len([coord for coord in self.coords if coord not in self.non_object_coords])
        return filled_cells / total_contour_cells if total_contour_cells > 0 else 0.0

    def calculate_inner_holes_share(self):
        """Calculate share of object that is related to inner_holes."""
        if not hasattr(self, 'inner_holes') or not self.inner_holes:
            return 0.0
        
        total_inner_hole_size = sum(hole.size for hole in self.inner_holes)
        total_object_area = len(self.inner_contour)  # Total area including holes
        return total_inner_hole_size / total_object_area if total_object_area > 0 else 0.0

    def define_positioning(self):
        """
        Identify if an object is located at specific positions on the grid.
        Supports grids from 1x1 to 30x30 and identifies various positioning attributes.
        """
        positioning = []

        # Get grid dimensions
        rows, cols = self.grid_shape

        grid_size = self.grid_shape
        ul = find_upper_left_corner(grid_size)
        bl = (ul[0]+grid_size[0]-1, ul[1])
        ur = (ul[0], ul[1]+grid_size[1]-1)
        br = (ul[0]+grid_size[0]-1, ul[1]+grid_size[1]-1)

        # Check if object is in corners
        if ul in self.coords:
            positioning.append('in_top_left_corner')
        if bl in self.coords:
            positioning.append('in_bottom_left_corner')  
        if ur in self.coords:
            positioning.append('in_top_right_corner')             
        if br in self.coords:
            positioning.append('in_bottom_right_corner') 
        
        # Get separate lists of row and column indices
        list_i, list_j = coords_transform(self.coords)
        
        # Check if object is at edges
        if ul[0] in list_i:
            positioning.append('at_top_edge')
        if ul[1] in list_j:
            positioning.append('at_left_edge')
        if br[0] in list_i:
            positioning.append('at_bottom_edge')
        if br[1] in list_j:
            positioning.append('at_right_edge')
        
        # Calculate halves
        top_half_rows = range(ul[0], ul[0]+(rows // 2))
        bottom_half_rows = range(ul[0]+(rows // 2), (ul[0]+rows))
        left_half_cols = range(ul[1], ul[1]+(cols//2))
        right_half_cols = range(ul[1]+(cols//2), (ul[1]+cols))
        
        # Check if object is entirely in quarters
        if all(i in top_half_rows for i in list_i) and all(j in left_half_cols for j in list_j):
            positioning.append('at_top_left_quarter')
        if all(i in bottom_half_rows for i in list_i) and all(j in left_half_cols for j in list_j):
            positioning.append('at_bottom_left_quarter')
        if all(i in top_half_rows for i in list_i) and all(j in right_half_cols for j in list_j):
            positioning.append('at_top_right_quarter')
        if all(i in bottom_half_rows for i in list_i) and all(j in right_half_cols for j in list_j):
            positioning.append('at_bottom_right_quarter')
        
        # Additional property: Check if the object forms a diagonal
        if len(self.coords) > 1:
            # Check for main diagonal (top-left to bottom-right)
            if all((i == j) for i, j in self.coords):
                positioning.append('on_main_diagonal')         
            # Check for counter diagonal (top-right to bottom-left)
            if all((i + j == rows - 1) for i, j in self.coords):
                positioning.append('on_counter_diagonal')
        
        return positioning
            
    def reinit_obj(self, new_coords:List[tuple], grid:np.array=None):
        """Updated object attributed based on new coordinates."""
        self.coords = tuple(sorted(new_coords, key=lambda x: (x[1],x[0]))) 
        self.coords_offsets = tuple((coord[0]-self.coords[0][0], coord[1]-self.coords[0][1]) for coord in self.coords)
        self.center = self.find_object_center()
        self.edges = self.define_edges()
        self.size = len(self.coords)
        self.positioning = self.define_positioning()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        if self.shape not in ['inner_hole', 'outer_hole']:
            self.inner_contour = self.define_inner_contour()
            self.inner_part, self.contour = self.inner_contour_split()
            self.non_object_coords = tuple(set(self.inner_contour).difference(set(self.coords)))
            self.inner_holes, self.outer_holes = self.define_holes(grid)
            self.hu_moments = self.calculate_hu_moments()
            self.compactness = self.calculate_compactness()
            self.inner_holes_share = self.calculate_inner_holes_share()
        # Color update if grid provided
        if grid is not None:
            colors = tuple(grid[i, j] for i, j in self.coords)
            self.color_numbers = tuple(set(colors))
            self.colors = tuple(colors_mapping[color] for color in self.color_numbers)
            self.color_homo = True if len(self.color_numbers) == 1 else False
    
    def check_equality(self, other_object):
        coords_1_shifted = [(tup[0]-self.min_i, tup[1]-self.min_j) for tup in self.coords]
        coords_1_shifted.sort(key=lambda x: x[0])
        coords_2_shifted = [(tup[0]-other_object.min_i, tup[1]-self.other_object.min_j) for tup in other_object]
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

    def inner_contour_split(self):
        """
        If object has contour around - define area surrounded by this contour.
        """
        inner_part = []
        contour = [] 
        contour.extend([(self.min_i, j) for j in range(self.min_j, self.max_j+1)])
        contour.extend([(i, self.min_j) for i in range(self.min_i, self.max_i+1)])
        contour.extend([(i, self.max_j) for i in range(self.min_i, self.max_i+1)]) 
        contour.extend([(self.max_i, j) for j in range(self.min_j, self.max_j+1)])
        inner_part = list(set(self.inner_contour).difference(set(contour)))
        return tuple(inner_part), tuple(contour)
    
    def define_holes(self, grid):
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
            hole_object = GridObject(hole_type, hole, [0], label, self.grid_shape, self.font_color, grid) # otherwise create candidate object
            if hole_type == 'inner_hole':
                inner_holes.append(hole_object)
            else:
                outer_holes.append(hole_object)
        return tuple(inner_holes), tuple(outer_holes)

    def find_object_center(self):
        """ Find the center cell of an object represented as a tuple of coordinate tuples."""
        # Calculate the geometric center (may be floating point)
        sum_x = sum(cell[0] for cell in self.coords)
        sum_y = sum(cell[1] for cell in self.coords)
        num_cells = self.size
        geometric_center_x = int(sum_x / num_cells)
        geometric_center_y = int(sum_y / num_cells)   
        return (geometric_center_x, geometric_center_y)
                
    def structure_analysis(self):
        """Get destributions describing object inner structure in terms of shapes and colors.""" 
        shape_types = ('line' ,'rectangle', 'diagonal', 'l_shape', 't_shape', 's_shape', 'tv_shape', 
                            'hs_shape', 'cross', 'flower', 'markup_matrix','markup_line', 'cell', 'complex')        
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
        objects_summary = {
            'size2shape': size2shape, 'shape2size': shape2size, 'hor_size2shape': hor_size2shape, 'shape2hor_size': shape2hor_size,
            'vert_size2shape': vert_size2shape, 'shape2vert_size': shape2vert_size, 'shapes': shapes, 'shape_colors': shape_colors,
            'colors': colors, 'shape_hor_size_description': hor_size2description, 'shape_vert_size_description': vert_size2description,
            'shape_size_description': size2description, 'shape_color_description': shape_color2description, 'color_description': color2description
        }
        self.objects_summary = objects_summary
    
    def create_embedding(self):
        """
        Creates a vector embedding representation of the GridObject with the following features:
        - color_shares: Color distribution based on actual color shares (up to 10 colors)
        - hor_size: Fraction of grid width [0-1]
        - vert_size: Fraction of grid height [0-1] 
        - size: Fraction of total grid size [0-1]
        - i_center: Normalized center i-coordinate [0-1]
        - j_center: Normalized center j-coordinate [0-1] 
        - min_i: Normalized minimum i-coordinate [0-1]
        - min_j: Normalized minimum j-coordinate [0-1]
        - max_i: Normalized maximum i-coordinate [0-1]
        - max_j: Normalized maximum j-coordinate [0-1]
        - inner_holes: Number of inner holes, normalized to range [0-1]
        - outer_holes: Number of outer holes, normalized to range [0-1]
        - symmetry_type: Boolean value indicating symmetry [0/1]
        - hu_moments: 7 Hu moments for shape identification
        - compactness: Share of rectangle contour filled with non-font colors [0-1]
        - inner_holes_share: Share of object related to inner holes [0-1]
        
        Returns:
            tuple: Flat vector representation of all features
        """
        # Get grid dimensions
        grid_rows, grid_cols = self.grid_shape

        # Validate grid dimensions
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError(f"Invalid grid dimensions: {self.grid_shape}")
        
        # Initialize the embedding dictionary
        embedding_dict = {}
        
        # 1. Color shares (based on actual color distribution)
        color_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 10 possible colors
        color_shares_vec = []
        for color_val in color_values:
            share = self.color_shares.get(color_val, 0.0)
            color_shares_vec.append(share)
        
        embedding_dict["color_shares"] = color_shares_vec
        
        # 2-4. Size metrics with bounds checking
        embedding_dict["hor_size"] = min(1.0, self.hor_size / grid_cols)
        embedding_dict["vert_size"] = min(1.0, self.vert_size / grid_rows)
        embedding_dict["size"] = min(1.0, self.size / (grid_rows * grid_cols))
        
        # 5-6. Center position with bounds checking
        i_center, j_center = self.center
        embedding_dict["i_center"] = min(1.0, max(0.0, i_center / grid_rows))
        embedding_dict["j_center"] = min(1.0, max(0.0, j_center / grid_cols))
        
        # 7-10. Boundary positions with bounds checking
        embedding_dict["min_i"] = min(1.0, max(0.0, self.min_i / grid_rows))
        embedding_dict["min_j"] = min(1.0, max(0.0, self.min_j / grid_cols))
        embedding_dict["max_i"] = min(1.0, max(0.0, self.max_i / grid_rows))
        embedding_dict["max_j"] = min(1.0, max(0.0, self.max_j / grid_cols))
        
        # 11-12. Holes
        if self.shape not in ['inner_hole', 'outer_hole']:
            n_inner_holes = len(self.inner_holes)
            n_outer_holes = len(self.outer_holes)
            embedding_dict["inner_holes"] = min(1.0, n_inner_holes / 5)  # Simplified
            embedding_dict["outer_holes"] = min(1.0, n_outer_holes / 5)  # Simplified
        else:
            embedding_dict["inner_holes"] = 0.0
            embedding_dict["outer_holes"] = 0.0
        
        # 13. Symmetry type
        if self.shape not in ['inner_hole', 'outer_hole']:
            has_symmetry = int(hasattr(self, 'symmetry') and self.symmetry != 'assymetry')
            embedding_dict["symmetry_type"] = has_symmetry
        else:
            embedding_dict["symmetry_type"] = 0

        # 14. Hu moments (7 values)
        if self.shape not in ['inner_hole', 'outer_hole'] and hasattr(self, 'hu_moments'):
            # Normalize Hu moments to [0,1] range using tanh
            hu_normalized = [max(0, min(1, (np.tanh(hu/10) + 1) / 2)) for hu in self.hu_moments]
            embedding_dict["hu_moments"] = hu_normalized
        else:
            embedding_dict["hu_moments"] = [0.0] * 7

        # 15. Compactness
        if self.shape not in ['inner_hole', 'outer_hole'] and hasattr(self, 'compactness'):
            embedding_dict["compactness"] = round(self.compactness, 3)
        else:
            embedding_dict["compactness"] = 0.0

        # 16. Inner holes share
        if self.shape not in ['inner_hole', 'outer_hole'] and hasattr(self, 'inner_holes_share'):
            embedding_dict["inner_holes_share"] = round(self.inner_holes_share, 3)
        else:
            embedding_dict["inner_holes_share"] = 0.0
        
        # Create flat vector from all features (immutable)
        flat_vector = []
        flat_vector.extend(color_shares_vec)  # 10 elements for color shares
        flat_vector.extend([
            embedding_dict["hor_size"],
            embedding_dict["vert_size"],
            embedding_dict["size"],
            embedding_dict["i_center"],
            embedding_dict["j_center"],
            embedding_dict["min_i"],
            embedding_dict["min_j"],
            embedding_dict["max_i"],
            embedding_dict["max_j"],
            embedding_dict["inner_holes"],
            embedding_dict["outer_holes"],
            embedding_dict["symmetry_type"]
        ])
        flat_vector.extend(embedding_dict["hu_moments"])  # 7 elements
        flat_vector.extend([
            embedding_dict["compactness"],
            embedding_dict["inner_holes_share"]
        ])
        
        # Store as immutable tuples
        self.embedding_dict = dict(embedding_dict)  # Shallow copy to prevent external mutation
        self.embedding_vector = tuple(flat_vector)  # Make immutable
        
        return tuple(flat_vector)  # Return immutable

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