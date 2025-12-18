import typing
import numpy as np
from typing import List
from collections import defaultdict
from symbolic.utils import find_upper_left_corner, coords_transform
from symbolic.patterns import find_connected_components_with_color

colors_mapping = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow', 
    5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown', 10: 'white'
}
class GridObject():
    """Class for storing identified objects on a grid."""
    def __init__(self, shape:str, coords:List[tuple], color:List[float], label:str, grid_shape:tuple, font_color=0, grid=None):
        self.shape = shape
        self.grid_shape = grid_shape
        self.coords = tuple(sorted(coords, key=lambda x: (x[1],x[0])))
        self.size = len(coords)
        self.center, self.precise_center = self.find_object_center()
        self.coords_offsets = tuple((coord[0]-self.coords[0][0], coord[1]-self.coords[0][1]) for coord in self.coords) # offsets in relation to top left corner
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
        self.color_homo = False if len(self.color_numbers) > 1 else True
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
        if self.shape == 'complex':
            self.classify_shape()
        
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
            cells_check = self.check_equality(other_object=other_GridObject)
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
            return [0.0] * 7
        
        # Correct normalization for central moments
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
                # Use log transform instead of tanh for better discrimination
                hu_normalized.append(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
        
        return hu_normalized

    def calculate_compactness(self):
        """Calculate what share of rectangle contour is filled with non-font colors."""
        total_contour_cells = self.hor_size * self.vert_size
        # Direct set difference is much faster than list comprehension with membership test
        filled_cells = len(self.coords)
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
        self.center, self.precise_center = self.find_object_center()
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
        coords_2_shifted = [(tup[0]-other_object.min_i, tup[1]-other_object.min_j) for tup in other_object.coords]
        coords_2_shifted.sort(key=lambda x: x[0])
        return coords_1_shifted == coords_2_shifted
    
    def define_edges(self):
        i_coords = [cell[0] for cell in self.coords]
        j_coords = [cell[1] for cell in self.coords]
        return (min(i_coords), max(i_coords), min(j_coords), max(j_coords))
    
    def check_symmetry(self):
        """Identify symmetric propetries for the object."""
        if self.size <= 1:
            return 'assymetry'
        
        # Convert to relative coordinates within the object's bounding box
        height = self.max_i - self.min_i + 1
        width = self.max_j - self.min_j + 1
        
        # Create a binary grid representing the object
        grid = np.zeros((height, width), dtype=int)
        for coord in self.coords:
            i_rel = coord[0] - self.min_i
            j_rel = coord[1] - self.min_j
            grid[i_rel, j_rel] = 1
        
        symmetries = []
        
        # Check horizontal symmetry (flip up-down)
        if np.array_equal(np.flipud(grid), grid):
            symmetries.append('horizontal_symmetry')
        
        # Check vertical symmetry (flip left-right)
        if np.array_equal(np.fliplr(grid), grid):
            symmetries.append('vertical_symmetry')
        
        if len(symmetries) == 0:
            return 'assymetry'
        elif 'horizontal_symmetry' in symmetries and 'vertical_symmetry' in symmetries:
            return 'horizontal_and_vertical_symmetry'
        else:
            return symmetries[0]

    def define_inner_contour(self):
        """Returns the minimal bounding rectangle around the shape."""
        height = self.max_i - self.min_i + 1
        width = self.max_j - self.min_j + 1
        
        # Pre-allocate arrays
        obj_mask = np.zeros((height, width), dtype=np.int8)
        obj_structure = np.zeros((height, width), dtype=np.int32)
        
        # Create coordinate lookup dict once - O(1) instead of O(n)
        coord_to_idx = {coord: idx + 1 for idx, coord in enumerate(self.coords)}
        
        # Vectorized coordinate generation
        i_coords = np.arange(self.min_i, self.max_i + 1)
        j_coords = np.arange(self.min_j, self.max_j + 1)
        contour = list(product(i_coords, j_coords))
        
        # Fill masks using dict lookup
        for i, j in self.coords:
            mask_i = i - self.min_i
            mask_j = j - self.min_j
            obj_mask[mask_i, mask_j] = 1
            obj_structure[mask_i, mask_j] = coord_to_idx[(i, j)]
        
        self.obj_mask = obj_mask
        self.obj_structure = obj_structure
        return tuple(contour)

    def inner_contour_split(self):
        """If object has contour around - define area surrounded by this contour."""
        inner_part = []
        contour = [] 
        contour.extend([(self.min_i, j) for j in range(self.min_j, self.max_j+1)])
        contour.extend([(i, self.min_j) for i in range(self.min_i, self.max_i+1)])
        contour.extend([(i, self.max_j) for i in range(self.min_i, self.max_i+1)]) 
        contour.extend([(self.max_i, j) for j in range(self.min_j, self.max_j+1)])
        inner_part = list(set(self.inner_contour).difference(set(contour)))
        return tuple(set(inner_part)), tuple(set(contour))
    
    def define_holes(self, grid):
        inner_holes = []
        outer_holes = []
        offset_i = self.min_i
        offset_j = self.min_j
        connected_components = find_connected_components_with_color(grid[self.min_i:self.max_i+1,self.min_j:self.max_j+1], self.font_color, folds=4)
        for idx, comp in enumerate(connected_components):
            hole = []
            hole_type = 'inner_hole'
            for coord in comp:
                if coord[0]+offset_i == self.min_i or coord[0]+offset_i == self.max_i or coord[1]+offset_j == self.min_j or coord[1]+offset_j == self.max_j:
                    hole_type = 'outer_hole'
                real_coord = (coord[0]+offset_i, coord[1]+offset_j)
                hole.append(real_coord)
            label = f'{hole_type}_hole_{idx}_of_{self.label}'
            hole_object = GridObject(hole_type, hole, [0], label, self.grid_shape, self.font_color, grid)
            if hole_type == 'inner_hole':
                inner_holes.append(hole_object)
            else:
                outer_holes.append(hole_object)
        return tuple(inner_holes), tuple(outer_holes)

    def find_object_center(self):
        """ Find the center cell of an object and precise geometric center each represented as a tuple of coordinate tuples."""
        sum_x = sum(cell[0] for cell in self.coords)
        sum_y = sum(cell[1] for cell in self.coords)
        num_cells = self.size
        # calculate precise geometric center, rounding for center cell
        geometric_center_x = sum_x / num_cells
        geometric_center_y = sum_y / num_cells  
        return (int(geometric_center_x), int(geometric_center_y)), (geometric_center_x, geometric_center_y)
                
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
        shape_colors = {colors_mapping[i]:0 for i in range(10)}
        colors = {colors_mapping[i]:0 for i in range(10)}
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
        - symmetry_type: Boolean value indicating symmetry [0/1]
        - compactness: Share of rectangle contour filled with non-font colors [0-1]
        - closure: Boolean value indicating closure [0/1]
        - inner_holes: Number of inner holes, normalized to range [0-1]
        - outer_holes: Number of outer holes, normalized to range [0-1]
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
        color_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 possible colors
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

        # 11. Symmetry type
        if self.shape not in ['inner_hole', 'outer_hole']:
            has_symmetry = int(hasattr(self, 'symmetry') and self.symmetry != 'assymetry')
            embedding_dict["symmetry_type"] = has_symmetry
        else:
            embedding_dict["symmetry_type"] = 0

        # 12. Compactness
        if self.shape not in ['inner_hole', 'outer_hole'] and hasattr(self, 'compactness'):
            embedding_dict["compactness"] = round(self.compactness, 3)
        else:
            embedding_dict["compactness"] = 0.0

        # 13. Closure
        if all(elem in self.coords for elem in self.contour):
            embedding_dict["closure"] = 1.0
        else:
            embedding_dict["closure"] = 0.0
        
        # 14-15. Holes
        if self.shape not in ['inner_hole', 'outer_hole']:
            n_inner_holes = len(self.inner_holes)
            n_outer_holes = len(self.outer_holes)
            embedding_dict["inner_holes"] = min(1.0, n_inner_holes / 5)  # Simplified
            embedding_dict["outer_holes"] = min(1.0, n_outer_holes / 5)  # Simplified
        else:
            embedding_dict["inner_holes"] = 0.0
            embedding_dict["outer_holes"] = 0.0        

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
        ])
        flat_vector.extend([embedding_dict["symmetry_type"]])
        flat_vector.extend([
            embedding_dict["compactness"],
        ])     
        flat_vector.extend([
            embedding_dict["closure"],
        ])     
        flat_vector.extend([
            embedding_dict["inner_holes"],
            embedding_dict["outer_holes"],
        ])
        flat_vector.extend([
            embedding_dict["inner_holes_share"]
        ])
        
        # Store as immutable tuples
        self.embedding_dict = dict(embedding_dict)  # Shallow copy to prevent external mutation
        self.embedding_vector = tuple(flat_vector)  # Make immutable
        
        return tuple(flat_vector)  # Return immutable

    def classify_shape(self) -> str:
        """Classify the shape type."""
        # Use the classification logic above
        shape_type = self._classify_shape()
        self.shape = shape_type  # Update the shape attribute
        if shape_type != 'complex':
            self.label = shape_type + '_' + self.label.split('_')[1]
        return shape_type

    def _classify_shape(self) -> str:
        """Improved shape classification using comprehensive pattern generation."""
        # Basic shapes first
        if self.hor_size == 1 and self.vert_size == 1:
            return 'cell'
        
        if self.hor_size == 1 or self.vert_size == 1:
            return 'line'
        
        if abs(1.0 - self.compactness) < 0.15 and self.size > 9 or abs(1.0 - self.compactness) < 0.02:
            return 'rectangle'
        
        # Enhanced diagonal detection
        if self._is_diagonal_shape():
            return 'diagonal'
        
        # Try pattern matching with generated shapes of same size
        shape_type = self._classify_complex_shapes()
        if shape_type != 'complex':
            return shape_type
        
        return 'complex'
            
    def _classify_complex_shapes(self) -> str:
        """Classify complex shapes based on symmetry, holes, and geometric properties."""
        if self.size < 3:
            return 'complex'
            
        if self.symmetry in ['horizontal_and_vertical_symmetry', 'both']:
            if self._is_cross_shape():
                return 'cross'
            elif self._is_flower_shape():
                return 'flower'  
                
        if self._is_tv_shape():
            return 'tv_shape'
    
        if self._is_hs_shape():
            return 'hs_shape'

        # Check for T-shape using intersection detection
        if self._is_t_shape():
            return 't_shape'
        
        # Check for L-shape using simple corner detection
        if self._is_l_shape():
            return 'l_shape'

        # Check for S-shape using zigzag detection
        if self._is_s_shape():
            return 's_shape'
                

        return 'complex'

    def _is_diagonal_shape(self) -> bool:
        """Enhanced diagonal detection."""
        if len(self.coords) < 2:
            return False
        
        if self._is_main_diagonal():
            return True
        
        if self._is_anti_diagonal():
            return True
        
        return False

    def _is_cross_shape(self) -> bool:
        """Check if shape is a cross (intersecting lines)."""
        if self.symmetry not in ['horizontal_and_vertical_symmetry', 'both']:
            return False
        
        # Cross typically has arms extending from center
        center_i, center_j = self.center
        center_cells = sum(1 for i, j in self.coords 
                          if abs(i - center_i) <= 1 and abs(j - center_j) <= 1)
        
        # Should have significant presence around center
        return center_cells >= 3 and self.compactness < 0.7 and not self._has_diagonal_connections()
    
    def _is_flower_shape(self) -> bool:
        """Check if shape is a flower (crossed diagonals)."""
        if not hasattr(self, 'hu_moments'):
            return False
        
        # Flower shapes have specific Hu moment patterns
        hu1, hu2, hu3, hu4, hu5, hu6, hu7 = self.hu_moments
        
        # Diagonal symmetry and specific moment patterns
        diagonal_symmetry = (self._check_diagonal_symmetry() and 
                            abs(hu3) > 2.0)  # Higher order moments for complex shapes
        
        return diagonal_symmetry and self.compactness < 0.6 and self.size < 13

    def _is_tv_shape(self) -> bool:
        """Check if shape is a TV shape (fully closed rectangle with interior holes)."""
        # TV shape should be fully closed rectangular contour
        if not (all(elem in self.contour for elem in self.coords) and all(elem in self.coords for elem in self.contour) and
                                  self.compactness < 0.9):
            return False
        
        # TV shape typically has 4 corners (a complete rectangle)
        corners = self._count_corners()
        if corners != 4: 
            return False
        
        # Should have significant hole area
        sufficient_holes = (self.inner_holes_share > 0.1 and 
                           len(self.inner_holes) >= 1)
        
        return len(self.contour) + 1 >= len(self.coords)
    
    def _is_hs_shape(self) -> bool:
        """Check if shape is HS shape (TV shape missing one edge)."""
        # if not self._is_approximately_rectangular():
        #     return False
    
        # HS shape should have exactly 2 proper corners
        corners = self._count_corners()
        if corners not in [2, 4]:
            return False
        
        # HS shape should not be fully closed (missing one edge)
        if all(elem in self.coords for elem in self.contour):
            return False     

        return all(elem in self.contour for elem in self.coords)
    
    def _is_l_shape(self) -> bool:
        """Simple L-shape detection using corner analysis."""
        pattern = self.obj_mask
        if pattern.shape[0] < 2 or pattern.shape[1] < 2 or self._has_diagonal_connections():
            return False
        
        height, width = pattern.shape
        
        # Count proper corners (cells that form 90-degree angles)
        proper_corners = 0
        
        for i in range(height):
            for j in range(width):
                if pattern[i, j] == 1:
                    # Check if this cell could be part of an L corner
                    neighbors = self._count_neighbors(i, j)
                    
                    # Corner cells typically have 2 neighbors in an L pattern
                    if neighbors == 2:
                        # Check if neighbors are perpendicular (not opposite)
                        has_up = i > 0 and pattern[i-1, j] == 1
                        has_down = i < height-1 and pattern[i+1, j] == 1
                        has_left = j > 0 and pattern[i, j-1] == 1
                        has_right = j < width-1 and pattern[i, j+1] == 1
                        
                        # Perpendicular if (up/down + left/right) but not both horizontal or both vertical
                        if (has_up or has_down) and (has_left or has_right):
                            proper_corners += 1
        
        # L-shape should have exactly one proper corner and reasonable size
        return proper_corners == 1 and 3 <= self.size <= 10
    
    def _is_t_shape(self) -> bool:
        """Simple T-shape detection using intersection point analysis."""
        pattern = self.obj_mask
        height, width = pattern.shape
        
        if height < 3 and width < 3 or self._has_diagonal_connections():
            return False
        
        # Find intersection points (cells with 3 neighbors)
        intersection_points = []
        
        for i in range(height):
            for j in range(width):
                if pattern[i, j] == 1:
                    neighbors = self._count_neighbors(i, j)
                    if neighbors == 3:
                        intersection_points.append((i, j))
                    elif neighbors > 3:
                        return False
        
        # T-shape should have exactly one intersection point
        if len(intersection_points) != 1:
            return False
        
        # Verify it's a proper T (not a cross)
        i, j = intersection_points[0]
        has_up = i > 0 and pattern[i-1, j] == 1
        has_down = i < height-1 and pattern[i+1, j] == 1
        has_left = j > 0 and pattern[i, j-1] == 1
        has_right = j < width-1 and pattern[i, j+1] == 1
        
        # Should have one arm in one direction and two in perpendicular direction
        vertical_arms = sum([has_up, has_down])
        horizontal_arms = sum([has_left, has_right])
        
        return (vertical_arms == 1 and horizontal_arms == 2) or (vertical_arms == 2 and horizontal_arms == 1)
    
    def _is_s_shape(self) -> bool:
        """Simple S-shape detection using curvature analysis."""
        pattern = self.obj_mask
        height, width = pattern.shape
        
        if (height < 2 or width < 2 or self.size < 4 or self.size > 25 
            or min(self.hor_size, self.vert_size) > 2 or self._has_diagonal_connections()):
            return False
        
        # Group coordinates by row and find the "center of mass" for each row
        rows = {}
        for i, j in self.coords:
            if i not in rows:
                rows[i] = []
            rows[i].append(j)
        
        # Sort by row and calculate average column position for each row
        sorted_rows = sorted(rows.keys())
        row_centers = []
        
        for row in sorted_rows:
            cols = rows[row]
            avg_col = sum(cols) / len(cols)
            row_centers.append((row, avg_col))
        
        # Now track the actual movement pattern
        if len(row_centers) < 3:
            return False
        
        # Calculate the direction changes between consecutive rows
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(row_centers)):
            current_diff = row_centers[i][1] - row_centers[i-1][1]
            
            if abs(current_diff) < 0.5:  # Too small to be significant
                current_direction = 'stable'
            elif current_diff > 0:
                current_direction = 'right'
            else:
                current_direction = 'left'
            
            if prev_direction is not None and current_direction != prev_direction and current_direction != 'stable':
                direction_changes += 1
            
            prev_direction = current_direction
        
        # For S-shape, we expect exactly 2 direction changes in a 3-part pattern
        return direction_changes == 2
    
    def _count_neighbors(self, i: int, j: int) -> int:
        """Count the number of adjacent neighbors (4-connected)."""
        pattern = self.obj_mask
        height, width = pattern.shape
        count = 0
        
        if i > 0 and pattern[i-1, j] == 1:  # up
            count += 1
        if i < height-1 and pattern[i+1, j] == 1:  # down
            count += 1
        if j > 0 and pattern[i, j-1] == 1:  # left
            count += 1
        if j < width-1 and pattern[i, j+1] == 1:  # right
            count += 1
        
        return count
        
    def _check_diagonal_symmetry(self) -> bool:
        """Check if shape has diagonal symmetry."""
        # Create binary pattern
        pattern = np.zeros((self.hor_size, self.vert_size), dtype=int)
        for coord in self.coords:
            i_rel = coord[0] - self.min_i
            j_rel = coord[1] - self.min_j
            pattern[i_rel, j_rel] = 1
        
        # Check main diagonal symmetry
        main_diag_symmetry = np.array_equal(pattern.T, pattern)
        
        # Check anti-diagonal symmetry  
        anti_diag_symmetry = np.array_equal(np.flipud(pattern).T, np.flipud(pattern))
        
        return main_diag_symmetry or anti_diag_symmetry
    
    def _is_main_diagonal(self) -> bool:
        """Check if shape is exactly on main diagonal (i = j + constant)."""
        # ... keep existing implementation unchanged ...
        differences = [i - j for i, j in self.coords]
        return len(set(differences)) == 1
    
    def _is_anti_diagonal(self) -> bool:
        """Check if shape is exactly on anti-diagonal (i + j = constant)."""
        # ... keep existing implementation unchanged ...
        sums = [i + j for i, j in self.coords]
        return len(set(sums)) == 1
    
    def _is_monotonic(self, x: List[int], y: List[int]) -> bool:
        """Check if coordinates form a monotonic sequence."""
        # ... keep existing implementation unchanged ...
        if len(x) < 2:
            return False
        
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        y_increasing = all(y_sorted[i] <= y_sorted[i+1] for i in range(len(y_sorted)-1))
        y_decreasing = all(y_sorted[i] >= y_sorted[i+1] for i in range(len(y_sorted)-1))
        
        return y_increasing or y_decreasing

    def _is_approximately_rectangular(self) -> bool:
        """Check if shape is approximately rectangular."""
        if not hasattr(self, 'hu_moments'):
            return False
        
        hu1, hu2, hu3, hu4, hu5, hu6, hu7 = self.hu_moments
        
        # Use absolute values
        return (abs(hu1) > 0.1 and  # Check absolute value
                abs(hu2) < 0.5 and 
                abs(hu3) < 0.3 and 
                abs(hu7) < 0.1)

    def _count_corners(self) -> int:
        """Count the number of proper 90-degree corners in the shape."""
        if not hasattr(self, 'obj_mask'):
            return 0
        
        pattern = self.obj_mask
        height, width = pattern.shape
        
        if height < 2 or width < 2:
            return 0
        
        corner_count = 0
        
        for i in range(height):
            for j in range(width):
                if pattern[i, j] == 1:
                    # Check if this cell forms a proper corner
                    if self._is_corner_cell(pattern, i, j):
                        corner_count += 1
        
        return corner_count
    
    def _is_corner_cell(self, pattern: np.ndarray, i: int, j: int) -> bool:
        """Check if a cell forms a proper 90-degree corner."""
        height, width = pattern.shape
        
        # Get neighborhood (handle boundaries)
        up = pattern[i-1, j] if i > 0 else 0
        down = pattern[i+1, j] if i < height-1 else 0
        left = pattern[i, j-1] if j > 0 else 0
        right = pattern[i, j+1] if j < width-1 else 0
        
        # Count neighbors
        neighbor_count = up + down + left + right
        
        # A corner cell should have exactly 2 neighbors that are perpendicular
        if neighbor_count != 2:
            return False
        
        # Check if neighbors are perpendicular (not opposite)
        has_vertical = (up or down)
        has_horizontal = (left or right)
        
        if not (has_vertical and has_horizontal):
            return False
        
        # Additional check: ensure it's a proper convex corner
        # by checking the diagonal cells (they should be empty for a sharp corner)
        if i > 0 and j > 0 and pattern[i-1, j-1] == 1 and up and left:
            return False  # Concave corner
        if i > 0 and j < width-1 and pattern[i-1, j+1] == 1 and up and right:
            return False  # Concave corner
        if i < height-1 and j > 0 and pattern[i+1, j-1] == 1 and down and left:
            return False  # Concave corner
        if i < height-1 and j < width-1 and pattern[i+1, j+1] == 1 and down and right:
            return False  # Concave corner
        
        return True

    def _has_diagonal_connections(self) -> bool:
        """Check if there are any cells connected only diagonally (no direct orthogonal connections between them)."""
        coord_set = set(self.coords)
        
        for i, j in self.coords:
            # Check diagonal neighbors
            diagonal_neighbors = [
                (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)  # diagonals
            ]
            
            for di, dj in diagonal_neighbors:
                if (di, dj) in coord_set:
                    # Check if these two diagonally connected cells lack direct orthogonal connection
                    orthogonal_connection_exists = (
                        (i, dj) in coord_set or  # vertical connection
                        (di, j) in coord_set     # horizontal connection
                    )
                    
                    # If no orthogonal connection exists between these specific cells, it's a pure diagonal connection
                    if not orthogonal_connection_exists:
                        return True
        return False