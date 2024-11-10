

class GridObject():
    """Class for storing identified objects on a grid."""
    def __init__(self, shape:str, coords:List[tuple], color:List[float], label:str, positioning=[]):
        self.shape = shape
        self.coords = coords
        self.size = len(coords)
        self.positioning = positioning
        self.edges = self.define_edges()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        self.colors_mapping = {0:'black', 0.1:'blue', 0.2:'red', 0.3:'green', 0.4:'yellow', 0.5:'gray', 0.6:'magenta',
                 0.7:'orange', 0.8:'sky', 0.9:'brown', 1:'white'}
        self.color_number = color
        self.color = [self.colors_mapping[color] for color in self.color_number]
        self.relations = defaultdict(list)
        self.label = label
        self.symmetry = self.check_symmetry
        self.positioning = positioning
        
    def __repr__(self):
        if self.shape != 'complex':
            return f'{self.color[0]} {self.shape} with horizontal size {self.hor_size} and vetrical size {self.vert_size} with coordinates {self.coords}'
        else:
            return self.label
    
    def __eq__(self, other_GridObject):
        isGridObject = isinstance(other_GridObject, self.__class__)
        if not isGridObject:
            return False
        else:
            return self.coords == other_GridObject.coords and self.color_number == other_GridObject.color_number
    
    def define_edges(self):
        """Calculate max and min values along each axis."""
        coords = self.coords
        max_i = 0
        max_j = 0
        min_i = 30
        min_j = 30
        for cell in coords:
            if cell[0] > max_i:
                max_i = cell[0]
            if cell[0] < min_i:
                min_i = cell[0]
            if cell[1] > max_j:
                max_j = cell[1]
            if cell[1] < min_j:
                min_j = cell[1]
        return (min_i, max_i, min_j, max_j)
    
    def check_symmetry(self):
        """Identify symmetric propetries for the object."""
        grid = np.zeros((self.max_i-self.min_i, self.max_j-self.min_j))
        symmetries = []
        for coord in self.coords:
            grid[coord] = 1
        if (np.flipud(grid)==grid).all():
            symmetries.append('horizontal_symmetry')
        if (np.fliplr(grid)==grid).all():
            symmetries.append('vertical_symmetry')
        if len(symmetries) == 0:
            symmetry = 'assymetry'
        elif len(symmetries) == 2:
            symmetry = 'horizontal_and_vertical_symmetry'
        else: 
            symmetry = symmetries[0]
        return symmetry

    def create_object_triples(self):
        """Create triples based on objects properties."""
        triples = [] 
        triples.append((self.label, 'has', self.symmetry))
        for position in self.positioning:
            triples.append((self.label, 'located', position))
        triples.append((self.label, 'has_shape', self.shape))
        triples.append((self.label, 'has_color', self.color))
        triples.append((self.label, 'has_size', self.size))
        return triples
    
    def plot(self):
        """Plot the object."""
        grid = np.zeros((30,30))
        for coord in self.coords:
            grid[coord] = self.color_number
        plot_grid(grid)

class ObjectsFilter():
    """Class for filtering out potentialy unimportant objects."""
    def __init__(self, objects:typing.Dict[str, List[GridObject]]):
        self.objects= objects

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
                        if obj_smaller[0].size != obj_larger[0].size and  obj_smaller[1].issubset(obj_larger[1]): # if smaller line in larger line - delete smaller
                            deletion_list.append((key_smaller, idx)) # save position of smaller line
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
        select_shapes = ['rectangle', 'cell']
        filtered_objects = defaultdict(list) | {k:v for k, v in objects.items() if k in select_shapes} 
        other_objects = {k:v for k, v in objects.items() if k not in select_shapes}
        for rect in filtered_objects['rectangle']: # iterate over each rectangle
            rect_set = set(rect.coords)
            for k, v in other_objects.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(rect_set, shape.coords): # delete shape if it is inside rectangle
                        deletion_list.append((k, idx))
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
        for markup in filtered_objects['markup']: # iterate over each markup
            markup_set = set(markup.coords)
            for k, v in other_shapes.items(): # iterate over each shape from select_shapes
                for idx, shape in enumerate(v):
                    if check_subset_condition(markup_set, shape.coords): # delete shape if it is inside markup
                        deletion_list.append((k, idx))
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
        objects_after_merging_in_rectangles = self.merge_in_rectangles(objects_after_lines_merging)
        objects_after_merging_in_t_shapes = self.merge_in_t_shapes(objects_after_merging_in_rectangles) 
        objects_after_merging_in_s_shapes = self.merge_in_s_shapes(objects_after_merging_in_t_shapes) 
        objects_after_merging_in_hs_shapes = self.merge_in_hs_shapes(objects_after_merging_in_s_shapes)  
        objects_after_merging_in_tv_shapes = self.merge_in_tv_shapes(objects_after_merging_in_hs_shapes)
        objects_after_merging_crosses = self.merge_in_crosses(objects_after_merging_in_tv_shapes)  
        objects_after_merging_in_markup = self.merge_in_markup(objects_after_merging_crosses)
        return objects_after_merging_in_markup

class RelationAnalyzer():
    """Class for setting relations between objects on a grid."""
    def __init__(self, object_1:GridObject=None, object_2:GridObject=None, shape:tuple=None):
        self.object_1 = object_1
        self.object_2 = object_2
        self.shape = shape
        self.triples = self.set_relations()
    
    @staticmethod
    def rotation_symmetry(coords_1, coords_2, shape):
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
    def translation_symmetry(coords_1, coords_2, shape):
        """Identify if each coordinate of object_1 equals each coordinate of object_2 after some shifting."""
        ul = find_upper_left_corner(shape)
        coords_1_shifted = [(tup[0]-ul[0], tup[1]-ul[1]) for tup in coords_1]
        coords_2_shifted = [(tup[0]-ul[0], tup[1]-ul[1]) for tup in coords_2]
        i_offset = 0
        j_offset = 0
        i_offsets = []
        j_offsets = []
        if len(coords_1) == len(coords_2):
            for coord_1 in coords_1_shifted:
                for coord_2 in coords_2_shifted:
                    i_offsets.append(coord_1[0]-coord_2[0])
                    j_offsets.append(coord_1[1]-coord_2[1])
        set_i_offsets = set(i_offsets)
        set_j_offsets = set(j_offsets)
        if len(set_i_offsets) == 1 and len(set_j_offsets) == 1:
            i_offset = list(set_i_offsets)[0]
            j_offset = list(set_j_offsets)[0]
        return (i_offset, j_offset)   

    @staticmethod
    def in_contour(object_1, object_2):
        """Identify if all coordinates of object_1 are surrounded by coordinates of object_2 or in the reverse order."""
        in_contour = None
        if object_2.max_i < object_1.max_i and object_2.max_j < object_1.max_j and object_2.min_i > object_1.min_i and object_2.min_j > object_1.min_j:
            in_contour = 'object_2'
        if object_1.max_i < object_2.max_i and object_1.max_j < object_2.max_j and object_1.min_i > object_2.min_i and object_1.min_j > object_2.min_j:
            in_contour = 'object_1'        
        return in_contour   

    @staticmethod
    def in_line(coords_1, coords_2):
        """Identify if object_1 and object_2 can be connected by line."""
        list_i_1, list_j_1 = coords_transform(coords_1)
        list_i_2, list_j_2 = coords_transform(coords_2)
        i_intersection = set(list_i_1) & set(list_i_2)
        j_intersection = set(list_j_1) & set(list_j_2)
        return set(list_i_1) & set(list_i_2) or set(list_j_1) & set(list_j_2)
            
    def set_relations(self):
        """Set all considered relations."""
        assert self.object_1!=None and self.object_2!=None and self.shape!=None, f"Object_1, Object_2 and grid shape should be specified"
        triples = []  

        if self.object_1.color == self.object_2.color:
            self.object_1.relations[self.object_2.label].append(f"same_color")
            self.object_2.relations[self.object_1.label].append(f"same_color")
            triples.append((self.object_2, f"same_color", self.object_1))
            triples.append((self.object_1, f"same_color", self.object_2))
            
        if self.object_1.shape == self.object_2.shape:
            self.object_1.relations[self.object_2.label].append(f"same_shape")
            self.object_2.relations[self.object_1.label].append(f"same_shape")
            triples.append((self.object_2, f"same_shape", self.object_1))
            triples.append((self.object_1, f"same_shape", self.object_2))
            
        if self.object_1.size == self.object_2.size:
            self.object_1.relations[self.object_2.label].append(f"same_size")
            self.object_2.relations[self.object_1.label].append(f"same_size")
            triples.append((self.object_2, f"same_size", self.object_1))
            triples.append((self.object_1, f"same_size", self.object_2))
            
        rotations = self.rotation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if rotations != []:
            for rotation in rotations:
                self.object_1.relations[self.object_2.label].append(rotation)
                self.object_2.relations[self.object_1.label].append(rotation)
                triples.append((self.object_1, rotation, self.object_2))
                triples.append((self.object_2, rotation, self.object_1))
        
        (i_offset, j_offset) = self.translation_symmetry(self.object_1.coords, self.object_2.coords, self.shape)
        if i_offset != 0 and j_offset != 0:
            self.object_1.relations[self.object_2.label].append(f"translation_symmetry")
            self.object_2.relations[self.object_1.label].append(f"translation_symmetry")
            triples.append((self.object_1, f"translation_symmetry", self.object_2))
            triples.append((self.object_2, f"translation_symmetry", self.object_1))

        in_contour = self.in_contour(self.object_1, self.object_2)
        if in_contour == "object_2":
            self.object_1.relations[self.object_2.label].append(f"has_in_contour")
            self.object_2.relations[self.object_1.label].append(f"in_contour")
            triples.append((self.object_2, f"in_contour", self.object_1))
            triples.append((self.object_1, f"has_in_contour", self.object_2))

        if in_contour == "object_1":
            self.object_1.relations[self.object_2.label].append(f"in_contour")
            self.object_2.relations[self.object_1.label].append(f"has_in_contour")
            triples.append((self.object_1, f"in_contour", self.object_2))
            triples.append((self.object_2, f"has_in_contour", self.object_1))

        if self.in_line(self.object_1.coords, self.object_2.coords):
            self.object_1.relations[self.object_2.label].append(f"in_line")
            self.object_2.relations[self.object_1.label].append(f"in_line") 
            triples.append((self.object_1, f"in_line", self.object_2))
            triples.append((self.object_2, f"in_line", self.object_1))             
        return triples

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