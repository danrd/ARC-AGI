class SubtaskSummary():
    def __init__(self, subtask:ARCSubtask, patterns:dict, train:bool=True):
        self.subtask = subtask
        self.subtask_label = self.subtask_label
        self.inp_grid_summary = GridSummary(self.subtask.train_inp, self.subtask.train_inp_shape, self.subtask_label)
        if train:
            self.out_grid_summary = GridSummary(subtask.train_out)

class GridSummary():
    def __init__(self, grid:np.array, shape:tuple):
        self.grid = grid
        self.shape = shape
        self.patterns = generate_patterns(shape[0])
        self.initial_objects = self.retrieve_objects(self.grid, self.patterns, self.shape)
        self.objects = self.filter_objects()
        self.objects_summary = self.create_objects_summary()
        self.triples = self.set_relations()
    
    @staticmethod
    def retrieve_objects(grid:np.array, patterns:typing.Dict['str', List[List[List[tuple]]]], shape:tuple)->typing.Dict[str, List[GridObject]]:
        """Retrieve all possible objects from the grid and return corresponding GridObject instances."""
        objects = defaultdict(list)
        candidate = False # flag indicating existance of candidate figure
        used_coordinates = [] # save occupied cells
        for k, v in patterns.items(): # iterating over shapes
            shape_patterns = v
            for idx, pattern_list in enumerate(shape_patterns):
                for pattern in pattern_list:
                    i, j = coords_tranform(pattern) # transform list of tuples into two lists for i and j coordinates
                    retrieval = set(grid[i, j]) # extract cells colors with lists of coordinates and keep only unique 
                    if len(retrieval) > 1: # if colors more than 1 - not a candidate
                        break
                    else:
                        label = f'{k}_{idx}'
                        obj = GridObject(k, pattern, retrieval.pop(), label) # otherwise create candidate object
                        used_coordinates.extend(pattern) # save occupied cells 
                        candidate = True
                if candidate:
                    objects[k].append(copy.deepcopy(obj))
                    candidate = False

        used_coordinates = set(used_coordinates) # keep only unique coordinates
        ul = find_upper_left_corner(shape[0])
        all_coordinates = set(product(range(ul,ul+shape[0]), range(ul,ul+shape[0])))
        cells_coordinates = list(all_coordinates.difference(used_coordinates))
        for idx, cell in enumerate(cells_coordinates): # if some cell is not belong to some figure - create cell object
            label = f'cell_{idx}'
            obj = GridObject('cell', [cell], grid[cell], label)
            objects['cell'].append(obj)
        return objects
    
    @staticmethod
    def merge_rectangles(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out smaller rectangles each of which is subset of some larger rectangle."""
        rects = defaultdict(list)
        deletion_list = [] 
        for obj in objects['rectangle']: # sort rectangles based on size
            key = obj.size
            rects[key].append(obj)
        keys = sorted(list(rects.keys()), reverse=True) # iterate over each possible size from larger to smaller
        for key_larger in keys[:-1]:
            for key_smaller in keys[1:]:
                for obj_larger in rects[key_larger]:
                    for idx, obj_smaller in enumerate(rects[key_smaller]):
                        if set(obj_smaller.coords).issubset(obj_larger.coords) and len(obj_smaller.coords)!=len(obj_larger.coords): # if smaller rectangle in larger rectangle - delete smaller
                            deletion_list.append((key_smaller, idx)) # save position of smaller rectangle 
        deletion_list = list(set(deletion_list))
        sorted_rects = [] # save rectangles that are not from deletion_list
        used_shapes = [] # save coordinates to exclude rectangles with the same coordinates
        for k, v in rects.items():
            for idx, rect in enumerate(v):
                if (k, idx) not in deletion_list and rect.coords not in used_shapes:
                    sorted_rects.append(rect)
                    used_shapes.append(rect.coords)
        objects['rectangle'] = sorted_rects
        return objects
    
    @staticmethod
    def merge_lines(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out smaller lines each of which is subset of some larger line."""
        lines = defaultdict(list)
        deletion_list = []
        for obj in objects['line']: # sort lines based on size
            key = obj.size
            lines[key].append(obj)
        keys = sorted(list(lines.keys()), reverse=True) # iterate over each possible size from larger to smaller
        for key_larger in keys[:-1]:
            for key_smaller in keys[1:]:
                for obj_larger in lines[key_larger]:
                    for idx, obj_smaller in enumerate(lines[key_smaller]):
                        if set(obj_smaller.coords).issubset(obj_larger.coords) and len(obj_smaller.coords)!=len(obj_larger.coords): # if smaller line in larger line - delete smaller
                            deletion_list.append((key_smaller, idx)) # save position of smaller line
        deletion_list = list(set(deletion_list))
        sorted_lines = [] # save rectangles that are not from deletion_list
        for k, v in lines.items():
            for idx, line in enumerate(v):
                if (k, idx) not in deletion_list and line.coords not in sorted_lines:
                    sorted_lines.append(line)
            objects['line'] = sorted_lines
        return objects
    
    @staticmethod
    def merge_in_rectangles(objects:typing.Dict[str, GridObject])->typing.Dict[str, GridObject]:
        """Filter out objects each of which is subset of some rectangle."""
        deletion_list = []
        filtered_objects = defaultdict(list) 
        filtered_objects['rectangle'] = objects['rectangle'] # don't need to filter rectangles and cells
        filtered_objects['cell'] = objects['cell']
        other_shapes = {k:v for k, v in objects.items() if k!='rectangle' and k!='cell'} # take all shapes without rectangles and cells
        for rect in objects['rectangle']: # iterate over each rectangle
            for k, v in other_shapes.items():
                for idx, shape in enumerate(v):
                    if set(shape.coords).issubset(rect.coords): # delete shape if it is inside rectangle
                        deletion_list.append((k, idx))
        deletion_list = list(set(deletion_list))
        for k, v in other_shapes.items():
            for idx, shape in enumerate(v):
                if (k, idx) not in deletion_list:
                    filtered_objects[k].append(shape)
        return filtered_objects
    
    def filter_objects(self):
        """Apply all filtration approaches for the objects."""
        objects_after_rectangle_merging = self.merge_rectangles(self.initial_objects)
        objects_after_lines_merging = self.merge_lines(objects_after_rectangle_merging)
        objects_after_merging_in_rectangles = self.merge_in_rectangles(objects_after_lines_merging)
        return objects_after_merging_in_rectangles
    
    def create_objects_summary(self):
        """Create a summary for grid objects to get aggregate information about their shapes, sizes, colors.""" 
        sizes = defaultdict(list)
        shapes = defaultdict(lambda: 0)
        colors = defaultdict(lambda: 0)
        for k, v in self.objects.items():
            for idx, obj in enumerate(v):
                sizes[obj.size].append(obj)
                shapes[obj.shape] += 1
                color = obj .colors_mapping[obj.color_number]
                colors[color] += 1
        sorted_keys = sorted(list(sizes.keys()), reverse=True)
        sizes = {k:sizes[k] for k in sorted_keys}
        summary = {'sizes':sizes, 'shape':shapes, 'colors':colors}
        return summary
    
    def set_relations(self):
        """Iterate over objects to identify relations between them.""" 
        triples = []
        complex_shapes = []
        used_pairs = []
        for k_1, v_1 in self.objects.items():
            for shape_1 in v_1:
                for k_2, v_2 in self.objects.items():
                    for shape_2 in v_2:
                        if shape_1 != shape_2 and (shape_1.label, shape_2.label) not in used_pairs:
                            used_pairs.append((shape_1.label, shape_2.label))
                            summary = RelationAnalyzer(shape_1, shape_2, self.shape)
                            triples.extend(summary.triples)
                            if summary.complex_shape:
                                complex_shapes.append(summary.complex_shape)
        while complex_shapes!=[]:
            complex_shape = complex_shapes.pop()
            self.objects['complex_shape'].append(complex_shape)
            for k, v in self.objects.items():
                for shape in v:
                    if complex_shape != shape_2 and (complex_shape.label, shape_2.label) not in used_pairs:
                        used_pairs.append((complex_shape.label, shape_2.label))
                        summary = RelationAnalyzer(complex_shape, shape_2, self.shape)
                        triples.extend(summary.triples) 
                        if summary.complex_shape:
                            complex_shapes.append(summary.complex_shape)
        return triples

class GridObject():
    def __init__(self, shape:str, coords:List[tuple], color:int, label:str):
        self.shape = shape
        self.coords = coords
        self.size = len(coords)
        self.edges = self.define_edges()
        self.hor_size = self.edges[1] - self.edges[0] + 1
        self.vert_size = self.edges[3] - self.edges[2] + 1
        self.min_i = self.edges[0]
        self.max_i = self.edges[1]
        self.min_j = self.edges[2]
        self.max_j = self.edges[3]
        self.colors_mapping = {0:'black', 1:'blue', 2:'red', 3:'green', 4:'yellow', 5:'gray', 6:'magenta',
                 7:'orange', 8:'sky', 9:'brown', -1:'white'}
        self.color_number = color
        self.color = self.colors_mapping[color]
        self.relations = defaultdict(list)
        self.label = f'{color}_{label}'
        
    
    def __repr__(self):
        return f'{self.color} {self.shape} with horizontal size {self.hor_size} and vetrical size {self.vert_size} with coordinates {self.coords}'
    
    def __eq__(self, other_GridObject):
        isGridObject = isinstance(other_GridObject, self.__class__)
        if not isGridObject:
            return False
        else:
            return self.coords == other_GridObject.coords and self.color_number == other_GridObject.color_number
    
    def define_edges(self):
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
    
    def plot(self):
        grid = np.zeros((30,30))
        for coord in self.coords:
            grid[coord] = self.color_number
        plot_grid(grid)          