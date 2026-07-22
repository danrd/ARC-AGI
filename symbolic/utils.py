import typing
import copy
from typing import List, Union
import sys
import numpy as np
from collections import defaultdict

def find_upper_left_corner(grid_size:tuple)->tuple:
    """Finds left upper corner of the grid to take into account padding."""
    i = min(14-(grid_size[0]%2)*((grid_size[0]//2)), 14-((grid_size[0]-1)%2)*(((grid_size[0]-1)//2)))
    j = min(14-(grid_size[1]%2)*((grid_size[1]//2)), 14-((grid_size[1]-1)%2)*(((grid_size[1]-1)//2)))
    return (0, 0)

def calculate_size(shapes):
    """Calculation to control a size of pregenerated shapes."""
    el_num = 0
    size = 0
    for k, v in shapes.items():
        for color in range(10):
            el_num += len(v[color])
            size += sys.getsizeof(v[color])/1048576
    print(f'number of elements in dictionary: {el_num}')
    print(f'size of dictionary: {size} mb')

def decompose_into_summands(number:int)->typing.Dict[int, List[np.array]]:
    summands = defaultdict(list)
    for k in range(number-1,0,-1):
        subsummands = []
        for i in range(0, k+1):
            j  = k-i
            subsummands.append((i, j))
        summands[i+j+1].append(subsummands)
    return summands

def dict_merge(dict_1:dict, dict_2:dict)->dict:
    keys = list(dict_1.keys())
    for k, v in dict_2.items():
        if k in keys:
            dict_1[k].extend(v)
        else:
            dict_1[k] = v
    return dict_1

def dict_to_list(dict):
    res_list = []
    keys = dict.keys()
    for key in keys:
        res_list.extend(dict[key])
    return res_list

def coords_transform(shape:List[tuple]):
    """Transform list of tuples into two lists for i and j coordinates."""
    return [tup[0] for tup in shape], [tup[1] for tup in shape]

def define_grid_cells(max_grid_size:int=30)->dict:
    """Defines admissible cells for all possible grid sizes."""
    grid_cells = {(1,1):[(14,14)]}
    i_right = 14
    j_right = 14
    i_left = 14
    j_left = 14
    cells = [(14,14)]
    for grid_size in range(2, max_grid_size+1):
        pattern = grid_size%2
        if pattern == 0:
            i_right += 1
            j_right += 1
            for j in range(j_left, j_right+1):
                cell = (i_right, j)
                cells.append(cell)
            for i in range(i_left, i_right):
                cell = (i, j_right)
                cells.append(cell)
        elif pattern == 1:
            i_left += -1
            j_left += -1
            for j in range(j_left, j_right+1):
                cell = (i_left, j)
                cells.append(cell)
            for i in range(i_left+1, i_right+1):
                cell = (i, j_left)
                cells.append(cell)
        cells_copy = copy.copy(cells)
        grid_cells[(grid_size, grid_size)] = cells_copy
    return grid_cells

grid_cells = define_grid_cells()

def grid_mapping(coords:List[tuple], grid_cells:dict)->tuple:
    """Maps a list of coordinates to minimal grid shape for which the list is admissible."""
    fig_size = len(coords)
    for k, v in grid_cells.items():
        for i in range(fig_size):
            if coords[i] in v:
                if i == fig_size-1:
                    return k
                else:
                    continue
            else:
                break

def create_figures(figures:dict)->typing.Dict[tuple, List[List[List[tuple]]]]:
    """Transforms a list of figures into corresponding colored grids."""
    colored_figures_dict = {}
    for k, v in figures.items(): # for eache figure
        figures = []
        for color in range(0, 10): # for eache color
            colored_figures = []
            for coord_list in v:
                grid = np.zeros((30,30))
                for coord in coord_list:
                    grid[coord] = color
                colored_figures.append(grid)
            figures.append(colored_figures)
        colored_figures_dict[k] = figures
    return colored_figures_dict

def count_unique_cells(shape:str, shape_coords:List[tuple], used_cells:List[tuple])->int:
    """Returns a number of shape cells that are already related to some other shape."""
    if shape != 'diagonal':
        return 0
    else:
        return sum(1 for cell in shape_coords if cell not in used_cells)

def shape_shift(shape:List[tuple], x_shift:int, y_shift:int):
    """Shifts each coordinate of the shape by x_shift and y_shift values."""
    return [(coord[0] + x_shift, coord[1] + y_shift) for coord in shape]

def perform_mapping(shape_dict:dict, shape:List[tuple], grid_cells)->dict:
    """Udpates dictionary with shapes adding new shape with correspinding key."""
    key = grid_mapping(shape, grid_cells)
    shape_dict[key].append(shape)
    return shape_dict

def is_admissible(shape:List[tuple], grid_size:tuple)->bool:
    """Defines possibility of placing shape inside grid."""
    ul = find_upper_left_corner(grid_size)
    i_coords = range(ul[0], grid_size[0]+ul[0])
    j_coords = range(ul[1], grid_size[1]+ul[1])
    shape.reverse()
    for coord in shape:
        if coord[0] in i_coords and coord[1] in j_coords:
            continue
        else:
            return False
    return True

def multiplicate_shapes(shapes, grid_size:tuple)->dict:
    """Multiplicates shapes shifting their coordinates inside grid."""
    multiplied_shapes = []
    for shape in shapes:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                new_shape = shape_shift(shape, i, j)
                if is_admissible(new_shape, grid_size):
                      multiplied_shapes.append([new_shape])
    return multiplied_shapes

def is_admissible2(grid_admissible:List[tuple], shape:List[tuple])->bool:
    """Defines possibility of placing shape inside grid."""
    for coord in shape:
        if coord not in grid_admissible:
            return False
        else:
            continue
    return True

def multiplicate_shapes2(shapes, grid_size:tuple)->dict:
    """Multiplicates shapes shifting their coordinates inside grid."""
    ul = find_upper_left_corner(grid_size)
    grid_admissible = [(ul[0]+i, ul[1]+j) for i in range(grid_size[0]) for j in range(grid_size[1])]
    multiplied_shapes = []
    for shape in shapes:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                new_shape = shape_shift(shape, i, j)
                if is_admissible(grid_admissible, new_shape):
                      multiplied_shapes.append([new_shape])
    return multiplied_shapes

def check_subset_condition(larger_obj:set, smaller_obj:list)->bool:
    """Check if all coordinates of smaller object are occupied be larger object."""
    for coord in smaller_obj:
        if coord in larger_obj:
            continue
        else:
            return False
    return True

def grid_formatting(grid:Union[np.array, List[list], List[tuple]])->np.array:
    """Unify grid format for processing as there is initial dataset format with ints and normalized from ARCDataset with floats."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    max_el = grid.max()
    if max_el >= 1 and type(max_el) in [np.int64, np.int32, np.int16, np.int8]:
       return grid.astype(int)
    else:
      return (grid*10).astype(int)

def crop_pad(grid: np.ndarray, pad_val=10) -> np.ndarray:
    """Return grid without padding."""
    # Find non-padding elements
    i, j = np.where(grid != pad_val)

    if len(i) == 0:  # Handle empty grids
        return grid

    # Find the boundaries
    min_i, max_i = min(i), max(i)
    min_j, max_j = min(j), max(j)

    # Extract the cropped region directly
    cropped_grid = grid[min_i:max_i+1, min_j:max_j+1]
    return cropped_grid

def adjust_grid_shape(grid:np.array, target_shape:tuple=(30,30), pad_value:int=10, normalize:bool=True)->np.array:
    """Transform any grid to target shape with padding."""
    shape_x = grid.shape[0]
    shape_y = grid.shape[1]
    target_x = target_shape[0]
    target_y = target_shape[1]
    reshaped_grid = copy.copy(grid)
    if shape_x!=target_x or shape_y!=target_y:
        left_pad = (target_x-shape_x)//2
        right_pad = target_x - shape_x - left_pad
        upper_pad = (target_y-shape_y)//2
        down_pad = target_y - shape_y - upper_pad
        reshaped_grid = np.pad(grid, pad_width=[(left_pad,right_pad), (upper_pad, down_pad)], constant_values=pad_value)
    if normalize:
        reshaped_grid = reshaped_grid/10
    return reshaped_grid

def augment_grid(grid:np.array)->List[np.array]:
    new_grids = []
    new_grid = crop_pad((grid*10).astype(int))
    new_grids.append(np.rot90(new_grid,k=1))
    new_grids.append(np.rot90(new_grid,k=2))
    new_grids.append(np.rot90(new_grid,k=3))
    new_grids.append(np.fliplr(new_grid))
    new_grids.append(np.flipud(new_grid))
    for inc in range(1, 10):
        grid_recolored = ((new_grid+inc)%10).astype(int)
        new_grids.append(grid_recolored)
    return new_grids

def check_grid_values(grid:np.array):
    """Check grid values for validity."""
    check = (grid >= 0) * (grid <= 1)
    for v in range(1, 10):
        check *= grid != v * 0.01
    return np.all(check)

def pad_grid(grid:np.array, target_shape, pad_val):
    """Create padded array of given shape using defined padding value."""
    shape_x, shape_y = grid.shape
    left_pad = (target_shape[0]-shape_x)//2
    right_pad = target_shape[0] - shape_x - left_pad
    upper_pad = (target_shape[1]-shape_y)//2
    down_pad = target_shape[1] - shape_y - upper_pad
    padded_grid = np.pad(grid, pad_width=[(left_pad,right_pad), (upper_pad, down_pad)], constant_values=pad_val)
    return padded_grid

