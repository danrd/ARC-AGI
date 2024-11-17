import typing
import copy
from typing import List
import sys
import numpy as np
from collections import defaultdict

def find_upper_left_corner(grid_size:tuple)->tuple:
    """Finds left upper corner of the grid to take into account padding."""
    i = min(14-(grid_size[0]%2)*((grid_size[0]//2)), 14-((grid_size[0]-1)%2)*(((grid_size[0]-1)//2)))
    j = min(14-(grid_size[1]%2)*((grid_size[1]//2)), 14-((grid_size[1]-1)%2)*(((grid_size[1]-1)//2)))
    return (i, j)

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
    center = 14
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