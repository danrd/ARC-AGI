import numpy as np
import typing
import functools
from typing import List, Tuple
from copy import copy
from symbolic.utils import find_upper_left_corner, multiplicate_shapes
from concurrent.futures import ThreadPoolExecutor
from collections import deque

def left_lines(grid_size:tuple, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for left lines of diagonals creations."""
    lines = []
    ul = find_upper_left_corner(grid_size)
    i_coords = range(ul[0], grid_size[0]+ul[0])
    j_coords = range(ul[1], grid_size[1]+ul[1])
    i, j = pos
    offset_i, offset_j, offset_ii, offset_jj = 0, 0, 0, 0
    base = [(i, j)]
    max_dim = max(grid_size)
    for _ in range(1, max_dim):
        offset_i += -1
        offset_j += -1
        new_cell = (i+offset_i, j+offset_j)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy(base))
        offset_ii += 1
        offset_jj += 1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy(base))
    return lines

def right_lines(grid_size:tuple, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for right lines of diagonals creations."""
    lines = []
    ul = find_upper_left_corner(grid_size)
    i_coords = range(ul[0], grid_size[0]+ul[0])
    j_coords = range(ul[1], grid_size[1]+ul[1])
    i, j = pos
    i, j = pos
    offset_i, offset_j, offset_ii, offset_jj = 0, 0, 0, 0
    base = [(i, j)]
    max_dim = max(grid_size)
    for _ in range(1, max_dim):
        offset_i += -1
        offset_j += 1
        new_cell = (i+offset_i, j+offset_j)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy(base))
        offset_ii += 1
        offset_jj += -1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy(base))
    return lines

def diagonals_coords(grid_size:tuple)->List[List[List[tuple]]]:
    """Create diagonal patterns for given grid size."""
    all_lines = []
    ul = find_upper_left_corner(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pos = (i+ul[0], j+ul[1])
            lefts = left_lines(grid_size, pos)
            rights = right_lines(grid_size, pos)
            if lefts != []:
                all_lines.append(lefts)
            if rights != []:
                all_lines.append(rights)
    return all_lines

def lines_coords(grid_size:tuple)->List[List[List[tuple]]]:
    """Create line patterns for given grid size."""
    coords = []
    def hor_lines_coords(grid_size)->List[List[List[tuple]]]:
        ul = find_upper_left_corner(grid_size)
        hor_lines = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]-1):
                lines = []
                line = [(i+ul[0], j+ul[1])]
                for k in range(1, grid_size[1]-j):
                    l = (i+ul[0], j+k+ul[1])
                    line.append(l)
                    lines.append((copy(line)))
                hor_lines.append(copy(lines))
        return hor_lines
    def vert_lines_coords(grid_size:tuple)->List[List[List[tuple]]]:
        ul = find_upper_left_corner(grid_size)
        vert_lines = []
        for i in range(grid_size[0]-1):
            for j in range(grid_size[1]):
                lines = []
                line = [(i+ul[0], j+ul[1])]
                for k in range(1, grid_size[0]-i):
                    l = (i+k+ul[0], j+ul[1])
                    line.append(l)
                    lines.append((copy(line)))           
                vert_lines.append(copy(lines))
        return vert_lines
    coords = hor_lines_coords(grid_size) + vert_lines_coords(grid_size)
    return coords

@functools.lru_cache(maxsize=128)
def rectangles_coords(grid_size:tuple)->List[List[List[tuple]]]:
    """Create rectangle patterns for given grid size."""
    def i_expansion(grid_size):
        """Patterns expanding along i axis.""" 
        ul = find_upper_left_corner(grid_size)
        coords = []
        for size_i in range(2, grid_size[0]+1):
            hor_rects = []
            hor_rects_2 = [[] for _ in range(2, size_i)]
            hor_rects_3 = [[] for _ in range(1, grid_size[1]-1)]
            for size_j in range(2, grid_size[1]+1):
                base_hor = [(ii+ul[0], jj+ul[1]) for jj in range(size_j) for ii in range(size_i)]
                base_hor_inv = [(ii+ul[0], jj+ul[1]) for ii in range(size_i) for jj in range(size_j)]
                hor_rects.append(copy(base_hor))
                for i in range(1, size_i-1):   
                   hor_rects_2[i-1].append(copy(base_hor[i*size_j:]))
                for j in range(1, size_j-1):
                   hor_rects_3[j-1].append(copy(base_hor_inv[j*size_i:]))           
            coords.append(hor_rects)
            coords.extend(hor_rects_2)
            coords.extend(hor_rects_3)
        return coords

    def j_expansion(grid_size):
        """Patterns expanding along j axis."""
        ul = find_upper_left_corner(grid_size)
        coords = []
        for size_j in range(2, grid_size[1]+1):
            hor_rects = []
            hor_rects_2 = [[] for _ in range(2, size_j)]
            hor_rects_3 = [[] for _ in range(1, grid_size[0]-1)]
            for size_i in range(2, grid_size[0]+1):
                base_hor = [(ii+ul[0], jj+ul[1]) for jj in range(size_j) for ii in range(size_i)]
                base_hor_inv = [(ii+ul[0], jj+ul[1]) for ii in range(size_i) for jj in range(size_j)]
                hor_rects.append(copy(base_hor))
                for j in range(1, size_j-1):   
                   hor_rects_2[j-1].append(copy(base_hor[j*size_i:]))
                for i in range(1, size_i-1):
                   hor_rects_3[i-1].append(copy(base_hor_inv[i*size_j:]))           
            coords.append(hor_rects)
            coords.extend(hor_rects_2)
            coords.extend(hor_rects_3)
        return coords
    
    def i_j_expansion(grid_size):
        """Square patterns."""
        ul = find_upper_left_corner(grid_size)
        coords = [] 
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                sq_rects = []
                min_dim = min(grid_size[0]-i+1, grid_size[1]-j+1)
                for size in range(2, min_dim):
                    base_sq = [(ii+ul[0]+i, jj+ul[1]+j) for ii in range(size) for jj in range(size)]
                    sq_rects.append(copy(base_sq))
                coords.append(copy(sorted(sq_rects, key=lambda x: x[0]))) 
        return coords

    coords = []
    coords.extend(i_expansion(grid_size))
    coords.extend(j_expansion(grid_size))
    coords.extend(i_j_expansion(grid_size))
    return coords

def generate_hs_shapes(grid_size:tuple, short:bool=True)->List[List[List[tuple]]]:
    """Create hs-shapes patterns for given grid size."""
    hs_shapes = []
    ul_init = find_upper_left_corner(grid_size)
    for i_pos in range(grid_size[0]):
        for j_pos in range(grid_size[1]):
            max_i_increment = min(4, grid_size[0]-i_pos+1) if short else grid_size[0] - i_pos+1
            max_j_increment = min(4, grid_size[1]-j_pos+1) if short else grid_size[1] - j_pos+1
            ul = (ul_init[0]+i_pos, ul_init[1]+j_pos)
            for j in range(3, max_j_increment):
                base = [(ul[0], ul[1]+jj) for jj in range(j)] # create base line
                cache = []
                max_expanding_increment = min(3, grid_size[0] - i_pos) if short else grid_size[0] - i_pos
                for i in range(1, max_expanding_increment): # expanding left and right lines 
                    new_cell_left = (i+ul[0], ul[1])
                    base.append(new_cell_left)
                    new_cell_right = (i+ul[0], j-1+ul[1])
                    base.append(new_cell_right)
                    cache.append(copy(base))
                if cache != []:
                    hs_shapes.append(copy(cache))
            for j in range(3, max_j_increment):
                cache = []
                max_expanding_increment =  min(3, grid_size[0] - i_pos) if short else grid_size[0] - i_pos
                for i in range(1, max_expanding_increment): # base line shifting
                    base = [(i+ul[0], jj+ul[1]) for jj in range(j)] # create base line
                    for ii in range(i): # expanding left and right lines 
                        new_cell_left = (ii+ul[0], 0+ul[1])
                        base.append(new_cell_left)
                        new_cell_right = (ii+ul[0], j-1+ul[1])
                        base.append(new_cell_right)
                    cache.append(copy(base))
                if cache != []: 
                    hs_shapes.append(copy(cache))
            for i in range(3, max_i_increment):
                base = [(k+ul[0], ul[1]) for k in range(i)] # create base line     
                cache = []
                max_expanding_increment =  min(3, grid_size[1] - j_pos) if short else grid_size[1] - j_pos
                for j in range(1, max_expanding_increment): # expanding upper and bottom lines 
                    new_cell_left = (ul[0], j+ul[1])
                    base.append(new_cell_left)
                    new_cell_right = (i-1+ul[0], j+ul[1]) 
                    base.append(new_cell_right)
                    cache.append(copy(base))
                if cache != []:
                    hs_shapes.append(copy(cache))
            for i in range(3, max_i_increment):
                cache = []
                max_expanding_increment = min(3, grid_size[1] - j_pos) if short else grid_size[1] - j_pos
                for j in range(1, max_expanding_increment): # base line shifting
                    base = [(k+ul[0], j+ul[1]) for k in range(i)] # create bottom line
                    for jj in range(j): # expanding upper and bottom lines 
                        new_cell_left = (i-1+ul[0], jj+ul[1])
                        base.append(new_cell_left)
                        new_cell_right = (ul[0], jj+ul[1])
                        base.append(new_cell_right)
                    cache.append(copy(base))
                if cache != []: 
                    hs_shapes.append(copy(cache))
    return hs_shapes

def generate_l_shapes(grid_size:tuple, regime:str='short')->List[List[List[tuple]]]:
    """Create l-shapes patterns for given grid size."""
    ul = find_upper_left_corner(grid_size)
    if regime == 'short': # create only most common l_shapes patterns
        l_shapes = []
        shape_ul = [(0+ul[0], ul[1]), (1+ul[0],0+ul[1]), (0+ul[0],1+ul[1])]
        l_shapes.append(shape_ul)  
        shape_ur = [(0+ul[0], 0+ul[1]), (0+ul[0], 1+ul[1]), (1+ul[0], 1+ul[1])]
        l_shapes.append(shape_ur)   
        shape_bl = [(0+ul[0], 0+ul[1]), (1+ul[0], 0+ul[1]), (1+ul[0], 1+ul[1])]
        l_shapes.append(shape_bl)   
        shape_br = [(0+ul[0], 1+ul[1]), (1+ul[0], 0+ul[1]), (1+ul[0], 1+ul[1])]
        l_shapes.append(shape_br)
        
        shape_ul_2 = [(0+ul[0], 0+ul[1]), (1+ul[0],0+ul[1]), (2+ul[0],0+ul[1]), (0+ul[0],1+ul[1]), (0+ul[0],2+ul[1])]
        l_shapes.append(shape_ul_2)  
        shape_ur_2 = [(0+ul[0], 0+ul[1]), (0+ul[0], 1+ul[1]), (0+ul[0], 2+ul[1]), (1+ul[0], 2+ul[1]), (2+ul[0], 2+ul[1])]
        l_shapes.append(shape_ur_2)   
        shape_bl_2 = [(0+ul[0], 0+ul[1]), (1+ul[0], 0+ul[1]), (2+ul[0], 0+ul[1]), (2+ul[0], 1+ul[1]), (2+ul[0], 2+ul[1])]
        l_shapes.append(shape_bl_2)   
        shape_br_2 = [(0+ul[0], 2+ul[1]), (1+ul[0], 2+ul[1]), (2+ul[0], 0+ul[1]), (2+ul[0], 1+ul[1]), (2+ul[0], 2+ul[1])]
        l_shapes.append(shape_br_2)
    
    if regime == 'full': # create all possible l_shapes patterns
        shape_ul_j = copy(shape_ul)
        for j in range(1, grid_size[1]):
            new_cell = (0+ul[0], j+ul[1])
            if new_cell not in shape_ul_j:
                shape_ul_j.append(new_cell)
                l_shapes.append(copy(shape_ul_j))
            shape_ul_i = copy(shape_ul_j)    
            for i in range(2, grid_size[0]):
                new_cell = (i+ul[0], 0+ul[1])
                shape_ul_i.append(new_cell)
                l_shapes.append(copy(shape_ul_i))
        for j in range(1, grid_size[1]):
            base = [(0+ul[0], k+ul[1]) for k in range(j+1)]
            shape_ur_i = copy(base)
            for i in range(1, grid_size[0]):
                new_cell = (i+ul[0], j+ul[1])
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy(shape_ur_i))
        for i in range(1, grid_size[0]):
            base = [(k+ul[0], 0+ul[1]) for k in range(i+1)]
            shape_ur_i = copy(base)
            for j in range(1, grid_size[1]):
                new_cell = (i+ul[0], j+ul[1])
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy(shape_ur_i))   
    l_shapes = multiplicate_shapes(l_shapes, grid_size)
    return l_shapes

def generate_cross_shapes(grid_size:tuple)->List[List[List[tuple]]]:
    """Create cross_shapes patterns for given grid size."""
    cross_shapes = []
    ul = find_upper_left_corner(grid_size)
    cross_3_3 = [(0+ul[0], 1+ul[1]), (1+ul[0],0+ul[1]), (1+ul[0],1+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],1+ul[1])]
    cross_shapes.append(cross_3_3)  
    cross_5_5 = [(0+ul[0], 2+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],2+ul[1]), (3+ul[0],2+ul[1]), (4+ul[0],2+ul[1]), (2+ul[0],0+ul[1]), (2+ul[0],1+ul[1]), (2+ul[0],3+ul[1]), (2+ul[0],4+ul[1])]
    cross_shapes.append(cross_5_5)
    cross_shapes = multiplicate_shapes(cross_shapes, grid_size)
    return cross_shapes

def generate_t_shapes(grid_size:tuple, shapes:List[str]=['3_1', '3_2'])->List[List[List[tuple]]]:
    """Create t-shapes patterns for given grid size."""
    t_shapes = []
    ul = find_upper_left_corner(grid_size)
    if '3_1' in shapes:
        t_shape_3_1_up = [(0+ul[0], 1+ul[1]), (1+ul[0],0+ul[1]), (1+ul[0],1+ul[1]), (1+ul[0],2+ul[1])]
        t_shapes.append(t_shape_3_1_up)
        t_shape_3_1_down = [(0+ul[0],0+ul[1]), (0+ul[0],1+ul[1]), (0+ul[0],2+ul[1]), (1+ul[0],1+ul[1])]
        t_shapes.append(t_shape_3_1_down) 
        t_shape_3_1_left = [(0+ul[0], 1+ul[1]), (1+ul[0],1+ul[1]), (2+ul[0],1+ul[1]), (1+ul[0],0+ul[1])]
        t_shapes.append(t_shape_3_1_left)
        t_shape_3_1_right = [(0+ul[0], 0+ul[1]), (1+ul[0],0+ul[1]), (2+ul[0],0+ul[1]), (1+ul[0],1+ul[1])]
        t_shapes.append(t_shape_3_1_right) 
    
    if '3_2' in shapes:
        t_shape_3_2_up = [(0+ul[0], 1+ul[1]), (1+ul[0],1+ul[1]), (2+ul[0],0+ul[1]), (2+ul[0],1+ul[1]), (2+ul[0],2+ul[1])]
        t_shapes.append(t_shape_3_2_up)
        t_shape_3_2_down = [(0+ul[0],0+ul[1]), (0+ul[0],1+ul[1]), (0+ul[0],2+ul[1]), (1+ul[0],1+ul[1]), (2+ul[0],1+ul[1])]
        t_shapes.append(t_shape_3_2_down) 
        t_shape_3_2_left = [(0+ul[0], 2+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],2+ul[1]), (1+ul[0],0+ul[1]), (1+ul[0],1+ul[1])]
        t_shapes.append(t_shape_3_2_left)
        t_shape_3_2_right = [(0+ul[0], 0+ul[1]), (1+ul[0],0+ul[1]), (2+ul[0],0+ul[1]), (1+ul[0],1+ul[1]), (1+ul[0],2+ul[1])]
        t_shapes.append(t_shape_3_2_right) 

    if '5_3' in shapes:
        t_shape_5_3_up = [(0+ul[0], 2+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],2+ul[1]), (3+ul[0],0+ul[1]), (3+ul[0],1+ul[1]), (3+ul[0],2+ul[1]), (3+ul[0],3+ul[1]), (3+ul[0],4+ul[1])]
        t_shapes.append(t_shape_5_3_up)
        t_shape_5_3_down = [(0+ul[0],0+ul[1]), (0+ul[0],1+ul[1]), (0+ul[0],2+ul[1]), (0+ul[0],3+ul[1]), (0+ul[0],4+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],2+ul[1]), (3+ul[0],2+ul[1])]
        t_shapes.append(t_shape_5_3_down) 
        t_shape_5_3_left = [(0+ul[0], 3+ul[1]), (1+ul[0],3+ul[1]), (2+ul[0],3+ul[1]), (3+ul[0],3+ul[1]), (4+ul[0],3+ul[1]), (2+ul[0],2+ul[1]), (2+ul[0],1+ul[1]), (2+ul[0],0+ul[1])]
        t_shapes.append(t_shape_5_3_left)
        t_shape_5_3_right = [(0+ul[0], 0+ul[1]), (1+ul[0],0+ul[1]), (2+ul[0],0+ul[1]), (3+ul[0],0+ul[1]), (4+ul[0],0+ul[1]), (2+ul[0],1+ul[1]), (2+ul[0],2+ul[1]), (2+ul[0],3+ul[1])]
        t_shapes.append(t_shape_5_3_right) 
    t_shapes = multiplicate_shapes(t_shapes, grid_size)
    return t_shapes

def generate_s_shapes(grid_size:tuple, shapes:List[str]=['4'])->List[List[List[tuple]]]:
    """Create s_shapes patterns for given grid_size."""
    s_shapes = []
    ul = find_upper_left_corner(grid_size)
    if '4' in shapes:
        s_shape_4_up = [(0+ul[0], 1+ul[1]), (1+ul[0], 1+ul[1]), (1+ul[0], 0+ul[1]), (2+ul[0], 0+ul[1])]
        s_shapes.append(s_shape_4_up)
        s_shape_4_down = [(0+ul[0], 0+ul[1]), (1+ul[0], 0+ul[1]), (1+ul[0], 1+ul[1]), (2+ul[0], 1+ul[1])]
        s_shapes.append(s_shape_4_down)
        s_shape_4_left = [(0+ul[0],0+ul[1]), (0+ul[0], 1+ul[1]), (1+ul[0], 1+ul[1]), (1+ul[0], 2+ul[1])]
        s_shapes.append(s_shape_4_left)
        s_shapes_4_right = [(1+ul[0],0+ul[1]), (1+ul[0], 1+ul[1]), (0+ul[0], 1+ul[1]), (0+ul[0], 2+ul[1])]
        s_shapes.append(s_shapes_4_right)
    
    if '6' in shapes:
        s_shape_6_up = [(0+ul[0], 1+ul[1]), (1+ul[0], 1+ul[1]), (2+ul[0], 1+ul[1]), (1+ul[0], 0+ul[1]), (2+ul[0], 0+ul[1]), (3+ul[0], 0+ul[1])]
        s_shapes.append(s_shape_6_up)
        s_shape_6_down = [(0+ul[0], 0+ul[1]), (1+ul[0], 0+ul[1]), (2+ul[0], 0+ul[1]), (1+ul[0], 1+ul[1]), (2+ul[0], 1+ul[1]), (3+ul[0], 1+ul[1])]
        s_shapes.append(s_shape_6_down)
        s_shape_6_left = [(0+ul[0], 0+ul[1]), (0+ul[0], 1+ul[1]), (0+ul[0], 2+ul[1]), (1+ul[0], 1+ul[1]), (1+ul[0], 2+ul[1]), (1+ul[0], 3+ul[1])]
        s_shapes.append(s_shape_6_left)
        s_shapes_6_right = [(1+ul[0], 0+ul[1]), (1+ul[0], 1+ul[1]), (1+ul[0], 2+ul[1]), (0+ul[0], 1+ul[1]), (0+ul[0], 2+ul[1]), (0+ul[0], 3+ul[1])]
        s_shapes.append(s_shapes_6_right)
    s_shapes = multiplicate_shapes(s_shapes, grid_size)
    return s_shapes

def generate_tv_shapes(grid_size:tuple)->List[List[List[tuple]]]:
    """Create tv-shapes patterns for given grid_size."""
    tv_shapes = []
    ul = find_upper_left_corner(grid_size)
    for k in range(2, min(grid_size)):
        tv = [(ul[0], ul[1]), (k+ul[0], k+ul[1])]
        for i in range(1, k+1):
            new_cell_1 = (ul[0], i+ul[1])
            tv.append(new_cell_1)
            new_cell_2 = (i+ul[0], ul[1])
            tv.append(new_cell_2)
            new_cell_3 = (k+ul[0], i+ul[1])
            tv.append(new_cell_3)
            new_cell_4 = (i+ul[0], k+ul[1])
            tv.append(new_cell_4)
        tv_shapes.append(copy(sorted(tv, key=lambda x: x[0])))
    tv_shapes = multiplicate_shapes(tv_shapes, grid_size)
    return tv_shapes

def generate_flowers(grid_size:tuple)->List[List[List[tuple]]]:
    """Create flower patterns for given grid_size."""
    flowers = []
    ul = find_upper_left_corner(grid_size)
    flower_2_2 = [(0+ul[0],1+ul[1]), (2+ul[0],1+ul[1]), (1+ul[0],0+ul[1]), (1+ul[0],2+ul[1])]
    flower_3_3 = [(0+ul[0],0+ul[1]), (0+ul[0],2+ul[1]), (1+ul[0],1+ul[1]), (2+ul[0],0+ul[1]), (2+ul[0],2+ul[1])]
    flowers.append(flower_2_2)
    flowers.append(flower_3_3)
    flowers = multiplicate_shapes(flowers, grid_size)
    return flowers

def lines_partition(grid_size:tuple)->List[List[tuple]]:
    """Create markup patterns with lines partition."""
    patterns = []
    ul = find_upper_left_corner(grid_size)
    if grid_size[0] >= 3 or grid_size[1] >= 3: # splitting in halves
        if grid_size[0] % 2 == 1:
            mid_i = grid_size[0]//2 + ul[0]
            patterns.append([[(mid_i, j+ul[1]) for j in range(grid_size[1])]])
        if grid_size[1] % 2 == 1:
            mid_j = grid_size[1]//2 + ul[1]
            patterns.append([[(i+ul[0], mid_j) for i in range(grid_size[0])]])
    if grid_size[0] >= 8 or grid_size[1] >= 8: # splitting in thirds
        if grid_size[0] % 3 == 2:
            mid_i_1 = grid_size[0]//3 + ul[0]
            patterns.append([[(mid_i_1, j+ul[1]) for j in range(grid_size[1])]])
            mid_i_2 = mid_i_1 + grid_size[0]//3 + 1
            patterns.append([[(mid_i_2, j+ul[1]) for j in range(grid_size[1])]])
        if grid_size[1] % 3 == 2:
            mid_j_1 = grid_size[1]//3 + ul[1]
            patterns.append([[(i+ul[0], mid_j_1) for i in range(grid_size[0])]])
            mid_j_2 = mid_j_1 + grid_size[1]//3 + 1
            patterns.append([[(i+ul[0], mid_j_2 ) for i in range(grid_size[0])]])
    if grid_size[0] >= 11 or grid_size[1] >= 11: # splitting in fourths
        if grid_size[0] % 4 == 3:
            mid_i_1 = grid_size[0]//4 + ul[0]
            patterns.append([[(mid_i_1, j+ul[1]) for j in range(grid_size[1])]])
            mid_i_2 = mid_i_1 + grid_size[0]//4 + 1
            patterns.append([[(mid_i_2, j+ul[1]) for j in range(grid_size[1])]])
            mid_i_3 = mid_i_2 + grid_size[0]//4 + 1
            patterns.append([[(mid_i_3, j+ul[1]) for j in range(grid_size[1])]])
        if grid_size[1] % 4 == 3:
            mid_j_1 = grid_size[1]//4 + ul[1]
            patterns.append([[(i+ul[0], mid_j_1) for i in range(grid_size[0])]])
            mid_j_2 = mid_j_1 + grid_size[1]//4 + 1
            patterns.append([[(i+ul[0], mid_j_2 ) for i in range(grid_size[0])]])
            mid_j_3 = mid_j_2 + grid_size[1]//4 + 1
            patterns.append([[(i+ul[0], mid_j_3 ) for i in range(grid_size[0])]])
    return patterns

def matrix_partition(grid_size:tuple)->List[List[tuple]]:
    """Create markup patterns with matrix partition."""
    patterns = []
    ul = find_upper_left_corner(grid_size)
    for size_1 in range(2, (grid_size[0]//2+1)):
        markup = []
        for size_2 in range(2, (grid_size[1]//2+1)): 
            if grid_size[0]//size_1 == (grid_size[0]%size_1)+1 == grid_size[1]//size_2 == (grid_size[1]%size_2)+1:
                    n = (grid_size[0] // size_1) - 1
                    k = (grid_size[1] // size_2) - 1
                    line_starts_i = [ul[0]+size_1*i+(i-1) for i in range(1, n+1)]
                    line_starts_j = [ul[1]+size_2*j+(j-1) for j in range(1, k+1)]       
                    markup.extend([(line_start_i, j+ul[1]) for j in range(grid_size[1]) for line_start_i in line_starts_i])
                    markup.extend([(i+ul[0], line_start_j) for i in range(grid_size[0]) for line_start_j in line_starts_j])
                    patterns.append([markup])   
    return patterns

def find_connected_components_with_color(grid, target_color, folds=8):
    """Find all connected components in a grid with specified color."""
    if isinstance(grid, np.ndarray):
        if grid.size == 0:
            return []
        rows, cols = grid.shape
        is_numpy = True
    else:
        if not grid or not grid[0]:
            return []
        rows, cols = len(grid), len(grid[0])
        is_numpy = False
    
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    # Precompute directions
    if folds == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, 1), (1, -1), (-1, 1)]
        
    def get_cell_value(r, c):
        return grid[r, c] if is_numpy else grid[r][c]
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and get_cell_value(r, c) == target_color:
                # Use BFS instead of DFS to avoid recursion depth issues
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    component.append((curr_r, curr_c))
                    
                    # Process neighbors without function call overhead
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not visited[nr, nc] and get_cell_value(nr, nc) == target_color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                components.append(component)
    
    return components

def find_connected_components_excluding_colors(grid, font_color=0, pad_val=10, folds=8):
    """
    Find all connected components in a grid where cells have any color except the specified font color.
    """
    if isinstance(grid, np.ndarray):
        if grid.size == 0:
            return []
        rows, cols = grid.shape
        is_numpy = True
    else:
        if not grid or not grid[0]:
            return []
        rows, cols = len(grid), len(grid[0])
        is_numpy = False
    
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    # Precompute directions
    if folds == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, 1), (1, -1), (-1, 1)]
    
    def get_cell_value(r, c):
        return grid[r, c] if is_numpy else grid[r][c]
    
    # Pre-mark excluded cells as visited
    for r in range(rows):
        for c in range(cols):
            cell_value = get_cell_value(r, c)
            if cell_value == font_color or cell_value == pad_val:
                visited[r, c] = True
    
    # Find connected components using BFS
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    component.append((curr_r, curr_c))
                    
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                components.append(component)
    
    return components

def generate_patterns(grid_size:tuple, shape_types:Tuple[str], multithreading:bool=True)->dict:
    """Generate specified types of patterns."""
    if multithreading:
        with ThreadPoolExecutor() as executor:
            lines = executor.submit(lines_coords, grid_size)
            rectangles = executor.submit(rectangles_coords, grid_size)
            diagonals = executor.submit(diagonals_coords, grid_size)
            l_shapes = executor.submit(generate_l_shapes, grid_size)
            t_shapes = executor.submit(generate_t_shapes, grid_size)
            s_shapes = executor.submit(generate_s_shapes, grid_size)
            tv_shapes = executor.submit(generate_tv_shapes, grid_size)
            hs_shapes = executor.submit(generate_hs_shapes, grid_size)
            crosses = executor.submit(generate_cross_shapes, grid_size)
            flowers = executor.submit(generate_flowers, grid_size)
            markup = executor.submit(matrix_partition, grid_size)
            partition_lines = executor.submit(lines_partition, grid_size)
            all_figures = {'line':lines.result(), 'rectangle':rectangles.result(), 'l_shape':l_shapes.result(), 
                           't_shape':t_shapes.result(), 's_shape':s_shapes.result(), 'tv_shape':tv_shapes.result(), 
                           'hs_shape':hs_shapes.result(), 'cross':crosses.result(), 'flower':flowers.result(), 
                           'diagonal':diagonals.result(), 'markup':markup.result(), 'partition_lines':partition_lines.result()}
            figures_after_filtering = {k:v for k,v in all_figures.items() if k in shape_types}
    else:
        lines = lines_coords(grid_size)
        rectangles = rectangles_coords(grid_size)
        diagonals = diagonals_coords(grid_size)
        l_shapes = generate_l_shapes(grid_size)
        t_shapes = generate_t_shapes(grid_size)
        s_shapes = generate_s_shapes(grid_size)
        tv_shapes = generate_tv_shapes(grid_size)
        hs_shapes = generate_hs_shapes(grid_size)
        crosses = generate_cross_shapes(grid_size)
        flowers = generate_flowers(grid_size)
        markup = matrix_partition(grid_size)
        partition_lines = lines_partition(grid_size)
        all_figures = {'line':lines, 'rectangle':rectangles, 'l_shape':l_shapes, 
                       't_shape':t_shapes, 's_shape':s_shapes, 'tv_shape':tv_shapes, 
                       'hs_shape':hs_shapes, 'cross':crosses, 'flower':flowers, 'diagonal':diagonals, 
                       'markup':markup, 'partition_lines':partition_lines}
        figures_after_filtering = {k:v for k,v in all_figures.items() if k in shape_types}
    return figures_after_filtering

def retrieve_shapes(grid, shape:tuple, shape_types:tuple, font_color=0):
    """Retrieve specified shape types patterns."""
    patterns = generate_patterns(shape, shape_types)
    objects = defaultdict(list)
    grid = copy(grid)
    for k, shape_patterns in patterns.items():
        for idx, pattern_list in enumerate(shape_patterns):
            for pattern in pattern_list:
                i, j = coords_transform(pattern)
                retrieval = set(grid[i, j])
                if len(retrieval) > 1:
                    break
                else:
                    color = retrieval.pop()
                    if color != font_color and pattern not in objects[k]:
                        objects[k].append(pattern)             
    return objects