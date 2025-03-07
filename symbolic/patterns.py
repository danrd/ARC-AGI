from copy import copy, deepcopy
import typing
from typing import List
from symbolic.utils import find_upper_left_corner, multiplicate_shapes
from concurrent.futures import ThreadPoolExecutor

def left_lines(grid_size:tuple, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for diagonals creations."""
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
            lines.append(copy.copy(base))
        offset_ii += 1
        offset_jj += 1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy.copy(base))
    return lines

def right_lines(grid_size:tuple, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for diagonals creations."""
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
            lines.append(copy.copy(base))
        offset_ii += 1
        offset_jj += -1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in i_coords and new_cell[1] in j_coords:
            base.append(new_cell)
            lines.append(copy.copy(base))
    return lines

def diagonals_coords(grid_size:tuple)->List[List[List[tuple]]]:
    """Create diagonal patterns for given grid_size."""
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
    """Create line patterns for given grid_size."""
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
                    lines.append((copy.copy(line)))
                hor_lines.append(copy.copy(lines))
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
                    lines.append((copy.copy(line)))           
                vert_lines.append(copy.copy(lines))
        return vert_lines
    coords = hor_lines_coords(grid_size) + vert_lines_coords(grid_size)
    return coords

def rectangles_coords(grid_size:tuple)->List[List[List[tuple]]]:
    """Create rectangle patterns for given grid_size."""
    def i_expansion(grid_size):
        """Patterns expanding along i axis.""" 
        ul = find_upper_left_corner(grid_size)
        coords = []
        for size_i in range(2, grid_size[0]+1):
            cache = []
            hor_rects = []
            for size_j in range(2, grid_size[1]+1):
                base_hor = [(ii+ul[0], jj+ul[1]) for ii in range(size_i) for jj in range(size_j)]
                hor_rects.append(copy.copy(base_hor))
            coords.append(copy.copy(hor_rects)) 
            cache.append(copy.copy(hor_rects)) 
            for i in range(size_i-2):
                hor_rects_2 = []
                for idx, rect in enumerate(hor_rects):
                    crop_rect = rect[(i+1)*(idx+2):]
                    if len(crop_rect) >= 4:
                        hor_rects_2.append(crop_rect)
                coords.append(copy.copy(hor_rects_2)) 
                cache.append(copy.copy(hor_rects_2)) 
            for hor_rect_list in cache:
                hor_rects_3 = [[] for i in range(2, grid_size[1])]
                for idx_1, rect in enumerate(hor_rect_list):
                    rect.sort(key=lambda x: x[1])
                    for idx_2 in range(idx_1):
                        crop_rect = rect[size_i*(idx_2+1):]
                        if len(crop_rect) >= 4:
                            hor_rects_3[idx_2].append(crop_rect)
                coords.extend(copy.copy(hor_rects_3))   
                size_i += -1
        return coords
    
    def j_expansion(grid_size):
        """Patterns expanding along j axis."""
        ul = find_upper_left_corner(grid_size)
        coords = []
        for size_j in range(2, grid_size[1]+1):
            cache = []
            hor_rects = []
            for size_i in range(2, grid_size[0]+1):
                base_hor = [(ii+ul[0], jj+ul[1]) for jj in range(size_j) for ii in range(size_i)]
                hor_rects.append(copy.copy(base_hor))
            coords.append(copy.copy(hor_rects)) 
            cache.append(copy.copy(hor_rects)) 
            for j in range(size_j-2):
                hor_rects_2 = []
                for idx, rect in enumerate(hor_rects):
                    crop_rect = rect[(j+1)*(idx+2):]
                    if len(crop_rect) >= 4:
                        hor_rects_2.append(crop_rect)
                coords.append(copy.copy(hor_rects_2)) 
                cache.append(copy.copy(hor_rects_2)) 
            for hor_rect_list in cache:
                hor_rects_3 = [[] for i in range(2, grid_size[0])]
                for idx_1, rect in enumerate(hor_rect_list):
                    rect.sort(key=lambda x: x[0])
                    for idx_2 in range(idx_1):
                        crop_rect = rect[size_j*(idx_2+1):]
                        if len(crop_rect) >= 4:
                            hor_rects_3[idx_2].append(crop_rect)
                coords.extend(copy.copy(hor_rects_3))   
                size_j += -1
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
                    sq_rects.append(copy.copy(base_sq))
                coords.append(copy.copy(sorted(sq_rects, key=lambda x: x[0]))) 
        return coords

    with ThreadPoolExecutor() as executor: # multithreading for better perfomance
        ul = find_upper_left_corner(grid_size)
        coords = []
        coords_1 = []
        coords_2 = []
        coords_3 = []
        i_coords = executor.submit(i_expansion, grid_size)
        j_coords = executor.submit(j_expansion, grid_size)
        i_j_coords = executor.submit(i_j_expansion, grid_size)
        coords_1.append(i_coords.result())
        coords_2.append(j_coords.result())
        coords_3.append(i_j_coords.result())
        coords.extend(coords_1[0])
        coords.extend(coords_2[0])
        coords.extend(coords_3[0])
    return coords

def rectangles_coords2(grid_size:tuple)->List[List[List[tuple]]]:
    ul = find_upper_left_corner(grid_size)
    coords = []
    for i in range(grid_size[0]-1):
        for j in range(grid_size[1]-1):
            for size_i in range(2, grid_size[0]-i+1):
                hor_rects = []
                for size_j in range(2, grid_size[1]-j+1):
                    base_hor = [(ii+ul[0]+i, jj+ul[1]+j) for ii in range(size_i) for jj in range(size_j)]
                    hor_rects.append(copy.copy(sorted(base_hor, key=lambda x: x[0])))
                coords.append(copy.copy(sorted(hor_rects, key=lambda x:x[0]))) 
            for size_j in range(2, grid_size[1]-j+1):
                vert_rects = []
                for size_i in range(2, grid_size[0]-i+1):
                    base_vert = [(ii+ul[0]+i, jj+ul[1]+j) for jj in range(size_j) for ii in range(size_i)]
                    vert_rects.append(copy.copy(sorted(base_vert, key=lambda x: x[0])))
                coords.append(copy.copy(sorted(vert_rects, key=lambda x: x[0]))) 
            sq_rects = []
            min_dim = min(grid_size[0]-i+1, grid_size[1]-j+1)
            for size in range(2, min_dim):
                base_sq = [(ii+ul[0]+i, jj+ul[1]+j) for ii in range(size) for jj in range(size)]
                sq_rects.append(copy.copy(base_sq))
            coords.append(copy.copy(sorted(sq_rects, key=lambda x: x[0]))) 
    return coords  

def generate_hs_shapes(grid_size:tuple, short:bool=True)->List[List[List[tuple]]]:
    """Create hs_shapes patterns for given grid_size."""
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
                    cache.append(copy.copy(base))
                if cache != []:
                    hs_shapes.append(copy.copy(cache))
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
                    cache.append(copy.copy(base))
                if cache != []: 
                    hs_shapes.append(copy.copy(cache))
            for i in range(3, max_i_increment):
                base = [(k+ul[0], ul[1]) for k in range(i)] # create base line     
                cache = []
                max_expanding_increment =  min(3, grid_size[1] - j_pos) if short else grid_size[1] - j_pos
                for j in range(1, max_expanding_increment): # expanding upper and bottom lines 
                    new_cell_left = (ul[0], j+ul[1])
                    base.append(new_cell_left)
                    new_cell_right = (i-1+ul[0], j+ul[1]) 
                    base.append(new_cell_right)
                    cache.append(copy.copy(base))
                if cache != []:
                    hs_shapes.append(copy.copy(cache))
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
                    cache.append(copy.copy(base))
                if cache != []: 
                    hs_shapes.append(copy.copy(cache))
    return hs_shapes

def generate_l_shapes(grid_size:tuple, regime:str='short')->List[List[List[tuple]]]:
    """Create l_shapes patterns for given grid_size."""
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
        shape_ul_j = copy.copy(shape_ul)
        for j in range(1, grid_size[1]):
            new_cell = (0+ul[0], j+ul[1])
            if new_cell not in shape_ul_j:
                shape_ul_j.append(new_cell)
                l_shapes.append(copy.copy(shape_ul_j))
            shape_ul_i = copy.copy(shape_ul_j)    
            for i in range(2, grid_size[0]):
                new_cell = (i+ul[0], 0+ul[1])
                shape_ul_i.append(new_cell)
                l_shapes.append(copy.copy(shape_ul_i))
        for j in range(1, grid_size[1]):
            base = [(0+ul[0], k+ul[1]) for k in range(j+1)]
            shape_ur_i = copy.copy(base)
            for i in range(1, grid_size[0]):
                new_cell = (i+ul[0], j+ul[1])
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy.copy(shape_ur_i))
        for i in range(1, grid_size[0]):
            base = [(k+ul[0], 0+ul[1]) for k in range(i+1)]
            shape_ur_i = copy.copy(base)
            for j in range(1, grid_size[1]):
                new_cell = (i+ul[0], j+ul[1])
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy.copy(shape_ur_i))   
    l_shapes = multiplicate_shapes(l_shapes, grid_size)
    return l_shapes

def generate_cross_shapes(grid_size:tuple)->List[List[List[tuple]]]:
    """Create cross_shapes patterns for given grid_size."""
    cross_shapes = []
    ul = find_upper_left_corner(grid_size)
    cross_3_3 = [(0+ul[0], 1+ul[1]), (1+ul[0],0+ul[1]), (1+ul[0],1+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],1+ul[1])]
    cross_shapes.append(cross_3_3)  
    cross_5_5 = [(0+ul[0], 2+ul[1]), (1+ul[0],2+ul[1]), (2+ul[0],2+ul[1]), (3+ul[0],2+ul[1]), (4+ul[0],2+ul[1]), (2+ul[0],0+ul[1]), (2+ul[0],1+ul[1]), (2+ul[0],3+ul[1]), (2+ul[0],4+ul[1])]
    cross_shapes.append(cross_5_5)
    cross_shapes = multiplicate_shapes(cross_shapes, grid_size)
    return cross_shapes

def generate_t_shapes(grid_size:tuple, shapes:List[str]=['3_1', '3_2'])->List[List[List[tuple]]]:
    """Create t_shapes patterns for given grid_size."""
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
    """Create tv_shapes patterns for given grid_size."""
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
        tv_shapes.append(copy.copy(sorted(tv, key=lambda x: x[0])))
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
    if grid_size >= (5,5): # splitting in halves
        if grid_size[0]%2 == 1:
            mid_i = grid_size[0]//2 + ul[0]
            patterns.append([(mid_i, j+ul[1]) for j in range(grid_size[1])])
        if grid_size[1]%2 == 1:
            mid_j = grid_size[1]//2 + ul[1]
            patterns.append([(i+ul[0], mid_j) for i in range(grid_size[0])])
    if grid_size >= (8,8): # splitting in thirds
        if grid_size[0]%3 == 2:
            pattern = []
            mid_i_1 = grid_size[0]//3 + ul[0]
            pattern.extend([(mid_i_1, j+ul[1]) for j in range(grid_size[1])])
            mid_i_2 = mid_i_1 + grid_size[0]//3 + 1
            pattern.extend([(mid_i_2, j+ul[1]) for j in range(grid_size[1])])
            patterns.append(pattern)
        if grid_size[1]%3 == 2:
            pattern = []
            mid_j_1 = grid_size[1]//3 + ul[1]
            pattern.extend([(i+ul[0], mid_j_1) for i in range(grid_size[0])])
            mid_j_2 = mid_j_1 + grid_size[1]//3 + 1
            pattern.extend([(i+ul[0], mid_j_2 ) for i in range(grid_size[0])])
            patterns.append(pattern)
    return patterns   

def matrix_partition(grid_size:tuple)->List[List[tuple]]:
    """Create markup patterns with matrix partition."""
    patterns = []
    ul = find_upper_left_corner(grid_size)
    if grid_size[0] == grid_size[1]:
        for size in range(2, grid_size[0]):
            if size*2+1 > grid_size[0]:
                break
            else:
                if (grid_size[0])%(size+1) == size:
                    n = (grid_size[0]//(size+1))
                    grid = []
                    lines_starts = [size+k*(size+1) for k in range(n)]
                    grid.extend([(line_start_i+ul[0], j+ul[1]) for j in range(grid_size[1]) for line_start_i in lines_starts])
                    grid.extend([(i+ul[0], line_start_j+ul[1]) for i in range(grid_size[0]) for line_start_j in lines_starts])
                    patterns.append([grid])   
    return patterns   

def genetate_markup(shape:tuple)->typing.Dict[str, List[List[tuple]]]:
    """Create patterns to identify possible grid markup."""
    definite = matrix_partition(shape)
    possible = lines_partition(shape)
    return {'definite':definite, 'possible':possible}

def generate_patterns(grid_size:tuple, multithreading:bool=True)->dict:
    """Generate all types of patterns."""
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
            all_figures = {'line':lines.result(), 'rectangle':rectangles.result(), 'l_shape':l_shapes.result(), 
                           't_shape':t_shapes.result(), 's_shape':s_shapes.result(), 'tv_shape':tv_shapes.result(), 
                           'hs_shape':hs_shapes.result(), 'cross':crosses.result(), 'flower':flowers.result(), 'diagonal':diagonals.result(), 'markup':markup.result()}
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
        all_figures = {'line':lines, 'rectangle':rectangles, 'l_shape':l_shapes, 
                       't_shape':t_shapes, 's_shape':s_shapes, 'tv_shape':tv_shapes, 
                       'hs_shape':hs_shapes, 'cross':crosses, 'flower':flowers, 'diagonal':diagonals, 'markup':markup}
    return all_figures