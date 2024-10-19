import copy
from typing import List
from symbolic.utils import find_upper_left_corner, multiplicate_figures

def left_lines(grid_size:int, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for diagonals creations."""
    lines = []
    ul = find_upper_left_corner(grid_size)
    grid_admissible = [ul+i for i in range(grid_size)]
    i, j = pos
    offset_i, offset_j, offset_ii, offset_jj = 0, 0, 0, 0
    base = [(i, j)]
    for _ in range(1, grid_size):
        offset_i += -1
        offset_j += -1
        new_cell = (i+offset_i, j+offset_j)
        if new_cell[0] in grid_admissible and new_cell[1] in grid_admissible:
            base.append(new_cell)
            lines.append(copy.deepcopy(base))
        offset_ii += 1
        offset_jj += 1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in grid_admissible and new_cell[1] in grid_admissible:
            base.append(new_cell)
            lines.append(copy.deepcopy(base))
    return lines

def right_lines(grid_size:int, pos:tuple)->List[List[tuple]]:
    """Auxiliary function for diagonals creations."""
    lines = []
    ul = find_upper_left_corner(grid_size)
    grid_admissible = [ul+i for i in range(grid_size)]
    i, j = pos
    offset_i, offset_j, offset_ii, offset_jj = 0, 0, 0, 0
    base = [(i, j)]
    for k in range(1, grid_size):
        offset_i += -1
        offset_j += 1
        new_cell = (i+offset_i, j+offset_j)
        if new_cell[0] in grid_admissible and new_cell[1] in grid_admissible:
            base.append(new_cell)
            lines.append(copy.deepcopy(base))
        offset_ii += 1
        offset_jj += -1
        new_cell = (i+offset_ii, j+offset_jj)
        if new_cell[0] in grid_admissible and new_cell[1] in grid_admissible:
            base.append(new_cell)
            lines.append(copy.deepcopy(base))
    return lines

def diagonals_coords(grid_size:int)->List[List[List[tuple]]]:
    all_lines = []
    ul = find_upper_left_corner(grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            pos = (i+ul, j+ul)
            lefts = left_lines(grid_size, pos)
            rights = right_lines(grid_size, pos)
            if lefts != []:
                all_lines.append(lefts)
            if rights != []:
                all_lines.append(rights)
    return all_lines

def lines_coords(grid_size:int)->List[List[List[tuple]]]:
    coords = []
    def hor_lines_coords(grid_size)->List[List[List[tuple]]]:
        ul = find_upper_left_corner(grid_size)
        hor_lines = []
        for i in range(grid_size):
            for j in range(grid_size-1):
                lines = []
                line = [(i+ul, j+ul)]
                for k in range(1, grid_size-j):
                    l = (i+ul, j+k+ul)
                    line.append(l)
                    lines.append((copy.deepcopy(line)))
                hor_lines.append(copy.deepcopy(lines))
        return hor_lines
    def vert_lines_coords(grid_size:int)->List[List[List[tuple]]]:
        ul = find_upper_left_corner(grid_size)
        vert_lines = []
        for i in range(grid_size-1):
            for j in range(grid_size):
                lines = []
                line = [(i+ul, j+ul)]
                for k in range(1, grid_size-i):
                    l = (i+k+ul, j+ul)
                    line.append(l)
                    lines.append((copy.deepcopy(line)))           
                vert_lines.append(copy.deepcopy(lines))
        return vert_lines
    coords = hor_lines_coords(grid_size) + vert_lines_coords(grid_size)
    return coords

def rectangles_coords(grid_size:int)->List[List[List[tuple]]]:
    ul = find_upper_left_corner(grid_size)
    coords = []
    for i in range(grid_size-1):
        for j in range(grid_size-1):
            for size_i in range(2, grid_size-i+1):
                hor_rects = []
                for size_j in range(2, grid_size-j+1):
                    base_hor = [(ii+ul+i, jj+ul+j) for ii in range(size_i) for jj in range(size_j)]
                    hor_rects.append(copy.deepcopy(sorted(base_hor, key=lambda x: x[0])))
                coords.append(copy.deepcopy(sorted(hor_rects, key=lambda x:x[0]))) 
            for size_j in range(2, grid_size-j+1):
                vert_rects = []
                for size_i in range(2, grid_size-i+1):
                    base_vert = [(ii+ul+i, jj+ul+j) for jj in range(size_j) for ii in range(size_i)]
                    vert_rects.append(copy.deepcopy(sorted(base_vert, key=lambda x: x[0])))
                coords.append(copy.deepcopy(sorted(vert_rects, key=lambda x: x[0]))) 
            sq_rects = []
            for size in range(2, min(grid_size-i+1, grid_size-j+1)):
                base_sq = [(ii+ul+i, jj+ul+j) for ii in range(size) for jj in range(size)]
                sq_rects.append(copy.deepcopy(base_sq))
            coords.append(copy.deepcopy(sorted(sq_rects, key=lambda x: x[0]))) 
    return coords  

def generate_hs_shapes(grid_size:int)->List[List[List[tuple]]]:
    hs_shapes = []
    ul = find_upper_left_corner(grid_size)
    for j in range(3, grid_size):
        base = [(ul, ul+jj) for jj in range(j)] # create bottom line
        cache = []
        for i in range(1, grid_size): # expanding left and right lines 
            new_cell_left = (i+ul, ul)
            base.append(new_cell_left)
            new_cell_right = (i+ul, j-1+ul)
            base.append(new_cell_right)
            cache.append(copy.deepcopy(base))
        hs_shapes.append(copy.deepcopy(cache))
    for j in range(3, grid_size):
        cache = []
        for i in range(1, grid_size): # bottom line shifting
            base = [(i+ul, jj+ul) for jj in range(j)] # create bottom line
            for ii in range(i): # expanding left and right lines 
                new_cell_left = (ii+ul, 0+ul)
                base.append(new_cell_left)
                new_cell_right = (ii+ul, j-1+ul)
                base.append(new_cell_right)
            cache.append(copy.deepcopy(base))
        hs_shapes.append(copy.deepcopy(cache))
    for i in range(3, grid_size):
        base = [(k+ul, ul) for k in range(i)] # create bottom line     
        cache = []
        for j in range(1, grid_size): # expanding left and right lines 
            new_cell_left = (ul, j+ul)
            base.append(new_cell_left)
            new_cell_right = (i-1+ul, j+ul)
            base.append(new_cell_right)
            cache.append(copy.deepcopy(base))
        hs_shapes.append(copy.deepcopy(cache))
    for i in range(3, grid_size):
        cache = []
        for j in range(1, grid_size): # bottom line shifting
            base = [(k+ul, j+ul) for k in range(i)] # create bottom line
            for jj in range(j): # expanding left and right lines 
                new_cell_left = (i-1+ul, jj+ul)
                base.append(new_cell_left)
                new_cell_right = (ul, jj+ul)
                base.append(new_cell_right)
            cache.append(copy.deepcopy(base))
        hs_shapes.append(copy.deepcopy(cache))
    return hs_shapes

def generate_l_shapes(grid_size:int, regime:str='short')->List[List[List[tuple]]]:
    ul = find_upper_left_corner(grid_size)
    l_shapes = []
    shape_ul = [(0+ul, ul), (1+ul,0+ul), (0+ul,1+ul)]
    l_shapes.append(shape_ul)  
    shape_ur = [(0+ul, 0+ul), (0+ul, 1+ul), (1+ul, 1+ul)]
    l_shapes.append(shape_ur)   
    shape_ll = [(0+ul, 0+ul), (1+ul, 0+ul), (1+ul, 1+ul)]
    l_shapes.append(shape_ll)   
    shape_lr = [(0+ul, 1+ul), (1+ul, 0+ul), (1+ul, 1+ul)]
    l_shapes.append(shape_lr)
    if regime == 'full':
        shape_ul_j = copy.deepcopy(shape_ul)
        for j in range(1, grid_size):
            new_cell = (0+ul, j+ul)
            if new_cell not in shape_ul_j:
                shape_ul_j.append(new_cell)
                l_shapes.append(copy.deepcopy(shape_ul_j))
            shape_ul_i = copy.deepcopy(shape_ul_j)    
            for i in range(2, grid_size):
                new_cell = (i+ul, 0+ul)
                shape_ul_i.append(new_cell)
                l_shapes.append(copy.deepcopy(shape_ul_i))
        for j in range(1, grid_size):
            base = [(0+ul, k+ul) for k in range(j+1)]
            shape_ur_i = copy.deepcopy(base)
            for i in range(1, grid_size):
                new_cell = (i+ul, j+ul)
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy.deepcopy(shape_ur_i))
        for i in range(1, grid_size):
            base = [(k+ul, 0+ul) for k in range(i+1)]
            shape_ur_i = copy.deepcopy(base)
            for j in range(1, grid_size):
                new_cell = (i+ul, j+ul)
                shape_ur_i.append(new_cell)
                if shape_ur_i not in l_shapes:
                    l_shapes.append(copy.deepcopy(shape_ur_i))   
    l_shapes = multiplicate_figures(l_shapes, grid_size)
    return l_shapes

def generate_cross_shapes(grid_size:int)->List[List[List[tuple]]]:
    cross_shapes = []
    ul = find_upper_left_corner(grid_size)
    cross_3_3 = [(0+ul, 1+ul), (1+ul,0+ul), (1+ul,1+ul), (1+ul,2+ul), (2+ul,1+ul)]
    cross_shapes.append(cross_3_3)  
    cross_5_5 = [(0+ul, 2+ul), (1+ul,2+ul), (2+ul,2+ul), (3+ul,2+ul), (4+ul,2+ul), (2+ul,0+ul), (2+ul,1+ul), (2+ul,3+ul), (2+ul,4+ul)]
    cross_shapes.append(cross_5_5)
    cross_shapes = multiplicate_figures(cross_shapes, grid_size)
    return cross_shapes

def generate_t_shapes(grid_size:int)->List[List[List[tuple]]]:
    t_shapes = []
    ul = find_upper_left_corner(grid_size)
    
    t_shape_3_1_up = [(0+ul, 1+ul), (1+ul,0+ul), (1+ul,1+ul), (1+ul,2+ul)]
    t_shapes.append(t_shape_3_1_up)
    t_shape_3_1_down = [(0+ul,0+ul), (0+ul,1+ul), (0+ul,2+ul), (1+ul,1+ul)]
    t_shapes.append(t_shape_3_1_down) 
    t_shape_3_1_left = [(0+ul, 1+ul), (1+ul,1+ul), (2+ul,1+ul), (1+ul,0+ul)]
    t_shapes.append(t_shape_3_1_left)
    t_shape_3_1_right = [(0+ul, 0+ul), (1+ul,0+ul), (2+ul,0+ul), (1+ul,1+ul)]
    t_shapes.append(t_shape_3_1_right) 

    t_shape_3_2_up = [(0+ul, 1+ul), (1+ul,1+ul), (2+ul,0+ul), (2+ul,1+ul), (2+ul,2+ul)]
    t_shapes.append(t_shape_3_2_up)
    t_shape_3_2_down = [(0+ul,0+ul), (0+ul,1+ul), (0+ul,2+ul), (1+ul,1+ul), (2+ul,1+ul)]
    t_shapes.append(t_shape_3_2_down) 
    t_shape_3_2_left = [(0+ul, 2+ul), (1+ul,2+ul), (2+ul,2+ul), (1+ul,0+ul), (1+ul,1+ul)]
    t_shapes.append(t_shape_3_2_left)
    t_shape_3_2_right = [(0+ul, 0+ul), (1+ul,0+ul), (2+ul,0+ul), (1+ul,1+ul), (1+ul,2+ul)]
    t_shapes.append(t_shape_3_2_right) 

    t_shape_5_3_up = [(0+ul, 2+ul), (1+ul,2+ul), (2+ul,2+ul), (3+ul,0+ul), (3+ul,1+ul), (3+ul,2+ul), (3+ul,3+ul), (3+ul,4+ul)]
    t_shapes.append(t_shape_5_3_up)
    t_shape_5_3_down = [(0+ul,0+ul), (0+ul,1+ul), (0+ul,2+ul), (0+ul,3+ul), (0+ul,4+ul), (1+ul,2+ul), (2+ul,2+ul), (3+ul,2+ul)]
    t_shapes.append(t_shape_5_3_down) 
    t_shape_5_3_left = [(0+ul, 3+ul), (1+ul,3+ul), (2+ul,3+ul), (3+ul,3+ul), (4+ul,3+ul), (2+ul,2+ul), (2+ul,1+ul), (2+ul,0+ul)]
    t_shapes.append(t_shape_5_3_left)
    t_shape_5_3_right = [(0+ul, 0+ul), (1+ul,0+ul), (2+ul,0+ul), (3+ul,0+ul), (4+ul,0+ul), (2+ul,1+ul), (2+ul,2+ul), (2+ul,3+ul)]
    t_shapes.append(t_shape_5_3_right) 
    t_shapes = multiplicate_figures(t_shapes, grid_size)
    return t_shapes

def generate_s_shapes(grid_size:int)->List[List[List[tuple]]]:
    s_shapes = []
    ul = find_upper_left_corner(grid_size)
    s_shape_4_up = [(0+ul,1+ul), (1+ul,1+ul), (1+ul,0+ul), (2+ul, 0+ul)]
    s_shapes.append(s_shape_4_up)
    s_shape_4_down = [(0+ul,0+ul), (1+ul,0+ul), (1+ul,1+ul), (2+ul,1+ul)]
    s_shapes.append(s_shape_4_down)
    s_shape_4_left = [(0+ul,0+ul), (0+ul,1+ul), (1+ul,1+ul), (1+ul,2+ul)]
    s_shapes.append(s_shape_4_left)
    s_shapes_4_right = [(1+ul,0+ul), (1+ul,1+ul), (0+ul,1+ul), (0+ul,2+ul)]
    s_shapes.append(s_shapes_4_right)

    s_shape_6_up = [(0+ul,1+ul), (1+ul,1+ul), (2+ul,1+ul), (1+ul,0+ul), (2+ul,0+ul), (3+ul,0+ul)]
    s_shapes.append(s_shape_6_up)
    s_shape_6_down = [(0+ul,0+ul), (1+ul,0+ul), (2+ul,0+ul), (1+ul,1+ul), (2+ul,1+ul), (3+ul,1+ul)]
    s_shapes.append(s_shape_6_down)
    s_shape_6_left = [(0+ul,0+ul), (0+ul,1+ul), (0+ul,2+ul), (1+ul,1+ul), (1+ul,2+ul), (1+ul,3+ul)]
    s_shapes.append(s_shape_6_left)
    s_shapes_6_right = [(1+ul,0+ul), (1+ul,1+ul), (1+ul,2+ul), (0+ul,1+ul), (0+ul,2+ul), (0+ul,3+ul)]
    s_shapes.append(s_shapes_6_right)
    s_shapes = multiplicate_figures(s_shapes, grid_size)
    return s_shapes

def generate_tv_shapes(grid_size:int)->List[List[List[tuple]]]:
    tv_shapes = []
    ul = find_upper_left_corner(grid_size)
    for k in range(3, grid_size+1):
        tv = [(ul,ul), (k+ul, k+ul)]
        for i in range(1, k+1):
            new_cell_1 = (ul, i+ul)
            tv.append(new_cell_1)
            new_cell_2 = (i+ul, ul)
            tv.append(new_cell_2)
            new_cell_3 = (k+ul, i+ul)
            tv.append(new_cell_3)
            new_cell_4 = (i+ul, k+ul)
            tv.append(new_cell_4)
        tv_shapes.append(copy.deepcopy(sorted(tv, key=lambda x: x[0])))
    tv_shapes = multiplicate_figures(tv_shapes, grid_size)
    return tv_shapes

def generate_flowers(grid_size:int)->List[List[List[tuple]]]:
    flowers = []
    ul = find_upper_left_corner(grid_size)
    flower_3_3 = [(0+ul,0+ul), (0+ul,2+ul), (1+ul,1+ul), (2+ul,0+ul), (2+ul,2+ul)]
    flowers.append(flower_3_3)
    flowers = multiplicate_figures(flowers, grid_size)
    return flowers

def generate_patterns(grid_size:int)->dict:
    lines = lines_coords(grid_size)
    rectangles = rectangles_coords(grid_size)
    diagonals = diagonals_coords(grid_size)
    l_shapes = generate_l_shapes(grid_size=grid_size, regime='short')
    t_shapes = generate_t_shapes(grid_size)
    s_shapes = generate_s_shapes(grid_size)
    tv_shapes = generate_tv_shapes(grid_size)
    hs_shapes = generate_hs_shapes(grid_size)
    crosses = generate_cross_shapes(grid_size)
    flowers = generate_flowers(grid_size)
    all_figures = {'line':lines, 'rectangle':rectangles, 'diagonal':diagonals, 'l_shape':l_shapes, 's_shape':s_shapes, 'tv_shape':tv_shapes, 'hs_shape':hs_shapes, 'cross':crosses, 'flower':flowers}
    return all_figures