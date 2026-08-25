import typing
import copy
from typing import List, Union
import sys
import numpy as np
from collections import Counter, defaultdict

def find_upper_left_corner(grid_size:tuple)->tuple:
    """Finds left upper corner of the grid to take into account padding.

    Grids are currently anchored at the top-left, so this is always (0, 0)
    regardless of grid_size. Kept as the single place ~30 call sites ask
    "where does this grid start", so re-introducing centered padding means
    changing this one function rather than all of them. (It used to compute
    a centered offset against a fixed 30x30 canvas; that computation was
    dead - the return ignored it - and has been removed.)
    """
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



# ---------------------------------------------------------------------------
# Background inference
# ---------------------------------------------------------------------------

# Thresholds measured over ARC-AGI-2's 1000 training tasks (6464 grids) -
# see infer_background's docstring for what the numbers are and why these
# two are where the signals stop agreeing.
BACKGROUND_MIN_DOMINANCE = 0.6
BACKGROUND_MIN_BORDER_PURITY = 0.9

# Share of a grid the task's canvas colour must still hold for that grid to
# be read as "the canvas, covered over" rather than as content that happens
# to share its colour. Of the 660 grids where the task establishes a canvas
# but the grid's own signals don't, the colour holds >=40% in 66.2% of them
# (median 0.44, and it is still the grid's most common colour in 66.2%),
# while 15.6% hold under 5% - content, not a covered canvas.
BACKGROUND_MIN_COVERED_SHARE = 0.4

# Fraction of a task's grids that must agree on a colour before it counts
# as that task's canvas.
BACKGROUND_TASK_CANVAS_MAJORITY = 0.5

# Stand-in for "no background colour was identified for this grid", for the
# code below the analyzer that has to compare cells against *some* colour.
# Outside ARC's 0-9 palette (and clear of patterns.py's pad_val=10), so
# nothing ever equals it and every colour is treated as foreground.
#
# Note what this does NOT claim. A grid always sits on some canvas; when it
# is painted over completely there is nothing left to observe, so the colour
# is unidentified rather than absent. Treating every colour as foreground is
# the right handling either way - there is no colour it would be safe to
# exclude - but the distinction matters for what gets reported: see
# infer_task_canvas, which recovers the colour from a task's other grids
# when a single grid can't show it.
BACKGROUND_NONE = -1


def _border_cells(grid: np.array) -> np.array:
    """Every cell on the grid's outer edge. Corners are counted twice (and
    a 1xN grid's whole row several times) - this is only ever used to ask
    which colour dominates the edge, which double-counting doesn't change."""
    return np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])


def infer_background(grid: np.array) -> Union[int, None]:
    """The grid's background colour, or None when this grid alone doesn't
    show which colour it is.

    Two independent signals have to agree: the most common colour overall,
    and the most common colour along the border. Measured over ARC-AGI-2's
    1000 training tasks (6464 grids), they agree 98.5% of the time once the
    most common colour covers at least 60% of the grid, but only 57.6% of
    the time when it covers under 40% - and there the border isn't uniform
    either (median purity 0.38).

    None means unidentified, not absent. Every grid sits on some canvas;
    a dense mosaic or a colour map on a 3x3 is one painted over so heavily
    that nothing about it can be read off this grid. Naming a colour anyway
    would hand every consumer a fabricated fact - and an expensive one,
    since whatever is called background stops being an object - so the
    answer is withheld, the rule findings.py applies to any parameter that
    wasn't established. Where the colour is recoverable from the task's
    other grids, infer_task_canvas recovers it; this function deliberately
    knows about one grid only.

    On the same data this names a colour for 67.5% of grids, and the 32.5%
    it declines sit squarely in the ambiguous band (median dominance 0.48,
    p90 0.57) rather than being clear cases wrongly refused.

    Per grid on purpose: assuming 0 (the hardcoded default this replaces)
    is wrong for 22.8% of tasks, and in 7.1% the background isn't even the
    same colour across one task's own examples.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return None

    values, counts = np.unique(grid, return_counts=True)
    dominant = int(values[counts.argmax()])
    dominance = counts.max() / grid.size

    border = _border_cells(grid)
    border_values, border_counts = np.unique(border, return_counts=True)
    border_dominant = int(border_values[border_counts.argmax()])
    border_purity = border_counts.max() / border.size

    if dominant != border_dominant:
        return None
    if dominance >= BACKGROUND_MIN_DOMINANCE or border_purity >= BACKGROUND_MIN_BORDER_PURITY:
        return dominant
    return None


def infer_task_canvas(grids) -> Union[int, None]:
    """The colour a task's grids agree is their canvas, or None if they
    don't agree on one.

    A grid whose own signals are inconclusive (see infer_background) is
    often just the canvas painted over, and the task's other grids still
    show which colour that is - evidence a per-grid rule throws away. Of
    the 2100 grids infer_background declines across ARC-AGI-2's training
    set, 31.4% belong to a task that establishes a canvas elsewhere.

    Two separate bars, because two different things can go wrong. A tie
    disqualifies outright: a task whose examples genuinely use different
    backgrounds (7.1% of them) has two grids naming two colours, and
    whichever Counter happens to return first is not the task's canvas -
    picking it would flatten a real difference into a wrong fact. Past
    that, the winner still has to be established by BACKGROUND_TASK_CANVAS_
    MAJORITY of the grids, so one confident grid out of five can't decide
    for the rest.
    """
    verdicts = [infer_background(grid) for grid in grids]
    established = [v for v in verdicts if v is not None]
    if not established:
        return None

    ranked = Counter(established).most_common()
    colour, agreeing = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == agreeing:
        return None  # tied - the examples disagree, so the task has no one canvas
    if agreeing / len(verdicts) < BACKGROUND_TASK_CANVAS_MAJORITY:
        return None
    return colour


def resolve_background(grid: np.array, task_canvas: Union[int, None] = None) -> Union[int, None]:
    """This grid's background colour, using the task's canvas to settle
    what the grid alone couldn't.

    The grid's own reading wins where it has one - that is what keeps a
    task whose examples use different backgrounds from being flattened
    onto a single colour. Only where the grid is inconclusive does the
    task's canvas fill in, and only while that colour still holds
    BACKGROUND_MIN_COVERED_SHARE of the grid: below that it is content
    that happens to share the canvas's colour, not the canvas showing
    through, and excluding it would delete real objects.

    A canvas covered *completely* (0% left) is left unidentified too. It
    costs nothing either way - excluding a colour with no cells is a no-op -
    but claiming a measurement no cell supports is not free.
    """
    own = infer_background(grid)
    if own is not None:
        return own
    if task_canvas is None:
        return None

    grid = np.asarray(grid)
    if grid.size and (grid == task_canvas).mean() >= BACKGROUND_MIN_COVERED_SHARE:
        return task_canvas
    return None
