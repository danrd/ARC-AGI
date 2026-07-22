import numpy as np
from copy import copy
from typing import Dict, List, Tuple
from collections import deque
from symbolic.objects_analysis import GridObject
from data.configs.env_configs import colors_mapping


def get_rotations(coords: List[tuple]) -> List[List[tuple]]:
    """
    Generate all possible 90-degree rotations of a set of coordinates.
    
    Args:
        coords: List of (x, y) coordinate tuples
    
    Returns:
        List of lists of coordinate tuples, each representing a rotation
    """
    if not coords:
        return []
    
    # Sort coordinates for consistency
    coords = sorted(coords, key=lambda x: (x[1], x[0]))
    
    # Get reference point (top-left)
    ref_x, ref_y = coords[0]
    
    # Normalize coordinates relative to reference point
    normalized = [(x - ref_x, y - ref_y) for x, y in coords]
    
    # Find the size of the bounding box
    max_x = max(x for x, y in normalized)
    max_y = max(y for x, y in normalized)
    
    rotations = []
    # Original orientation
    rotations.append(coords.copy())
    
    # 90 degrees clockwise - (x, y) -> (y, -x + max_x)
    rot_90 = [(ref_x + y, ref_y + (max_x - x)) for x, y in normalized]
    rotations.append(sorted(rot_90, key=lambda x: (x[1], x[0])))
    
    # 180 degrees - (x, y) -> (-x + max_x, -y + max_y)
    rot_180 = [(ref_x + (max_x - x), ref_y + (max_y - y)) for x, y in normalized]
    rotations.append(sorted(rot_180, key=lambda x: (x[1], x[0])))
    
    # 270 degrees clockwise - (x, y) -> (-y + max_y, x)
    rot_270 = [(ref_x + (max_y - y), ref_y + x) for x, y in normalized]
    rotations.append(sorted(rot_270, key=lambda x: (x[1], x[0])))
    
    return rotations

def calculate_adjacency_positions(obj: 'GridObject') -> List[tuple]:
    """
    Calculate all possible positions adjacent to an object where another object could be placed.
    
    Args:
        obj: GridObject to find adjacency positions for
    
    Returns:
        List of (x, y) coordinate tuples representing adjacent positions
    """
    adjacent_positions = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for x, y in obj.coords:
        for dx, dy in directions:
            adjacent_pos = (x + dx, y + dy)
            if adjacent_pos not in obj.coords:
                adjacent_positions.add(adjacent_pos)
    return list(adjacent_positions)

def count_holes(obj: 'GridObject') -> int:
    """Count the total number of holes in a GridObject."""
    if obj.shape in ['inner_hole', 'outer_hole']:
        return 0
    return len(obj.inner_holes) + len(obj.outer_holes)

def check_intersection(coords1: List[tuple], coords2: List[tuple]) -> bool:
    """Check if two sets of coordinates intersect."""
    return bool(set(coords1).intersection(set(coords2)))

def evaluate_match_configuration(obj1: 'GridObject', obj2: 'GridObject', 
                                 position: tuple, rotation_idx: int,
                                 all_grid_objects: List['GridObject'], 
                                 grid_shape:tuple) -> Dict:
    """
    Evaluate a potential match configuration between two objects.
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        position: Position to place obj2 (reference point)
        rotation_idx: Index of rotation to use for obj2
        all_grid_objects: List of all objects on the grid to check for intersections
    Returns:
        Dictionary with evaluation metrics:
        - valid: Whether the configuration is valid
        - hole_reduction: Reduction in holes (higher is better)
        - compactness: Measure of how compact the resulting shape is
    """
    # Get the rotated coordinates for obj2
    rotation_coords = get_rotations(obj2.coords)[rotation_idx]
    
    # Calculate the offset based on position and reference point 
    offset_x = position[0] - rotation_coords[0][0]
    offset_y = position[1] - rotation_coords[0][1]
    
    # Apply offset to all coordinates in the rotation
    shifted_coords = [(x + offset_x, y + offset_y) for x, y in rotation_coords]
    
    # Check if the shifted obj2 intersects with obj1 coords
    if not check_intersection(obj1.coords, shifted_coords):
        # Check if shifted obj2 intersects with any other grid objects
        for other_obj in all_grid_objects:
            # Fix the object equality check - avoid direct comparison
            if id(other_obj) != id(obj1) and id(other_obj) != id(obj2):
                if check_intersection(other_obj.coords, shifted_coords):
                    return {"valid": False, "hole_reduction": 0, "compactness": 0}
        
        # Create a temporary merged object to evaluate
        merged_coords = list(set(obj1.coords + shifted_coords))
        merged_obj = GridObject(
            shape="complex",
            coords=merged_coords,
            color=obj1.color_numbers+obj2.color_numbers,  # Assume we keep the color of obj1
            label=f"merged_{obj1.label}_{obj2.label}",
            grid_shape=grid_shape  # Assume we keep the positioning of obj1
        )
        
        # Calculate hole reduction
        original_holes = count_holes(obj1) + count_holes(obj2)
        merged_holes = count_holes(merged_obj)
        hole_reduction = original_holes - merged_holes
        
        # Calculate compactness (area of bounding box / number of cells)
        merged_area = merged_obj.hor_size * merged_obj.vert_size
        merged_cells = len(merged_coords)
        compactness = merged_cells / merged_area if merged_area > 0 else 0
        
        return {
            "valid": True,
            "hole_reduction": hole_reduction,
            "compactness": compactness,
            "shifted_coords": shifted_coords,
            "merged_obj": merged_obj
        }
    
    return {"valid": False, "hole_reduction": 0, "compactness": 0}

def find_best_object_match(obj1: 'GridObject', obj2: 'GridObject', 
                           all_grid_objects: List['GridObject'], grid_shape:tuple) -> Dict:
    """
    Find the best match configuration between two grid objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        all_grid_objects: List of all objects on the grid
    
    Returns:
        Dictionary with the best match configuration, or None if no valid match exists
    """
    best_match = None
    best_score = -float('inf')
    
    # Get all possible adjacent positions for obj1
    adjacent_positions = calculate_adjacency_positions(obj1)
    
    # Try all rotations of obj2
    for rotation_idx in range(4):
        # Try placing obj2 at each adjacent position of obj1
        for position in adjacent_positions:
            match_config = evaluate_match_configuration(
                obj1, obj2, position, rotation_idx, all_grid_objects, grid_shape
            )
            
            if match_config["valid"]:
                # Calculate a score based on hole reduction and compactness
                # Weight hole reduction higher than compactness
                score = match_config["hole_reduction"] * 10 + match_config["compactness"]
                
                if score > best_score:
                    best_score = score
                    best_match = match_config
                    best_match["rotation_idx"] = rotation_idx
                    best_match["position"] = position
    
    return best_match

def merge_objects(grid:np.ndarray, obj1:GridObject, obj2:GridObject, match_config:Dict, font_color:int):
    """
    Merge two objects based on a match configuration.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        match_config: Match configuration from find_best_object_match
        grid: The current grid array
    
    Returns:
        Tuple of (merged_object, updated_grid)
    """
    # Create the merged object
    shifted_coords = match_config["shifted_coords"]

    height, width = grid.shape
    # Update the grid
    updated_grid = grid.copy()
    
    for x, y in obj2.coords:
        updated_grid[x, y] = font_color
    
    # Add the merged object to the grid
    for x, y in shifted_coords:
        if x < height and y < width:
            updated_grid[x, y] = obj2.color_numbers[0]  # Use the color of obj2

    updated_obj = copy(obj2)
    updated_obj.reinit_obj(shifted_coords, updated_grid)
    
    return updated_obj, updated_grid

def find_most_probable_merge(grid: np.ndarray, obj1: 'GridObject', obj2: 'GridObject', 
                           all_grid_objects: List['GridObject']):
    """
    Find the most probable merge configuration between two specific objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        all_grid_objects: List of all GridObjects on the grid (to check for intersections)
        grid: The current grid array
    
    Returns:
        Dictionary with the best match configuration, or None if no valid match exists
    """
    # Skip if either object is a hole
    if obj1.shape in ['inner_hole', 'outer_hole'] or obj2.shape in ['inner_hole', 'outer_hole']:
        return None
    
    # Use a filtered list of grid objects that excludes the two objects we're working with
    # to avoid the equality comparison issue
    filtered_grid_objects = [obj for obj in all_grid_objects if id(obj) != id(obj1) and id(obj) != id(obj2)]
    
    # Try matching in both directions
    match = find_best_object_match(obj1, obj2, [obj1, obj2] + filtered_grid_objects, grid.shape)
    
    best_match = None
    best_score = -float('inf')
    
    if match and match["valid"]:
        match["obj1"] = obj1
        match["obj2"] = obj2
        match["score"] = match["hole_reduction"] * 10 + match["compactness"]
        if match["score"] > best_score:
            best_score = match["score"]
            best_match = match
    
    return best_match

def perform_merge(grid:np.ndarray, obj1:GridObject, obj2:GridObject, 
                  all_grid_objects:List[GridObject], font_color:int):
    """
    Find and perform the best merge between two specific objects.
    
    Args:
        obj1: First GridObject
        obj2: Second GridObject
        all_grid_objects: List of all GridObjects on the grid
        grid: The current grid array
    
    Returns:
        Tuple of (updated_grid_objects, updated_grid, match_info)
    """
    # Find the best match configuration
    best_match = find_most_probable_merge(grid, obj1, obj2, all_grid_objects)
    
    if not best_match:
        return all_grid_objects, grid, {"status": "no_valid_match", "message": "No valid match found"}
    
    # Extract the objects from the match
    first_obj = best_match["obj1"]
    second_obj = best_match["obj2"]
    
    # Perform the merge
    updated_obj, updated_grid = merge_objects(grid, first_obj, second_obj, best_match, font_color)
    
    return updated_grid

def inverse_obj_color(grid:np.array, obj:GridObject, font_color:int):
    if len(obj.colors) == 1 and (len(obj.inner_holes) + len(obj.outer_holes) > 0):
        base_color = obj.color_numbers[0]
        coords_copy = copy(obj.coords)
        non_object_coords_copy = copy(obj.non_object_coords)
        for i, j in obj.non_object_coords:
            grid[i, j] = base_color
        for i, j in obj.coords:
            grid[i, j] = font_color
        obj.coords = non_object_coords_copy
        obj.non_object_coords = coords_copy
    elif len(obj.colors) == 2:
        color_1 = obj.color_numbers[0]
        color_2 = obj.color_numbers[1]
        inversion_dict = {color_1:color_2, color_2:color_1}
        for i, j in obj.coords:
            grid[i, j] = inversion_dict[grid[i, j]]
    return grid

def find_shortest_distance(obj1, obj2, allow_diagonals=False):
    """
    Find the shortest distance between two objects and return the pair(s) of cells with minimal distance.
    
    Args:
    -----------
    obj1 : list of tuples
        List of (x, y) coordinates representing cells of the first object
    obj2 : list of tuples
        List of (x, y) coordinates representing cells of the second object
    allow_diagonals : bool, default=False
        If True, use Chebyshev distance (allows diagonals)
        If False, use Manhattan distance
    
    Returns:
    --------
    list of tuples:
        List of (cell1, cell2) pairs with the minimum distance, where:
        - cell1: coordinates (x, y) from object1
        - cell2: coordinates (x, y) from object2
    """
    if not obj1 or not obj2:
        raise ValueError("Both objects must contain at least one cell")
    
    min_distance = float('inf')
    closest_pairs = []
    
    for cell1 in obj1:
        for cell2 in obj2:
            # Calculate the distance based on the specified method
            if allow_diagonals:
                # Chebyshev distance (allows diagonals) - maximum of absolute differences
                distance = max(abs(cell1[0] - cell2[0]), abs(cell1[1] - cell2[1]))
            else:
                # Manhattan distance (no diagonals) - sum of absolute differences
                distance = abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])
                
            # If we found a new minimum distance, clear the previous closest pairs
            if distance < min_distance:
                min_distance = distance
                closest_pairs = [(cell1, cell2)]
            # If we found another pair with the same minimum distance, add it
            elif distance == min_distance:
                closest_pairs.append((cell1, cell2))
    
    return closest_pairs

def add_color_to_object(obj, color):
    """Safely add color to object, avoiding None values."""
    if color is not None and color not in obj.color_numbers:
        if isinstance(obj.color_numbers, list):
            obj.color_numbers.append(color)
        else:
            obj.color_numbers = obj.color_numbers + tuple([color])
        
        # Update colors, filtering out None values
        if isinstance(obj.color_numbers, list):
            obj.colors = tuple([colors_mapping[c] for c in obj.color_numbers if c is not None])
        else:
            obj.colors = tuple([colors_mapping[c] for c in obj.color_numbers if c is not None])

def find_shortest_path(grid, start, end):
    """Find shortest path between two points using BFS."""
    if start == end:
        return [start]
        
    rows, cols = grid.shape
    visited = set()
    queue = deque([(start, [start])])
    
    max_iterations = rows * cols * 5 # Maximum possible states
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            return path
            
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        # Check all four adjacent cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and grid[nx, ny] != 1.0:
                queue.append(((nx, ny), path + [(nx, ny)]))
                
    return []  # No path found

def find_path_through_background(grid, start, end, font_color):
    """
    Find path between two points.
    First checks for 1 or 2 straight line solutions, then uses A* algorithm.
    """
    import heapq
    rows, cols = grid.shape

    # Basic boundary and equality checks
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and
            0 <= end[0] < rows and 0 <= end[1] < cols):
        # Start or end is out of bounds
        return []

    if start == end:
        return [start]

    # Helper function to check and construct a straight path segment
    # p1: start of the segment
    # p2: end of the segment
    # is_p2_overall_end: True if p2 is the global 'end' point of the entire path
    def get_straight_path_if_valid(p1, p2, is_p2_overall_end):
        # Path starts with p1
        path = [p1]
        
        # If p1 and p2 are the same point
        if p1 == p2:
            # If this single point is the overall end, it's a valid path of one point.
            if is_p2_overall_end:
                return [p1]
            # If it's an intermediate point (like a corner), it must be of font_color.
            # (p1 is assumed to be valid if it's the global start or a previously validated corner)
            # This check is more for when p1=p2 is a corner being evaluated.
            if 0 <= p1[0] < rows and 0 <= p1[1] < cols and grid[p1[0], p1[1]] == font_color:
                return [p1]
            return None # Invalid single-point segment if not overall end and not font_color

        curr_x, curr_y = p1

        if p1[0] == p2[0]:  # Vertical line
            x = p1[0]
            y_step = 1 if p2[1] > p1[1] else -1
            next_y = p1[1] + y_step
            
            while True:
                current_point = (x, next_y)
                # Check bounds for the current point
                if not (0 <= x < rows and 0 <= next_y < cols):
                    return None # Path goes out of bounds

                is_current_point_p2 = (current_point == p2)
                
                # Validity check for current_point:
                # If current_point is p2 AND p2 is the overall_end, it's allowed (color doesn't matter for overall end).
                # Otherwise (it's an intermediate point, or p2 is a corner), it must be font_color.
                if not (is_current_point_p2 and is_p2_overall_end):
                    if grid[x, next_y] != font_color:
                        return None # Invalid intermediate point
                
                path.append(current_point)

                if is_current_point_p2: # Reached the end of the segment
                    break 
                
                next_y += y_step
            return path

        elif p1[1] == p2[1]:  # Horizontal line
            y = p1[1]
            x_step = 1 if p2[0] > p1[0] else -1
            next_x = p1[0] + x_step

            while True:
                current_point = (next_x, y)
                if not (0 <= next_x < rows and 0 <= y < cols):
                    return None # Path goes out of bounds

                is_current_point_p2 = (current_point == p2)

                if not (is_current_point_p2 and is_p2_overall_end):
                    if grid[next_x, y] != font_color:
                        return None # Invalid intermediate point
                
                path.append(current_point)

                if is_current_point_p2: # Reached the end of the segment
                    break
                
                next_x += x_step
            return path
        
        return None # Not a straight line (neither vertical nor horizontal)

    # 1. Check for 1-line path (direct straight path from start to end)
    # The 'end' point is allowed to not be font_color.
    path_1_line = get_straight_path_if_valid(start, end, True)
    if path_1_line:
        return path_1_line

    # 2. Check for 2-line paths
    # The corner point must be within bounds and be of font_color.
    # It must also be distinct from start and end to form a true "bend".

    # Path Type 1: Horizontal then Vertical (start -> corner1 -> end)
    # Corner1 is (end_x, start_y)
    corner1 = (end[0], start[1]) 
    if corner1 != start and corner1 != end and \
       0 <= corner1[0] < rows and 0 <= corner1[1] < cols and \
       grid[corner1[0], corner1[1]] == font_color:
        
        # Segment from start to corner1. Corner1 is an intermediate point, so is_p2_overall_end is False.
        segment1_to_c1 = get_straight_path_if_valid(start, corner1, False) 
        if segment1_to_c1:
            # Segment from corner1 to end. End is the overall end, so is_p2_overall_end is True.
            segment2_from_c1_to_end = get_straight_path_if_valid(corner1, end, True)
            if segment2_from_c1_to_end:
                # Combine paths, removing duplicate corner1 from the end of segment1
                return segment1_to_c1[:-1] + segment2_from_c1_to_end

    # Path Type 2: Vertical then Horizontal (start -> corner2 -> end)
    # Corner2 is (start_x, end_y)
    corner2 = (start[0], end[1])
    if corner2 != start and corner2 != end and \
       0 <= corner2[0] < rows and 0 <= corner2[1] < cols and \
       grid[corner2[0], corner2[1]] == font_color:

        segment1_to_c2 = get_straight_path_if_valid(start, corner2, False)
        if segment1_to_c2:
            segment2_from_c2_to_end = get_straight_path_if_valid(corner2, end, True)
            if segment2_from_c2_to_end:
                return segment1_to_c2[:-1] + segment2_from_c2_to_end

    # 3. If no 1 or 2-line path found, proceed with A* algorithm
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    visited = set()
    parent_map = {}
    # Priority queue stores: (f_score, tie_breaker_step_count, position)
    # f_score = g_score + h_score (heuristic)
    # Using step_count as a tiebreaker for items with equal f_score
    open_set = [(manhattan_distance(start, end), 0, start)] 
    g_score = {start: 0}  # Cost from start to current position (number of steps)
    
    # step_count is used to ensure stable sorting in heapq if f_scores are equal
    # It does not represent path length for g_score.
    a_star_step_counter = 0 

    while open_set:
        current_f_score, _, current_pos = heapq.heappop(open_set)

        if current_pos in visited:
            continue
        
        if current_pos == end:
            # Reconstruct path
            path = [current_pos]
            while path[-1] != start:
                path.append(parent_map[path[-1]])
            return path[::-1] # Return reversed path (start to end)

        visited.add(current_pos)
        x, y = current_pos

        # Explore neighbors (up, down, left, right)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            neighbor_pos = (nx, ny)

            # Check if neighbor is valid
            if (0 <= nx < rows and 0 <= ny < cols and 
                neighbor_pos not in visited and 
                (grid[nx, ny] == font_color or neighbor_pos == end)): # Valid move condition
                
                tentative_g_score = g_score[current_pos] + 1

                if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                    # This path to neighbor is better than any previous one found
                    parent_map[neighbor_pos] = current_pos
                    g_score[neighbor_pos] = tentative_g_score
                    h_score = manhattan_distance(neighbor_pos, end)
                    f_score = tentative_g_score + h_score
                    
                    a_star_step_counter += 1
                    heapq.heappush(open_set, (f_score, a_star_step_counter, neighbor_pos))
    
    return []  # No path found
    
def filter_paths(paths, obj1, obj2, preference='right'):
    """
    Filter paths based on specified criteria.
    
    Args:
        paths: List of paths, where each path is a list of coordinate tuples [(x1, y1), (x2, y2), ...].
        preference: String 'left' or 'right' for tie-breaking when paths have the same number of turns.
    
    Returns:
        Either a single path or list of paths based on the filtering criteria.
    """
    # Case 1: If there is only 1 path, return it
    if len(paths) == 1:
        return [paths[0]]
    contours = set(obj1.inner_contour + obj2.inner_contour)
    paths = [path for path in paths if len(contours.intersection(set(path[1:-1])))==0]
    # Check if all starting and ending points are different
    start_points = [path[0] for path in paths]
    end_points = [path[-1] for path in paths]
    
    # Case 2: If all starting and ending coordinates are different
    if len(set(start_points)) == len(paths) and len(set(end_points)) == len(paths):
        return paths
    
    # Case 3: Return the path with the least number of turns
    def count_turns(path):
        """Count the number of turns in a path."""
        if len(path) < 3:
            return 0
            
        turns = 0
        for i in range(1, len(path) - 1):
            # Calculate direction changes
            dx1, dy1 = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            dx2, dy2 = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
            
            # If direction changes, it's a turn
            if dx1 != dx2 or dy1 != dy2:
                turns += 1
                
        return turns
    
    # Calculate turns for each path
    turn_counts = [count_turns(path) for path in paths]
    min_turns = min(turn_counts)
    
    # Find paths with minimum turns
    min_turn_paths = [path for path, turns in zip(paths, turn_counts) if turns == min_turns]
    
    # If only one path has minimum turns, return it
    if len(min_turn_paths) == 1:
        return [min_turn_paths[0]]
    
    # Case 3a: Tie-breaking based on preference
    # For this tie-breaker, we'll interpret 'left' as the first path in the list
    # and 'right' as the last path in the list with the minimum number of turns
    if preference.lower() == 'left':
        return [min_turn_paths[0]]
    else:  # 'right' or any other va
        return [min_turn_paths[-1]]

def gravity(grid:np.array, obj1:GridObject, obj2:GridObject, font_color:int):
    """Object 2 is moved to connect with Object 1 based on the shortest distance."""
    # Find the closest points between the two objects
    new_coords = []
    closest_pairs = find_shortest_distance(obj2.coords, obj1.coords)
    if closest_pairs == []:
        return grid  # No change if no valid pair found
    start, end = closest_pairs[0]
    path = find_shortest_path(grid, start, end)
    if len(path) < 3: # Objeact are already close enough
        return grid
    # Calculate the direction and distance to move obj_2
    obj1_coords = obj1.coords
    obj2_coords = obj2.coords
    
    # Calculate the shift needed to make the closest points touch
    shift_x = end[0] - start[0]
    shift_y = end[1] - start[1]
    
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Store original colors of obj_2
    colors = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions of object_2
    for x, y in obj2_coords:
        new_grid[x, y] = font_color
    
    # Place object_2 in new position
    for x, y in obj2_coords:
        new_x, new_y = x + shift_x, y + shift_y
        if 0 <= new_x < new_grid.shape[0] and 0 <= new_y < new_grid.shape[1] and new_grid[new_x, new_y]==font_color:
            new_coords.append((new_x, new_y))
            new_grid[new_x, new_y] = colors[(x, y)]
        else:
            break
    obj2.reinit_obj(new_coords, new_grid)
    return new_grid

def edge_gravity(grid:np.array, obj1:GridObject, font_color:int, bottom_only=False):
    """
    Object 1 is moved towards the closest edge or optionally towards bottom edge.
    If bottom_only is True, only move towards the bottom edge.
    """
    new_coords = []
    obj_coords = obj1.coords
    min_x, max_x, min_y, max_y = obj1.min_i, obj1.max_i, obj1.min_j, obj1.max_j
    
    # Get grid dimensions
    grid_height, grid_width = grid.shape
    
    # If bottom_only, we just need to calculate movement to bottom
    if bottom_only:
        shift_x = grid_height - 1 - max_x  # Distance to bottom edge
        shift_y = 0
    else:
        # Calculate distances to each edge
        dist_to_top = min_x
        dist_to_bottom = grid_height - 1 - max_x
        dist_to_left = min_y
        dist_to_right = grid_width - 1 - max_y
        
        # Find the nearest edge
        min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)
        
        # Determine shift direction based on closest edge
        if min_dist == dist_to_top:
            shift_x = -dist_to_top
            shift_y = 0
        elif min_dist == dist_to_bottom:
            shift_x = dist_to_bottom
            shift_y = 0
        elif min_dist == dist_to_left:
            shift_x = 0
            shift_y = -dist_to_left
        else:  # Bottom is closest
            shift_x = 0
            shift_y = dist_to_right
    
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Store original colors
    colors = {(x, y): grid[x, y] for x, y in obj_coords}
    
    # Clear original positions
    for x, y in obj_coords:
        new_grid[x, y] = font_color
    
    # Place object in new position
    for x, y in obj_coords:
        new_x, new_y = x + shift_x, y + shift_y
        if 0 <= new_x < new_grid.shape[0] and 0 <= new_y < new_grid.shape[1]:
            new_coords.append((new_x, new_y))
            new_grid[new_x, new_y] = colors[(x, y)]
    obj1.reinit_obj(new_coords, new_grid)
    return new_grid

def x_alignment(grid:np.array, obj1:GridObject, obj2:GridObject, font_color:int):
    """Align Object 2 with Object 1 along x axis."""
    new_coords = []
    obj1_center_x = obj1.center[0]
    obj2_center_x = obj2.center[0]
    obj2_coords = obj2.coords
    
    # Calculate shift needed to align centers on x-axis
    shift_x = obj1_center_x - obj2_center_x
    shift_y = 0  # No vertical shift
    
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Store original colors of obj_2
    colors = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions of object_2
    for x, y in obj2_coords:
        new_grid[x, y] = font_color
    
    # Place object_2 in new position
    for x, y in obj2_coords:
        new_x, new_y = x + shift_x, y + shift_y
        if 0 <= new_x < new_grid.shape[0] and 0 <= new_y < new_grid.shape[1]:
            new_coords.append((new_x, new_y))
            new_grid[new_x, new_y] = colors[(x, y)]
    obj2.reinit_obj(new_coords, new_grid)
    return new_grid

def y_alignment(grid:np.array, obj1:GridObject, obj2:GridObject, font_color:int):
    """Align Object 2 with Object 1 along y axis."""
    new_coords = []
    obj1_center_y = obj1.center[1]
    obj2_center_y = obj2.center[1]
    obj2_coords = obj2.coords
    
    # Calculate shift needed to align centers on y-axis
    shift_x = 0  # No horizontal shift
    shift_y = obj1_center_y - obj2_center_y
    
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Store original colors of obj_2
    colors = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions of object_2
    for x, y in obj2_coords:
        new_grid[x, y] = font_color
    
    # Place object_2 in new position
    for x, y in obj2_coords:
        new_x, new_y = x + shift_x, y + shift_y
        if 0 <= new_x < new_grid.shape[0] and 0 <= new_y < new_grid.shape[1]:
            new_coords.append((new_x, new_y))
            new_grid[new_x, new_y] = colors[(x, y)]
    obj2.reinit_obj(new_coords, new_grid)
    return new_grid

def contour_connection(grid:np.array, obj1:GridObject, obj2:GridObject, contour_color:float, rectangle_color:float):
    """
    If Object 1 and Object 2 are 'cell' type, connect them with contour and 
    color it with specified color to create rectangle.
    """
    # Get bounding box coordinates
    min_x = min(obj1.min_i, obj2.min_i)
    max_x = max(obj1.max_i, obj2.max_i)
    min_y = min(obj1.min_j, obj2.min_j)
    max_y = max(obj1.max_j, obj2.max_j)
      
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Draw the rectangle edges (contour)
    for x in range(min_x, max_x + 1):
        # Top and bottom edges
        if 0 <= x < grid.shape[0]:
            if 0 <= min_y < grid.shape[1]:
                new_grid[x, min_y] = contour_color
            if 0 <= max_y < grid.shape[1]:
                new_grid[x, max_y] = contour_color
    
    for y in range(min_y, max_y + 1):
        # Left and right edges
        if 0 <= y < grid.shape[1]:
            if 0 <= min_x < grid.shape[0]:
                new_grid[min_x, y] = contour_color
            if 0 <= max_x < grid.shape[0]:
                new_grid[max_x, y] = contour_color

    for i in range(min_x+1, max_x):
        for j in range(min_y+1, max_y):
            new_grid[i, j] = rectangle_color
    return new_grid

def define_emission_center(obj:GridObject, direction):
    center_x = (obj.min_i + obj.max_i) / 2
    center_y = (obj.min_j + obj.max_j) / 2
    adjustments = {
    'N': (-0.5, 0),    # North: adjust row up by 0.5
    'NE': (-0.5, 0.5), # Northeast: adjust row up, column right
    'E': (0, 0.5),     # East: adjust column right by 0.5
    'SE': (0.5, 0.5),  # Southeast: adjust row down, column right
    'S': (0.5, 0),     # South: adjust row down by 0.5
    'SW': (0.5, -0.5), # Southwest: adjust row down, column left
    'W': (0, -0.5),    # West: adjust column left by 0.5
    'NW': (-0.5, -0.5) # Northwest: adjust row up, column left
    }
    adjustment = adjustments[direction]
    if type(center_x) is not int:
      center_x += adjustment[0]
    if type(center_y) is not int:
      center_y += adjustment[1]    
    return int(center_x), int(center_y)

def emission(grid:np.array, obj1:GridObject, color:float, direction:str):
    """Emit lines with specified color for 8 directions from Object 1."""
    # Get center point of object
    center_x, center_y = obj1.center
    new_coords = []
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Define the 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = {
        'N':(-1, 0),  # North
        'NE':(-1, 1),  # Northeast
        'E':(0, 1),   # East
        'SE':(1, 1),   # Southeast
        'S':(1, 0),   # South
        'SW':(1, -1),  # Southwest
        'W':(0, -1),  # West
        'NW':(-1, -1)  # Northwest
    }
    dx, dy = directions[direction]
    x, y = define_emission_center(obj1, direction)
    
    # Continue in direction until we hit the grid boundary
    while 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1]:
        x += dx
        y += dy
        # Don't overwrite original object
        if (x, y) not in obj1.coords:
            new_coords.append((x,y))
            new_grid[x, y] = color
            add_color_to_object(obj1, color)
    return new_grid


def emission_with_collision(grid:np.array, obj1:GridObject, emission_color:float, 
                            font_color:int, direction:str, collision_type:str, 
                            collision_color=None, cell2obj:Dict[Tuple[int, int], int]=None,
                            objects: List[GridObject]=None):
    """Emit lines with specified color for 8 directions from Object 1 with collision handling."""
    # Get center point of object
    new_coords = list(copy(obj1.coords))
    collision_cells = []
    collision_objects = []
    # Create a new grid to modify
    new_grid = grid.copy()
    
    # Define the 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = {
        'N':(-1, 0),  # North
        'NE':(-1, 1),  # Northeast
        'E':(0, 1),   # East
        'SE':(1, 1),   # Southeast
        'S':(1, 0),   # South
        'SW':(1, -1),  # Southwest
        'W':(0, -1),  # West
        'NW':(-1, -1)  # Northwest
    }

    turn_directions = {
        'N':['W', 'E'],  # North
        'NE':['NW', 'SE'],  # Northeast
        'E':['N', 'S'],   # East
        'SE':['NE', 'SW'],   # Southeast
        'S':['E', 'W'],   # South
        'SW':['SE', 'NW'],  # Southwest
        'W':['S', 'N'],  # West
        'NW':['SW', 'NE']  # Northwest
    }
    
    dx, dy = directions[direction]
    dx_l, dy_l = directions[turn_directions[direction][0]]
    dx_r, dy_r = directions[turn_directions[direction][1]]
    x, y = define_emission_center(obj1, direction)
    
    # Continue in direction until we hit the grid boundary
    while 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1]:
        x += dx
        y += dy
        # Don't overwrite original object
        if (x, y) not in obj1.coords and grid[x, y] == font_color:
            new_coords.append((x, y))
            new_grid[x, y] = emission_color
            add_color_to_object(obj1, emission_color)
            
        elif (x, y) not in obj1.coords and grid[x, y] != font_color:
            if collision_type == 'stop':
                break
                
            elif collision_type == 'recolor':
                # Ensure collision_color is set, default to emission_color if None
                if collision_color is None:
                    collision_color = emission_color
                new_coords.append((x, y))
                new_grid[x, y] = collision_color
                add_color_to_object(obj1, collision_color)
                
            elif collision_type == 'object_recolor': 
                # Ensure collision_color is set for object recoloring
                if collision_color is None:
                    collision_color = emission_color
                    
                collision_obj = None
                try:
                    collision_obj = objects[cell2obj[(x,y)]]
                except (KeyError, TypeError):
                    if objects:  # Safety check
                        for obj in objects:
                            if (x,y) in obj.coords:
                                collision_obj = obj  
                                break
                                
                if collision_obj and collision_obj.label not in collision_objects:
                    collision_objects.append(collision_obj.label)
                    if isinstance(collision_obj.color_numbers, list):
                        collision_obj.color_numbers = [collision_color]
                    else:
                        collision_obj.color_numbers = tuple([collision_color])
                    collision_obj.colors = tuple([colors_mapping[collision_color]])
                    for cx, cy in collision_obj.coords:
                        new_grid[cx, cy] = collision_color
                        
            elif collision_type == 'contour':
                collision_cells.append((x, y))
                
            elif collision_type in ['turn_left', 'turn_right']:
                # Fixed turning logic with proper boundary checks and loop prevention
                virt_x_l = x - dx
                virt_y_l = y - dy
                left_path = []
                visited_left = set()
                
                # Explore left path with safety checks
                while (0 <= virt_x_l + dx_l < grid.shape[0] and 
                       0 <= virt_y_l + dy_l < grid.shape[1] and 
                       (virt_x_l, virt_y_l) not in visited_left and
                       len(left_path) < grid.shape[0] * grid.shape[1]):  # Prevent infinite loops
                    
                    visited_left.add((virt_x_l, virt_y_l))
                    virt_x_l += dx_l
                    virt_y_l += dy_l
                    
                    # Check if the next cell in the new direction is valid
                    if (0 <= virt_x_l < grid.shape[0] and 
                        0 <= virt_y_l < grid.shape[1] and
                        grid[virt_x_l, virt_y_l] == font_color):
                        left_path.append((virt_x_l, virt_y_l))
                    else:
                        break
                
                virt_x_r = x - dx
                virt_y_r = y - dy
                right_path = []
                visited_right = set()
                
                # Explore right path with safety checks
                while (0 <= virt_x_r + dx_r < grid.shape[0] and 
                       0 <= virt_y_r + dy_r < grid.shape[1] and 
                       (virt_x_r, virt_y_r) not in visited_right and
                       len(right_path) < grid.shape[0] * grid.shape[1]):  # Prevent infinite loops
                    
                    visited_right.add((virt_x_r, virt_y_r))
                    virt_x_r += dx_r
                    virt_y_r += dy_r
                    
                    # Check if the next cell in the new direction is valid
                    if (0 <= virt_x_r < grid.shape[0] and 
                        0 <= virt_y_r < grid.shape[1] and
                        grid[virt_x_r, virt_y_r] == font_color):
                        right_path.append((virt_x_r, virt_y_r))
                    else:
                        break
                
                # Choose path and apply changes
                if len(left_path) < len(right_path) or collision_type == 'turn_left':
                    for i, j in left_path:
                        new_coords.append((i, j))
                        new_grid[i, j] = emission_color
                        add_color_to_object(obj1, emission_color)
                    if left_path:
                        x, y = left_path[-1]
                        dx, dy = dx_l, dy_l  # Update direction for continued emission
                else:
                    for i, j in right_path:
                        new_coords.append((i, j))
                        new_grid[i, j] = emission_color
                        add_color_to_object(obj1, emission_color)
                    if right_path:
                        x, y = right_path[-1]
                        dx, dy = dx_r, dy_r  # Update direction for continued emission
                
                # If no valid path found, stop
                if not left_path and not right_path:
                    break
                    
    # Handle contour collision
    if collision_cells:  
        # Ensure collision_color is set for contour
        if collision_color is None:
            collision_color = emission_color
            
        contour_cells = []
        for cell in collision_cells:
            collision_contour = get_outer_contour(grid, [cell], font_color, recolor=True)
            contour_cells.extend(collision_contour)
        for x, y in collision_contour:
            new_grid[x, y] = collision_color
        collision_cells.extend(contour_cells)
        
    # Update obj1 coordinates
    obj1.coords = tuple(new_coords)
    
    return new_grid

def get_outer_contour(grid:np.array, obj_coords:List[tuple], font_color:int, recolor=False) -> List[Tuple[int, int]]:
    """
    Returns the minimal bounding rectangle around the shape.
    """
    i_coords = [cell[0] for cell in obj_coords]
    j_coords = [cell[1] for cell in obj_coords]
    
    height, width = grid.shape
    
    min_i = min(i_coords)
    min_j = min(j_coords)
    max_i = max(i_coords)
    max_j = max(j_coords)
    
    contour_min_i = max(min(i_coords) - 1, 0) 
    contour_min_j = max(min(j_coords) - 1, 0)
    contour_max_i = min(max(i_coords) + 1, height-1)
    contour_max_j = min(max(j_coords) + 1, width-1)
    
    contour = []
    
    if min_j > contour_min_j:
        left_line = [(i, contour_min_j) for i in range(contour_min_i, contour_max_i+1)]
        values_check = [grid[i, j]==font_color for i, j in left_line]
        if False not in values_check or recolor:
            contour.extend(left_line)
        else:
            return []
            
    if max_j < contour_max_j:
        right_line = [(i, contour_max_j) for i in range(contour_min_i, contour_max_i+1)]
        values_check = [grid[i, j]==font_color for i, j in right_line]
        if False not in values_check or recolor:
            contour.extend(right_line)
        else:
            return []
           
    if min_i > contour_min_i:
        upper_line = [(contour_min_i, j) for j in range(contour_min_j, contour_max_j+1)]
        values_check = [grid[i, j]==font_color for i, j in upper_line]
        if False not in values_check or recolor:
            contour.extend(upper_line)
        else:
            return []
            
    if max_i < contour_max_i:
        bottom_line = [(contour_max_i, j) for j in range(contour_min_j, contour_max_j+1)]
        values_check = [grid[i, j]==font_color for i, j in bottom_line]
        if False not in values_check or recolor:
            contour.extend(bottom_line)
        else:
            return []
    
    contour = list(set(contour))  
    
    return contour

def color_inner_holes(grid:np.array, obj1:GridObject, color:float):
    """Change color of inner holes of Object 1."""
    inner_holes_coords = []
    for inner_hole in obj1.inner_holes:
        inner_holes_coords.extend(inner_hole.coords)
        inner_hole.color_numbers = tuple(color)
        inner_hole.colors = tuple([colors_mapping[color]])
    # Create a new grid to modify
    new_grid = grid.copy()
    for x, y in inner_holes_coords:
        new_grid[x, y] = color
    return new_grid

def color_outer_holes(grid:np.array, obj1:GridObject, color:float):
    """Change color of outer holes (concavities) of Object 1."""
    outer_holes_coords = []
    for outer_hole in obj1.outer_holes:
        outer_holes_coords.extend(outer_hole.coords)
        outer_hole.color_numbers = tuple([color])
        outer_hole.colors = tuple([colors_mapping[color]])
    # Create a new grid to modify
    new_grid = grid.copy()
    for x, y in outer_holes_coords:
        new_grid[x, y] = color
    return new_grid

def objects_swap(grid:np.array, obj1:GridObject, obj2:GridObject, font_color:int):
    """Exchange of objects positions: object 1 will be placed on object 2 position and visa versa."""
    new_coords1 = []
    new_coords2 = []
    obj1_coords = obj1.coords
    obj2_coords = obj2.coords
    obj1_center_x, obj1_center_y = obj1.center[0], obj1.center[1]
    obj2_center_x, obj2_center_y = obj2.center[0], obj2.center[1]
    shift1_to_2_x = round(obj2_center_x - obj1_center_x)
    shift1_to_2_y = round(obj2_center_y - obj1_center_y)
    shift2_to_1_x = round(obj1_center_x - obj2_center_x)
    shift2_to_1_y = round(obj1_center_y - obj2_center_y)
    colors1 = {(x, y): grid[x, y] for x, y in obj1_coords}
    colors2 = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions
    for x, y in obj1_coords:
        grid[x, y] = font_color
    for x, y in obj2_coords:
        grid[x, y] = font_color
    
    # Place object_1 at object_2's position
    for x, y in obj1_coords:
        new_x, new_y = x + shift1_to_2_x, y + shift1_to_2_y
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            new_coords1.append((new_x, new_y))
            grid[new_x, new_y] = colors1[(x, y)]
            
    # Place object_2 at object_1's position
    for x, y in obj2_coords:
        new_x, new_y = x + shift2_to_1_x, y + shift2_to_1_y
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            new_coords2.append((new_x, new_y))
            grid[new_x, new_y] = colors2[(x, y)]
    obj1.reinit_obj(new_coords1, grid)
    obj2.reinit_obj(new_coords2, grid)
    return grid

def upscale(grid:np.array, obj1:GridObject, font_color:int):
    coords = obj1.coords
    new_coords = []
    min_x, max_x, min_y, max_y = obj1.min_i, obj1.max_i, obj1.min_j, obj1.max_j
    colors = {(x, y): grid[x, y] for x, y in coords}
    # Clear original positions
    for x, y in coords:
        grid[x, y] = font_color
                        
    # Place upscaled object - each cell becomes a 2x2 square
    for orig_x, orig_y in coords:
        color = colors[(orig_x, orig_y)]
        # Create 2x2 square
        for dx in range(2):
            for dy in range(2):
                new_x, new_y = orig_x * 2 + dx - (max_x-min_x+1) // 2, orig_y * 2 + dy - (max_y-min_y+1) // 2
                if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
                    new_coords.append((new_x, new_y))
                    grid[new_x, new_y] = color
    if len(new_coords) > 0:
        obj1.reinit_obj(new_coords, grid)
    return grid

def define_coords(obj:GridObject, new_obj_structure:np.array):
    old_coords = obj.coords
    new_coords = []
    coords_mapping = {}
    max_size = max(obj.vert_size, obj.hor_size)
    old_max_i, old_max_j = obj.obj_structure.shape
    old_extended_structure = np.zeros((max_size, max_size)) 
    old_extended_structure[max_size-old_max_i:, max_size-old_max_j:] = obj.obj_structure
    new_max_i, new_max_j = new_obj_structure.shape
    new_extended_structure = np.zeros((max_size, max_size)) 
    new_extended_structure[max_size-new_max_i:, max_size-new_max_j:] = new_obj_structure 
    if obj.center in obj.coords:
        center_idx = obj.coords.index(obj.center) + 1
        center_in_old_structure = np.argwhere(old_extended_structure==center_idx)[0]
        center_in_new_structure = np.argwhere(new_extended_structure==center_idx)[0]
    else:
        center_in_old_structure = (old_extended_structure.shape[0]//2, old_extended_structure.shape[1]//2)
        center_in_new_structure = (new_extended_structure.shape[0]//2, new_extended_structure.shape[1]//2)
    for j in range(max_size):
        for i in range(max_size):
            val = new_extended_structure[i, j]
            if val > 0:
                coords = old_coords[int(val-1)]
                old_center_offset = np.argwhere(old_extended_structure==val)[0] - center_in_old_structure
                new_center_offset = np.argwhere(new_extended_structure==val)[0] - center_in_new_structure
                offset = new_center_offset - old_center_offset
                new_coord = (coords[0]+offset[0], coords[1]+offset[1])
                new_coords.append(new_coord)
                coords_mapping[(new_coord[0], new_coord[1])] = (coords[0], coords[1])
    return new_coords, coords_mapping

def define_coords_offsets(obj_mask:np.array):
    shape = obj_mask.shape
    new_coords_offsets = []
    start_coord = (0, 0)
    for j in range(shape[1]):
        for i in range(shape[0]):
            if new_coords_offsets == [] and obj_mask[i, j] == 1:
                new_coords_offsets.append((0,0))
                start_coord = (i, j)
            elif obj_mask[i, j] == 1:
                offset = (i-start_coord[0], j-start_coord[1])
                new_coords_offsets.append(offset)
    return new_coords_offsets

def object_rotation(obj:GridObject, transf_type:str):
    if transf_type == "rot90":
        new_mask = np.rot90(obj.obj_mask, k=-1)
        new_obj_structure = np.rot90(obj.obj_structure, k=-1)
    elif transf_type == "flipud":
        new_mask = np.flipud(obj.obj_mask)
        new_obj_structure = np.flipud(obj.obj_structure)
    elif transf_type == "fliplr":
        new_mask = np.fliplr(obj.obj_mask)
        new_obj_structure = np.fliplr(obj.obj_structure)    
    new_coords_offsets = define_coords_offsets(new_mask)
    new_coords, coords_mapping = define_coords(obj, new_obj_structure)
    return (new_mask, new_obj_structure, new_coords_offsets, new_coords, coords_mapping)

def symmetry_transformation(grid:np.array, obj1:GridObject, font_color:int, transf_type:str):
    """Symmetry transform of object on grid with corresponding grid changes."""
    coords = obj1.coords
    new_mask, new_obj_structure, new_coords_offsets, new_coords, coords_mapping = object_rotation(obj1, transf_type)
    new_coords_on_grid = []
    old_center = obj1.center
    # Store original colors
    colors = {(new_coords[0], new_coords[1]): grid[(old_coords[0], old_coords[1])] 
              for new_coords, old_coords in coords_mapping.items()}
    # Clear the original object
    for x, y in coords:
        grid[x, y] = font_color
    for x, y in new_coords:
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            new_coords_on_grid.append((x, y))
            grid[x, y] = colors[(x, y)]
    obj1.reinit_obj(new_coords, grid)
    obj1.center = old_center # keep center when rotating
    return grid

def shift_object(grid: np.array, obj1: GridObject, direction: str, font_color:int):
    """Shift each cell of object 1 cell in given direction."""
    # Define the 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = {
        'N': (-1, 0),    # North
        'NE': (-1, 1),   # Northeast
        'E': (0, 1),     # East
        'SE': (1, 1),    # Southeast
        'S': (1, 0),     # South
        'SW': (1, -1),   # Southwest
        'W': (0, -1),    # West
        'NW': (-1, -1)   # Northwest
    }
    
    # Get direction vector
    dx, dy = directions[direction]
    
    # Create new coordinates list for the shifted object
    new_coords = []
    obj_coords = copy(obj1.coords)
    colors = {(x, y): grid[x, y] for x, y in obj_coords}
    
    # Check if shift is possible (won't go out of bounds)
    valid_shift = True
    for x, y in obj_coords:
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]):
            valid_shift = False
            break
    
    if valid_shift:
        # Clear original positions
        for x, y in obj_coords:
            grid[x, y] = font_color
        
        # Place at new positions
        for x, y in obj_coords:
            new_x, new_y = x + dx, y + dy
            new_coords.append((new_x, new_y))
            grid[new_x, new_y] = colors[(x, y)]
        
        # Update object coordinates
        obj1.reinit_obj(new_coords, grid)
    
    return grid

def color_inner_part(grid: np.array, obj1: GridObject, color: float):
    """Change color of Object inner_part property."""
    # Create a new grid to modify
    new_grid = grid.copy()
    # If object has inner_part property, color it
    if hasattr(obj1, 'inner_part') and obj1.inner_part:
        inner_part_coords = obj1.inner_part
        for x, y in inner_part_coords:
            new_grid[x, y] = color
        # Update object color information
        add_color_to_object(obj1, color)
    return new_grid

def center_merge(grid: np.array, obj1: GridObject, obj2: GridObject, font_color:int):
    """Object 2 to be placed in center of Object_1."""
    # Get the center coordinates of object_1
    center_x, center_y = obj1.center
    
    # Get the center coordinates of object_2
    obj2_center_x, obj2_center_y = obj2.center
    
    # Calculate the shift needed to place object_2's center at object_1's center
    shift_x = round(center_x - obj2_center_x)
    shift_y = round(center_y - obj2_center_y)
    
    # Create new coordinates list for the shifted object_2
    new_coords = []
    obj2_coords = copy(obj2.coords)
    colors = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions of object_2
    for x, y in obj2_coords:
        grid[x, y] = font_color
    
    # Place object_2 with its center at object_1's center
    for x, y in obj2_coords:
        new_x, new_y = x + shift_x, y + shift_y
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            new_coords.append((new_x, new_y))
            grid[new_x, new_y] = colors[(x, y)]
    # Update object_2 coordinates
    obj2.reinit_obj(new_coords, grid)
    
    return grid

def color_merge(grid: np.array, obj1: GridObject, obj2: GridObject, font_color:int):
    """If Object 1 has cells with the same color as in Object 2 - Object 2 to be placed in such cell closest to center."""
    # Find colors in object_2
    obj2_colors = set(obj2.color_numbers)
    
    # Find cells in object_1 with matching colors
    matching_cells = []
    for x, y in obj1.coords:
        if grid[x, y] in obj2_colors:
            matching_cells.append((x, y))
    
    if not matching_cells:
        # No matching colors found, return unchanged grid
        return grid
    
    # Find the matching cell closest to object_1's center
    obj1_center_x, obj1_center_y = obj1.center
    obj2_center_x, obj2_center_y = obj2.center
    
    closest_cell = min(matching_cells, 
                      key=lambda cell: abs(cell[0] - obj1_center_x) + abs(cell[1] - obj1_center_y))
    
    # Calculate shift to place object_2 with its center at the chosen cell
    shift_x = round(closest_cell[0] - obj2_center_x)
    shift_y = round(closest_cell[1] - obj2_center_y)
    
    # Create new coordinates list for the shifted object_2
    new_coords = []
    obj2_coords = copy(obj2.coords)
    colors = {(x, y): grid[x, y] for x, y in obj2_coords}
    
    # Clear original positions of object_2
    for x, y in obj2_coords:
        grid[x, y] = font_color
    
    # Place object_2 centered at the matching cell
    for x, y in obj2_coords:
        new_x, new_y = x + shift_x, y + shift_y
        new_coords.append((new_x, new_y))
        grid[new_x, new_y] = colors[(x, y)]
    
    # Update object_2 coordinates
    obj2.reinit_obj(new_coords, grid)
    
    return grid