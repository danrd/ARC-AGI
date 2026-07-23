import numpy as np
from typing import List, Tuple, Dict
from symbolic.objects_analysis import GridObject
from data.configs.env_configs import COLORS_MAPPING
from rl.ARC_transformators import (
symmetry_transformation, upscale, get_outer_contour, inverse_obj_color, edge_gravity, emission, emission_with_collision, color_inner_holes, color_outer_holes,
shift_object, color_inner_part, gravity, x_alignment, y_alignment, contour_connection, find_shortest_distance, find_shortest_path, filter_paths,
find_path_through_background, perform_merge, objects_swap, center_merge, color_merge
)
class World:
    def __init__(self, objects, actions_dict, font_color=0, ):
        self.objects = []
        self.font_color = font_color
        self.actions_dict = actions_dict
        self.COLORS_MAPPING = {0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
                               5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown', 10: 'white'
}
        self.inverse_colors_mapping =  {v:k for k, v in self.COLORS_MAPPING.items()}
        self.paded_cells = set()

    def parse_action(self, action):
        """
        Parse actions from MultiDiscrete action space:
        action[0]: Action type
        action[1]: index of the first object in self.objects
        action[2]: index of the second object in self.objects
        if action[1] == action[2] - transform the object
        """
        action_type = action[0]
        add = -1
        if self.actions_dict[action_type].split("_")[0] in self.inverse_colors_mapping.keys():
            color = self.actions_dict[action_type].split("_")[0]
            transform = self.actions_dict[action_type][len(color)+1:]
            add = self.inverse_colors_mapping[color]
        else:
           transform = self.actions_dict[action_type]
        return add, transform

    def apply_transform(self, add, transform, obj1, obj2, grid, objects, cell2obj):
        """Apply the specified transformation to the grid."""
        if transform is None:
            return grid, False  # No change
        new_grid = grid.copy()

        # 1 OBJECT
        if obj1.label == obj2.label:
            if transform == "rotate90":
                new_grid = symmetry_transformation(new_grid, obj1, self.font_color, "rot90")

            elif transform == "fliplr":
                """Flip left-right the object."""
                new_grid = symmetry_transformation(new_grid, obj1, self.font_color, "fliplr")

            elif transform == "flipud":
                """Flip up-down the object."""
                new_grid = symmetry_transformation(new_grid, obj1, self.font_color, "flipud")

            elif transform == "recolor":
                """Recolor object with the specified color."""
                coords = obj1.coords
                for x, y in coords:
                    new_grid[x, y] = add
                obj1.color_numbers = [add]
                obj1.colors = [COLORS_MAPPING[color] for color in obj1.color_numbers]

            elif transform == "upscale4":
                """Upscale object by 4x: each cell becomes a 2x2 square."""
                new_grid = upscale(new_grid, obj1, self.font_color)

            elif transform == "outer_contour":
                """Create a colored contour over the object."""
                contour = get_outer_contour(new_grid, obj1.coords, self.font_color)
                for x, y  in contour:
                    new_grid[x, y] = add
                new_coords = list(obj1.coords) + contour
                obj1.reinit_obj(new_coords, new_grid)

            elif transform == "color_inversion":
                """Inverse colors of an object."""
                new_grid = inverse_obj_color(new_grid, obj1, self.font_color)

            elif transform == "edge_gravity":
                """Object 1 is moved towards the closest edge or optionally towards bottom edge."""
                new_grid = edge_gravity(new_grid, obj1, self.font_color, bottom_only=False)

            elif transform == "edge_gravity_bottom":
                """Object 1 is moved towards bottom edge"""
                new_grid = edge_gravity(new_grid, obj1, self.font_color, bottom_only=True)

            elif "emission" in transform and "emission_with" not in transform:
                """Emit lines with specified color for 8 directions from Object 1."""
                direction = transform.split("emission_")[1]
                new_grid = emission(new_grid, obj1, add, direction)

            elif "emission_with_turn_left_collision" in transform:
                """Emit lines with specified color for 8 directions from Object 1 with collision handling."""
                direction = transform.split("emission_with_turn_left_collision_")[1]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction, "turn_left")

            elif "emission_with_turn_right_collision" in transform:
                """Emit lines with specified color for 8 directions from Object 1 with turn collision handling."""
                direction = transform.split("emission_with_turn_right_collision_")[1]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction, "turn_right")

            elif "object_recolor" in transform:
                """Emit lines with specified color for 8 directions from Object 1 with object recolor collision handling."""
                direction = transform.split("object_recolor_")[1]
                collision_color_name = transform[transform.find("with_") + 5 : transform.find("_object_recolor")]
                collision_color = self.inverse_colors_mapping[collision_color_name]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction,
                            "object_recolor", collision_color, cell2obj, objects)

            elif "recolor_collision" in transform:
                """Emit lines with specified color for 8 directions from Object 1 with recolor collision handling."""
                direction = transform.split("recolor_collision_")[1]
                collision_color_name = transform[transform.find("with_") + 5: transform.find("_recolor_collision")]
                collision_color = self.inverse_colors_mapping[collision_color_name]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction,
                            "recolor", collision_color)

            elif "contour_collision" in transform:
                """Emit lines with specified color for 8 directions from Object 1 with contour collision handling."""
                direction = transform.split("contour_collision_")[1]
                collision_color_name = transform[transform.find("with_") + 5: transform.find("_contour_collision")]
                collision_color = self.inverse_colors_mapping[collision_color_name]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction,
                            "contour", collision_color)

            elif "emission_with_collision_stop" in transform:
                """Emit lines with specified color for given direction from Object 1 with stop collision handling."""
                direction = transform.split("emission_with_collision_stop_")[1]
                new_grid = emission_with_collision(new_grid, obj1, add, self.font_color, direction, "stop")

            elif transform == "color_inner_holes":
                """Change color of inner holes of Object 1."""
                new_grid = color_inner_holes(new_grid, obj1, add)

            elif transform == "color_outer_holes":
                """Change color of outer holes of Object 1."""
                new_grid = color_outer_holes(new_grid, obj1, add)

            elif transform == "shift":
                direction = transform.split("shift_")[1]
                new_grid = shift_object(new_grid, obj1, direction, self.font_color)

            elif transform == "color_inner_part":
                new_grid = color_inner_part(new_grid, obj1, add)

            # FOR FURTHER IMPLEMENTATION
            elif transform == "copy":
                """Copy the selected object to clipboard."""
                return new_grid

            elif transform == "copy_input":
                """Copy from input grid if input pattern is available."""
                return new_grid

            elif transform == "paste":
                """Paste the clipboard contents at object_1's position."""
                return new_grid

            elif transform == "cut":
                """Copy then clear the original object."""
                coords = obj1.coords
                return new_grid

        elif obj1.label != obj2.label:
            # 2 OBJECTS
            if transform == "gravity":
                """Object 2 is moved to connect with Object 1 based on the shortest distance."""
                new_grid = gravity(new_grid, obj1, obj2, self.font_color)

            elif transform == "x_alignment":
                """Align Object 2 with Object 1 along x axis."""
                new_grid = x_alignment(new_grid, obj1, obj2, self.font_color)

            elif transform == "y_alignment":
                """Align Object 2 with Object 1 along y axis."""
                new_grid = y_alignment(new_grid, obj1, obj2, self.font_color)

            elif "contour_connection" in transform and obj1.shape == 'cell' and obj2.shape == 'cell':
                """If Object 1 and Object 2 are 'cell' type, connect them with contour
                and color it with specified color to create rectangle."""
                rectangle_color_name = transform.split("contour_connection_")[1]
                rectangle_color = self.inverse_colors_mapping[rectangle_color_name]
                new_grid = contour_connection(new_grid, obj1, obj2, add, rectangle_color)

            elif transform == "shortest_path" and "background" not in transform:
                # Connect two objects on grid by drawing the shortest path with the specified color
                closest_pairs = find_shortest_distance(obj1.coords, obj2.coords)
                if closest_pairs != []:
                    paths = []
                    for closest_pair in closest_pairs:
                        start, end = closest_pair
                        path = find_shortest_path(new_grid, start, end)
                        if path != []:
                            paths.append(path)
                    filtered_paths =  filter_paths(paths, obj1, obj2, preference='left')
                    for path in filtered_paths:
                        for x, y in path[1:-1]:
                            new_grid[x, y] = add

            elif "shortest_path_left" in transform:
                """Connect two objects via background cells with the shortest path with specified color."""
                closest_pairs = find_shortest_distance(obj1.coords, obj2.coords)
                if closest_pairs != []:
                    paths = []
                    for closest_pair in closest_pairs:
                        start, end = closest_pair
                        path = find_path_through_background(new_grid, start, end, self.font_color)
                        if path != []:
                            paths.append(path)
                    filtered_paths =  filter_paths(paths, obj1, obj2, preference='left')
                    for path in filtered_paths:
                        for x, y in path[1:-1]:
                            new_grid[x, y] = add

            elif "shortest_path_right" in transform:
                """Connect two objects via background cells with the shortest path with specified color."""
                closest_pairs = find_shortest_distance(obj1.coords, obj2.coords)
                if closest_pairs != []:
                    paths = []
                    for closest_pair in closest_pairs:
                        start, end = closest_pair
                        path = find_path_through_background(new_grid, start, end, self.font_color)
                        if path != []:
                            paths.append(path)
                    filtered_paths =  filter_paths(paths, obj1, obj2, preference='right')
                    for path in filtered_paths:
                        for x, y in path[1:-1]:
                            new_grid[x, y] = add

            elif transform == "merge":
                """Merge object_2 into object_1's position by calculating relative shift."""
                new_grid = perform_merge(new_grid, obj1, obj2, objects, self.font_color)

            elif transform == "swap":
                """Exchange of objects positions: object 1 will be placed on object 2 position and visa versa."""
                new_grid = objects_swap(new_grid, obj1, obj2, self.font_color)

            elif transform == "center_merge":
                """Exchange of objects positions: object 1 will be placed on object 2 position and visa versa."""
                new_grid = center_merge(new_grid, obj1, obj2, self.font_color)

            elif transform == "color_merge":
                """If Object 1 has cells with the same color as in Object 2 - Object 2 to be placed in such cell closest to center."""
                new_grid = color_merge(new_grid, obj1, obj2, self.font_color)

        return new_grid

    def step(self, add:int, transform:str, obj1:GridObject, obj2:GridObject,
             grid:np.array, objects:List[GridObject], cell2obj:Dict[Tuple[int, int], int]):
        """Execute one step of the world dynamics"""
        # Apply grid transformations
        if grid is not None and transform is not None:
            new_grid = self.apply_transform(add, transform, obj1, obj2, grid, objects, cell2obj)
            return new_grid
        return grid
