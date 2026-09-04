#: Declared, dispatched, and doing nothing: World.apply_transform returns the
#: grid untouched for each of these (verified over 43 applications apiece on
#: real tasks). They are kept in ACTION_TYPES so the dispatch-coverage tests
#: still account for them, and excluded from the vocabulary a search is given
#: - four of 89 action slots that the search would otherwise keep trying.
UNIMPLEMENTED_ACTIONS = {"copy", "copy_input", "paste", "cut"}

ACTION_TYPES = {
                "color": ["recolor", "color_inversion", "color_inner_part", "color_inner_holes", "color_outer_holes", "dense_outer_contour"],
                "modification": ["upscale_4"],
                "shift": ["shift_object", "swap"],
                "gravity": ["gravity", "edge_gravity", "edge_gravity_bottom"],
                "emission": ["emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision", "emission_with_recolor_collision",
                             "emission_with_contour_collision", "emission_with_collision_stop", "emission_with_object_recolor"],
                "merge": ["merge", "center_merge", "color_merge"],
                "rotation": ["rotate90", "fliplr", "flipud"],
                "symmetry": ["symmetry_reflection", "symmetric_restoration"],
                "exchange": ["color_swap", "shape_swap", "color_copy", "shape_copy"],
                "edit": ["copy", "copy_input", "paste", "cut"],
                "alignment": ["x_alignment", "y_alignment"],
                "connection": ["shortest_path", "background_shortest_path_left", "background_shortest_path_right", "contour_connection"],
}
TWO_OBJECTS_ACTION_TYPES = ["swap", "merge", "center_merge", "color_merge", "x_alignment", "y_alignment", "shortest_path", "background_shortest_path_left", "background_shortest_path_right", "contour_connection",
                            "color_swap", "shape_swap", "color_copy", "shape_copy", "gravity"]
COLORS_MAPPING = {0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
                  5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown', 10: 'white'
}
ALL_COLORS = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
ALL_DIRECTIONS = ["N", "W", "E", "S", "NW", "NE", "SW", "SE"]
MAIN_DIRECTIONS = ["N", "W", "E", "S"]
COLOR_DEPENDENT_ACTIONS = ["recolor", "shortest_path", "background_shortest_path_left", "background_shortest_path_right",
                           "outer_contour", "contour_connection",
                           "emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision",
                           "emission_with_recolor_collision", "emission_with_contour_collision", "color_inner_holes",
                           "color_outer_holes", "color_inner_part", "emission_with_collision_stop", "emission_with_object_recolor",
                           "dense_outer_contour"
                          ]
DOUBLE_COLOR_DEPENDENT_ACTIONS = ["contour_connection", "emission_with_object_recolor", "emission_with_recolor_collision", "emission_with_contour_collision"]
DIRECTION_DEPENDENT_ACTIONS = ["emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision", "emission_with_object_recolor",
                               "emission_with_recolor_collision", "emission_with_contour_collision", "emission_with_collision_stop",
                               "shift_object"
                              ]
AGENT2ACTIONS = {
    'highlighter': ["submit", "recolor", "color_inversion"],
    'modifier': ["submit", "color_inner_part", "color_inner_holes", "color_outer_holes",
                 "recolor", "outer_contour", "color_inversion", "emission"],
    'connector': ["submit", "recolor", "shift_object", "outer_contour", "color_inner_part", "shortest_path",
                  "background_shortest_path_left", "background_shortest_path_right", "contour_connection",
                  "emission",],
    'shifter': ["submit", "recolor", "gravity", "edge_gravity", "edge_gravity_bottom", "x_alignment", "y_alignment",
                "shift_object", "swap", "merge", "center_merge", "color_merge"],
    'connector_extended': ["submit", "recolor", "shift_object", "outer_contour", "color_inner_part", "shortest_path",
              "background_shortest_path_left", "background_shortest_path_right", "contour_connection",
              "emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision",
              "emission_with_recolor_collision", "emission_with_contour_collision",
              "emission_with_collision_stop", "emission_with_object_recolor",],
}


#: What each transform does, in words, for a reader who has never seen the
#: vocabulary. Not derived from the names: several of them mislead. Read
#: against the implementations in rl/arc_world.py's dispatch and
#: rl/arc_transformators.py, and that is the only thing that makes them
#: true - `color_outer_holes` fills the concave notches of an outline
#: rather than any hole outside it, and `dense_outer_contour` paints the
#: unoccupied edge of the object's bounding rectangle rather than a
#: contour of the object.
#:
#: `{colour}` is the colour the action paints with, `{second}` a second
#: colour where a transform takes one, `{direction}` a compass direction,
#: and `{other}` the second object. rl.search_hints.describe_action fills
#: whichever the name carries.
TRANSFORM_DESCRIPTIONS = {
    "submit": "hand in the grid as it stands",

    # One object, painting
    "recolor": "repaint every cell of the shape to {colour}",
    "color_inner_holes": "fill the enclosed empty regions inside the shape with {colour}",
    "color_outer_holes": "fill the concave notches along the shape's outline with {colour}",
    "color_inner_part": "paint the shape's interior - its cells other than its border - {colour}",
    "outer_contour": "draw a border of {colour} around the shape",
    "dense_outer_contour": "paint {colour} on the cells of the shape's bounding "
                           "rectangle edge that the shape does not already occupy",
    "color_inversion": "swap the shape with its holes: the cells it occupies become "
                       "background and the empty cells around and inside it take its colour",

    # One object, moving or reshaping
    "rotate90": "rotate the shape a quarter turn in place",
    "fliplr": "mirror the shape left to right in place",
    "flipud": "mirror the shape top to bottom in place",
    "shift_object": "move the shape one cell {direction}",
    "edge_gravity": "move the shape until it meets the nearest edge of the grid",
    "edge_gravity_bottom": "move the shape down until it meets the bottom edge",
    "upscale_4": "double the shape in both directions, each cell becoming a 2x2 block",
    "symmetric_restoration": "mirror the shape left to right, then mirror the result "
                             "top to bottom, so a quarter of a symmetric figure "
                             "becomes the whole of it",
    "symmetry_reflection": "add a mirrored copy of the shape on whichever of its four "
                           "sides leaves the result compact and overlapping nothing",

    # One object, emitting
    "emission": "shoot a line of {colour} out of the shape towards {direction}",
    "emission_with_collision_stop": "shoot a line of {colour} out of the shape towards "
                                    "{direction}, stopping where it meets something",
    "emission_with_turn_left_collision": "shoot a line of {colour} out of the shape "
                                         "towards {direction}, turning left where it "
                                         "meets something",
    "emission_with_turn_right_collision": "shoot a line of {colour} out of the shape "
                                          "towards {direction}, turning right where it "
                                          "meets something",
    "emission_with_recolor_collision": "shoot a line of {colour} out of the shape "
                                       "towards {direction}, continuing in {second} "
                                       "after it meets something",
    "emission_with_contour_collision": "shoot a line of {colour} out of the shape "
                                       "towards {direction}, drawing a {second} border "
                                       "around whatever it meets",
    "emission_with_object_recolor": "shoot a line of {colour} out of the shape towards "
                                    "{direction}, repainting to {second} whatever it meets",

    # Two objects
    "shortest_path": "join the two shapes with a line of {colour} along the shortest "
                     "route between them",
    "background_shortest_path_left": "join the two shapes with a line of {colour} along "
                                     "the shortest route that stays on background cells, "
                                     "keeping left where the route forks",
    "background_shortest_path_right": "join the two shapes with a line of {colour} along "
                                      "the shortest route that stays on background cells, "
                                      "keeping right where the route forks",
    "contour_connection": "draw a rectangle of {colour} with the two cells at opposite "
                          "corners, filling it with {second}",
    "gravity": "move {other} until it touches the shape, by the shortest distance",
    "merge": "move {other} onto the shape's position",
    "center_merge": "move {other} to the centre of the shape",
    "color_merge": "move {other} onto the cell of the shape that shares its colour, "
                   "nearest the shape's centre",
    "swap": "exchange the positions of the shape and {other}",
    "color_swap": "exchange the colours of the shape and {other}",
    "shape_swap": "exchange the outlines of the shape and {other}, each keeping its "
                  "own position and colour",
    "color_copy": "repaint the shape in the colour of {other}",
    "shape_copy": "give the shape the outline of {other}, keeping its own position "
                  "and colour",
    "x_alignment": "move {other} to line up with the shape along the x axis",
    "y_alignment": "move {other} to line up with the shape along the y axis",
}
