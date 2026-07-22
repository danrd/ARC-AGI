action_types = {
                "color": ["recolor", "color_inversion", "color_inner_part", "color_inner_holes", "color_outer_holes"],
                "modification": ["upscale_4"],
                "shift": ["shift_object", "swap"],
                "gravity": ["gravity", "edge_gravity", "edge_gravity_bottom"],
                "emission": ["emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision", "emission_with_recolor_collision", 
                             "emission_with_contour_collision", "emission_with_collision_stop", "emission_with_object_recolor"],
                "merge": ["merge", "center_merge", "color_merge"],
                "rotation": ["rotate90", "fliplr", "flipud"],
                "edit": ["copy", "copy_input", "paste", "cut"],
                "alignment": ["x_alignment", "y_alignment"],
                "connection": ["shortest_path", "background_shortest_path_left", "background_shortest_path_right", "contour_connection"],
}
two_objects_action_types = ["swap", "merge", "center_merge", "color_merge", "x_alignment", "y_alignment", "shortest_path", "background_shortest_path_left", "background_shortest_path_right", "contour_connection"]
colors_mapping = {0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow', 
                  5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown', 10: 'white'
}
all_colors = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
all_directions = ["N", "W", "E", "S", "NW", "NE", "SW", "SE"]
main_directions = ["N", "W", "E", "S"]
color_dependent_actions = ["recolor", "shortest_path", "background_shortest_path_left", "background_shortest_path_right",
                           "outer_contour", "contour_connection",
                           "emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision",
                           "emission_with_recolor_collision", "emission_with_contour_collision", "color_inner_holes",
                           "color_outer_holes", "color_inner_part", "emission_with_collision_stop", "emission_with_object_recolor"                                   
                          ]
double_color_dependent_actions = ["contour_connection", "emission_with_object_recolor", "emission_with_recolor_collision", "emission_with_contour_collision"]
direction_dependent_actions = ["emission", "emission_with_turn_left_collision", "emission_with_turn_right_collision", "emission_with_object_recolor",
                               "emission_with_recolor_collision", "emission_with_contour_collision", "emission_with_collision_stop", "shift"
                              ]
agent2actions = {
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