"""What a cell of a grid is, beyond its colour and position: the features the feature probe measured and the
coordinate policy reads."""
import numpy as np

from symbolic import cell_features as cf


def test_the_background_is_the_commonest_colour_and_black_on_a_tie():
    assert cf.background(np.array([[3, 3, 0], [3, 1, 0]])) == 3
    assert cf.background(np.array([[3, 0], [0, 3]])) == 0


def test_the_nearest_foreground_is_found_along_each_direction_with_its_distance():
    grid = np.array([[0, 0, 5],
                     [2, 0, 0],
                     [0, 0, 0]])
    along = cf.nearest_along(grid, 0)
    assert along["nearest_right"][0, 0] == 5 and along["distance_right"][0, 0] == 2
    assert along["nearest_left"][1, 2] == 2 and along["distance_left"][1, 2] == 2
    assert along["nearest_up"][2, 2] == 5 and along["distance_up"][2, 2] == 2
    assert along["nearest_down"][0, 1] == cf.OUTSIDE and along["distance_down"][0, 1] == 0
    assert along["nearest_up_right"][1, 1] == 5 and along["nearest_down_left"][0, 1] == 2


def test_an_object_knows_its_size_its_box_and_where_a_cell_sits_in_it():
    grid = np.array([[0, 0, 0, 0],
                     [0, 4, 4, 0],
                     [0, 4, 0, 0],
                     [0, 0, 0, 0]])
    objects = cf.object_features(grid, 0)
    assert objects["object_size"][1, 1] == 3 and objects["box_height"][1, 2] == 2 and objects["box_width"][2, 1] == 2
    assert objects["from_left"][1, 2] == 1 and objects["from_bottom"][1, 1] == 1
    assert objects["on_box_border"][1, 1] == 1
    assert objects["object_touches_edge"][1, 1] == 0 and objects["object_touches_edge"][0, 0] == 1


def test_a_cell_between_two_of_one_colour_is_marked_and_one_beside_a_single_cell_is_not():
    grid = np.array([[3, 0, 0, 3, 0],
                     [0, 0, 6, 0, 0],
                     [0, 0, 0, 0, 0]])
    features, _ = cf.cell_features(grid, "structure")
    assert features["between_left_right"][0, 1] == 1 and features["between_left_right"][0, 4] == 0
    assert features["between_left_right"][1, 0] == 0 and features["between_left_right"][1, 3] == 0
    assert not features["between_left_right"][2].any()         # nothing on either side is not "between"


def test_the_rich_set_counts_neighbours_by_colour_and_reads_mirrors_periods_and_blobs():
    grid = np.array([[2, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 5, 5],
                     [0, 0, 5, 7]])
    features, colours = cf.cell_features(grid, "rich")
    assert features["diagonal_2"][1, 1] == 1 and features["orthogonal_2"][1, 1] == 0
    assert features["orthogonal_5"][2, 1] == 1 and features["orthogonal_foreground"][3, 2] == 2
    assert features["mirror_both"][0, 0] == 7 and features["mirror_left_right"][0, 3] == 2
    assert features["mirror_up_down"][3, 0] == 2
    square = cf.cell_features(np.array([[1, 2], [3, 4]]), "rich")[0]
    assert square["mirror_diagonal"].tolist() == [[1, 3], [2, 4]]
    assert features["col_mod_3"][0, 3] == 0 and features["col_from_right_mod_2"][0, 2] == 1
    assert features["blob_size"][2, 2] == 4 and features["blob_colours"][3, 3] == 2
    assert features["blob_majority"][3, 3] == 5 and features["blob_minority"][2, 2] == 7
    assert features["blob_majority"][1, 1] == cf.OUTSIDE
    assert features["grid_majority"][0, 0] == 5 and features["grid_minority"][0, 0] in (2, 7)
    assert {"mirror_both", "blob_minority", "grid_majority"} <= colours and "blob_size" not in colours
    wide = cf.cell_features(np.zeros((2, 3), dtype=int), "rich")[0]
    assert (wide["mirror_diagonal"] == cf.OUTSIDE).all()


def test_a_cell_closed_in_by_markers_on_four_sides_is_known_and_by_which_colours():
    grid = np.array([[0, 0, 3, 0, 0],
                     [3, 0, 0, 0, 3],
                     [0, 0, 3, 0, 0]])
    features, _ = cf.cell_features(grid, "rich")
    assert features["closed_in"][1, 2] == 1 and features["closed_in_by_one_colour"][1, 2] == 1
    assert features["closed_in"][0, 0] == 0
    grid[2, 2] = 4
    features, _ = cf.cell_features(grid, "rich")
    assert features["closed_in"][1, 2] == 1 and features["closed_in_by_one_colour"][1, 2] == 0


def test_a_blob_holds_cells_that_touch_only_at_a_corner():
    grid = np.array([[3, 0, 0],
                     [0, 4, 0],
                     [0, 0, 0]])
    features, _ = cf.cell_features(grid, "rich")
    assert features["blob_size"][0, 0] == 2 and features["blob_colours"][1, 1] == 2



def test_a_grid_is_encoded_cell_by_cell_in_schema_order_padded_with_nothing():
    grid = np.array([[0, 2, 0],
                     [0, 0, 0]])
    names, colours = cf.schema()
    assert "colour" in colours and "row" not in names and "col" not in names     # nothing absolute about position
    out = cf.encoded(grid, shape=(4, 5))
    assert out.shape == (4, 5, len(names)) and out.dtype == np.uint8
    at = {name: i for i, name in enumerate(names)}
    assert out[0, 1, at["colour"]] == 3 and out[0, 0, at["colour"]] == 1            # a colour is one more than itself
    assert out[0, 0, at["next_up"]] == 0                                            # off the grid is nothing
    assert out[0, 0, at["nearest_right"]] == 3 and out[0, 0, at["distance_right"]] == 1
    assert not out[2:].any() and not out[:, 3:].any()                               # past the grid, all zeros
    big = np.zeros((20, 20), dtype=int)
    assert cf.encoded(big)[0, 0, at["object_size"]] == 255                           # a count stops at 255


def test_the_core_set_is_the_plainest_part_of_the_rich_one_and_is_read_from_its_own_key():
    rich, rich_colours = cf.schema("rich")
    core, core_colours = cf.schema("core")
    assert len(core) == 25 and set(core) <= set(rich) and set(core_colours) == set(core) & set(rich_colours)
    assert "mirror_both" in core and "col_mod_3" not in core and "blob_minority" not in core
    grid = np.array([[0, 3, 0], [0, 0, 5]])
    full, small = cf.encoded(grid, feature_set="rich"), cf.encoded(grid, feature_set="core")
    assert small.shape == (2, 3, 25)
    assert (small[..., core.index("nearest_right")] == full[..., rich.index("nearest_right")]).all()
    assert cf.observed_set("cell_features") == "rich" and cf.observed_set("cell_features_core") == "core"
    assert cf.feature_key(["grid", "cell_features_core"]) == "cell_features_core"
    assert cf.feature_key(["grid", "delta_input"]) is None
