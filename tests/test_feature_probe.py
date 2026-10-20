"""The feature probe: which cell features let a small tree carry a coordinate task's rule from the training pairs to
the test, with no RL in between."""
import numpy as np

from scripts import feature_probe as fp


def test_the_background_is_the_commonest_colour_and_black_on_a_tie():
    assert fp.background(np.array([[3, 3, 0], [3, 1, 0]])) == 3
    assert fp.background(np.array([[3, 0], [0, 3]])) == 0


def test_the_nearest_foreground_is_found_along_each_direction_with_its_distance():
    grid = np.array([[0, 0, 5],
                     [2, 0, 0],
                     [0, 0, 0]])
    along = fp.nearest_along(grid, 0)
    assert along["nearest_right"][0, 0] == 5 and along["distance_right"][0, 0] == 2
    assert along["nearest_left"][1, 2] == 2 and along["distance_left"][1, 2] == 2
    assert along["nearest_up"][2, 2] == 5 and along["distance_up"][2, 2] == 2
    assert along["nearest_down"][0, 1] == fp.OUTSIDE and along["distance_down"][0, 1] == 0
    assert along["nearest_up_right"][1, 1] == 5 and along["nearest_down_left"][0, 1] == 2


def test_an_object_knows_its_size_its_box_and_where_a_cell_sits_in_it():
    grid = np.array([[0, 0, 0, 0],
                     [0, 4, 4, 0],
                     [0, 4, 0, 0],
                     [0, 0, 0, 0]])
    objects = fp.object_features(grid, 0)
    assert objects["object_size"][1, 1] == 3 and objects["box_height"][1, 2] == 2 and objects["box_width"][2, 1] == 2
    assert objects["from_left"][1, 2] == 1 and objects["from_bottom"][1, 1] == 1
    assert objects["on_box_border"][1, 1] == 1
    assert objects["object_touches_edge"][1, 1] == 0 and objects["object_touches_edge"][0, 0] == 1


def test_a_colour_is_split_by_equality_where_a_threshold_would_need_two_cuts():
    x = np.array([[1, 5], [2, 5], [3, 5], [2, 5]])          # the second feature says nothing
    y = np.array([0, 1, 0, 1])
    tree, fits = fp.fitted_tree(x, y, [True, False], max_depth=1)
    assert fits and tree[1:4] == (0, 2, True) and tree[4] == ("leaf", 1) and tree[5] == ("leaf", 0)
    assert (fp.predict(tree, x) == y).all()
    assert not fp.fitted_tree(x, y, [False, False], max_depth=1)[1]
    assert fp.fitted_tree(x, y, [True, False], max_depth=0) == (("leaf", 0), False)


def test_a_cell_between_two_of_one_colour_is_marked_and_one_beside_a_single_cell_is_not():
    grid = np.array([[3, 0, 0, 3, 0],
                     [0, 0, 6, 0, 0],
                     [0, 0, 0, 0, 0]])
    features, _ = fp.cell_features(grid, "structure")
    assert features["between_left_right"][0, 1] == 1 and features["between_left_right"][0, 4] == 0
    assert features["between_left_right"][1, 0] == 0 and features["between_left_right"][1, 3] == 0
    assert not features["between_left_right"][2].any()         # nothing on either side is not "between"


def test_the_colour_features_of_a_task_are_split_by_equality():
    def mark(colours):
        grid = np.zeros((3, 4), dtype=int)
        grid[:, 0] = colours
        out = grid.copy()
        out[np.array(colours) == 2, 1] = 2                       # only the middle colour of three spreads
        return {"input": grid.tolist(), "output": out.tolist()}

    train = [mark([1, 2, 3]), mark([3, 1, 2]), mark([2, 3, 1])]
    test = mark([3, 2, 1])
    shallow = fp.probe_task(train, test["input"], test["output"], "neighbours", max_depth=2)
    assert shallow["fits_train"] and shallow["test_solved"]


def test_a_new_colour_is_labelled_by_the_role_that_holds_it_before_the_colour_itself():
    names = ["colour", "nearest_left", "nearest_right"]
    x = np.array([[0, 3, 9], [0, 3, 9], [0, 4, 8], [0, 1, 8]])
    targets = np.array([3, 3, 4, 8])
    labels = fp.new_colour_labels(x, names, {"colour", "nearest_left", "nearest_right"}, targets)
    assert labels.tolist() == ["src:nearest_left"] * 3 + ["src:nearest_right"]
    assert (fp.apply_colours(labels, x, names) == targets).all()
    alone = fp.new_colour_labels(np.array([[0, 1, 1]]), names, {"colour", "nearest_left"}, np.array([5]))
    assert alone.tolist() == ["abs:5"]


def between_task(width, row, left, right, colour):
    """A row of background with two `colour` cells; the output fills the cells between them with 4."""
    grid = np.zeros((5, width), dtype=int)
    grid[row, left] = grid[row, right] = colour
    out = grid.copy()
    out[row, left + 1:right] = 4
    return {"input": grid.tolist(), "output": out.tolist()}


def test_a_rule_about_structure_carries_to_a_test_in_a_new_place_and_one_about_position_does_not():
    train = [between_task(9, 1, 1, 5, 2), between_task(9, 3, 2, 7, 2), between_task(10, 0, 4, 8, 2)]
    test = between_task(10, 4, 0, 3, 2)
    structure = fp.probe_task(train, test["input"], test["output"], "structure")
    position = fp.probe_task(train, test["input"], test["output"], "position")
    assert structure["fits_train"] and structure["test_solved"] and structure["test_changed_cells_right"]
    assert not position["test_solved"]


def test_a_colour_no_training_output_has_comes_through_a_role():
    def recolour(colour, row):
        grid = np.zeros((4, 6), dtype=int)
        grid[row, 0] = colour
        out = grid.copy()
        out[row, 1:] = colour                          # the row is painted in its marker's colour
        return {"input": grid.tolist(), "output": out.tolist()}

    train = [recolour(3, 0), recolour(5, 2), recolour(3, 3)]
    test = recolour(7, 1)
    result = fp.probe_task(train, test["input"], test["output"], "structure")
    assert result["test_solved"] and result["colour_labels"] == ["src:nearest_left"]


def test_the_rich_set_counts_neighbours_by_colour_and_reads_mirrors_periods_and_blobs():
    grid = np.array([[2, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 5, 5],
                     [0, 0, 5, 7]])
    features, colours = fp.cell_features(grid, "rich")
    assert features["diagonal_2"][1, 1] == 1 and features["orthogonal_2"][1, 1] == 0
    assert features["orthogonal_5"][2, 1] == 1 and features["orthogonal_foreground"][3, 2] == 2
    assert features["mirror_both"][0, 0] == 7 and features["mirror_left_right"][0, 3] == 2
    assert features["mirror_up_down"][3, 0] == 2
    square = fp.cell_features(np.array([[1, 2], [3, 4]]), "rich")[0]
    assert square["mirror_diagonal"].tolist() == [[1, 3], [2, 4]]
    assert features["col_mod_3"][0, 3] == 0 and features["col_from_right_mod_2"][0, 2] == 1
    assert features["blob_size"][2, 2] == 4 and features["blob_colours"][3, 3] == 2
    assert features["blob_majority"][3, 3] == 5 and features["blob_minority"][2, 2] == 7
    assert features["blob_majority"][1, 1] == fp.OUTSIDE
    assert features["grid_majority"][0, 0] == 5 and features["grid_minority"][0, 0] in (2, 7)
    assert {"mirror_both", "blob_minority", "grid_majority"} <= colours and "blob_size" not in colours
    wide = fp.cell_features(np.zeros((2, 3), dtype=int), "rich")[0]
    assert (wide["mirror_diagonal"] == fp.OUTSIDE).all()


def test_a_cell_closed_in_by_markers_on_four_sides_is_known_and_by_which_colours():
    grid = np.array([[0, 0, 3, 0, 0],
                     [3, 0, 0, 0, 3],
                     [0, 0, 3, 0, 0]])
    features, _ = fp.cell_features(grid, "rich")
    assert features["closed_in"][1, 2] == 1 and features["closed_in_by_one_colour"][1, 2] == 1
    assert features["closed_in"][0, 0] == 0
    grid[2, 2] = 4
    features, _ = fp.cell_features(grid, "rich")
    assert features["closed_in"][1, 2] == 1 and features["closed_in_by_one_colour"][1, 2] == 0


def test_a_rule_that_holds_on_every_pair_is_reproduced_on_each_pair_held_out():
    train = [between_task(9, 1, 1, 5, 2), between_task(9, 3, 2, 7, 2), between_task(10, 0, 4, 8, 2)]
    assert fp.held_out_pairs(train, "structure") == (3, 1.0)
    assert fp.held_out_pairs(train, "position")[0] < 3
    assert fp.held_out_pairs(train[:1], "structure") == (0, 0.0)


def test_the_set_chosen_is_the_one_that_does_best_on_held_out_pairs_and_the_simplest_on_a_tie():
    results = {"neighbours": {"held_out_pairs": [0, 0.9]}, "structure": {"held_out_pairs": [2, 0.95]},
               "rich": {"held_out_pairs": [2, 0.95]}, "position": {"held_out_pairs": [3, 1.0]}}
    assert fp.chosen(results) == "structure"
    results["rich"]["held_out_pairs"] = [2, 0.97]
    assert fp.chosen(results) == "rich"
    assert fp.chosen({"neighbours": {"held_out_pairs": [0, 0.5]}}) == "neighbours"


def test_a_pair_held_out_is_predicted_from_the_others_only(monkeypatch):
    pairs = [{"input": [[k]], "output": [[k + 1]]} for k in range(3)]
    seen = []

    def fake_fit(train, feature_set, max_depth=6):
        seen.append([p["input"][0][0] for p in train])
        return train

    def fake_apply(model, grid):                         # a memoriser: right only on a pair it was fitted on
        known = {p["input"][0][0]: p["output"] for p in model}
        return np.array(known.get(grid[0][0], [[-1]])), None

    monkeypatch.setattr(fp, "fit", fake_fit)
    monkeypatch.setattr(fp, "apply", fake_apply)
    assert fp.held_out_pairs(pairs, "structure") == (0, 0.0)
    assert seen == [[1, 2], [0, 2], [0, 1]]


def test_a_blob_holds_cells_that_touch_only_at_a_corner():
    grid = np.array([[3, 0, 0],
                     [0, 4, 0],
                     [0, 0, 0]])
    features, _ = fp.cell_features(grid, "rich")
    assert features["blob_size"][0, 0] == 2 and features["blob_colours"][1, 1] == 2
