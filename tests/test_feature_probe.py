"""The feature probe: which cell features let a small tree carry a coordinate task's rule from the training pairs to
the test, with no RL in between."""
import numpy as np

from scripts import feature_probe as fp


def test_a_colour_is_split_by_equality_where_a_threshold_would_need_two_cuts():
    x = np.array([[1, 5], [2, 5], [3, 5], [2, 5]])          # the second feature says nothing
    y = np.array([0, 1, 0, 1])
    tree, fits = fp.fitted_tree(x, y, [True, False], max_depth=1)
    assert fits and tree[1:4] == (0, 2, True) and tree[4] == ("leaf", 1) and tree[5] == ("leaf", 0)
    assert (fp.predict(tree, x) == y).all()
    assert not fp.fitted_tree(x, y, [False, False], max_depth=1)[1]
    assert fp.fitted_tree(x, y, [True, False], max_depth=0) == (("leaf", 0), False)


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
