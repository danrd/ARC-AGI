"""Why a rule is missing or wrong."""
import numpy as np

from symbolic import rule_diagnosis as rd


def test_cells_that_follow_from_their_features_have_no_conflict():
    x = np.array([[0, 1], [0, 1], [1, 0]])
    assert rd.conflicts(x, np.array([5, 5, 7])) == 0.0


def test_the_cells_of_a_feature_row_with_two_targets_are_the_conflict():
    x = np.array([[0, 1], [0, 1], [1, 0], [1, 1]])
    assert rd.conflicts(x, np.array([5, 6, 7, 7])) == 0.5
    assert rd.conflicts(np.zeros((0, 2), int), np.zeros(0, int)) == 0.0


def test_a_cell_changed_that_stays_a_cell_left_that_changes_and_a_wrong_colour_are_told_apart():
    start = np.array([[0, 0, 0, 0]])
    truth = np.array([[0, 5, 5, 5]])
    predicted = np.array([[3, 0, 6, 5]])
    assert rd.error_kinds(predicted, truth, start) == {"false_change": 1, "missed": 1, "wrong_colour": 1}


def test_another_shape_cannot_be_compared():
    assert rd.error_kinds(np.zeros((1, 2)), np.zeros((2, 2)), np.zeros((2, 2))) == {}
    assert rd.kind_of({}) == "shape"


def test_the_kind_of_a_wrong_answer_is_the_one_error_it_has_or_mixed():
    assert rd.kind_of({"false_change": 3, "missed": 0, "wrong_colour": 0}) == "false_change"
    assert rd.kind_of({"false_change": 3, "missed": 1, "wrong_colour": 0}) == "mixed"
    assert rd.kind_of({"false_change": 0, "missed": 0, "wrong_colour": 0}) == "equal"
