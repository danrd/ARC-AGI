"""What the examples agree on, and what an answer that disagrees is told."""
import numpy as np

from symbolic.invariants import STRONG, WEAK, Invariants, check, learn, refuses


def g(rows):
    return np.array(rows)


SAME = [(g([[0, 1], [0, 0]]), g([[0, 1], [0, 2]])), (g([[0, 0, 1], [0, 0, 0]]), g([[0, 0, 1], [0, 0, 2]]))]


class TestShape:
    def test_a_shape_all_examples_share_is_expected_of_the_answer(self):
        inv = learn(SAME)
        assert inv.shape_rule == ("same", ())
        assert check(inv, g([[0, 1, 0]]), g([[0, 1, 2]])) == []
        (violation,) = [v for v in check(inv, g([[0, 1, 0]]), g([[0, 1], [2, 0]])) if v.name == "shape"]
        assert violation.severity == STRONG and "2x2" in violation.message and "1x3" in violation.message

    def test_a_fixed_output_a_scale_and_a_shrink_are_told_apart(self):
        fixed = learn([(g([[1, 2, 3]]), g([[1]])), (g([[1, 2], [3, 4]]), g([[1]]))])
        assert fixed.shape_rule == ("fixed", (1, 1))
        scale = learn([(g([[1, 2]]), g([[1, 1, 2, 2]])), (g([[3]]), g([[3, 3]]))])
        assert scale.shape_rule is not None
        up = learn([(g([[1, 2]]), g([[1, 1, 2, 2], [1, 1, 2, 2]])), (g([[3]]), g([[3, 3], [3, 3]]))])
        assert up.shape_rule == ("scale", (2, 2))
        assert [v.name for v in check(up, g([[0, 6, 7]]), g(np.zeros((2, 3), int)))] == ["shape"]
        assert check(up, g([[0, 6, 7]]), g(np.zeros((2, 6), int))) == []
        down = learn([(g(np.zeros((4, 4), int)), g(np.zeros((2, 2), int))), (g(np.zeros((6, 2), int)), g(np.zeros((3, 1), int)))])
        assert down.shape_rule == ("shrink", (2, 2))
        assert check(down, g(np.zeros((8, 8), int)), g(np.zeros((4, 4), int))) == []

    def test_pairs_that_agree_on_no_shape_rule_expect_nothing(self):
        inv = learn([(g([[1, 2]]), g([[1]])), (g([[1, 2, 3]]), g([[1, 2]]))])
        assert inv.shape_rule is None
        assert [v for v in check(inv, g([[1, 2, 3, 4]]), g([[5]])) if v.name == "shape"] == []


class TestColours:
    def test_an_answer_may_bring_in_the_colours_the_examples_bring_in_and_no_other(self):
        inv = learn(SAME)
        assert inv.introduced == frozenset({2})
        assert check(inv, g([[0, 1, 0]]), g([[0, 1, 2]])) == []
        (violation,) = [v for v in check(inv, g([[0, 1, 0]]), g([[0, 1, 7]])) if v.name == "colours"]
        assert violation.severity == STRONG and "[7]" in violation.message and "[2]" in violation.message

    def test_a_colour_the_input_already_has_is_not_new(self):
        inv = learn(SAME)
        assert [v for v in check(inv, g([[0, 7, 0]]), g([[0, 7, 7]])) if v.name == "colours"] == []

    def test_examples_that_introduce_nothing_forbid_anything_new(self):
        inv = learn([(g([[1, 0]]), g([[0, 1]])), (g([[2, 0]]), g([[0, 2]]))])
        assert inv.introduced == frozenset()
        (violation,) = [v for v in check(inv, g([[3, 0]]), g([[0, 4]])) if v.name == "colours"]
        assert "no example brings in a colour that was not already in its input" in violation.message


class TestSymmetry:
    PAIRS = [(g([[1, 0, 0], [0, 0, 0]]), g([[1, 0, 1], [0, 0, 0]])), (g([[2, 0, 0], [0, 2, 0]]), g([[2, 0, 2], [0, 2, 0]]))]

    def test_a_symmetry_every_output_has_and_no_input_has_is_asked_of_the_answer(self):
        inv = learn(self.PAIRS)
        assert "left-right mirror" in inv.symmetries
        assert [v.name for v in check(inv, g([[3, 0, 0]]), g([[3, 0, 3]])) if v.name.startswith("symmetry")] == []
        (violation,) = [v for v in check(inv, g([[3, 0, 0]]), g([[3, 0, 0]])) if v.name.startswith("symmetry")]
        assert violation.severity == STRONG

    def test_a_symmetry_the_inputs_already_have_says_nothing_about_the_work(self):
        mirrored = [(g([[1, 0, 1]]), g([[1, 0, 1]])), (g([[2, 0, 2]]), g([[2, 0, 2]]))]
        assert learn(mirrored).symmetries == ()

    def test_one_example_is_not_enough_to_call_a_symmetry_a_law(self):
        assert learn(self.PAIRS[:1]).symmetries == ()


class TestWeakOnes:
    def test_how_much_is_left_alone_and_which_counts_are_kept_are_weak(self):
        pairs = [(g(np.zeros((4, 4), int)), g(np.eye(4, dtype=int) * 3)), (g(np.zeros((4, 4), int)), g(np.eye(4, dtype=int) * 3))]
        inv = learn(pairs)
        assert inv.unchanged_range == (0.75, 0.75)
        wrong = check(inv, g(np.zeros((4, 4), int)), g(np.ones((4, 4), int) * 3))
        assert {v.name for v in wrong} >= {"unchanged"} and all(v.severity == WEAK for v in wrong if v.name == "unchanged")

    def test_a_count_no_example_changes_is_asked_to_stay(self):
        pairs = [(g([[1, 1, 0]]), g([[0, 1, 1]])), (g([[1, 0, 0, 1]]), g([[0, 1, 1, 0]]))]
        inv = learn(pairs)
        assert 1 in inv.kept_counts
        names = [v.name for v in check(inv, g([[1, 1, 1, 0]]), g([[0, 0, 0, 0]]))]
        assert "count:1" in names

    def test_only_strong_violations_refuse(self):
        weak = [v for v in check(learn(SAME), g([[0, 0, 0, 0, 0, 1]]), g([[2, 2, 2, 2, 2, 1]])) if v.severity == WEAK]
        assert weak and not refuses(weak)
        strong = check(learn(SAME), g([[0, 1, 0]]), g([[0, 1, 7]]))
        assert refuses(strong) and not refuses([])


def test_no_pairs_expect_nothing():
    assert learn([]) == Invariants() and check(learn([]), g([[1]]), g([[2, 3]])) == []
