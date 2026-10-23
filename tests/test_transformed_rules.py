"""The transformation library, and the cell rule search behind a transformation."""
import numpy as np

from symbolic import grid_transforms as gt
from symbolic import transformed_rules as tr

G = np.array([[1, 2, 0], [0, 3, 0]])


class TestTransforms:
    def test_the_rotations_and_flips_are_the_numpy_ones(self):
        assert np.array_equal(gt.transformed("rot90", G), [[0, 0], [2, 3], [1, 0]])
        assert np.array_equal(gt.transformed("rot180", G), [[0, 3, 0], [0, 2, 1]])
        assert np.array_equal(gt.transformed("flip_rows", G), [[0, 3, 0], [1, 2, 0]])
        assert np.array_equal(gt.transformed("flip_columns", G), [[0, 2, 1], [0, 3, 0]])
        assert np.array_equal(gt.transformed("transpose", G), [[1, 0], [2, 3], [0, 0]])

    def test_anti_transpose_reflects_over_the_other_diagonal(self):
        assert gt.transformed("anti_transpose", np.array([[1, 2], [3, 4]])).tolist() == [[4, 2], [3, 1]]

    def test_upscaling_repeats_each_cell_and_tiling_repeats_the_grid(self):
        assert gt.transformed("up2", np.array([[1, 2]])).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2]]
        assert gt.transformed("tile_1x2", np.array([[1, 2]])).tolist() == [[1, 2, 1, 2]]
        assert gt.transformed("tile_2x1", np.array([[1, 2]])).tolist() == [[1, 2], [1, 2]]

    def test_downscaling_only_applies_to_a_grid_of_blocks(self):
        assert gt.transformed("down2", np.kron(np.array([[1, 2], [3, 4]]), np.ones((2, 2), int))).tolist() == [[1, 2], [3, 4]]
        assert gt.transformed("down2", np.array([[1, 2, 1, 2], [1, 1, 1, 1], [3, 3, 3, 3], [3, 3, 3, 3]])) is None
        assert gt.transformed("down2", np.ones((3, 3), int)) is None

    def test_mirroring_appends_the_reflection(self):
        assert gt.transformed("mirror_right", np.array([[1, 2]])).tolist() == [[1, 2, 2, 1]]
        assert gt.transformed("mirror_down", np.array([[1], [2]])).tolist() == [[1], [2], [2], [1]]
        assert gt.transformed("mirror_both", np.array([[1, 2]])).shape == (2, 4)

    def test_cropping_to_content_cuts_the_box_round_what_is_not_background(self):
        grid = np.zeros((5, 5), int)
        grid[1, 2] = 4
        grid[3, 3] = 5
        assert gt.transformed("crop_content", grid).tolist() == [[4, 0], [0, 0], [0, 5]]
        assert gt.transformed("crop_content", np.zeros((3, 3), int)) is None

    def test_the_largest_and_smallest_piece_are_told_apart(self):
        grid = np.zeros((6, 6), int)
        grid[0:2, 0:3] = 1
        grid[5, 5] = 2
        assert gt.transformed("crop_largest", grid).tolist() == [[1, 1, 1], [1, 1, 1]]
        assert gt.transformed("crop_smallest", grid).tolist() == [[2]]

    def test_pieces_touching_at_a_corner_are_one_piece(self):
        grid = np.zeros((4, 4), int)
        grid[0, 0] = 1
        grid[1, 1] = 2
        grid[3, 3] = 3
        assert gt.transformed("crop_largest", grid).tolist() == [[1, 0], [0, 2]]

    def test_halves_take_the_side_and_leave_the_middle_of_an_odd_grid(self):
        grid = np.arange(15).reshape(3, 5)
        assert gt.transformed("left_half", grid).shape == (3, 2) and gt.transformed("right_half", grid).shape == (3, 2)
        assert gt.transformed("right_half", grid).tolist() == [[3, 4], [8, 9], [13, 14]]
        assert gt.transformed("top_half", grid).shape == (1, 5) and gt.transformed("bottom_half", grid).tolist() == [[10, 11, 12, 13, 14]]

    def test_trim_and_pad_undo_each_other(self):
        grid = np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]])
        assert gt.transformed("trim", grid).tolist() == [[7]]
        assert gt.transformed("pad", gt.transformed("trim", grid)).shape == (3, 3)
        assert gt.transformed("pad", np.array([[7]])).tolist() == [[7, 7, 7], [7, 7, 7], [7, 7, 7]]
        assert gt.transformed("trim", np.ones((2, 2), int)) is None

    def test_a_transformation_does_not_change_its_argument(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        before = grid.copy()
        for name in gt.TRANSFORMS:
            gt.transformed(name, grid)
        assert np.array_equal(grid, before)


def pairs(function, grids):
    return [{"input": g, "output": function(g)} for g in grids]


GRIDS = [np.array([[1, 0, 0], [0, 2, 0]]), np.array([[0, 3, 0], [4, 0, 0]]), np.array([[0, 0, 5], [0, 6, 0]])]


class TestRulesBehindATransformation:
    def test_a_turn_followed_by_a_recolouring_is_found_and_solves_the_test(self):
        def task(g):
            out = np.rot90(g).copy()
            out[out == 0] = 8
            return out
        found = tr.candidates(pairs(task, GRIDS), np.array([[0, 7, 0], [9, 0, 0]]), ("core",))
        assert found and found[0].transform != "identity"
        assert np.array_equal(tr.attempts(found)[0], task(np.array([[0, 7, 0], [9, 0, 0]])))

    def test_a_task_the_plain_search_solves_is_found_behind_the_identity(self):
        def task(g):
            out = g.copy()
            out[out == 0] = 8
            return out
        found = tr.candidates(pairs(task, GRIDS), GRIDS[0], ("core",))
        assert found[0].transform == "identity" and found[0].cost == found[0].rules[0].cost

    def test_a_transformation_that_misses_the_output_shape_is_not_tried(self):
        found = tr.candidates(pairs(lambda g: g[:1, :1] * 0 + 4, GRIDS), GRIDS[0], ("core",))
        assert all(c.transform in ("identity",) or c.rules[0].apply(c.test_input).shape == (1, 1) for c in found)

    def test_a_transformation_costs_its_penalty_over_the_same_rule_without_one(self):
        c = tr.Candidate("rot90", "core", [type("R", (), {"cost": 2.0})()], GRIDS[0])
        assert c.cost == 2.0 + tr.PENALTY
        assert tr.Candidate("identity", "core", c.rules, GRIDS[0]).cost == 2.0

    def test_the_time_given_to_the_searches_cuts_off_the_transformations_not_tried(self):
        ticks = iter([0, 10, 100, 200, 300])
        found = tr.candidates(pairs(lambda g: g, GRIDS), GRIDS[0], ("core",), seconds=50, clock=lambda: next(ticks))
        assert [c.transform for c in found] == ["identity"]

    def test_the_two_attempts_differ(self):
        def task(g):
            out = g.copy()
            out[out == 0] = 8
            return out
        found = tr.candidates(pairs(task, GRIDS), GRIDS[0], ("core",))
        guesses = tr.attempts(found)
        assert 1 <= len(guesses) <= 2
        if len(guesses) == 2:
            assert not np.array_equal(*guesses)


class FakeRule:
    def __init__(self, cost, result):
        self.cost, self.result = cost, result

    def apply(self, grid):
        return self.result


class TestTheRanking:
    def test_a_rule_behind_a_transformation_is_ranked_with_the_penalty_in_both_the_candidates_and_the_attempts(self):
        plain = tr.Candidate("identity", "core", [FakeRule(3.0, np.array([[1]]))], GRIDS[0])
        moved = tr.Candidate("rot90", "core", [FakeRule(2.5, np.array([[2]]))], GRIDS[0])
        assert [c.transform for c in sorted([moved, plain], key=lambda c: c.cost)] == ["identity", "rot90"]
        assert [g.tolist() for g in tr.attempts([moved, plain])] == [[[1]], [[2]]]

    def test_a_cheaper_rule_behind_a_transformation_wins_when_it_is_cheaper_by_more_than_the_penalty(self):
        plain = tr.Candidate("identity", "core", [FakeRule(5.0, np.array([[1]]))], GRIDS[0])
        moved = tr.Candidate("rot90", "core", [FakeRule(2.5, np.array([[2]]))], GRIDS[0])
        assert tr.attempts([plain, moved], tries=1)[0].tolist() == [[2]]

    def test_the_search_is_only_asked_about_pairs_whose_shapes_agree(self, monkeypatch):
        seen = []

        def spy(pairs, feature_set):
            seen.extend(p["input"].shape == p["output"].shape for p in pairs)
            return []

        monkeypatch.setattr(tr.rule_search, "search", spy)
        tr.candidates(pairs(lambda g: g.T, GRIDS), GRIDS[0], ("core",))
        assert seen and all(seen)


class TestDownscaling:
    def test_a_grid_of_one_block_is_not_shrunk_to_a_cell(self):
        assert gt.transformed("down2", np.ones((2, 2), int)) is None


class TestTheLargerLibrary:
    def test_stretched_rows_and_columns_are_merged(self):
        g = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 4]])
        assert gt.transformed("dedup_rows", g).tolist() == [[1, 1, 2], [3, 3, 4]]
        assert gt.transformed("dedup_columns", g).tolist() == [[1, 2], [1, 2], [3, 4]]
        assert gt.transformed("dedup_both", g).tolist() == [[1, 2], [3, 4]]
        assert gt.transformed("dedup_rows", np.array([[1, 2], [2, 1]])) is None
        assert gt.transformed("dedup_both", np.array([[1, 2], [2, 1]])) is None

    def test_a_row_equal_to_the_one_before_only_is_merged(self):
        assert gt.transformed("dedup_rows", np.array([[1], [2], [1]])) is None

    def test_down4_and_down5_need_blocks_of_that_size(self):
        small = np.array([[1, 2], [3, 4]])
        assert gt.transformed("down4", np.kron(small, np.ones((4, 4), int))).tolist() == small.tolist()
        assert gt.transformed("down5", np.kron(small, np.ones((5, 5), int))).tolist() == small.tolist()
        assert gt.transformed("down4", np.kron(small, np.ones((2, 2), int))) is None

    def test_the_rarest_and_the_commonest_colour_are_cropped_to(self):
        g = np.zeros((5, 5), int)
        g[0, 0], g[4, 4] = 7, 7
        g[2, 1:4] = 3
        assert gt.transformed("crop_rarest_colour", g).shape == (5, 5)
        assert gt.transformed("crop_commonest_colour", g).tolist() == [[3, 3, 3]]
        assert gt.transformed("crop_rarest_colour", np.zeros((3, 3), int)) is None

    def test_a_grid_is_cut_in_two_at_a_divider_of_one_colour_or_exactly_in_half(self):
        g = np.array([[1, 0, 5, 0, 2], [0, 0, 5, 3, 0]])
        a, b = gt.parts(g, False)
        assert a.tolist() == [[1, 0], [0, 0]] and b.tolist() == [[0, 2], [3, 0]]
        assert gt.parts(np.array([[1, 2, 3, 4]]), False)[1].tolist() == [[3, 4]]
        assert gt.parts(np.array([[1, 2, 3]]), False) is not None            # one cell's divider is one colour
        assert gt.parts(np.array([[1, 0, 2], [1, 5, 2]]), False) is None     # a middle that is not a divider
        assert gt.parts(np.array([[1]]), False) is None

    def test_vertical_cuts_stack_the_parts(self):
        g = np.array([[1, 0], [5, 5], [0, 2]])
        a, b = gt.parts(g, True)
        assert a.tolist() == [[1, 0]] and b.tolist() == [[0, 2]]

    def test_the_parts_laid_over_each_other(self):
        g = np.array([[1, 0, 9, 0, 2], [0, 0, 9, 3, 0], [4, 0, 9, 4, 0]])
        assert gt.transformed("side_or_first", g).tolist() == [[1, 2], [3, 0], [4, 0]]
        assert gt.transformed("side_and", g).tolist() == [[0, 0], [0, 0], [4, 0]]
        assert gt.transformed("side_xor", g).tolist() == [[1, 2], [3, 0], [0, 0]]
        nor = gt.transformed("side_nor", g)
        assert nor.tolist() == [[0, 0], [0, 1], [0, 1]]

    def test_the_first_part_wins_in_or_first_and_the_second_in_or_second(self):
        g = np.array([[1, 0, 9, 3, 2], [0, 0, 9, 0, 0], [0, 0, 9, 0, 0]])
        assert gt.transformed("side_or_first", g).tolist() == [[1, 2], [0, 0], [0, 0]]
        assert gt.transformed("side_or_second", g).tolist() == [[3, 2], [0, 0], [0, 0]]

    def test_stack_cuts_across_rows(self):
        g = np.array([[1, 0], [9, 9], [0, 2]])
        assert gt.transformed("stack_or_first", g).tolist() == [[1, 2]]
        assert gt.transformed("stack_xor", g).tolist() == [[1, 2]]
        assert gt.transformed("stack_and", g).tolist() == [[0, 0]]

    def test_the_library_has_all_its_names_and_none_changes_its_argument(self):
        assert len(gt.TRANSFORMS) == 46
        g = np.array([[1, 0, 9, 0, 2], [0, 0, 9, 3, 0], [4, 0, 9, 4, 0]])
        before = g.copy()
        for name in gt.TRANSFORMS:
            gt.transformed(name, g)
        assert np.array_equal(g, before)


class TestReachable:
    def test_a_transformation_is_reachable_when_every_pair_matches_the_output_shape(self):
        reach = tr.reachable(pairs(lambda g: g.T, GRIDS), GRIDS[0])
        assert "transpose" in [n for n, _, _ in reach] and "identity" not in [n for n, _, _ in reach]
        assert all(p["input"].shape == p["output"].shape for _, ps, _ in reach for p in ps)

    def test_the_identity_comes_first(self):
        assert tr.reachable(pairs(lambda g: g, GRIDS), GRIDS[0])[0][0] == "identity"
