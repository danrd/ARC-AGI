"""The shortest cell rule (symbolic.rule_search): atoms, formulas for the changed cells, colour models, and the whole
search on small tasks whose rule is known."""
import numpy as np

from symbolic import rule_search as rs


def pair(grid, out):
    return {"input": np.asarray(grid).tolist(), "output": np.asarray(out).tolist()}


def test_atoms_are_distinct_masks_with_equalities_on_colours_and_thresholds_between_counts():
    x = np.array([[0, 0], [1, 3], [2, 5], [1, 5]])
    found, masks = rs.atoms(x, ["colour", "size"], {"colour"})
    text = [str(a) for a in found]
    assert "colour == 1" in text and "colour != 1" in text and "size <= 3" in text and "size > 3" in text
    assert len({m.tobytes() for m in masks}) == len(masks)              # no two atoms with the same cells
    assert "size <= 0" not in text                                      # the same cells as colour == 0, said first
    assert not any(m.all() or not m.any() for m in masks)
    assert "size == 3" not in text                                      # a count of three values gets thresholds


def test_the_cheapest_formula_comes_first_and_a_disjunction_is_found_when_nothing_shorter_fits():
    x = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [2, 3]])
    found, masks = rs.atoms(x, ["a", "b"], {"a", "b"})
    target = np.array([False, False, True, True, False])
    best = rs.formulas_for(target, found, masks)
    assert best[0][0] == 1.0 and [str(found[i]) for i in best[0][1][0]] == ["a == 1"]
    either = np.array([True, False, False, False, True])               # (a == 0 and b == 1) or a == 2
    rules = rs.formulas_for(either, found, masks)
    assert rules and all(np.array_equal(rs.holds(r, found, x, ["a", "b"]), either) for _, r in rules)
    assert rs.formulas_for(np.zeros(5, dtype=bool), found, masks) == [(0.0, ())]


def between_task(width, row, left, right, colour):
    grid = np.zeros((5, width), dtype=int)
    grid[row, left] = grid[row, right] = colour
    out = grid.copy()
    out[row, left + 1:right] = 4
    return pair(grid, out)


def test_a_rule_about_structure_is_found_and_carries_to_a_test_in_a_new_place():
    train = [between_task(9, 1, 1, 5, 2), between_task(9, 3, 2, 7, 2), between_task(10, 0, 4, 8, 2)]
    test = between_task(10, 4, 0, 3, 2)
    rules = rs.search(train)
    assert rules and "between_left_right" in rules[0].describe()
    assert np.array_equal(rs.attempts(rules, test["input"])[0], test["output"])


def test_a_colour_no_training_output_has_comes_from_a_role():
    def recolour(colour, row):
        grid = np.zeros((4, 6), dtype=int)
        grid[row, 0] = colour
        out = grid.copy()
        out[row, 1:] = colour
        return pair(grid, out)

    rules = rs.search([recolour(3, 0), recolour(5, 2), recolour(3, 3)])
    assert rules and "src:" in rules[0].describe()
    test = recolour(7, 1)
    assert np.array_equal(rules[0].apply(test["input"]), test["output"])


def test_three_labels_split_by_atoms_colour_the_cells_between_two_markers():
    def bridge(row, left_colour, right_colour):
        grid = np.zeros((3, 11), dtype=int)
        grid[row, 0], grid[row, 10] = left_colour, right_colour
        out = grid.copy()
        out[row, 1:5], out[row, 5], out[row, 6:10] = left_colour, 5, right_colour
        return pair(grid, out)

    train = [bridge(0, 1, 2), bridge(2, 3, 4)]
    rules = rs.search(train)
    test = bridge(1, 6, 7)
    assert rules and any(np.array_equal(g, test["output"]) for g in rs.attempts(rules, test["input"]))


def test_a_task_that_changes_shape_has_no_rule_and_the_attempts_differ():
    assert rs.search([pair(np.zeros((2, 2), dtype=int), np.zeros((3, 3), dtype=int))]) == []
    train = [between_task(9, 1, 1, 5, 2), between_task(9, 3, 2, 7, 2)]
    rules = rs.search(train)
    tries = rs.attempts(rules, between_task(10, 4, 0, 3, 2)["input"])
    assert 1 <= len(tries) <= 2 and (len(tries) == 1 or not np.array_equal(tries[0], tries[1]))


def test_a_rule_that_does_not_reproduce_every_pair_is_not_returned():
    train = [between_task(9, 1, 1, 5, 2), pair(np.eye(4, dtype=int), np.eye(4, dtype=int)[::-1])]
    for rule in rs.search(train):
        assert all(np.array_equal(rule.apply(p["input"]), p["output"]) for p in train)


def test_a_strategy_chooses_among_the_rules_that_fit_and_an_unknown_one_is_refused():
    import pytest
    train = [between_task(9, 1, 1, 5, 2), between_task(9, 3, 2, 7, 2), between_task(10, 0, 4, 8, 2)]
    test = between_task(10, 4, 0, 3, 2)
    rules = rs.search(train)
    for strategy in rs.STRATEGIES:
        tries = rs.attempts(rules, test["input"], strategy=strategy, train=train)
        assert 1 <= len(tries) <= 2 and np.array_equal(tries[0], test["output"]), strategy
    with pytest.raises(ValueError, match="strategy"):
        rs.attempts(rules, test["input"], strategy="best")
    with pytest.raises(ValueError, match="training pairs"):
        rs.attempts(rules, test["input"], strategy="loo")


class FixedRule:
    """A rule that gives one output whatever it is shown."""

    def __init__(self, cost, value):
        self.cost, self.value = cost, value

    def apply(self, grid):
        return np.full((1, 1), self.value)


def test_many_long_rules_that_agree_outvote_one_short_rule_that_stands_alone():
    rules = [FixedRule(1.0, 7)] + [FixedRule(3.0, 5) for _ in range(4)]
    assert [g[0, 0] for g in rs.attempts(rules, None, strategy="shortest")] == [7, 5]
    assert [g[0, 0] for g in rs.attempts(rules, None, strategy="vote")] == [5, 7]       # 4 x 0.25 against 0.5
    assert [g[0, 0] for g in rs.attempts(rules, None, strategy="vote", tries=1)] == [5]


def test_the_threshold_cost_decides_whether_a_threshold_or_an_equality_is_the_shorter_rule(monkeypatch):
    atom = rs.Atom("size", "<=", 3)
    monkeypatch.setattr(rs, "THRESHOLD_COST", 0.25)
    assert atom.cost == 1.25 and rs.Atom("size", "==", 3).cost == 1.0 and rs.Atom("colour", "!=", 3).cost == 1.0
    monkeypatch.setattr(rs, "THRESHOLD_COST", 1.0)
    assert atom.cost == 2.0


def test_a_cheap_rule_outweighs_two_dear_ones_and_an_output_that_survives_leaving_a_pair_out_comes_first(monkeypatch):
    cheap_and_dear = [FixedRule(0.0, 7), FixedRule(9.0, 5), FixedRule(9.0, 5)]
    assert [g[0, 0] for g in rs.attempts(cheap_and_dear, None, strategy="vote")] == [7, 5]    # 1.0 against 0.2
    steady = [FixedRule(1.0, 7), FixedRule(3.0, 5)]
    asked = []
    monkeypatch.setattr(rs, "search", lambda train, feature_set="core", keep=12: asked.append(list(train)) or
                        [FixedRule(1.0, 5)])
    pairs = [{"n": 0}, {"n": 1}, {"n": 2}]
    assert [g[0, 0] for g in rs.attempts(steady, None, strategy="vote")] == [7, 5]
    assert [g[0, 0] for g in rs.attempts(steady, None, strategy="loo", train=pairs)] == [5, 7]
    assert asked == [[{"n": 1}, {"n": 2}], [{"n": 0}, {"n": 2}], [{"n": 0}, {"n": 1}]]       # each pair left out once
    monkeypatch.setattr(rs, "search", lambda train, feature_set="core", keep=12: [])
    assert [g[0, 0] for g in rs.attempts(steady, None, strategy="loo", train=pairs)] == [7, 5]   # nothing to vote with


def test_a_condition_can_compare_two_colour_features_when_asked_and_not_otherwise(monkeypatch):
    x = np.array([[1, 1, 5], [2, 3, 5], [4, 4, 5], [1, 2, 6], [3, 3, 6], [2, 2, 7]])
    names, colours = ["colour", "mirror_both", "next_left"], {"colour", "mirror_both", "next_left"}
    monkeypatch.setattr(rs, "RELATIONAL", False)
    assert not any(a.op in ("==f", "!=f") for a in rs.atoms(x, names, colours)[0])
    monkeypatch.setattr(rs, "RELATIONAL", True)
    found, masks = rs.atoms(x, names, colours)
    text = [str(a) for a in found]
    assert "colour ==f mirror_both" in text and "colour !=f mirror_both" in text
    atom = found[text.index("colour ==f mirror_both")]
    assert atom.holds(x[:, 0], x[:, 1]).tolist() == [True, False, True, False, True, True] and atom.cost == 1.0
    assert rs.holds(((text.index("colour !=f mirror_both"),),), found, x, names).tolist() == [False, True, False, True, False, False]


def test_a_symmetry_task_is_solved_by_comparing_a_cell_with_its_mirror_only_when_conditions_may_compare(monkeypatch):
    def mirrored(seed):
        rng = np.random.default_rng(seed)
        half = rng.integers(1, 4, (4, 3))
        grid = np.concatenate([half, np.zeros((4, 3), dtype=int)], axis=1)
        return pair(grid, np.concatenate([half, half[:, ::-1]], axis=1))

    train = [mirrored(1), mirrored(2), mirrored(3)]
    test = mirrored(9)
    monkeypatch.setattr(rs, "RELATIONAL", True)
    rules = rs.search(train, "rich")
    assert rules and np.array_equal(rules[0].apply(test["input"]), test["output"])
    assert "mirror_left_right" in rules[0].describe() or "==f" in rules[0].describe() or "!=f" in rules[0].describe()


def test_the_width_and_the_number_of_rules_kept_are_read_when_the_search_runs(monkeypatch):
    x = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [2, 3]])
    found, masks = rs.atoms(x, ["a", "b"], {"a", "b"})
    target = np.array([True, False, False, False, True])
    monkeypatch.setattr(rs, "KEEP", 1)
    assert len(rs.formulas_for(target, found, masks)) == 1
    monkeypatch.setattr(rs, "KEEP", 12)
    assert len(rs.formulas_for(target, found, masks)) > 1
