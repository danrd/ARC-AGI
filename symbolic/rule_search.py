"""The shortest cell rule that turns every training input into its output, found by search rather than grown greedily.

The feature probe's decision tree showed which cell features carry a coordinate task's rule to its test (7 of 23 on
the right ones, none on position); RL on the same features solved none. A greedy tree is still a poor solver: it takes
the split that is best at one step, so a spurious one - "the box is at most 3 high" - can win over the rule, and among
the many rules that fit two or three pairs it has no way to prefer the plain one. This enumerates the rules instead,
shortest first, and keeps the ones that reproduce every training pair exactly:

    which cells change   a formula over atoms ("nearest_left == red", "distance_up <= 2"): one atom, a conjunction of
                         up to three, or a disjunction of up to three conjunctions of up to two
    what they become     one label for all of them - a colour, or a role such as "the colour of the nearest cell to
                         the left" - or two labels with a condition between them, or one colour per group

A rule is scored by its length (atoms, and a threshold costs a little more than an equality), the shortest first, and
the two shortest that give different test outputs are the two attempts ARC allows. Only tasks whose outputs keep their
inputs' shape, and where each cell follows from its own features, can be solved this way.
"""
import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from symbolic.cell_features import cell_features

#: How many of the shortest rules for "which cells change" and for "what they become" are combined.
KEEP = 12
#: A threshold reads more into the examples than an equality does: it costs this much more in a rule's length.
THRESHOLD_COST = 0.25
#: Conditions that compare two colour features of the cell ("colour == mirror_both") besides those that compare one with
#: a value; off by default, for the features in RELATIONAL below.
RELATIONAL = False
#: The colour features the relational conditions are made from: the cell, its neighbours and nearest colours, and its
#: mirror images. All pairs of more than these would be the product of two hundred features with each other.
RELATIONAL_FEATURES = ("colour", "next_left", "next_right", "next_up", "next_down", "nearest_left", "nearest_right",
                       "nearest_up", "nearest_down", "mirror_left_right", "mirror_up_down", "mirror_both",
                       "mirror_diagonal", "box_mirror_lr", "box_mirror_ud", "box_mirror_both", "turn_90",
                       "turn_270", "turn_anti")
#: How many terms (and how many supersets) the formulas for "which cells change" are built from.
WIDE = 400
SUPERSET_CAP = 120


@dataclass(frozen=True)
class Atom:
    """One condition on one feature of a cell: equal to a value, at most a value, or more than it."""
    feature: str
    op: str            # "==", "!=" (colours: "nearest_left != none"), "<=", ">", "==f" and "!=f" (against `other`)
    value: int
    other: str = ""

    def holds(self, column, other=None):
        if self.op == "==f":
            return column == other
        if self.op == "!=f":
            return column != other
        if self.op == "==":
            return column == self.value
        if self.op == "!=":
            return column != self.value
        if self.op == "<=":
            return column <= self.value
        return column > self.value

    @property
    def cost(self):
        return 1.0 if self.op in ("==", "!=", "==f", "!=f") else 1.0 + THRESHOLD_COST

    def __str__(self):
        return f"{self.feature} {self.op} {self.other if self.other else self.value}"


def table(grids, feature_set):
    """(cells, features) matrix over every cell of the grids, the feature names, and which of them are colours."""
    blocks, names, colours = [], None, None
    for grid in grids:
        features, colours = cell_features(np.asarray(grid), feature_set)
        names = sorted(features)
        blocks.append(np.stack([features[name].ravel() for name in names], axis=1))
    return np.concatenate(blocks), names, colours


def atoms(x, names, colours):
    """Every distinct condition the training cells can tell apart, the cheapest description of each mask kept: an
    equality on every value of a colour or of a feature with two values, a threshold between the values of any other.
    Returns the atoms and their (atoms, cells) masks."""
    found, masks, seen = [], [], set()

    def add(atom, mask):
        if mask.all() or not mask.any():
            return
        key = np.packbits(mask).tobytes()
        if key in seen:
            return
        seen.add(key)
        found.append(atom)
        masks.append(mask)

    for j, name in enumerate(names):
        column = x[:, j]
        values = np.unique(column)
        if name in colours or len(values) <= 2:
            for value in values:
                add(Atom(name, "==", int(value)), column == value)
        if name in colours:
            for value in values:
                add(Atom(name, "!=", int(value)), column != value)
    if RELATIONAL:
        usable = [j for j, name in enumerate(names) if name in RELATIONAL_FEATURES and name in colours]
        for a, b in itertools.combinations(usable, 2):
            same = x[:, a] == x[:, b]
            add(Atom(names[a], "==f", 0, names[b]), same)
            add(Atom(names[a], "!=f", 0, names[b]), ~same)
    for j, name in enumerate(names):
        if name in colours:
            continue
        column = x[:, j]
        values = np.unique(column)
        if len(values) <= 2:
            continue
        for value in values[:-1]:
            add(Atom(name, "<=", int(value)), column <= value)
            add(Atom(name, ">", int(value)), column > value)
    return found, (np.array(masks) if masks else np.zeros((0, len(x)), dtype=bool))


# ---------------------------------------------------------------------------
# which cells change
# ---------------------------------------------------------------------------

def term_cost(term, found):
    return sum(found[i].cost for i in term)


def formulas_for(target, found, masks, keep=None, wide=None, superset_cap=None):
    """The cheapest formulas in disjunctive form (tuples of terms, a term a tuple of atom indices) that are true on
    exactly the `target` cells: a single atom, a conjunction of two or three, or a disjunction of two or three
    conjunctions of at most two. Returns [(cost, formula)], cheapest first."""
    keep = KEEP if keep is None else keep
    wide = WIDE if wide is None else wide
    superset_cap = SUPERSET_CAP if superset_cap is None else superset_cap
    found_rules = []
    if not target.any():
        return [(0.0, ())]                                     # nothing changes
    n_target = target.sum()
    inside = (masks & target).sum(axis=1)
    outside = (masks & ~target).sum(axis=1)
    # Single atoms and pure atoms (true on target cells only).
    for i in np.nonzero((inside == n_target) & (outside == 0))[0]:
        found_rules.append((found[i].cost, ((int(i),),)))
    pure_terms = [((int(i),), masks[i]) for i in np.nonzero((outside == 0) & (inside > 0))[0]]
    # Conjunctions of two: both atoms hold on every target cell (supersets), or the pair is pure.
    candidates = np.nonzero(inside > 0)[0]
    supersets = np.nonzero(inside == n_target)[0]
    supersets = supersets[np.argsort(masks[supersets].sum(axis=1))][:superset_cap]
    for a in candidates:
        both = masks[a] & masks[candidates]
        out = (both & ~target).any(axis=1)
        cover = (both & target).sum(axis=1)
        for k in np.nonzero(~out & (cover > 0))[0]:
            b = candidates[k]
            if b <= a:
                continue
            term = (int(a), int(b))
            if cover[k] == n_target:
                found_rules.append((term_cost(term, found), (term,)))
            pure_terms.append((term, both[k]))
    # Conjunctions of three supersets.
    for a, b in itertools.combinations(supersets, 2):
        ab = masks[a] & masks[b]
        if (ab & ~target).sum() == 0:
            continue
        rest = ab & masks[supersets]
        for k in np.nonzero(~(rest & ~target).any(axis=1))[0]:
            c = supersets[k]
            if c > b:
                term = (int(a), int(b), int(c))
                found_rules.append((term_cost(term, found), (term,)))
    # Disjunctions of two or three pure terms.
    pure_terms.sort(key=lambda item: (-item[1].sum(), term_cost(item[0], found)))
    pure_terms = pure_terms[:wide]
    if pure_terms:
        cover = np.array([mask for _, mask in pure_terms])
        for i in range(len(pure_terms)):
            union = cover[i] | cover[i + 1:]
            for k in np.nonzero(union.sum(axis=1) == n_target)[0]:
                j = i + 1 + k
                rule = (pure_terms[i][0], pure_terms[j][0])
                found_rules.append((sum(term_cost(t, found) for t in rule), rule))
        small = cover[:min(len(cover), 80)]
        for i, j in itertools.combinations(range(len(small)), 2):
            union = small[i] | small[j]
            if union.sum() == n_target:
                continue
            rest = union | small[j + 1:]
            for k in np.nonzero(rest.sum(axis=1) == n_target)[0]:
                rule = (pure_terms[i][0], pure_terms[j][0], pure_terms[j + 1 + k][0])
                found_rules.append((sum(term_cost(t, found) for t in rule), rule))
    return cheapest(found_rules, keep)


def cheapest(rules, keep):
    """The `keep` cheapest distinct rules."""
    out, seen = [], set()
    for cost, rule in sorted(rules, key=lambda item: (item[0], str(item[1]))):
        if rule in seen:
            continue
        seen.add(rule)
        out.append((cost, rule))
        if len(out) == keep:
            break
    return out


def holds(formula, found, x, names):
    """Where a formula is true on the cells of `x`, a table over `names`."""
    result = np.zeros(len(x), dtype=bool)
    for term in formula:
        part = np.ones(len(x), dtype=bool)
        for i in term:
            atom = found[i]
            part &= atom.holds(x[:, names.index(atom.feature)], x[:, names.index(atom.other)] if atom.other else None)
        result |= part
    return result


# ---------------------------------------------------------------------------
# what they become
# ---------------------------------------------------------------------------

def label_candidates(x, names, colours, targets):
    """label -> (cells) mask of the changed cells the label gives the right colour: "abs:<c>" for a colour, and
    "src:<feature>" for each colour feature holding the new colour there."""
    out = {f"abs:{c}": targets == c for c in np.unique(targets)}
    for name in sorted(colours):
        mask = x[:, names.index(name)] == targets
        if mask.any():
            out[f"src:{name}"] = mask
    return out


def label_cost(label):
    """A role is no dearer than a colour; it is what carries over to a test whose colours are new."""
    return 0.0


def colour_models(x, names, colours, targets, found, masks, keep=None):
    """The cheapest ways to give the changed cells (`x`, `targets` restricted to them) their colours: one label for
    all; two labels split by a term of one or two atoms; or one colour per group, each group but the largest picked
    out by a term. A model is a list of (term or None, label), read in order, the last with no condition."""
    keep = KEEP if keep is None else keep
    models = []
    allowed = label_candidates(x, names, colours, targets)
    for label, mask in allowed.items():
        if mask.all():
            models.append((label_cost(label), [(None, label)]))
    if models:
        return cheapest_models(models, keep)
    # Two labels: cells where the term holds take the first.
    restricted = masks
    for first, a1 in allowed.items():
        for second, a2 in allowed.items():
            if first == second or not (a1 | a2).all():
                continue
            must = ~a2                                         # these can only be the first
            if not must.any():
                continue
            fits = (restricted & ~a1).sum(axis=1) == 0
            fits &= (restricted & must).sum(axis=1) == must.sum()
            for i in np.nonzero(fits)[0]:
                models.append((found[i].cost, [(((int(i),),), first), (None, second)]))
            supers = np.nonzero(((restricted & must).sum(axis=1) == must.sum()))[0][:150]
            for a, b in itertools.combinations(supers, 2):
                both = restricted[a] & restricted[b]
                if not (both & ~a1).any():
                    term = (int(a), int(b))
                    models.append((term_cost(term, found), [((term,), first), (None, second)]))
    # Three labels: a first one where an atom holds, then two split by another atom (single atoms only), e.g. the
    # nearest colour on the left, on the right, and a colour of its own between them.
    if len(models) < keep:
        for first, a1 in allowed.items():
            pure = np.nonzero(((restricted & ~a1).sum(axis=1) == 0) & (restricted.sum(axis=1) > 0))[0]
            pure = pure[np.argsort(-restricted[pure].sum(axis=1))][:30]
            for i in pure:
                left = ~restricted[i]
                for second, a2 in allowed.items():
                    if second == first:
                        continue
                    for third, a3 in allowed.items():
                        if third in (first, second) or not (left <= (a2 | a3)).all():
                            continue
                        must = left & ~a3
                        fits = ((restricted & left & ~a2).sum(axis=1) == 0)
                        fits &= ((restricted & must).sum(axis=1) == must.sum())
                        for j in np.nonzero(fits)[0][:3]:
                            models.append((found[i].cost + found[j].cost,
                                           [(((int(i),),), first), (((int(j),),), second), (None, third)]))
    # One colour per group (the colours themselves), each group but the largest picked out exactly by a term.
    colours_here, counts = np.unique(targets, return_counts=True)
    if 2 <= len(colours_here) <= 4:
        order = np.argsort(-counts)
        rest_colour = colours_here[order[0]]
        steps, total = [], 0.0
        for c in colours_here[order[1:]]:
            group = targets == c
            rules = formulas_for(group, found, restricted, keep=1)
            if not rules:
                steps = None
                break
            total += rules[0][0]
            steps.append((rules[0][1], f"abs:{c}"))
        if steps is not None:
            models.append((total, steps + [(None, f"abs:{rest_colour}")]))
    return cheapest_models(models, keep)


def cheapest_models(models, keep):
    out, seen = [], set()
    for cost, model in sorted(models, key=lambda item: (item[0], str(item[1]))):
        key = str(model)
        if key not in seen:
            seen.add(key)
            out.append((cost, model))
        if len(out) == keep:
            break
    return out


def paint(model, found, x, names):
    """The colours a model gives cells (rows of `x`)."""
    out = np.full(len(x), -1)
    left = np.ones(len(x), dtype=bool)
    for condition, label in model:
        where = left.copy() if condition is None else left & holds(condition, found, x, names)
        kind, what = label.split(":", 1)
        out[where] = int(what) if kind == "abs" else x[where, names.index(what)]
        left &= ~where
    return out


# ---------------------------------------------------------------------------
# the whole rule
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    cost: float
    change: tuple
    colour: list
    found: list
    names: list
    feature_set: str

    def apply(self, grid):
        grid = np.asarray(grid)
        x, names, _ = table([grid], self.feature_set)
        will = holds(self.change, self.found, x, names)
        out = grid.ravel().copy()
        if will.any():
            out[will] = paint(self.colour, self.found, x[will], names)
        return out.reshape(grid.shape)

    def describe(self):
        def term(t):
            return " and ".join(str(self.found[i]) for i in t)
        change = " or ".join(f"({term(t)})" for t in self.change) or "nothing"
        colour = "; ".join(f"{'otherwise' if c is None else ' or '.join(term(t) for t in c)} -> {label}"
                           for c, label in self.colour)
        return f"change where {change}; {colour}"


def search(train, feature_set="core", keep=None) -> List[Rule]:
    """The rules that reproduce every training pair, cheapest first. Empty when the pairs change shape, when a
    cell's change does not follow from its features, or when no rule short enough is found."""
    keep = KEEP if keep is None else keep
    inputs = [np.asarray(p["input"]) for p in train]
    outputs = [np.asarray(p["output"]) for p in train]
    if any(i.shape != o.shape for i, o in zip(inputs, outputs)):
        return []
    x, names, colours = table(inputs, feature_set)
    before = np.concatenate([g.ravel() for g in inputs])
    after = np.concatenate([g.ravel() for g in outputs])
    changed = after != before
    found, masks = atoms(x, names, colours)
    rules = []
    for change_cost, change in formulas_for(changed, found, masks, keep):
        if not changed.any():
            rules.append(Rule(change_cost, change, [], found, names, feature_set))
            continue
        sub = x[changed]
        sub_found, sub_masks = atoms(sub, names, colours)
        for colour_cost, model in colour_models(sub, names, colours, after[changed], sub_found, sub_masks, keep):
            rules.append(Rule(change_cost + colour_cost, change, model_with(model, sub_found, found),
                              found, names, feature_set))
    # Every rule reproduces the pairs by construction: the change formula is exact on the training cells and the
    # colour model on the changed ones.
    rules.sort(key=lambda rule: rule.cost)
    return rules


def model_with(model, sub_found, found):
    """A colour model whose atoms index `sub_found` re-indexed into `found` (one list for the rule to keep)."""
    out = []
    for condition, label in model:
        if condition is None:
            out.append((None, label))
            continue
        terms = []
        for term in condition:
            indices = []
            for i in term:
                atom = sub_found[i]
                if atom not in found:
                    found.append(atom)
                indices.append(found.index(atom))
            terms.append(tuple(indices))
        out.append((tuple(terms), label))
    return out


STRATEGIES = ("shortest", "vote", "loo")


def attempts(rules: Sequence[Rule], grid, tries=2, strategy="shortest", train=None,
             feature_set="core") -> List[np.ndarray]:
    """The outputs to hand in for `grid` out of the rules that fit the training pairs (cheapest first).

    shortest  the outputs of the cheapest rules, the first `tries` that differ
    vote      the outputs grouped, each rule voting 1 / (1 + cost) for its output; the heaviest groups first. Many
              long rules that agree outweigh one short rule that stands alone
    loo       the same groups, ranked first by how many of the leave-one-pair-out searches (the rules that fit all
              the pairs but one) give that output as their cheapest, then by their weight: an output that survives
              losing any one example is less likely to hang on a coincidence of it. Needs `train`
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy is one of {STRATEGIES}, not {strategy!r}")
    if strategy == "shortest":
        out = []
        for rule in rules:
            guess = rule.apply(grid)
            if not any(np.array_equal(guess, g) for g in out):
                out.append(guess)
            if len(out) == tries:
                break
        return out
    groups = []                                       # [output, weight, votes from leave-one-out]
    for rule in rules:
        guess = rule.apply(grid)
        for group in groups:
            if np.array_equal(group[0], guess):
                group[1] += 1.0 / (1.0 + rule.cost)
                break
        else:
            groups.append([guess, 1.0 / (1.0 + rule.cost), 0])
    if strategy == "loo":
        if train is None:
            raise ValueError("the loo strategy needs the training pairs")
        for i in range(len(train)):
            fewer = search(list(train[:i]) + list(train[i + 1:]), feature_set)
            if not fewer:
                continue
            guess = fewer[0].apply(grid)
            for group in groups:
                if np.array_equal(group[0], guess):
                    group[2] += 1
                    break
    groups.sort(key=lambda group: (-group[2], -group[1]))
    return [group[0] for group in groups[:tries]]


def solve(train, test_input, feature_sets=("core", "rich"), tries=2) -> Tuple[List[np.ndarray], Optional[Rule]]:
    """The attempts for a test input from the first feature set that has a rule, and the rule behind the first."""
    for feature_set in feature_sets:
        rules = search(train, feature_set)
        if rules:
            return attempts(rules, test_input, tries), rules[0]
    return [], None
