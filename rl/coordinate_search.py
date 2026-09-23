"""The search for coordinate-addressed tasks: which strokes repaint a pair.

The object search walks the action space an env offers and keeps what moved
the grid. Under coordinates that space is every pair of cells for every
stroke in every colour - on a 21x21 grid 53,361 rectangles, 176,400
triangles and some 5,000 lines per colour - and a tree over it learns
nothing in the iterations a search has. So the search is not over the
space. It is over strokes read off the answer.

On a training pair the output is known, and a stroke is worth trying only
if it is *safe*: every cell it paints is one the output wants in that
colour. A safe stroke never breaks a cell, so safe strokes commute and any
set of them that covers the wrong cells is a solution, in any order. That
turns the search into covering the wrong cells, and a cover only needs the
strokes no other safe stroke contains:

  fill      the maximal rectangles inside the colour's cells
  line      the maximal runs along the two diagonals - a run along a row
            or column is a one-cell-thick rectangle, which the fills
            already hold
  triangle  the largest right isosceles triangle at each corner and each
            of its four orientations

and of those, only the ones whose wrong cells no other candidate's contain.
What is left is a pool of tens to a few hundred actions, and MCTS runs over
it with the machinery the object search uses.

What this does not find: a shorter solution that paints a cell wrong and
paints over it later - a red square and then a blue dot in it, where the
safe cover has to take the red round the dot in pieces. Every stroke of
such a solution but the last is unsafe, and a pool of safe ones cannot
contain it.

What it is for: which colours and which strokes the task's pairs are
repainted with - narrowed_for_task keeps those, as it keeps the object
actions the object search moved the grid with - and the solutions
themselves, as traces a policy could be started from.
"""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl.mcts as mcts
from data.configs.env_configs import COLORS_MAPPING
from rl.arc_env import ARCGridWorld
from rl.arc_transformators import draw_line, fill_rectangle, fill_triangle

_PAINT = {"fill": fill_rectangle, "line": draw_line, "triangle": fill_triangle}
#: Which of two candidates with the same wrong cells is kept - the simpler
#: stroke, so a one-cell-thick box is a fill and not a line.
_PREFERENCE = {"fill": 0, "line": 1, "triangle": 2}
_ORIENTATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class CoordinateSearchSettings:
    """What the coordinate search is given.

    `rollouts` and `iterations` are the object search's defaults. The pool
    is a hundred times smaller than the object search's widest and every
    action in it helps somewhere, so they buy more here; and the greedy
    cover runs first, so the tree only has to beat an answer it already
    has.
    """
    rollouts: int = 5
    iterations: int = 50
    episode_len: int = 25
    c: float = 1.414
    #: Seconds one pair may take, 0 for no cap - the object search's
    #: default. What the search found before the cut is kept.
    timeout: int = 120
    #: A pair whose wrong cells make a pool larger than this is searched by
    #: the greedy cover alone. MCTS scores the whole pool at every
    #: expansion, and a pool this size is a grid the search will not get
    #: through in any sensible budget.
    max_pool: int = 2000


def stroke_mask(shape, transform: str, first, second) -> np.ndarray:
    """The cells one stroke paints, from the transform the env runs - so a
    candidate paints exactly what the action would."""
    painted = _PAINT[transform](np.zeros(shape, dtype=int), first, second, 1)
    return painted == 1


def maximal_rectangles(allowed: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """(top, left, bottom, right) of every rectangle inside `allowed` that
    no larger rectangle inside it contains.

    Row band by row band: for rows top..bottom, the columns true in all of
    them split into runs, each run a rectangle that cannot widen. It is
    maximal when it cannot grow a row up or down either.
    """
    rows, cols = allowed.shape
    found = []
    for top in range(rows):
        columns = np.ones(cols, dtype=bool)
        for bottom in range(top, rows):
            columns &= allowed[bottom]
            if not columns.any():
                break
            for left, right in _runs(columns):
                span = slice(left, right + 1)
                if top > 0 and allowed[top - 1, span].all():
                    continue
                if bottom + 1 < rows and allowed[bottom + 1, span].all():
                    continue
                found.append((top, left, bottom, right))
    return found


def _runs(flags: np.ndarray):
    """(start, end) of every run of True, both ends inclusive."""
    padded = np.concatenate([[False], flags, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2] - 1))


def diagonal_runs(allowed: np.ndarray, min_length: int = 2):
    """((i0, j0), (i1, j1)) of every maximal run of `allowed` along a
    diagonal or an anti-diagonal, at least `min_length` cells long."""
    rows, cols = allowed.shape
    segments = []
    for step_j in (1, -1):
        starts = ([(0, j) for j in range(cols)] + [(i, 0 if step_j == 1 else cols - 1)
                                                   for i in range(1, rows)])
        for i0, j0 in starts:
            cells = []
            i, j = i0, j0
            while 0 <= i < rows and 0 <= j < cols:
                cells.append((i, j))
                i, j = i + 1, j + step_j
            flags = np.array([allowed[c] for c in cells])
            for start, end in _runs(flags):
                if end - start + 1 >= min_length:
                    segments.append((cells[start], cells[end]))
    return segments


def largest_triangles(allowed: np.ndarray):
    """(first, second) for the largest right isosceles triangle inside
    `allowed` at each corner cell and orientation, legs at least one cell
    long - in fill_triangle's terms, so the right angle lands on the corner:
    it sits at (first's row, second's column)."""
    rows, cols = allowed.shape
    found = []
    for r, c in zip(*np.nonzero(allowed)):
        for down, across in _ORIENTATIONS:
            size = 0
            while True:
                n = size + 1
                edge = [(r + down * a, c + across * (n - a)) for a in range(n + 1)]
                if not all(0 <= i < rows and 0 <= j < cols and allowed[i, j]
                           for i, j in edge):
                    break
                size = n
            if size:
                found.append(((int(r), int(c + across * size)),
                              (int(r + down * size), int(c))))
    return found


def candidate_strokes(grid: np.ndarray, target: np.ndarray, pad_val: int = 10):
    """The safe, undominated strokes for turning `grid` into `target`, as
    (colour, transform, first, second) - see the module docstring.

    A candidate has to fix at least one wrong cell; of candidates whose
    wrong cells are the same, the simplest stroke is kept, and one whose
    wrong cells another candidate's strictly contain is dropped.
    """
    grid, target = np.asarray(grid), np.asarray(target)
    shape = target.shape
    # The working grid is the input padded to the output where the output
    # is larger; where it is smaller the grid keeps the input's size and
    # only the part the target covers is compared.
    grid = grid[:shape[0], :shape[1]]
    strokes = []
    for colour in sorted(int(v) for v in np.unique(target) if v != pad_val):
        allowed = target == colour
        wrong = allowed & (grid != target)
        if not wrong.any():
            continue
        shapes = [("fill", (top, left), (bottom, right))
                  for top, left, bottom, right in maximal_rectangles(allowed)]
        shapes += [("line", first, second) for first, second in diagonal_runs(allowed)]
        shapes += [("triangle", first, second) for first, second in largest_triangles(allowed)]
        for transform, first, second in shapes:
            mask = stroke_mask(shape, transform, first, second)
            if not (mask & ~allowed).any() and (mask & wrong).any():
                strokes.append((colour, transform, tuple(map(int, first)),
                                tuple(map(int, second)), mask & wrong))
    return _undominated(strokes)


def _undominated(strokes):
    """Drop every stroke whose wrong cells another stroke's contain."""
    keyed = {}
    for colour, transform, first, second, fixes in strokes:
        key = fixes.tobytes()
        known = keyed.get(key)
        if known is None or _PREFERENCE[transform] < _PREFERENCE[known[1]]:
            keyed[key] = (colour, transform, first, second, fixes)
    kept = sorted(keyed.values(), key=lambda s: -int(s[4].sum()))
    result = []
    for index, stroke in enumerate(kept):
        fixes = stroke[4]
        size = int(fixes.sum())
        if any(int(other[4].sum()) > size and not (fixes & ~other[4]).any()
               for other in kept[:index]):
            continue
        result.append(stroke)
    return [(colour, transform, first, second)
            for colour, transform, first, second, _fixes in result]


def to_actions(strokes, actions_dict) -> List[List[int]]:
    """Strokes as [action, i1, j1, i2, j2] in `actions_dict`'s indices.
    A stroke whose colour and transform the vocabulary has no name for is
    left out rather than guessed at."""
    index = {name: i for i, name in actions_dict.items()}
    out = []
    for colour, transform, first, second in strokes:
        name = f"{COLORS_MAPPING[colour]}_{transform}"
        if name in index:
            out.append([index[name], first[0], first[1], second[0], second[1]])
    return out


def greedy_cover(env, pool) -> List[List[int]]:
    """Strokes from `pool`, each the one fixing the most cells still wrong,
    until nothing is or nothing helps. Run on the simulator, from the env's
    reset state; a sequence that does not solve is returned as far as it
    got."""
    env.reset()
    state = mcts.env_state_snapshot(env)
    sequence = []
    for _ in range(env.max_episode_len):
        if state["max_int"] == env.target_int:
            break
        best, best_state = None, None
        for action in pool:
            grid, objects, reached, _reward, _done = env.simulate_action(
                np.asarray(action), state["objects"], state["grid"], state["max_int"], None)
            if reached > (best_state["max_int"] if best_state else state["max_int"]):
                best = action
                best_state = {"grid": grid, "objects": objects, "max_int": reached,
                              "prev_action": np.asarray(action)}
        if best is None:
            break
        sequence.append(list(best))
        state = best_state
    return sequence


def _solves(env, sequence) -> bool:
    return mcts.replay_solution(env, sequence) is not None


def shortened(env, sequence) -> List[List[int]]:
    """The sequence with every step removed that it still solves without,
    left to right - one pass, as search_hints.minimise does."""
    body = [list(a) for a in sequence]
    index = 0
    while index < len(body):
        trial = body[:index] + body[index + 1:]
        if trial and _solves(env, trial):
            body = trial
        else:
            index += 1
    return body


def pair_env(subtask, actions_dict, settings: CoordinateSearchSettings):
    """A coordinate env holding one pair, sized to it."""
    shape = np.asarray(subtask.train_out).shape
    inp_shape = np.asarray(subtask.train_inp).shape
    shape = (max(shape[0], inp_shape[0]), max(shape[1], inp_shape[1]))
    env = ARCGridWorld(max_episode_len=settings.episode_len, feasible_actions=actions_dict,
                       reward_approach=3, repr_level=1, input_pattern="start",
                       addressing="coordinates", coordinate_shape=shape,
                       observation_space_elements=["delta_input"])
    env.set_subtask(subtask)
    env.reset()
    return env


def search_pair(subtask, actions_dict, settings: Optional[CoordinateSearchSettings] = None
                ) -> Dict[str, Any]:
    """Search one pair. Returns the pool it searched, the shortest verified
    solution (or None), the greedy cover as it ran ("greedy") and shortened
    ("cover", None unless it solved), the shortest MCTS found ("searched"),
    and how long it took.

    The greedy cover first: it is exact when the wrong cells of each colour
    are one stroke apiece, it costs one pass over the pool a step, and it
    gives the tree something to beat. Then MCTS over the same pool, when the
    pool is small enough for it and the cover was longer than one stroke;
    every sequence it or the cover found is replayed on the real env and
    shortened before it counts.
    """
    from rl.search_hints import SearchTimedOut, time_limit

    settings = settings or CoordinateSearchSettings()
    started = time.perf_counter()
    env = pair_env(subtask, actions_dict, settings)
    if env.max_int == env.target_int:
        # Nothing to repaint: solved by no strokes at all, and a pool built
        # from no wrong cells is empty rather than a failure.
        return {"pool": [], "solution": [], "greedy": [], "cover": [], "searched": None,
                "seconds": time.perf_counter() - started}
    pool = to_actions(candidate_strokes(env.grid, env.train_out, env.pad_val), actions_dict)
    found = []
    greedy = greedy_cover(env, pool) if pool else []
    cover = shortened(env, greedy) if greedy and _solves(env, greedy) else None
    if cover is not None:
        found.append(cover)
    searched = []
    if pool and len(pool) <= settings.max_pool and not (found and len(found[0]) <= 1):
        rollouts = []
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    time_limit(settings.timeout):
                rollouts = mcts.collect_mcts_rollouts(
                    env, n_rollouts=settings.rollouts, mcts_iterations=settings.iterations,
                    max_episode_len=settings.episode_len, actions=pool, c=settings.c,
                    playout="weighted")
        except SearchTimedOut:
            pass
        for rollout in rollouts:
            if rollout.get("solved"):
                sequence = [[int(x) for x in np.asarray(a).reshape(-1)]
                            for a in rollout["actions"]
                            if actions_dict.get(int(np.asarray(a).reshape(-1)[0])) != "submit"]
                if _solves(env, sequence):
                    searched.append(shortened(env, sequence))
    found = sorted(found + searched, key=len)
    searched.sort(key=len)
    return {"pool": pool, "solution": found[0] if found else None,
            "greedy": greedy, "cover": cover,
            "searched": searched[0] if searched else None,
            "seconds": time.perf_counter() - started}


def feasible_from_coordinate_search(task, actions_dict,
                                    settings: Optional[CoordinateSearchSettings] = None
                                    ) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """The coordinate vocabulary cut to what the task's pairs are painted
    with, and the searches it came from.

    Every training pair, not the first: a colour or a stroke the second
    pair needs is still the task's. What is kept is every name a pair's
    shortest solution uses - or, for a pair nothing solved, every name its
    greedy cover got anywhere with. Nothing found anywhere keeps the whole
    vocabulary, as feasible_from_search does: an env left with submit alone
    is not a narrower version of the task.

    Returns (feasible_actions, {subtask label: search_pair result}).
    """
    settings = settings or CoordinateSearchSettings()
    results = {}
    used = set()
    for subtask in task.subtasks:
        result = search_pair(subtask, actions_dict, settings)
        results[subtask.label] = result
        steps = result["solution"] if result["solution"] is not None else result["greedy"]
        used |= {actions_dict[int(step[0])] for step in steps}
    kept = sorted(used - {"submit"})
    if not kept:
        return dict(actions_dict), results
    return {0: "submit", **{i: name for i, name in enumerate(kept, start=1)}}, results
