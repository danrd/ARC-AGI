"""Every transform arc_world dispatches, exercised under its real name.

Nothing covered this: the RL tests build environments with two or three
actions, so most of the vocabulary is never called from them, and several
transforms raised on every invocation without anything noticing. MCTS
enumerates the whole action space, so one of these takes a whole rollout
down.

Names are not free-form, and a wrong one looks exactly like a broken
transform - it matches no branch, falls through, and returns the grid
untouched. So the names here are not written out. They come from
`define_feasible_actions` over the repo's own action config, which is what
an env is actually configured with, and the two ends are checked against
each other in test_the_generated_vocabulary_matches_the_dispatch.

That check is the point of the file as much as the transforms are. The
generator decorates a base action with a colour, a direction, or a second
colour, and the dispatch takes those apart again:

    rotate90                                     bare
    red_recolor                                  <colour>_<name>
    red_emission_N                               <colour>_<name>_<direction>
    red_emission_with_blue_object_recolor_N      two colours and a direction
    red_contour_connection_blue                  a second colour, appended

Either end can drift. An action listed as direction-dependent under a name
the roster spells differently never gets its direction, and the dispatch
reads whatever is in that position as one - which is how `shift_object`
came to raise KeyError('object') for three of the five agents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from data.configs.env_configs import (ACTION_TYPES, AGENT2ACTIONS,
                                      COLOR_DEPENDENT_ACTIONS,
                                      DIRECTION_DEPENDENT_ACTIONS,
                                      DOUBLE_COLOR_DEPENDENT_ACTIONS,
                                      TWO_OBJECTS_ACTION_TYPES)
from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask
from rl.utils import define_feasible_actions

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Two colours and two directions rather than all ten and all eight. Every
#: branch is reached either way - the decorations pick arguments, not
#: branches - and the full cross product is 2926 names, which is a sweep
#: over colour rather than over the vocabulary.
COLOURS = ["red", "blue"]
DIRECTIONS = ["N", "E"]


def _vocabulary():
    """Names as the env gets them, paired with whether they take two objects.

    Generated one base action at a time, because a generated name does not
    always contain its base: `emission_with_object_recolor` comes back as
    `red_emission_with_blue_object_recolor_N`, with the second colour spliced
    into the middle. Asking per base keeps the association exact instead of
    recovering it from the string.
    """
    bases = ({a for roster in AGENT2ACTIONS.values() for a in roster}
             | {a for group in ACTION_TYPES.values() for a in group})
    entries = []
    for base in sorted(bases):
        if base == "submit":
            continue
        generated = define_feasible_actions(
            [base], COLOURS, DIRECTIONS, COLOR_DEPENDENT_ACTIONS,
            DOUBLE_COLOR_DEPENDENT_ACTIONS, DIRECTION_DEPENDENT_ACTIONS)
        for name in generated.values():
            if name != "submit":
                entries.append((name, base in TWO_OBJECTS_ACTION_TYPES))
    return entries


VOCABULARY = _vocabulary()

#: Dispatched, but deliberately empty - each returns the grid untouched under
#: a "FOR FURTHER IMPLEMENTATION" comment. Kept out of the effect test rather
#: than out of the crash test: they cost a slot in every search that
#: enumerates the action space, which is worth seeing.
STUBS = ["copy", "copy_input", "paste", "cut"]

#: Transforms known to raise, with the diagnosis. Listed so the suite is
#: green on a known state rather than silent about it - remove an entry when
#: it is fixed, and the test starts demanding it stay fixed.
KNOWN_BROKEN: dict = {}

ACTION_NAMES = [name for name, _ in VOCABULARY]
PAIRWISE = {name for name, is_pair in VOCABULARY if is_pair}


def _dispatch_names():
    """Transform names read out of arc_world's own if/elif chain."""
    source = (REPO_ROOT / "rl" / "arc_world.py").read_text()
    body = source.split("def apply_transform")[1]
    single, pair = body.split("elif obj1.label != obj2.label:")
    def names(chunk):
        return set(re.findall(r'transform == "([a-z_0-9]+)"', chunk)) | \
               set(re.findall(r'transform\.startswith\("([a-z_0-9]+)"', chunk)) | \
               set(re.findall(r'"([a-z_0-9]+)" in transform', chunk))
    return names(single), names(pair)


def _task_with(min_objects):
    """A real ARC training pair whose grids keep their shape and whose input
    parses into at least `min_objects` objects.

    Hand-built grids would not do: several of these transforms only reach
    their branch for particular object shapes (contour_connection wants two
    cells), and holes, contours and symmetry are what they operate on.
    """
    with open(REPO_ROOT / "data" / "datasets" / "ARC" / "training_challenges.json") as f:
        challenges = json.load(f)
    for task_id in sorted(challenges):
        pair = challenges[task_id]["train"][0]
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        if inp.shape != out.shape:
            continue
        env = _env(task_id, inp, out)
        if env.visible_object_count() >= min_objects:
            return env
    pytest.skip(f"no shape-preserving task with {min_objects} objects")


def _env(task_id, inp, out):
    actions = {0: "submit", **{i + 1: name for i, name in enumerate(ACTION_NAMES)}}
    env = ARCGridWorld(max_episode_len=25, feasible_actions=actions, reward_approach=1,
                       repr_level=1, input_pattern="start",
                       observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask(f"{task_id}_0", inp, out))
    return env


@pytest.fixture(scope="module")
def envs():
    """Several tasks, not one. These transforms fail on particular object
    shapes rather than on all of them - gravity used to raise on 13 of 17
    tasks and rotate90 on 1 of 27 - so a single fixture task reports them
    green while they are broken everywhere else.
    """
    with open(REPO_ROOT / "data" / "datasets" / "ARC" / "training_challenges.json") as f:
        challenges = json.load(f)
    built = []
    for task_id in sorted(challenges):
        pair = challenges[task_id]["train"][0]
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        if inp.shape != out.shape:
            continue
        env = _env(task_id, inp, out)
        if env.visible_object_count() >= 2:
            built.append((task_id, env))
        if len(built) == 15:
            break
    return built


def test_the_generated_vocabulary_matches_the_dispatch():
    """Both directions, because both are silent failures.

    A branch no generated name reaches is dead code the env can never run.
    A generated name no branch answers to is worse: it occupies an index in
    the action space, and every search that enumerates that space spends
    rollouts on an action that returns the grid untouched - which reads as
    an action that had nothing to do.
    """
    single_names, pair_names = _dispatch_names()
    dispatched = single_names | pair_names

    unreachable = [b for b in dispatched
                    if not any(b in name for name in ACTION_NAMES)]
    assert not unreachable, (
        f"dispatched by arc_world but never generated: {sorted(unreachable)}")

    unanswered = [name for name in ACTION_NAMES
                   if not any(b in name for b in dispatched)]
    assert not unanswered, (
        f"generated but no dispatch branch answers to them: {sorted(unanswered)}")


def test_every_name_parses_the_way_it_reads():
    """parse_action is a dict lookup now, filled once per World instead of
    re-splitting the same strings ~117k times a search. The table has to
    hold what reading the name gives: a leading colour word names what the
    transform paints with and is stripped off, and a name without one keeps
    add at -1.

    Checked over the whole generated vocabulary rather than a sample, since
    a name that parses wrong does not fail - it reaches a branch that paints
    with the wrong colour, or no branch at all, and returns the grid.
    """
    from rl.arc_world import World

    actions = {0: "submit", **{i + 1: name for i, name in enumerate(ACTION_NAMES)}}
    world = World(objects=[], actions_dict=actions)

    for index, name in actions.items():
        add, transform = world.parse_action([index, 0, 0])
        colour = name.split("_")[0]
        if colour in world.inverse_colors_mapping:
            assert add == world.inverse_colors_mapping[colour], name
            assert transform == name[len(colour) + 1:], name
        else:
            assert add == -1, name
            assert transform == name, name


def test_the_parse_table_belongs_to_its_world():
    """Built from the actions_dict the World was given, so two Worlds with
    different vocabularies do not share an index's meaning."""
    from rl.arc_world import World

    first = World(objects=[], actions_dict={0: "submit", 1: "red_recolor"})
    second = World(objects=[], actions_dict={0: "submit", 1: "blue_recolor"})

    assert first.parse_action([1, 0, 0]) == (2, "recolor")
    assert second.parse_action([1, 0, 0]) == (1, "recolor")


def test_the_parse_table_is_built_once_and_not_reread():
    """The table is filled at construction, so a vocabulary swapped in
    afterwards is not picked up. Pinned rather than left to be discovered:
    it is the one behavioural difference caching makes, and a caller
    rewriting actions_dict on a live World would otherwise get the old
    meanings back silently.
    """
    from rl.arc_world import World

    world = World(objects=[], actions_dict={0: "submit", 1: "red_recolor"})
    world.actions_dict = {0: "submit", 1: "blue_recolor"}

    assert world.parse_action([1, 0, 0]) == (2, "recolor")


def test_parsing_an_action_beats_the_splitting_it_replaced():
    """What the table is for. Correctness cannot tell it from the string
    splitting it replaced - that is the point of an optimisation - so the
    cost is asserted instead: measured at 0.49us per call against 0.09us,
    over ~117k calls in one search.

    Against the old form rather than a wall-clock bound, because a fixed
    number of seconds is a claim about the machine. Both are timed here,
    back to back, over the same indices - a loaded runner slows them
    equally and the ratio survives it.
    """
    import time

    from rl.arc_world import World

    actions = {0: "submit", **{i + 1: name for i, name in enumerate(ACTION_NAMES)}}
    world = World(objects=[], actions_dict=actions)
    indices = [i % len(actions) for i in range(100_000)]

    def by_splitting(action):
        name = world.actions_dict[action[0]]
        colour = name.split("_")[0]
        if colour in world.inverse_colors_mapping:
            return world.inverse_colors_mapping[colour], name[len(colour) + 1:]
        return -1, name

    start = time.perf_counter()
    for index in indices:
        world.parse_action([index, 0, 0])
    cached = time.perf_counter() - start

    start = time.perf_counter()
    for index in indices:
        by_splitting([index, 0, 0])
    splitting = time.perf_counter() - start

    assert cached * 2 < splitting, (
        f"{splitting / cached:.1f}x faster than re-splitting, expected at "
        f"least 2x - is parse_action reading the table?")


def test_the_two_object_list_matches_the_pair_block():
    """apply_transform picks its half of the chain by comparing the two
    object indices, so an action in the wrong list is handed to the half
    that has no branch for it - and comes back with the grid untouched,
    reading as a transform that had nothing to do rather than as one that
    was never dispatched.

    `gravity` sat outside the list while the pair block dispatched it.
    """
    single_names, pair_names = _dispatch_names()

    # By containment, because the two ends spell some of these differently:
    # the config's "background_shortest_path_left" is dispatched by a branch
    # testing for "shortest_path_left" in the name.
    missing = [b for b in pair_names
                if not any(b in t for t in TWO_OBJECTS_ACTION_TYPES)]
    assert not missing, (
        f"dispatched in the pair block but not in TWO_OBJECTS_ACTION_TYPES: "
        f"{sorted(missing)}")

    stray = [t for t in TWO_OBJECTS_ACTION_TYPES if t in single_names]
    assert not stray, (
        f"listed as two-object but dispatched in the single block: {sorted(stray)}")


@pytest.mark.parametrize("name", ACTION_NAMES)
def test_a_transform_does_not_raise(envs, name):
    """On any task, not just a convenient one. One raising transform takes a
    whole MCTS rollout down, and MCTS enumerates the whole action space."""
    index = ACTION_NAMES.index(name) + 1
    indices = (0, 1) if name in PAIRWISE else (0, 0)

    failures = []
    for task_id, env in envs:
        env.reset()
        try:
            env.step(np.array([index, *indices]))
        except Exception as exc:
            failures.append(f"{task_id}: {type(exc).__name__}: {exc}")

    if failures and name in KNOWN_BROKEN:
        # Not pytest.raises: these raise on *some* object shapes, not all, so
        # demanding a failure would be as wrong as ignoring one.
        pytest.xfail(f"{KNOWN_BROKEN[name]} - {len(failures)}/{len(envs)}: {failures[0]}")
    assert not failures, (
        f"{name!r} raised on {len(failures)} of {len(envs)} tasks:\n  "
        + "\n  ".join(failures[:3])
    )


@pytest.mark.parametrize("name", [n for n in ACTION_NAMES
                                   if n not in STUBS and n not in KNOWN_BROKEN])
def test_a_transform_is_reachable_on_some_task(name):
    """It changes the grid somewhere. A name that matches no branch falls
    through and returns the grid untouched, which is indistinguishable from
    a transform that had nothing to do - so "never changes anything on any
    task" is the signal that the name is wrong.
    """
    with open(REPO_ROOT / "data" / "datasets" / "ARC" / "training_challenges.json") as f:
        challenges = json.load(f)

    index = ACTION_NAMES.index(name) + 1
    indices = (0, 1) if name in PAIRWISE else (0, 0)
    tried = 0
    for task_id in sorted(challenges)[:40]:
        pair = challenges[task_id]["train"][0]
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        if inp.shape != out.shape:
            continue
        env = _env(task_id, inp, out)
        if env.visible_object_count() < (2 if name in PAIRWISE else 1):
            continue
        env.reset()
        before = env.grid.copy()
        tried += 1
        try:
            env.step(np.array([index, *indices]))
        except Exception:
            continue
        if not np.array_equal(env.grid, before):
            return
    pytest.fail(f"{name!r} changed nothing on any of {tried} tasks - "
                 "either the name misses its branch or the transform is inert")


@pytest.mark.parametrize("name", ACTION_NAMES)
def test_a_transform_returns_a_grid(envs, name):
    """Every branch hands back an array of the grid's shape - including the
    branches that decline to do anything.

    Two did not: apply_transform's `transform is None` returned
    `(grid, False)`, and perform_merge's no-match branch returned
    `(objects, grid, {...})`. The caller assigns whatever comes back to
    new_grid, so the failure surfaces two steps later in
    maximal_intersection as numpy refusing to build an array out of a
    3-tuple - a message that names neither the transform nor the branch.
    """
    index = ACTION_NAMES.index(name) + 1
    indices = (0, 1) if name in PAIRWISE else (0, 0)

    for task_id, env in envs:
        env.reset()
        expected = env.grid.shape
        env.step(np.array([index, *indices]))

        assert isinstance(env.grid, np.ndarray), \
            f"{name!r} on {task_id} left {type(env.grid).__name__} in env.grid"
        assert env.grid.shape == expected, \
            f"{name!r} on {task_id} changed the grid shape to {env.grid.shape}"


def test_merge_returns_a_grid_when_it_finds_no_configuration(monkeypatch):
    """Reached through the env only after several steps put the objects
    somewhere no placement fits, so it is asked of the function directly.

    Every other path out of perform_merge returns the grid, and the caller
    assigns whatever it gets to new_grid. A branch handing back anything
    else - `(all_grid_objects, grid, {...})`, say - kills the run two steps
    later inside maximal_intersection, with numpy complaining about an
    inhomogeneous array and naming neither merge nor the branch.
    """
    from rl import arc_transformators
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 1
    grid[4, 4] = 2
    obj1 = GridObject(shape="cell", coords=[(0, 0)], color=[1], label="a",
                       grid_shape=grid.shape, grid=grid)
    obj2 = GridObject(shape="cell", coords=[(4, 4)], color=[2], label="b",
                       grid_shape=grid.shape, grid=grid)
    monkeypatch.setattr(arc_transformators, "find_most_probable_merge",
                         lambda *args, **kwargs: None)

    result = arc_transformators.perform_merge(grid, obj1, obj2, [obj1, obj2], font_color=0)

    assert isinstance(result, np.ndarray)
    assert result.shape == grid.shape


def test_a_rotation_that_would_leave_the_grid_is_refused():
    """Not clipped. Rotating a non-square object about its own centre can put
    cells outside the grid, and the three ways out are not equal: clipping
    keeps the object but loses cells, so every size, contour and hole read
    off it afterwards describes something that is not on the grid. Refusing
    costs one action that does nothing, which the env already scores as
    ineffective.

    A five-cell horizontal line centred at (0, 2) becomes a vertical one
    spanning rows -2..2 - two rows above the grid.
    """
    grid = np.zeros((5, 5), dtype=int)
    grid[0, :] = 3
    out = grid.copy()
    out[2, :] = 3

    names = {0: "submit", 1: "rotate90"}
    env = ARCGridWorld(max_episode_len=4, feasible_actions=names, repr_level=1,
                       input_pattern="start", observation_space_elements=["objects_emb"])
    env.set_subtask(ARCSubtask("edge_line", grid, out))
    env.reset()
    before = env.grid.copy()

    env.step(np.array([1, 0, 0]))

    assert np.array_equal(env.grid, before)
    # And the object still describes what is on the grid.
    obj = env.objects[0]
    assert all(0 <= i < 5 and 0 <= j < 5 for i, j in obj.coords)


@pytest.mark.parametrize("name", [n for n in ACTION_NAMES if n not in STUBS])
def test_a_transform_survives_being_applied_repeatedly(envs, name):
    """Everything above steps once from reset. These break later: after a
    few steps the objects have been rewritten, moved to edges, emptied - and
    a transform that is fine on a freshly parsed grid raises on that.

    Measured before this test existed: 1 of 12 tasks crashed a rollout at
    12 steps, 7 of 12 at 25.
    """
    index = ACTION_NAMES.index(name) + 1
    indices = (0, 1) if name in PAIRWISE else (0, 0)

    failures = []
    for task_id, env in envs:
        env.reset()
        try:
            for _ in range(12):
                _, _, done, truncated, _ = env.step(np.array([index, *indices]))
                if done or truncated:
                    break
        except Exception as exc:
            failures.append(f"{task_id}: {type(exc).__name__}: {exc}")

    if failures and name in KNOWN_BROKEN:
        pytest.xfail(f"{KNOWN_BROKEN[name]} - {len(failures)}/{len(envs)}: {failures[0]}")
    assert not failures, (
        f"{name!r} raised when applied repeatedly on {len(failures)} of "
        f"{len(envs)} tasks:\n  " + "\n  ".join(failures[:3])
    )


@pytest.mark.parametrize("name", [n for n in ACTION_NAMES if n not in STUBS])
def test_a_transform_leaves_colours_as_tuples(envs, name):
    """GridObject.__init__ writes `tuple(color)`, and consumers concatenate
    these - merge does `obj1.color_numbers + obj2.color_numbers`. Two sites
    wrote a list instead, and one of them (`recolor`) travelled with the
    object, so the mismatch surfaced inside a different transform entirely,
    as `can only concatenate list (not "tuple") to list`.
    """
    index = ACTION_NAMES.index(name) + 1
    indices = (0, 1) if name in PAIRWISE else (0, 0)

    for task_id, env in envs:
        env.reset()
        env.step(np.array([index, *indices]))
        for obj in env.objects:
            assert isinstance(obj.color_numbers, tuple), \
                f"{name!r} on {task_id} left {obj.label}.color_numbers a " \
                f"{type(obj.color_numbers).__name__}"


def test_an_alignment_that_would_leave_the_grid_is_refused():
    """Directly, because the fixture tasks do not reach it: the alignment
    transforms placed the cells that happened to land on the grid and
    dropped the rest, which erases the object from where it was and draws
    part of it where it was going.
    """
    from rl.arc_transformators import x_alignment
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 1
    for row in (2, 3, 4):
        grid[row, 4] = 2
    anchor = GridObject(shape="cell", coords=[(0, 0)], color=[1], label="a",
                         grid_shape=grid.shape, grid=grid)
    tall = GridObject(shape="line", coords=[(2, 4), (3, 4), (4, 4)], color=[2], label="b",
                       grid_shape=grid.shape, grid=grid)
    before = grid.copy()

    # Aligning the tall object's centre row (3) with the anchor's (0) shifts
    # it up by three, putting its top cell at row -1.
    result = x_alignment(grid, anchor, tall, font_color=0)

    assert np.array_equal(result, before)
    assert all(0 <= i < 6 and 0 <= j < 6 for i, j in tall.coords)


@pytest.mark.parametrize("transform", ["center_merge", "color_merge", "objects_swap"])
def test_a_placement_that_would_leave_the_grid_is_refused(transform):
    """Same all-or-nothing rule as the alignments, and reached the same way:
    these transforms move object 2 by the distance between the two centres,
    which for a distant or a much larger partner puts part of it past the
    border.

    They fail differently. objects_swap can leave an object with no cells at
    all, which takes reinit_obj down on min() of an empty sequence.
    color_merge is quieter and worse: a negative index is not an error to
    numpy, so it writes at the far edge and the run continues on a grid
    nobody asked for, until a shift large enough to raise finally names the
    wrong place.
    """
    from rl import arc_transformators
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 2
    for row in (2, 3, 4):
        grid[row, 4] = 2
    anchor = GridObject(shape="cell", coords=[(0, 0)], color=[2], label="a",
                         grid_shape=grid.shape, grid=grid)
    tall = GridObject(shape="line", coords=[(2, 4), (3, 4), (4, 4)], color=[2], label="b",
                       grid_shape=grid.shape, grid=grid)
    before = grid.copy()

    # Every one of them sends the tall object's centre (3, 4) to the
    # anchor's cell (0, 0): a shift of (-3, -4), putting its top cell at
    # row -1. The anchor's own half of the swap fits, which is the point -
    # one side being possible does not make the move possible.
    result = getattr(arc_transformators, transform)(grid, anchor, tall, font_color=0)

    assert np.array_equal(result, before)
    assert all(0 <= i < 6 and 0 <= j < 6 for i, j in tall.coords)
    assert all(0 <= i < 6 and 0 <= j < 6 for i, j in anchor.coords)


def test_an_upscale_that_would_leave_the_grid_is_refused():
    """Doubling puts a cell at twice its distance from the origin, so an
    object anywhere but the top left leaves the grid readily - and unlike
    the moves, an upscale that keeps only what fits is not even the right
    shape.

    A 2x2 block in the bottom-right corner of a 6x6 grid wants rows and
    columns 7 through 10.
    """
    from rl.arc_transformators import upscale
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((6, 6), dtype=int)
    block = [(4, 4), (4, 5), (5, 4), (5, 5)]
    for i, j in block:
        grid[i, j] = 3
    obj = GridObject(shape="square", coords=block, color=[3], label="a",
                      grid_shape=grid.shape, grid=grid)
    before = grid.copy()

    result = upscale(grid, obj, font_color=0)

    assert np.array_equal(result, before)
    assert sorted(obj.coords) == block


def _ring_inverted():
    """A ring keeps its hole and loses its eight-cell body to
    inverse_obj_color."""
    from rl.arc_transformators import inverse_obj_color
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((5, 5), dtype=int)
    ring = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    for i, j in ring:
        grid[i, j] = 1
    obj = GridObject(shape="ring", coords=ring, color=[1], label="a",
                      grid_shape=grid.shape, grid=grid)
    inverse_obj_color(grid, obj, font_color=0)
    return grid, obj


def _cell_emitted():
    """A single cell grows the three cells it emits before hitting the
    blocker."""
    from rl.arc_transformators import emission_with_collision
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((5, 5), dtype=int)
    grid[2, 0] = 1
    grid[2, 4] = 3
    obj = GridObject(shape="cell", coords=[(2, 0)], color=[1], label="a",
                      grid_shape=grid.shape, grid=grid)
    blocker = GridObject(shape="cell", coords=[(2, 4)], color=[3], label="b",
                          grid_shape=grid.shape, grid=grid)
    grid = emission_with_collision(grid, obj, emission_color=2, font_color=0,
                                   direction="E", collision_type="stop",
                                   collision_color=4,
                                   cell2obj={(2, 0): 0, (2, 4): 1},
                                   objects=[obj, blocker])
    return grid, obj


@pytest.mark.parametrize("build", [_ring_inverted, _cell_emitted],
                          ids=["inverse_obj_color", "emission_with_collision"])
def test_an_object_that_changes_shape_keeps_its_structure(build):
    """obj_structure numbers an object's cells by their index in coords, so
    the two are only meaningful together and every cell has to carry a
    number. Transforms that give an object a different set of cells go
    through reinit_obj, which recomputes both; assigning coords alone leaves
    a structure describing the shape the object no longer has.

    Either direction of the mismatch is a bug, and they fail differently. A
    structure numbering more cells than coords holds sends object_rotation
    past the end of the list. One numbering fewer is quieter: the rotation
    runs and silently leaves the unnumbered cells behind.
    """
    from rl.arc_transformators import symmetry_transformation

    grid, obj = build()

    assert obj.obj_structure.max() == len(obj.coords)
    # And the rotation that reads it gets through, with nothing dropped.
    symmetry_transformation(grid, obj, font_color=0, transf_type="rot90")
    assert obj.obj_structure.max() == len(obj.coords)


def test_a_shift_with_no_direction_in_its_name_is_refused():
    """The dispatch reads the direction off the end of the name, so a name
    that carries none leaves whatever sits in that position to be read as
    one. Refusing costs an action that does nothing, which the env already
    scores as ineffective; a KeyError takes the whole rollout down.

    Asked of the function directly, because the generated vocabulary now
    always appends a direction - this guards the case where the two ends
    disagree again, which is the case that produced KeyError('object').
    """
    from rl.arc_transformators import shift_object
    from symbolic.objects_analysis import GridObject

    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    obj = GridObject(shape="cell", coords=[(2, 2)], color=[1], label="a",
                      grid_shape=grid.shape, grid=grid)
    before = grid.copy()

    result = shift_object(grid, obj, direction="object", font_color=0)

    assert np.array_equal(result, before)
    assert obj.coords == ((2, 2),)


def test_the_stubs_are_still_stubs():
    """copy/copy_input/paste/cut return the grid untouched by design. Pinned
    so that becomes a deliberate state rather than a suspicion, and so
    implementing one is noticed here."""
    env = _task_with(min_objects=1)
    for name in STUBS:
        index = ACTION_NAMES.index(name) + 1
        env.reset()
        before = env.grid.copy()
        env.step(np.array([index, 0, 0]))
        assert np.array_equal(env.grid, before), f"{name} now does something"
