"""Every transform arc_world dispatches, exercised under its real name.

Nothing covered this: the RL tests build environments with two or three
actions, so most of the vocabulary was never called from a test, and five
transforms raised on every invocation without anything noticing. MCTS
enumerates the whole action space, so one of these takes a whole rollout
down.

Names are not free-form, and getting one wrong looks exactly like a broken
transform. `World.parse_action` strips a leading colour word to get the
`add` argument, and several branches then split a direction or a second
colour out of what remains:

    rotate90                                     bare
    red_recolor                                  <colour>_<name>
    red_emission_N                               <colour>_<name>_<direction>
    red_emission_with_blue_object_recolor_N      two colours and a direction
    shift_N                                      direction, no colour
    contour_connection_red                       colour *after* the name

A name that matches no branch falls through and returns the grid unchanged,
which is why NAMES below is derived from the dispatch itself in
test_the_vocabulary_here_matches_the_dispatch - a transform added there and
not here would silently go untested.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from rl.arc_env import ARCGridWorld
from rl.arc_task import ARCSubtask

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Single-object transforms: dispatched when both action indices name the
#: same object.
SINGLE = [
    "rotate90", "fliplr", "flipud", "red_recolor", "upscale4",
    "red_outer_contour", "color_inversion", "edge_gravity",
    "edge_gravity_bottom", "red_color_inner_holes", "red_color_outer_holes",
    "shift_N", "red_color_inner_part", "symmetry_reflection",
    "symmetric_restoration", "red_dense_outer_contour",
    "red_emission_N", "red_emission_with_turn_left_collision_N",
    "red_emission_with_turn_right_collision_N",
    "red_emission_with_collision_stop_N",
    "red_emission_with_blue_object_recolor_N",
    "red_emission_with_blue_recolor_collision_N",
    "red_emission_with_blue_contour_collision_N",
]

#: Two-object transforms: dispatched when the indices name different objects.
PAIR = [
    "gravity", "x_alignment", "y_alignment", "shortest_path", "merge", "swap",
    "center_merge", "color_merge", "color_swap", "shape_swap", "color_copy",
    "shape_copy", "contour_connection_red", "shortest_path_left",
    "shortest_path_right",
]

#: Dispatched, but deliberately empty - each returns the grid untouched under
#: a "FOR FURTHER IMPLEMENTATION" comment. Kept out of the effect test rather
#: than out of the crash test: they cost a slot in every search that
#: enumerates the action space, which is worth seeing.
STUBS = ["copy", "copy_input", "paste", "cut"]

#: Transforms known to raise, with the diagnosis. Listed so the suite is
#: green on a known state rather than silent about it - remove an entry when
#: it is fixed, and the test starts demanding it stay fixed.
KNOWN_BROKEN = {
    "rotate90": "IndexError in GridObject.reinit_obj - rotation puts coords "
                 "outside the grid on non-square objects",
}

ACTION_NAMES = SINGLE + PAIR + STUBS
PAIRWISE = set(PAIR)


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
    """Several tasks, not one. These transforms raise on particular object
    shapes - gravity on 13 of 17 tasks, rotate90 on 1 of 27 - so a single
    fixture task reports them green while they are broken everywhere else.
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


def test_the_vocabulary_here_matches_the_dispatch():
    """A transform added to arc_world and not here would be tested by
    nothing, which is the state this file exists to end."""
    single_names, pair_names = _dispatch_names()
    covered = {n for n in ACTION_NAMES}

    for dispatched in single_names | pair_names:
        assert any(dispatched in name for name in covered), \
            f"{dispatched!r} is dispatched by arc_world but no test name reaches it"


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
