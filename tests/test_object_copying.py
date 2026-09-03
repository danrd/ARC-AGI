"""GridObject's copy semantics, and the equality that had to be fixed to
express them.

The search copies an object on every simulated step, and MCTS simulates
~118k of them for one task - measured at 88% of a run's wall clock under
the generic deepcopy, against 7% for the transforms themselves. So the copy
is not an implementation detail here; it is most of what the search does,
and __deepcopy__ shares everything immutable rather than walking it.

What that buys is only safe if the copy is still a copy. The tests below
pin both halves: the clone carries the same values, and nothing done to it
reaches back into the original.
"""
from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import pytest

from symbolic.objects_analysis import GridObject

#: Attributes that are meant to be independent buffers on a clone. The rest
#: are immutable and deliberately shared.
COPIED = ("obj_mask", "obj_structure", "color_structure")


def _ring(grid_size=5):
    """A ring - eight cells around a hole - so the object carries nested
    hole GridObjects, which are the part a shallow copy would share.

    Coordinates are numpy ints, as they are on a real object: the ones the
    grid analysis produces come out of numpy, and so do the ones every
    transform hands to reinit_obj. It matters for what the sharing test can
    see - copy._deepcopy_tuple returns the original tuple when every element
    copied to itself, which plain Python ints do and numpy ints do not, so a
    fixture built from `range` would pass that test under either
    implementation.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    coords = [(np.int64(i), np.int64(j))
              for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    for i, j in coords:
        grid[i, j] = 1
    return grid, GridObject(shape="ring", coords=coords, color=[1], label="a",
                             grid_shape=grid.shape, grid=grid)


def test_a_copy_carries_the_same_values():
    grid, obj = _ring()

    clone = deepcopy(obj)

    for name, value in vars(obj).items():
        other = getattr(clone, name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other), name
        elif name in ("inner_holes", "outer_holes"):
            assert len(value) == len(other), name
            assert all(a.coords == b.coords for a, b in zip(value, other)), name
        else:
            assert value == other, name


def test_a_copy_is_independent_of_the_original():
    """The property the simulator depends on: it steps a copied object and
    must not disturb the state the tree will branch from again."""
    grid, obj = _ring()
    clone = deepcopy(obj)
    before = obj.coords

    clone.reinit_obj([(0, 0), (0, 1)], grid)

    assert obj.coords == before
    assert clone.coords != before


@pytest.mark.parametrize("name", COPIED)
def test_the_arrays_are_separate_buffers(name):
    """Shared and then written to, these would corrupt the original in a way
    no assertion about values would catch until much later."""
    grid, obj = _ring()

    clone = deepcopy(obj)

    assert getattr(obj, name) is not getattr(clone, name)


def test_the_nested_holes_are_copied_too():
    grid, obj = _ring()
    assert obj.inner_holes, "the fixture is meant to have a hole"

    clone = deepcopy(obj)

    for original_hole, cloned_hole in zip(obj.inner_holes, clone.inner_holes):
        assert original_hole is not cloned_hole


def test_the_immutable_attributes_are_shared():
    """Not an implementation detail to be tolerated - it is the point. A
    tuple of tuples of ints cannot be changed, so a copy of one differs from
    the original in nothing but the time it took to make.
    """
    grid, obj = _ring()

    clone = deepcopy(obj)

    assert clone.coords is obj.coords
    assert clone.coords_offsets is obj.coords_offsets
    assert clone.color_numbers is obj.color_numbers


def test_copying_beats_the_generic_walk_it_replaced():
    """The one test that can tell the two implementations apart.

    Everything above holds under the generic deepcopy too, and has to -
    __deepcopy__ is an optimisation, and an optimisation that changed the
    answer would be a bug. What it changes is the cost, and the cost is why
    it exists: measured at 240us per object against 8us, and the search
    copies one on every simulated step.

    Against the generic walk rather than a wall-clock bound, because a
    fixed number of seconds is a claim about the machine. Both are timed
    here, back to back, on the same object - a loaded runner slows them
    equally and the ratio survives it.
    """
    grid, obj = _ring()
    copies = 300

    start = time.perf_counter()
    for _ in range(copies):
        deepcopy(obj)
    with_override = time.perf_counter() - start

    original = GridObject.__deepcopy__
    del GridObject.__deepcopy__
    try:
        start = time.perf_counter()
        for _ in range(copies):
            deepcopy(obj)
        generic = time.perf_counter() - start
    finally:
        GridObject.__deepcopy__ = original

    assert with_override * 3 < generic, (
        f"{generic / with_override:.1f}x faster than the generic walk, "
        f"expected at least 3x - is __deepcopy__ still being used?")


def test_two_objects_of_the_same_shape_and_colour_are_equal():
    """__eq__ compared self.color, which is not an attribute of anything -
    every comparison of two objects raised AttributeError instead of
    answering."""
    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 1
    grid[3, 3] = 1
    first = GridObject(shape="cell", coords=[(0, 0)], color=[1], label="a",
                        grid_shape=grid.shape, grid=grid)
    second = GridObject(shape="cell", coords=[(3, 3)], color=[1], label="b",
                         grid_shape=grid.shape, grid=grid)

    assert first == second
    assert first == deepcopy(first)
    assert first != "not an object"


def test_a_different_colour_is_not_equal():
    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 1
    grid[3, 3] = 2
    first = GridObject(shape="cell", coords=[(0, 0)], color=[1], label="a",
                        grid_shape=grid.shape, grid=grid)
    second = GridObject(shape="cell", coords=[(3, 3)], color=[2], label="b",
                         grid_shape=grid.shape, grid=grid)

    assert first != second


def test_objects_can_go_in_a_set():
    """Defining __eq__ without __hash__ leaves the class unhashable, so any
    set or dict keyed on objects raised TypeError. Hashing is by identity:
    two distinct objects can be equal here, and they are not
    interchangeable."""
    grid, obj = _ring()
    clone = deepcopy(obj)

    assert len({obj, clone}) == 2
