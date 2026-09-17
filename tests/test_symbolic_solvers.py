"""What the solvers in symbolic/symbolic_module.py refuse, and how fast.

A solver's contract is that `.solve(task)` comes back with a grid or with a
reason. Nothing in that contract says it comes back at all, and
MixerSolver's colour-mix search did not: it tries every ordering of the
segments against every augmentation, so its cost is
`len(segments)! * len(AUGS)`. Two training tasks hand it sixteen segments
and one hands it thirty-six. Sixteen is 20,922,789,888,000 orderings; at
the 12-15 thousand candidates a second measured on this machine, that is
some thirty thousand years, and every run of this module so far has worked
around it with an alarm in the calling script.

The bound is counted, not timed, and these tests hold it to that: a
refusal has to be a property of the task, so that the same task gives the
same answer in a different process on a different day.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from rl.arc_task import ARCSubtask, ARCTask
from symbolic.symbolic_module import MixerSolver, SolveResult


def _segments(count, side=6, colour=3):
    """`count` little grids, each with one non-background cell.

    Distinct enough to be segments and small enough that a search over them
    costs nothing - the tests here are about how many orderings are tried,
    not about what any one of them computes.
    """
    built = []
    for index in range(count):
        segment = np.zeros((side, side), dtype=int)
        segment[index % side, (index * 2) % side] = colour
        built.append(segment)
    return built


class TestTheColourMixSearchDeclinesWhatItCannotFinish:
    def test_a_search_it_can_afford_is_run(self):
        """The bound must not be a blanket refusal - four segments is 24
        orderings, which is the size 11 of the 38 training tasks that reach
        this search actually have."""
        segments = _segments(4)
        target = np.zeros((6, 6), dtype=int)

        result = MixerSolver()._color_mix_search(segments, target, None)

        # False means "searched and found nothing", which is a real answer
        # and not a refusal - the refusal path raises.
        assert result is False

    def test_too_many_segments_is_declined_rather_than_attempted(self):
        """The count comes from the module, not from a number written here.

        A hard-coded count is what the first version of this test used, and
        it was a trap: raise the bound and the test stops failing and
        starts *running the search* - an hour of it - so the mutation that
        broke the bound was caught by the suite hanging rather than by a
        failure. Asking the module which counts it declines keeps this
        fast whatever the bound is set to.
        """
        from symbolic.symbolic_module import (_TooManyOrderings,
                                              affordable_orderings)

        declined = next(n for n in range(2, 40) if not affordable_orderings(n))
        segments = _segments(declined)
        target = np.zeros((6, 6), dtype=int)

        began = time.perf_counter()
        with pytest.raises(_TooManyOrderings) as refusal:
            MixerSolver()._color_mix_search(segments, target, None)
        took = time.perf_counter() - began

        assert took < 1.0, f"the refusal itself took {took:.1f}s"
        assert f"{declined} segments" in str(refusal.value)

    @pytest.mark.parametrize("count,affordable", [
        (6, True),    # 720 orderings x augmentations - 0.3s measured
        (7, True),    # 5,040 - about 2.4s
        (8, False),   # 40,320 - 20.6s measured, and no training task solves
        (9, False),   # 362,880 - about three minutes
        (16, False),  # what 06df4c85 and bda2d7a6 hand it
    ])
    def test_the_bound_falls_between_seven_and_eight_segments(self, count,
                                                              affordable):
        """Where the line sits is the whole decision, so it is pinned here
        rather than left implicit in a constant. The training split has 32
        tasks at five segments or fewer, nothing at six or seven, and six
        tasks above - all of which the solver refuses anyway.

        Asked of the module's own function. The first version of this test
        recomputed `factorial(count) * len(AUGS)` on this side and compared
        that to the constant, which is two copies of one formula agreeing
        with each other: it passed unchanged when the module was made to
        count segments linearly instead of factorially.
        """
        from symbolic.symbolic_module import affordable_orderings

        assert affordable_orderings(count) is affordable

    def test_a_declined_search_becomes_a_failed_result_not_a_raise(self):
        """The module's contract is a SolveResult either way. A refusal that
        escaped as an exception would reach the dispatcher as a crash, and
        a solver declining a task it cannot afford is not a crash."""
        grids = [np.zeros((12, 12), dtype=int) for _ in range(3)]
        for index, grid in enumerate(grids):
            grid[::2, ::2] = index + 1
        task = ARCTask(
            label="declined",
            subtasks=[ARCSubtask(f"declined_{i}", g, g.copy())
                      for i, g in enumerate(grids)],
            test_inp=grids[0].copy(), test_out=grids[0].copy())

        solver = MixerSolver()
        result = solver.solve(task)

        assert isinstance(result, SolveResult)
        assert isinstance(result.debug, str) and result.debug

    def test_the_bound_is_counted_and_not_timed(self):
        """A wall-clock cutoff would make the answer depend on how loaded
        the machine is, and two runs of this pipeline on the same seed give
        the same number today. Asked of the refusal itself: it names a
        count, and it fires on a slow machine and a fast one alike."""
        from symbolic.symbolic_module import (_TooManyOrderings,
                                              affordable_orderings)

        declined = next(n for n in range(2, 40) if not affordable_orderings(n))
        segments = _segments(declined)
        target = np.zeros((6, 6), dtype=int)

        messages = set()
        for _ in range(3):
            with pytest.raises(_TooManyOrderings) as refusal:
                MixerSolver()._color_mix_search(segments, target, None)
            messages.add(str(refusal.value))

        assert len(messages) == 1, f"the refusal varies between runs: {messages}"
        assert "orderings" in messages.pop()
