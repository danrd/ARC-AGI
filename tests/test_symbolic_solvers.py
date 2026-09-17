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


class _Solver:
    """A solver whose answers are scripted, so the check can be tested
    without a real task the real solvers happen to get right."""

    def __init__(self, on_test, on_held_out):
        self.on_test = on_test
        #: label of the held-out task -> what to answer for it. A callable
        #: gets the task and returns the answer.
        self.on_held_out = on_held_out
        self.calls = []

    def solve(self, task):
        self.calls.append(task.label)
        answer = (self.on_test if "holding-out" not in task.label
                  else self.on_held_out)
        if callable(answer):
            answer = answer(task)
        if answer is None:
            return SolveResult.fail("declined")
        return SolveResult.ok(answer)


def _task(label="t", pairs=3):
    """`pairs` training examples, each output the input plus one, and a
    test pair the same way - so a solver that has actually found the rule
    reproduces every held-out example."""
    from rl.arc_task import ARCSubtask, ARCTask

    subtasks = []
    for index in range(pairs):
        inp = np.full((3, 3), index, dtype=int)
        subtasks.append(ARCSubtask(f"{label}_{index}", inp, inp + 1))
    test_inp = np.full((3, 3), 9, dtype=int)
    return ARCTask(label=label, subtasks=subtasks, test_inp=test_inp,
                   test_out=test_inp + 1)


class TestAClaimIsHeldToTheTaskSOwnExamples:
    def test_a_rule_that_reproduces_every_example_is_kept(self):
        from symbolic.symbolic_module import checked_solve

        task = _task()
        solver = _Solver(on_test=task.test_out,
                         on_held_out=lambda t: t.test_out)

        result = checked_solve(solver, task)

        assert result.success
        assert np.array_equal(result.grid, task.test_out)

    def test_a_rule_that_contradicts_an_example_is_refused(self):
        from symbolic.symbolic_module import checked_solve

        task = _task()
        solver = _Solver(on_test=task.test_out,
                         on_held_out=lambda t: t.test_out + 5)

        result = checked_solve(solver, task)

        assert not result.success
        assert "wrong" in result.debug

    def test_a_rule_that_declines_a_held_out_example_is_refused(self):
        """The strict reading, and it is the one the numbers pick. Forgiving
        a decline keeps one more claim on the training split and it is a
        wrong one - 100.0% against 95.5% - and changes nothing on
        evaluation."""
        from symbolic.symbolic_module import checked_solve

        task = _task()
        solver = _Solver(on_test=task.test_out, on_held_out=None)

        result = checked_solve(solver, task)

        assert not result.success
        assert "declined" in result.debug

    def test_a_solver_that_already_failed_is_not_checked(self):
        """Nothing to verify, and the check costs a solve per example."""
        from symbolic.symbolic_module import checked_solve

        class _Refuses:
            def __init__(self):
                self.calls = 0

            def solve(self, task):
                self.calls += 1
                return SolveResult.fail("nope")

        solver = _Refuses()
        result = checked_solve(solver, _task())

        assert not result.success
        assert solver.calls == 1, "the check ran anyway"

    def test_the_check_never_sees_the_test_answer(self):
        """The whole reason it is available at inference. A solver asked
        for a held-out pair is handed the other examples and that pair's
        input - never the task's own test output."""
        from symbolic.symbolic_module import _holding_out

        task = _task(pairs=3)

        for index in range(3):
            variant = _holding_out(task, index)
            assert len(variant.subtasks) == 2
            assert not any(np.array_equal(s.train_out, task.test_out)
                           for s in variant.subtasks)
            assert np.array_equal(variant.test_inp,
                                  task.subtasks[index].train_inp)
            assert not np.array_equal(variant.test_out, task.test_out)

    def test_the_dispatcher_applies_the_check(self):
        """The wiring, asked of the dispatcher and not of the checker.

        Every other test here passes with _dispatch_symbolic calling
        `solve` directly - which is exactly the state this change is
        undoing, and it is the one mutation the rest of the suite did not
        notice.
        """
        from types import SimpleNamespace

        from orchestration.graph import _dispatch_symbolic

        task = _task()
        claims_and_contradicts = _Solver(on_test=task.test_out,
                                         on_held_out=lambda t: t.test_out + 5)
        refuses = _Solver(on_test=None, on_held_out=None)
        module = SimpleNamespace(mixer=claims_and_contradicts,
                                 upscale_or_covering=refuses,
                                 color_restore=refuses)

        answer = _dispatch_symbolic(task, module)

        assert answer["solution"] == "", (
            "a claim its own examples contradict reached the dispatcher's "
            "output")
        assert "module_results" in answer

    def test_the_dispatcher_still_returns_a_claim_that_checks_out(self):
        """The other half: the check must not make the dispatcher answer
        nothing at all."""
        from types import SimpleNamespace

        from orchestration.graph import _dispatch_symbolic

        task = _task()
        sound = _Solver(on_test=task.test_out, on_held_out=lambda t: t.test_out)
        refuses = _Solver(on_test=None, on_held_out=None)
        module = SimpleNamespace(mixer=sound, upscale_or_covering=refuses,
                                 color_restore=refuses)

        answer = _dispatch_symbolic(task, module)

        assert np.array_equal(answer["solution"], task.test_out)

    def test_every_example_is_held_out_in_turn(self):
        from symbolic.symbolic_module import checked_solve

        task = _task(pairs=4)
        solver = _Solver(on_test=task.test_out,
                         on_held_out=lambda t: t.test_out)

        checked_solve(solver, task)

        held = [c for c in solver.calls if "holding-out" in c]
        assert len(held) == 4, f"only {held} were checked"


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
