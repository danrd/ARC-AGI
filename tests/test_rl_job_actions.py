"""The action set a pipeline run actually trains on.

data.configs.rl_configs ships feasible_actions={0: 'submit'} - a
placeholder, not a vocabulary - and rl.rl_job._rl_training_worker passed it
through untouched, so a run started by the pipeline trained an agent whose
only move was to give up. Every measurement in this repository built the
action set by hand instead, which meant the thing measured was never the
thing the pipeline ran.

rl.rl_job.narrowed_for_task is the seam that closes that: the search
already knows which of a task's 137-607 generated actions ever moved the
grid (20-135 of them), and the roster of the agent a task is labelled with
narrows it to 9-13 where one exists.
"""
from __future__ import annotations


from rl.rl_job import narrowed_for_task


class _Subtask:
    def __init__(self, shape, grid=None):
        import numpy as np
        self.train_inp = np.zeros(shape, dtype=int) if grid is None else grid
        self.train_inp_shape = self.train_inp.shape


class _Task:
    """Two examples of one size and a held-out pair, unless told otherwise."""

    def __init__(self, agent=None, shapes=((3, 3), (3, 3)), test_shape=(3, 3)):
        if agent is not None:
            self.agent = agent
        self.subtasks = [_Subtask(shape) for shape in shapes]
        self.test_subtask = _Subtask(test_shape)


CONFIG = {"feasible_actions": {0: "submit"}, "seed": 42, "repr_level": 1,
          "max_objects": 16, "observation_grid_shape": None}


class TestWhatThePipelineTrainsOn:
    def test_the_placeholder_vocabulary_is_replaced(self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        config = narrowed_for_task(_Task(), CONFIG)

        assert config["feasible_actions"] == {0: "submit", 1: "red_recolor"}

    def test_the_rest_of_the_config_is_carried_through(self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        config = narrowed_for_task(_Task(), CONFIG)

        assert config["seed"] == 42

    def test_the_config_it_was_given_is_not_mutated(self, monkeypatch):
        """The shipped rl_config is a module-level dict: writing into it
        would change the vocabulary of every later run in the process."""
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        narrowed_for_task(_Task(), CONFIG)

        assert CONFIG["feasible_actions"] == {0: "submit"}

    def test_the_task_s_agent_label_reaches_the_search(self, monkeypatch):
        import rl.search_hints as hints
        seen = {}

        def capture(task, settings, agent=None):
            seen["agent"] = agent
            return {0: "submit", 1: "red_recolor"}

        monkeypatch.setattr(hints, "feasible_from_search", capture)

        narrowed_for_task(_Task(agent="connector"), CONFIG)

        assert seen["agent"] == "connector"

    def test_a_task_with_no_agent_label_still_searches(self, monkeypatch):
        import rl.search_hints as hints
        seen = {}

        def capture(task, settings, agent=None):
            seen["agent"] = agent
            return {0: "submit", 1: "red_recolor"}

        monkeypatch.setattr(hints, "feasible_from_search", capture)

        config = narrowed_for_task(_Task(), CONFIG)

        assert seen["agent"] is None
        assert config["feasible_actions"] == {0: "submit", 1: "red_recolor"}

    def test_a_failed_search_leaves_the_config_alone(self, monkeypatch):
        """Training over a wider space is a worse run; taking the pipeline
        down because a search raised is a lost one."""
        import rl.search_hints as hints

        def explode(*a, **k):
            raise RuntimeError("search died")

        monkeypatch.setattr(hints, "feasible_from_search", explode)

        config = narrowed_for_task(_Task(), CONFIG)

        assert config["feasible_actions"] == {0: "submit"}


def test_the_worker_narrows_before_it_trains(monkeypatch):
    """The wiring itself: _rl_training_worker is what the spawned process
    runs, and it is where the placeholder used to go through untouched."""
    import rl.rl_job as job
    import rl.rl_module as module
    seen = {}

    monkeypatch.setattr(job, "narrowed_for_task",
                        lambda task, config, settings=None:
                        {**config, "feasible_actions": {0: "submit", 1: "x"}})

    class _Module:
        def __init__(self, rl_config, ppo_config=None, *a, **k):
            seen["actions"] = rl_config.feasible_actions

        def solve(self, task):
            return {"solution": None}

    monkeypatch.setattr(module, "RLModule", _Module)

    job._rl_training_worker(_Task())

    assert seen["actions"] == {0: "submit", 1: "x"}


class TestSizingTheObjectSlots:
    """The action space is (transform, object, object) over max_objects
    slots whatever the task holds, so a slot past the objects a grid has is
    a legal action that does nothing. Over the 262 shape-preserving training
    tasks the median holds 3 objects against 16 slots, 251 need fewer than
    16, and only 3.5% of the (object, object) pairs name two real objects on
    the median task - a median factor of 28 in the pair space.
    """

    @staticmethod
    def _task(*grids):
        import numpy as np

        class _Sub:
            def __init__(self, grid):
                self.train_inp = np.array(grid)
                self.train_inp_shape = self.train_inp.shape

        class _T:
            pass

        task = _T()
        task.subtasks = [_Sub(g) for g in grids[:-1]]
        task.test_subtask = _Sub(grids[-1])
        return task

    def test_the_slots_are_the_objects_the_task_holds(self):
        from rl.rl_job import object_slots

        one = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        two = [[5, 0, 5], [0, 0, 0], [0, 0, 0]]

        assert object_slots(self._task(one, one), 1) == 1
        assert object_slots(self._task(one, two), 1) == 2

    def test_the_held_out_grid_counts_too(self):
        """One agent is scored on it, so a slot it needs and the training
        examples do not is a slot the space has to have."""
        from rl.rl_job import object_slots

        one = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        three = [[5, 0, 5], [0, 0, 0], [5, 0, 0]]

        assert object_slots(self._task(one, one, three), 1) == 3

    def test_the_config_gets_the_task_s_own_slot_count(self):
        from rl.rl_job import narrowed_for_task
        import rl.search_hints as hints
        import pytest

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "x"})
        one = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        try:
            config = narrowed_for_task(self._task(one, one), CONFIG)
        finally:
            monkeypatch.undo()

        assert config["max_objects"] == 1


class TestSizingTheObservation:
    """Examples of different sizes cannot share a rollout buffer while the
    observation carries a grid sized to one subtask - 133 of the 262
    shape-preserving training tasks, which never reached their first step.
    The observation is padded to a common shape and cropped back inside the
    extractor; the env's own grid is untouched.
    """

    def test_examples_of_one_size_need_no_padding(self, monkeypatch):
        """A shape here would only make the observation bigger than the
        task: ARC's 30x30 maximum is 7 times the median task's need."""
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "x"})

        config = narrowed_for_task(_Task(shapes=((3, 3), (3, 3)),
                                         test_shape=(3, 3)), CONFIG)

        assert config["observation_grid_shape"] is None

    def test_examples_of_different_sizes_are_padded_to_their_own_largest(
            self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "x"})

        config = narrowed_for_task(_Task(shapes=((9, 9), (10, 8)),
                                         test_shape=(12, 11)), CONFIG)

        assert config["observation_grid_shape"] == (12, 11)

    def test_the_held_out_grid_is_covered(self, monkeypatch):
        """It is bigger than every training example here, and an agent that
        cannot observe it cannot be scored on it."""
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "x"})

        config = narrowed_for_task(_Task(shapes=((3, 3), (4, 4)),
                                         test_shape=(20, 20)), CONFIG)

        assert config["observation_grid_shape"] == (20, 20)

    def test_a_failed_search_still_leaves_the_slots_and_the_shape(
            self, monkeypatch):
        """They are read off the task and cannot fail the way a search can,
        so losing the search should not cost them."""
        import rl.search_hints as hints

        def explode(*a, **k):
            raise RuntimeError("search died")

        monkeypatch.setattr(hints, "feasible_from_search", explode)

        config = narrowed_for_task(_Task(shapes=((9, 9), (10, 8)),
                                         test_shape=(12, 11)), CONFIG)

        assert config["observation_grid_shape"] == (12, 11)
        assert config["feasible_actions"] == {0: "submit"}
