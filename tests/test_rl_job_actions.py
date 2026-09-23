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

import numpy as np


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
        """Above the floor of two - see
        TestATaskWhoseGridsHoldOneObject, which is why there is one."""
        from rl.rl_job import object_slots

        two = [[5, 0, 5], [0, 0, 0], [0, 0, 0]]
        three = [[5, 0, 5], [0, 0, 0], [5, 0, 0]]

        assert object_slots(self._task(two, two), 1) == 2
        assert object_slots(self._task(two, three), 1) == 3

    def test_the_held_out_grid_counts_too(self):
        """One agent is scored on it, so a slot it needs and the training
        examples do not is a slot the space has to have."""
        from rl.rl_job import object_slots

        one = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        three = [[5, 0, 5], [0, 0, 0], [5, 0, 0]]

        assert object_slots(self._task(one, one, three), 1) == 3

    def test_the_config_gets_the_task_s_own_slot_count(self):
        import rl.search_hints as hints
        import pytest

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "x"})
        four = [[5, 0, 5], [0, 0, 0], [5, 0, 5]]
        try:
            config = narrowed_for_task(self._task(four, four), CONFIG)
        finally:
            monkeypatch.undo()

        assert config["max_objects"] == 4


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


class TestATaskWhoseGridsHoldOneObject:
    """An action is (transform, object, object) and the relations an
    observation carries are shaped (slots, (slots - 1) * RELATION_DIM): at
    one slot that block has width zero, and everything that reads it fails.

    Measured over the arm sweep: all eight tasks whose grids hold a single
    object crashed in all three arms carrying relations - ValueError
    "cannot reshape array of size 0 into shape (1,0)" for the combined and
    GNN extractors, RuntimeError from index_select for the separate one -
    and no task with two or more objects crashed in any arm.
    """

    @staticmethod
    def _one_object_task():
        import numpy as np

        class _Sub:
            def __init__(self, grid):
                self.train_inp = np.array(grid)
                self.train_inp_shape = self.train_inp.shape

        class _T:
            pass

        lone = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        task = _T()
        task.subtasks = [_Sub(lone), _Sub(lone)]
        task.test_subtask = _Sub(lone)
        return task

    def test_the_slots_never_fall_below_a_pair(self):
        from rl.rl_job import object_slots

        assert object_slots(self._one_object_task(), 1) == 2

    def test_the_relation_block_is_not_empty(self, monkeypatch):
        """The property the floor exists for, checked against the env that
        declares the block rather than against the number."""
        import rl.search_hints as hints
        from rl.arc_task import ARCSubtask
        from rl.rl_job import narrowed_for_task
        from rl.training import create_ARC_env
        import numpy as np

        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "black_recolor"})
        task = self._one_object_task()
        config = narrowed_for_task(task, CONFIG)
        subtask = ARCSubtask("lone_0", task.subtasks[0].train_inp,
                             np.array([[5, 5, 5], [5, 0, 5], [5, 5, 5]]))

        env = create_ARC_env(
            subtask, max_episode_len=5, feasible_actions={0: "submit"},
            observation_space_elements=["objects_emb", "relations_emb"],
            max_objects=config["max_objects"])
        env.reset()

        width = env.observation_space["relations_emb"].shape[1]
        assert width > 0, "a zero-width relation block is what broke"

    def test_a_task_with_more_objects_is_untouched(self):
        from rl.rl_job import object_slots
        import numpy as np

        class _Sub:
            def __init__(self, grid):
                self.train_inp = np.array(grid)
                self.train_inp_shape = self.train_inp.shape

        class _T:
            pass

        task = _T()
        three = [[5, 0, 5], [0, 0, 0], [5, 0, 0]]
        task.subtasks = [_Sub(three)]
        task.test_subtask = _Sub(three)

        assert object_slots(task, 1) == 3


class TestTheAgentDecidesWhatTheActionsName:
    """Object actions name connected components, coordinate actions name
    cells, and a task needs one or the other. The label decides the base
    case - see AGENT2ADDRESSING for the measurement behind it.
    """

    def test_constructor_gets_coordinates(self):
        """44% of its tasks cannot be moved by any single object action."""
        from rl.rl_job import addressing_for

        assert addressing_for("constructor") == "coordinates"

    def test_connector_stays_on_objects(self):
        """Its roster paints background by construction - paths between
        objects - and object actions move 69% of its tasks. Deciding it by
        the share of changed cells that were background put it on the wrong
        side once already."""
        from rl.rl_job import addressing_for

        assert addressing_for("connector") == "objects"

    def test_every_other_agent_and_no_agent_get_objects(self):
        from rl.rl_job import addressing_for

        for agent in ("highlighter", "modifier", "shifter", "generalizer", None):
            assert addressing_for(agent) == "objects"

    def test_no_roster_carries_a_coordinate_action(self):
        """An object-addressed env has no cells to hand these, so one in a
        roster would reach the object dispatch and do nothing - a slot in
        every search that enumerates the vocabulary, wasted."""
        from data.configs.env_configs import (ACTION_TYPES, AGENT2ACTIONS,
                                              COORDINATE_ACTIONS)

        rostered = {a for roster in AGENT2ACTIONS.values() for a in roster}
        typed = {a for group in ACTION_TYPES.values() for a in group}
        assert not (rostered | typed) & set(COORDINATE_ACTIONS)


class TestWhatACoordinateTaskIsNarrowedTo:

    @staticmethod
    def _task(agent, shapes=((5, 5), (5, 5)), test_shape=(5, 5), out_shapes=None):
        task = _Task(agent=agent, shapes=shapes, test_shape=test_shape)
        for index, subtask in enumerate(task.subtasks):
            subtask.train_out = np.zeros(
                out_shapes[index] if out_shapes else subtask.train_inp.shape, dtype=int)
            subtask.train_out[0, 0] = 2
        task.test_subtask.train_out = np.zeros(test_shape, dtype=int)
        return task

    def test_the_observation_drops_objects_and_carries_both_deltas(self):
        from rl.rl_job import observation_for

        assert observation_for(["objects_emb", "relations_emb"], "coordinates") == \
            ["delta_input", "delta_target"]

    def test_object_addressing_keeps_what_was_asked_for(self):
        from rl.rl_job import observation_for

        asked = ["objects_emb", "relations_emb", "target"]
        assert observation_for(asked, "objects") == asked

    def test_whatever_else_was_asked_for_is_kept(self):
        from rl.rl_job import observation_for

        kept = observation_for(["input_pattern", "objects_emb", "delta_input"], "coordinates")
        assert kept == ["input_pattern", "delta_input", "delta_target"]

    def test_the_shape_is_the_largest_grid_any_pair_works_on(self):
        """Input and output both, since an output larger than its input is
        what the working grid is padded out to - and the held-out pair."""
        from rl.rl_job import coordinate_shape

        task = self._task("constructor", shapes=((3, 4), (5, 2)), test_shape=(4, 6),
                          out_shapes=[(3, 7), (5, 2)])
        assert coordinate_shape(task) == (5, 7)

    def test_the_vocabulary_is_the_coordinate_one_in_the_outputs_colours(self, monkeypatch):
        """No object search - its findings are object actions and none of
        them belongs here."""
        import rl.search_hints as hints

        def refuse(*_a, **_k):
            raise AssertionError("the object search ran for a coordinate task")

        monkeypatch.setattr(hints, "feasible_from_search", refuse)
        config = narrowed_for_task(self._task("constructor"),
                                   dict(CONFIG, observation_space_elements=["objects_emb"]))
        names = set(config["feasible_actions"].values())
        assert names == {"submit", "black_fill", "red_fill", "black_line", "red_line",
                         "black_triangle", "red_triangle"}
        assert config["feasible_actions"][0] == "submit"

    def test_the_observation_is_padded_to_the_action_space(self, monkeypatch):
        """Even with every grid one size: the heads score the observed
        grid's rows and columns, and those have to be the action space's."""
        config = narrowed_for_task(self._task("constructor"), dict(CONFIG))
        assert config["addressing"] == "coordinates"
        assert config["coordinate_shape"] == (5, 5)
        assert config["observation_grid_shape"] == config["coordinate_shape"]

    def test_the_narrowed_config_is_one_the_worker_accepts(self, monkeypatch):
        """RlConfig forbids keys it does not declare, and _rl_training_worker
        builds one out of whatever narrowed_for_task returns. Adding a key
        to one and not the other broke every task the orchestration sent to
        RL - and the worker test above did not see it, because it replaces
        narrowed_for_task with a stub."""
        import rl.search_hints as hints
        from data.configs.rl_configs import rl_config
        from rl.rl_module import RlConfig

        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})
        for agent in ("constructor", "highlighter"):
            RlConfig(**narrowed_for_task(self._task(agent), rl_config))

    def test_the_config_and_its_pydantic_twin_declare_the_same_keys(self):
        from data.configs.rl_configs import rl_config
        from rl.rl_module import RlConfig

        assert set(rl_config) == set(RlConfig.model_fields)
