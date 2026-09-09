"""The action set a pipeline run actually trains on.

data.configs.rl_configs ships feasible_actions={0: 'submit'} - a
placeholder, not a vocabulary - and rl.rl_job._rl_training_worker passed it
through untouched, so a run started by the pipeline trained an agent whose
only move was to give up. Every measurement in this repository built the
action set by hand instead, which meant the thing measured was never the
thing the pipeline ran.

rl.rl_job.with_searched_actions is the seam that closes that: the search
already knows which of a task's 137-607 generated actions ever moved the
grid (20-135 of them), and the roster of the agent a task is labelled with
narrows it to 9-13 where one exists.
"""
from __future__ import annotations


from rl.rl_job import with_searched_actions


class _Task:
    def __init__(self, agent=None):
        if agent is not None:
            self.agent = agent


CONFIG = {"feasible_actions": {0: "submit"}, "seed": 42}


class TestWhatThePipelineTrainsOn:
    def test_the_placeholder_vocabulary_is_replaced(self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        config = with_searched_actions(_Task(), CONFIG)

        assert config["feasible_actions"] == {0: "submit", 1: "red_recolor"}

    def test_the_rest_of_the_config_is_carried_through(self, monkeypatch):
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        config = with_searched_actions(_Task(), CONFIG)

        assert config["seed"] == 42

    def test_the_config_it_was_given_is_not_mutated(self, monkeypatch):
        """The shipped rl_config is a module-level dict: writing into it
        would change the vocabulary of every later run in the process."""
        import rl.search_hints as hints
        monkeypatch.setattr(hints, "feasible_from_search",
                            lambda *a, **k: {0: "submit", 1: "red_recolor"})

        with_searched_actions(_Task(), CONFIG)

        assert CONFIG["feasible_actions"] == {0: "submit"}

    def test_the_task_s_agent_label_reaches_the_search(self, monkeypatch):
        import rl.search_hints as hints
        seen = {}

        def capture(task, settings, agent=None):
            seen["agent"] = agent
            return {0: "submit", 1: "red_recolor"}

        monkeypatch.setattr(hints, "feasible_from_search", capture)

        with_searched_actions(_Task(agent="connector"), CONFIG)

        assert seen["agent"] == "connector"

    def test_a_task_with_no_agent_label_still_searches(self, monkeypatch):
        import rl.search_hints as hints
        seen = {}

        def capture(task, settings, agent=None):
            seen["agent"] = agent
            return {0: "submit", 1: "red_recolor"}

        monkeypatch.setattr(hints, "feasible_from_search", capture)

        config = with_searched_actions(_Task(), CONFIG)

        assert seen["agent"] is None
        assert config["feasible_actions"] == {0: "submit", 1: "red_recolor"}

    def test_a_failed_search_leaves_the_config_alone(self, monkeypatch):
        """Training over a wider space is a worse run; taking the pipeline
        down because a search raised is a lost one."""
        import rl.search_hints as hints

        def explode(*a, **k):
            raise RuntimeError("search died")

        monkeypatch.setattr(hints, "feasible_from_search", explode)

        config = with_searched_actions(_Task(), CONFIG)

        assert config["feasible_actions"] == {0: "submit"}


def test_the_worker_narrows_before_it_trains(monkeypatch):
    """The wiring itself: _rl_training_worker is what the spawned process
    runs, and it is where the placeholder used to go through untouched."""
    import rl.rl_job as job
    import rl.rl_module as module
    seen = {}

    monkeypatch.setattr(job, "with_searched_actions",
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
