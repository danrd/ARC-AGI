"""Tests for scripts/rl_compare.py - ways of building an agent's observation
and features, compared on the same tasks and seeds.

What is pinned is what an experiment needs to be worth reading: every arm is
what it says it is (the wide arm was once a copy of another and nothing
noticed), the grid of runs is split between shards with none run twice or
missed, a run cut short resumes where it stopped, and the gate precision the
orchestrator's RL step depends on is computed as it is defined.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rl_compare", ROOT / "scripts" / "rl_compare.py")
compare = importlib.util.module_from_spec(SPEC)
sys.modules["rl_compare"] = compare
SPEC.loader.exec_module(compare)


class TestTheArms:
    def test_every_arm_names_observations_and_settings(self):
        for arm, (elements, settings) in compare.ARMS.items():
            assert isinstance(elements, list) and isinstance(settings, dict), arm

    def test_the_grid_only_arm_has_no_observation_beyond_the_grid(self):
        assert compare.arm_settings("g")[0] == []

    def test_the_deltas_arms_differ_by_where_the_deltas_go(self):
        gd, ch = compare.arm_settings("gd"), compare.arm_settings("gd_ch")
        assert gd[0] == ch[0] == compare.DELTAS
        assert "delta_in_grid" not in gd[1] and ch[1]["delta_in_grid"] is True

    def test_the_wide_arm_builds_a_wider_grid_encoder_than_the_narrow_one(self):
        """It ran as gd_ch once: its width was a setting nothing read."""
        from rl.features import ARCCombinedExtractor

        def width(arm):
            _elements, settings = compare.arm_settings(arm)
            space = spaces.Dict({"grid": spaces.Box(0, 10, shape=(6, 6), dtype=np.int64),
                                 "delta_input": spaces.Box(0, 1, shape=(6, 6), dtype=np.int64)})
            extractor = ARCCombinedExtractor(space, extr_arch=settings.get("extr_arch"),
                                             delta_in_grid=settings.get("delta_in_grid", False))
            return extractor.features_dim

        assert compare.arm_settings("gd_wide")[1]["extr_arch"] is not None
        assert width("gd_wide") == 32 * 9 and width("gd_ch") == 16 * 9

    def test_the_default_arm_is_what_rl_config_observes(self):
        from data.configs.rl_configs import rl_config
        assert compare.arm_settings("default")[0] == list(rl_config["observation_space_elements"])

    def test_the_task_lists_are_tasks_the_first_holds(self):
        assert set(compare.SHORT_TASKS) <= set(compare.DEFAULT_TASKS)


class TestTheGrid:
    CONFIGS = {"b": {}, "a": {}, "c": {}}

    def test_tasks_come_first_so_a_partial_file_covers_every_arm_on_the_tasks_it_reached(self):
        grid = compare.run_grid(self.CONFIGS, ["g", "gd"], seeds=(0, 1))
        assert [task for task, _arm, _seed in grid[:4]] == ["a"] * 4
        assert len(grid) == 3 * 2 * 2 and len(set(grid)) == len(grid)

    def test_the_shards_between_them_run_everything_once(self):
        grid = compare.run_grid(self.CONFIGS, ["g", "gd", "gd_ch"])
        shards = [grid[shard::4] for shard in range(4)]
        assert sorted(run for shard in shards for run in shard) == sorted(grid)
        assert all(not set(a) & set(b) for a in shards for b in shards if a is not b)


class TestResuming:
    def test_a_torn_last_line_is_ignored_and_a_missing_file_is_empty(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        path.write_text(json.dumps({"task": "a", "arm": "g", "seed": 0, "held_out": 1.0, "train": []})
                        + '\n{"task": "b", "ar')
        assert set(compare.read_runs(path)) == {("a", "g", 0)}
        assert compare.read_runs(tmp_path / "none.jsonl") == {}

    def test_a_run_already_in_the_file_is_not_run_again(self, tmp_path, monkeypatch):
        path = tmp_path / "runs.jsonl"
        path.write_text(json.dumps({"task": "a", "arm": "g", "seed": 0, "held_out": 1.0, "train": [1.0]}) + "\n")
        ran = []

        def fake(config, arm, seed, steps, task_id):
            ran.append((task_id, arm, seed))
            return {"task": task_id, "arm": arm, "seed": seed, "held_out": 0.0, "train": [0.0]}

        monkeypatch.setattr(compare, "one_run", fake)
        compare.train({"a": {}}, ["g"], 10, 0, 1, path, seeds=(0, 1))
        assert ran == [("a", "g", 1)]
        assert len(path.read_text().splitlines()) == 2

    def test_a_shard_runs_only_its_share_and_the_shards_together_run_the_grid(self, tmp_path, monkeypatch):
        ran = []
        monkeypatch.setattr(compare, "one_run", lambda config, arm, seed, steps, task_id: (
            ran.append((task_id, arm, seed)) or
            {"task": task_id, "arm": arm, "seed": seed, "held_out": 0.0, "train": [0.0]}))
        configs = {"a": {}, "b": {}}
        for shard in range(3):
            compare.train(configs, ["g", "gd"], 10, shard, 3, tmp_path / f"{shard}.jsonl", seeds=(0, 1))
        assert sorted(ran) == sorted(compare.run_grid(configs, ["g", "gd"], seeds=(0, 1)))
        assert len(ran) == len(set(ran)) == 8


class TestNarrowing:
    def test_each_task_is_written_with_the_vocabulary_it_was_narrowed_to(self, tmp_path, monkeypatch):
        import rl.rl_job as job
        monkeypatch.setattr(compare, "load_task", lambda task_id: type("T", (), {"agent": "shifter"})())
        monkeypatch.setattr(job, "narrowed_for_task", lambda task, config: {
            "addressing": "objects", "feasible_actions": {0: "submit", 1: "blue_recolor"}})
        out = tmp_path / "c.json"
        compare.narrow(["x", "y"], out)
        written = json.loads(out.read_text())
        assert set(written) == {"x", "y"} and written["x"]["feasible_actions"] == {"0": "submit", "1": "blue_recolor"}


def run(task, arm, held_out, train):
    return {"task": task, "arm": arm, "seed": 0, "held_out": held_out, "train": train}


class TestSummary:
    RUNS = [run("a", "g", 1.0, [1.0, 1.0]), run("a", "g", -0.4, [1.0, 1.0]), run("b", "g", 0.0, [0.0, 1.0]),
            run("a", "gd", 1.0, [1.0]), run("b", "gd", 1.0, [1.0])]

    def test_the_mean_is_over_tasks_and_the_solved_count_over_runs(self):
        summary = compare.summarise(self.RUNS)
        assert summary["g"]["tasks"] == 2 and summary["g"]["runs"] == 3 and summary["g"]["solved_runs"] == 1
        assert summary["g"]["mean_over_tasks"] == round((np.mean([1.0, -0.4]) + 0.0) / 2, 3)

    def test_the_gate_is_every_training_pair_closed_and_its_precision_is_the_held_out_share(self):
        summary = compare.summarise(self.RUNS)
        assert summary["g"]["gated"] == 2 and summary["g"]["gate_precision"] == 0.5
        assert summary["gd"]["gate_precision"] == 1.0

    def test_an_arm_no_run_passed_the_gate_has_no_precision(self):
        assert compare.summarise([run("a", "g", 0.0, [0.0])])["g"]["gate_precision"] is None

    def test_the_table_lists_every_arm(self):
        text = compare.render(compare.summarise(self.RUNS))
        assert "\ng " in text and "\ngd " in text and "precision" in text


class TestSelectingTasks:
    RECORDS = [{"task": "d", "peak": 1.0},
               {"task": "a", "peak": 0.0},                       # nothing reachable: nothing to compare
               {"task": "c", "agents": {"x": {"peak": 0.0}, "y": {"peak": 0.7}}},   # best over rosters
               {"task": "b", "peak": 0.4},
               {"task": "e", "skipped": "the grids change size"},
               {"task": "f", "agents": {}}]

    def test_the_tasks_the_search_got_somewhere_on_are_chosen_sorted(self):
        assert compare.select_tasks(self.RECORDS) == ["b", "c", "d"]

    def test_the_tasks_symbolic_solves_are_left_out(self):
        assert compare.select_tasks(self.RECORDS, excluded=["c"]) == ["b", "d"]

    def test_a_census_peak_and_a_roster_peak_are_read_alike(self):
        assert compare.best_peak({"peak": 0.3}) == 0.3
        assert compare.best_peak({"agents": {"x": {"peak": 0.3}, "y": {"peak": 0.9}}}) == 0.9
        assert compare.best_peak({}) == 0.0

    def test_a_run_is_the_full_length_by_default(self):
        assert compare.STEPS == 250_000
