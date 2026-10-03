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
import types
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

    def test_the_deltas_are_the_only_difference_between_objrel_and_default(self):
        """Default against objonly changes the relations and the deltas at once;
        objrel is the arm that separates them."""
        assert set(compare.ARMS["default"][0]) - set(compare.ARMS["objrel"][0]) == set(compare.DELTAS)
        assert set(compare.ARMS["objrel"][0]) - set(compare.ARMS["objonly"][0]) == {"relations_emb"}
        assert compare.ARMS["default"][1] == compare.ARMS["objrel"][1] == {}

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

    def test_the_shares_between_them_run_everything_once(self):
        grid = compare.run_grid({str(i): {} for i in range(20)}, ["g", "gd", "gd_ch"])
        shares = [compare.share(grid, index, 4) for index in range(4)]
        assert sorted(run for share in shares for run in share) == sorted(grid)
        assert all(not set(a) & set(b) for a in shares for b in shares if a is not b)

    def test_a_share_holds_every_arm_and_keeps_the_tasks_first_order(self):
        """By stride, three arms and three workers would hand each worker one
        arm: lose a notebook and an arm is gone."""
        grid = compare.run_grid({str(i): {} for i in range(30)}, ["g", "gd", "gd_ch"])
        for index in range(3):
            share = compare.share(grid, index, 3)
            assert {arm for _task, arm, _seed in share} == {"g", "gd", "gd_ch"}
            assert share == [run for run in grid if run in set(share)]

    def test_a_share_is_the_same_on_every_machine(self):
        """crc32 of the run, not hash(): the latter differs between processes."""
        import zlib
        assert compare.share([("a", "g", 0)], zlib.crc32(b"a|g|0") % 5, 5) == [("a", "g", 0)]
        assert compare.share([("a", "g", 0)], (zlib.crc32(b"a|g|0") + 1) % 5, 5) == []


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
            return {"task": task_id, "arm": arm, "seed": seed, "held_out": 0.0, "train": [0.0],
                    "seconds": 1.0}

        monkeypatch.setattr(compare, "one_run", fake)
        compare.train({"a": {}}, ["g"], 10, 0, 1, path, seeds=(0, 1))
        assert ran == [("a", "g", 1)]
        assert len(path.read_text().splitlines()) == 2

    def _fake(self, ran, seconds=1.0):
        def fake(config, arm, seed, steps, task_id):
            ran.append((task_id, arm, seed))
            return {"task": task_id, "arm": arm, "seed": seed, "held_out": 0.0, "train": [0.0],
                    "seconds": seconds}
        return fake

    def test_a_shard_runs_only_its_share_and_the_shards_together_run_the_grid(self, tmp_path, monkeypatch):
        ran = []
        monkeypatch.setattr(compare, "one_run", self._fake(ran))
        configs = {"a": {}, "b": {}}
        for shard in range(3):
            compare.train(configs, ["g", "gd"], 10, shard, 3, tmp_path / f"{shard}.jsonl", seeds=(0, 1))
        assert sorted(ran) == sorted(compare.run_grid(configs, ["g", "gd"], seeds=(0, 1)))
        assert len(ran) == len(set(ran)) == 8

    def test_workers_split_the_shard_between_them(self, tmp_path, monkeypatch):
        """Two notebooks of two workers: worker w of shard k is number 2k+w of 4."""
        ran = []
        monkeypatch.setattr(compare, "one_run", self._fake(ran))
        configs = {str(i): {} for i in range(6)}
        grid = compare.run_grid(configs, ["g", "gd"])
        for shard in range(2):
            for w in range(2):
                compare._work(configs, ["g", "gd"], 10, shard * 2 + w, 4, tmp_path / f"{shard}{w}.jsonl",
                              (0, 1, 2), None)
        assert sorted(ran) == sorted(grid)

    def test_it_stops_before_a_run_that_would_end_after_the_deadline(self, tmp_path, monkeypatch):
        """1000 s left: the first run is assumed to take FIRST_RUN_SECONDS (900), takes 500, and
        the next fits exactly; with 0 left the longest seen (500) does not."""
        clock = [0.0]
        ran = []

        def fake(config, arm, seed, steps, task_id):
            ran.append((task_id, arm, seed))
            clock[0] += 500.0
            return {"task": task_id, "arm": arm, "seed": seed, "held_out": 0.0, "train": [0.0],
                    "seconds": 500.0}

        monkeypatch.setattr(compare, "one_run", fake)
        monkeypatch.setattr(compare, "time", types.SimpleNamespace(time=lambda: clock[0]))
        compare._work({"a": {}, "b": {}}, ["g"], 10, 0, 1, tmp_path / "o.jsonl", (0, 1, 2), 1000.0)
        assert len(ran) == 2

    def test_runs_in_the_skip_files_are_done(self, tmp_path, monkeypatch):
        ran = []
        monkeypatch.setattr(compare, "one_run", self._fake(ran))
        other = tmp_path / "other.jsonl"
        other.write_text("".join(json.dumps({"task": "a", "arm": "g", "seed": seed, "held_out": 0.0,
                                             "train": []}) + "\n" for seed in (0, 1)))
        compare._work({"a": {}}, ["g"], 10, 0, 1, tmp_path / "o.jsonl", (0, 1, 2), None, skip=[other])
        assert ran == [("a", "g", 2)]

    def test_each_worker_of_a_notebook_gets_its_own_number_of_all_the_workers(self, tmp_path, monkeypatch):
        """Notebook 1 of 3 with 2 workers: they are workers 2 and 3 of 6, and the
        deadline and the skip files go to both."""
        started = []

        class Process:
            def __init__(self, target, args):
                self.args = args

            def start(self):
                started.append(self.args)

            def join(self):
                pass

        monkeypatch.setattr(compare.multiprocessing, "get_context",
                            lambda method: types.SimpleNamespace(Process=Process))
        compare.train({"a": {}}, ["g"], 10, 1, 3, tmp_path / "o.jsonl", workers=2, hours=2.0, skip=["s"])
        assert [(args[3], args[4]) for args in started] == [(2, 6), (3, 6)]
        assert all(args[7] is not None and args[8] == ["s"] for args in started)

    def test_no_deadline_means_every_run(self, tmp_path, monkeypatch):
        ran = []
        monkeypatch.setattr(compare, "one_run", self._fake(ran, seconds=10 ** 6))
        compare._work({"a": {}}, ["g"], 10, 0, 1, tmp_path / "o.jsonl", (0, 1, 2), None)
        assert len(ran) == 3


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
