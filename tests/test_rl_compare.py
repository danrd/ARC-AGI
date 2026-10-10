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

    def test_every_object_arm_is_one_change_from_default(self):
        """Same observation as default, and one PPO setting that default does not set;
        the settings are keys the PPO config has."""
        from data.configs.rl_configs import load_PPO_config
        ppo = load_PPO_config()
        assert compare.OBJECT_ARMS and set(compare.OBJECT_ARMS) <= set(compare.ARMS)
        for arm in compare.OBJECT_ARMS:
            elements, settings = compare.arm_settings(arm)
            assert elements == compare.arm_settings("default")[0], arm
            assert len(settings) == 1 and set(settings) <= set(ppo), arm
            assert settings != {key: ppo[key] for key in settings}, arm

    def test_what_train_runs_by_default_is_the_first_series(self):
        """The object arms are run only on request: a command without --arms must not
        start the second series on every task."""
        series = (compare.OBSERVATION_ARMS, compare.OBJECT_ARMS, compare.GRID_ARMS, compare.MAP_ARMS,
                  compare.READ_ARMS)
        assert set().union(*map(set, series)) == set(compare.ARMS)
        assert sum(map(len, series)) == len(compare.ARMS)

    def test_the_third_series_changes_what_the_grid_and_deltas_do_and_nothing_else(self):
        from data.configs.rl_configs import load_PPO_config
        ppo = load_PPO_config()
        default = compare.arm_settings("default")[0]
        assert compare.GRID_ARMS and set(compare.GRID_ARMS) <= set(compare.ARMS)
        elements, settings = compare.arm_settings("s_ch")
        assert elements == default and settings == {"delta_in_grid": True} and ppo["delta_in_grid"] is False
        elements, settings = compare.arm_settings("s_nodelta")
        assert set(default) - set(elements) == set(compare.DELTAS) and settings == {"spatial_channels": 32}
        elements, settings = compare.arm_settings("s_chmap")
        assert elements == default and settings == {"delta_in_grid": True, "spatial_channels": 32}
        elements, settings = compare.arm_settings("s_grid")
        assert elements == compare.DELTAS and settings == {"spatial_channels": 32} and ppo["spatial_channels"] == 0

    def test_every_map_arm_is_o_spatial_with_at_most_one_change(self):
        """o_spatial is default with the map; the fourth series moves one thing from it."""
        base_elements, base_settings = compare.arm_settings("default")[0], {"spatial_channels": 32}
        assert compare.MAP_ARMS
        for arm in compare.MAP_ARMS:
            elements, settings = compare.arm_settings(arm)
            changed = {key for key in {*settings, *base_settings} if settings.get(key) != base_settings.get(key)}
            moved = len(changed) + (elements != base_elements)
            assert moved == 1, (arm, changed)
        assert set(compare.arm_settings("m_norel")[0]) == {"objects_emb", *compare.DELTAS}
        assert compare.arm_settings("m_nopos")[1]["object_arch"] == {"use_position": False}

    def test_every_reading_arm_is_o_spatial_with_one_spatial_arch_key_changed(self):
        from rl.features import SPATIAL_ARCH
        base_elements = compare.arm_settings("default")[0]
        assert set(compare.READ_ARMS) == {"r_deep", "r_shallow", "r_peak", "r_noctx", "r_nobox", "r_cells", "r_ring", "r_attn"}
        for arm in compare.READ_ARMS:
            elements, settings = compare.arm_settings(arm)
            assert elements == base_elements and settings["spatial_channels"] == 32, arm
            arch = settings["spatial_arch"]
            assert len(arch) == 1 and set(arch) <= set(SPATIAL_ARCH), arm
            assert arch != {key: SPATIAL_ARCH[key] for key in arch}, arm

    def test_a_train_command_without_arms_runs_the_first_series(self, tmp_path, monkeypatch):
        configs = tmp_path / "c.json"
        configs.write_text(json.dumps({"a": {}}))
        seen = {}
        monkeypatch.setattr(compare, "train", lambda configs, arms, *args, **kwargs: seen.update(arms=arms))
        monkeypatch.setattr(sys, "argv", ["rl_compare.py", "train", "--configs", str(configs)])
        compare.main()
        assert tuple(seen["arms"]) == compare.OBSERVATION_ARMS

    def test_a_train_command_runs_the_seeds_it_is_given_and_three_by_default(self, tmp_path, monkeypatch):
        configs = tmp_path / "c.json"
        configs.write_text(json.dumps({"a": {}}))
        seen = []
        monkeypatch.setattr(compare, "train", lambda configs, arms, *args, **kwargs: seen.append(kwargs["seeds"]))
        for extra in ([], ["--seeds", "4"], ["--seeds", "1", "2"]):
            monkeypatch.setattr(sys, "argv", ["rl_compare.py", "train", "--configs", str(configs), *extra])
            compare.main()
        assert seen == [(0, 1, 2), (4,), (1, 2)]

    def test_an_arms_settings_are_its_own_copy(self):
        """A run that edited its object_arch must not change the next arm's."""
        _elements, settings = compare.arm_settings("o_nopos")
        settings["object_arch"]["use_position"] = True
        assert compare.arm_settings("o_nopos")[1]["object_arch"]["use_position"] is False

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

    def test_the_shares_differ_in_size_by_at_most_one_however_few_runs_there_are(self):
        """The leftover of a wave is a few dozen runs; a hash taken modulo the workers
        left one worker with none and another with a third of them."""
        for runs in (7, 71, 100):
            grid = [(str(i), "g", 0) for i in range(runs)]
            sizes = [len(compare.share(grid, index, 8)) for index in range(8)]
            assert max(sizes) - min(sizes) <= 1 and sum(sizes) == runs

    def test_a_share_holds_every_arm_and_keeps_the_tasks_first_order(self):
        """By stride, three arms and three workers would hand each worker one
        arm: lose a notebook and an arm is gone."""
        grid = compare.run_grid({str(i): {} for i in range(30)}, ["g", "gd", "gd_ch"])
        for index in range(3):
            share = compare.share(grid, index, 3)
            assert {arm for _task, arm, _seed in share} == {"g", "gd", "gd_ch"}
            assert share == [run for run in grid if run in set(share)]

    def test_a_share_is_the_same_whatever_order_or_process_asks(self):
        """Hash by crc32 and not hash(), and by the runs themselves and not their position."""
        grid = compare.run_grid({str(i): {} for i in range(10)}, ["g", "gd"])
        assert compare.share(grid, 1, 4) == compare.share(list(reversed(grid)), 1, 4)[::-1]
        assert compare.share(grid, 1, 4) == compare.share(grid, 1, 4)


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

    def test_the_shares_are_cut_from_what_the_skip_files_leave_not_from_what_this_worker_wrote(
            self, tmp_path, monkeypatch):
        """Two notebooks of one wave, given the same skip file, between them do exactly
        what it leaves - and a notebook restarted with its own output does not
        move the cuts."""
        ran = []
        monkeypatch.setattr(compare, "one_run", self._fake(ran))
        configs = {str(i): {} for i in range(6)}
        other = tmp_path / "other.jsonl"
        other.write_text("".join(json.dumps({"task": str(i), "arm": "g", "seed": 0, "held_out": 0.0,
                                             "train": []}) + "\n" for i in range(3)))
        for worker in range(2):
            compare._work(configs, ["g"], 10, worker, 2, tmp_path / f"w{worker}.jsonl", (0, 1), None, skip=[other])
        left = [run for run in compare.run_grid(configs, ["g"], (0, 1)) if run[2] != 0 or int(run[0]) >= 3]
        assert sorted(ran) == sorted(left)
        before = len(ran)
        compare._work(configs, ["g"], 10, 0, 2, tmp_path / "w0.jsonl", (0, 1), None, skip=[other])
        assert len(ran) == before

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

    def test_a_run_records_how_long_it_trained(self, monkeypatch):
        """Runs of 250k and of 500k steps share a task, an arm and a seed: the
        record is what tells them apart."""
        import rl.training
        seen = {}

        def fake(task, config, ppo, show_plots):
            seen["steps"] = config["total_steps"]
            return {0: 1.0}, {0: 2.0}, None, {"test_acc": 0.5}

        monkeypatch.setattr(rl.training, "train_on_task", fake)
        monkeypatch.setattr(compare, "load_task", lambda task_id: None)
        config = {"feasible_actions": {"0": [0]}}
        record = compare.one_run(config, "default", 0, 500_000, "a")
        assert seen["steps"] == 500_000 and record["steps"] == 500_000


class TestWhatIsReadAsARun:
    def test_a_line_that_is_not_a_run_is_ignored_like_a_torn_one(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text('{"task": "a", "arm": "g", "seed": 0, "held_out": 1.0}\n'
                        '{"task": "a", "solved": true, "options": {}}\n'          # a line of the system's runs
                        '[1, 2]\n{"task": "b"\n')
        assert list(compare.read_runs(path)) == [("a", "g", 0)]

    def test_workers_that_die_make_the_shard_fail_and_not_look_done(self, tmp_path, monkeypatch):
        import pytest

        class Dead:
            exitcode = 1

            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def join(self):
                pass

        class Context:
            Process = Dead

        monkeypatch.setattr(compare.multiprocessing, "get_context", lambda name: Context)
        with pytest.raises(RuntimeError, match="2 of 2 workers died"):
            compare.train({"a": {}}, ["g"], 10, 0, 1, tmp_path / "o.jsonl", workers=2)
