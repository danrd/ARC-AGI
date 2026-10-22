"""scripts/rl_evaluation.py, the evaluation-split RL runner: each shard's tasks, narrowed in parallel and then trained."""
import json

import pytest

from scripts import rl_evaluation as ev


def test_a_shard_takes_every_shards_th_task():
    tasks = list("abcdefg")
    assert ev.my_tasks(tasks, 1, 3) == ["b", "e"] and ev.my_tasks(tasks, 0, 1) == tasks
    assert sorted(sum((ev.my_tasks(tasks, k, 3) for k in range(3)), [])) == tasks


def fake_narrow(chunk, path, split):
    assert split == "evaluation"
    with open(path, "w") as handle:
        json.dump({t: {"split": split} for t in chunk}, handle)


def test_the_configs_of_all_the_processes_are_merged(tmp_path, monkeypatch):
    import multiprocessing

    class Inline:
        def __init__(self, target, args):
            self.target, self.args, self.exitcode = target, args, 0

        def start(self):
            self.target(*self.args)

        def join(self):
            pass

    context = type("C", (), {"Process": staticmethod(lambda target, args: Inline(target, args))})
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: context)
    monkeypatch.setattr(ev, "narrow", fake_narrow)
    configs = ev.narrowed(list("abcde"), 2, tmp_path)
    assert set(configs) == set("abcde") and all(c["split"] == "evaluation" for c in configs.values())
    assert len(list(tmp_path.glob("narrow_*.json"))) == 2


def test_more_processes_than_tasks_start_no_idle_ones(tmp_path, monkeypatch):
    import multiprocessing
    started = []

    class Inline:
        def __init__(self, target, args):
            self.target, self.args, self.exitcode = target, args, 0

        def start(self):
            started.append(self.args[0])
            self.target(*self.args)

        def join(self):
            pass

    context = type("C", (), {"Process": staticmethod(lambda target, args: Inline(target, args))})
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: context)
    monkeypatch.setattr(ev, "narrow", fake_narrow)
    ev.narrowed(["a", "b"], 4, tmp_path)
    assert started == [["a"], ["b"]]


def test_a_dead_narrowing_process_fails_the_run(tmp_path, monkeypatch):
    import multiprocessing

    class Dead:
        exitcode = 1

        def __init__(self, target, args):
            pass

        def start(self):
            pass

        def join(self):
            pass

    context = type("C", (), {"Process": staticmethod(lambda target, args: Dead(target, args))})
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: context)
    with pytest.raises(RuntimeError, match="narrowing processes died"):
        ev.narrowed(["a"], 1, tmp_path)


def test_main_trains_this_shards_tasks_with_the_time_narrowing_left(tmp_path, monkeypatch):
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps(list("abcd")))
    seen = {}
    monkeypatch.setattr(ev, "narrowed", lambda chunk, workers, directory: {t: {} for t in chunk})
    monkeypatch.setattr(ev, "train", lambda configs, arms, steps, shard, shards, out, seeds, workers, hours:
                        seen.update(configs=sorted(configs), arms=arms, shard=shard, shards=shards, seeds=seeds,
                                    hours=hours, out=out))
    monkeypatch.setattr("sys.argv", ["x", "--tasks-file", str(tasks), "--shard", "1", "--shards", "2", "--hours", "10",
                                     "--out", "o.jsonl"])
    ev.main()
    assert seen["configs"] == ["b", "d"] and seen["arms"] == ["o_spatial"] and seen["seeds"] == (0,)
    assert (seen["shard"], seen["shards"]) == (0, 1) and 9.9 < seen["hours"] <= 10 and seen["out"] == "o.jsonl"


def test_the_hour_spent_narrowing_is_taken_off_the_training_time(tmp_path, monkeypatch):
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps(["a"]))
    clock = iter([1000.0, 1000.0 + 3600, 1000.0 + 3600])
    monkeypatch.setattr(ev.time, "time", lambda: next(clock))
    seen = {}
    monkeypatch.setattr(ev, "narrowed", lambda chunk, workers, directory: {"a": {}})
    monkeypatch.setattr(ev, "train", lambda *a, **k: seen.update(hours=a[-1] if not k else k["hours"]))
    monkeypatch.setattr("sys.argv", ["x", "--tasks-file", str(tasks), "--hours", "10"])
    ev.main()
    assert seen["hours"] == pytest.approx(9.0)
