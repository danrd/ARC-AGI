"""The notebook's worker: the command of a step, and the runs going to a branch and coming back."""
import json
import subprocess

from scripts import kaggle_worker as worker


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


def make_clone(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    git(clone, "remote", "add", "origin", str(origin))
    (clone / "README").write_text("x")
    git(clone, "add", "README")
    git(clone, "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-qm", "init")
    return origin, clone


class TestTheCommand:
    def test_a_step_gets_its_shard_and_its_own_file(self):
        step = {"name": "panel", "args": ["--arms", "default"]}
        command = worker.train_command("j", step, 2, 5)
        assert command[1:4] == ["scripts/rl_compare.py", "train", "--arms"]
        assert command[-6:] == ["--shard", "2", "--shards", "5", "--out", str(worker.out_path("j", step, 2))]
        assert worker.out_path("j", step, 2) != worker.out_path("j", step, 3)
        assert worker.out_path("j", step, 2) != worker.out_path("j", {"name": "other"}, 2)

    def test_a_step_that_is_another_script_gets_its_places_filled_in(self):
        step = {"name": "p", "command": ["scripts/x.py", "--out", "{out}", "--shard", "{shard}/{shards}"]}
        command = worker.train_command("j", step, 1, 3)
        assert command[1:3] == ["scripts/x.py", "--out"] and command[4:] == ["--shard", "1/3"]
        assert command[3] == str(worker.out_path("j", step, 1).with_suffix("")) and "." not in command[3].split("/")[-1]

    def test_a_secret_is_read_under_the_name_the_step_gives(self, monkeypatch):
        import sys
        import types
        client = types.SimpleNamespace(get_secret=lambda name: {"OpenRouter": "k"}[name])
        monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=lambda: client))
        env = worker.secrets_env({"secrets": {"OPENROUTER_API_KEY": "OpenRouter"}}, None)
        assert env["OPENROUTER_API_KEY"] == "k"

    def test_a_notebook_has_its_own_branch(self):
        assert worker.branch_of("j", 0) != worker.branch_of("j", 1)

    def test_a_job_is_its_file_of_steps(self, tmp_path):
        (tmp_path / "data/experiments/jobs").mkdir(parents=True)
        (tmp_path / "data/experiments/jobs/j.json").write_text(json.dumps({"steps": [{"name": "a", "args": []}]}))
        assert worker.load_job("j", tmp_path) == [{"name": "a", "args": []}]


class TestThePushes:
    def test_the_runs_reach_the_branch_and_a_fresh_clone_gets_them_back(self, tmp_path):
        origin, clone = make_clone(tmp_path)
        (clone / worker.RUNS).mkdir(parents=True)
        (clone / worker.RUNS / "j_a_0.jsonl").write_text('{"task": "t"}\n')
        assert worker.publish(clone, "j", 0, remote=str(origin))
        assert worker.branch_of("j", 0) in git(origin, "branch", "--list", "kaggle/*")

        again = tmp_path / "again"
        subprocess.run(["git", "clone", "-q", str(origin), str(again)], check=True)
        git(again, "remote", "set-url", "origin", str(origin))
        worker.restore(again, "j", 0)
        assert (again / worker.RUNS / "j_a_0.jsonl").read_text() == '{"task": "t"}\n'

    def test_nothing_new_is_not_pushed(self, tmp_path):
        origin, clone = make_clone(tmp_path)
        (clone / worker.RUNS).mkdir(parents=True)
        (clone / worker.RUNS / "j_a_0.jsonl").write_text("x\n")
        assert worker.publish(clone, "j", 0, remote=str(origin))
        assert not worker.publish(clone, "j", 0, remote=str(origin))
        (clone / worker.RUNS / "j_a_0.jsonl").write_text("x\ny\n")
        assert worker.publish(clone, "j", 0, remote=str(origin))

    def test_restoring_with_no_branch_is_not_an_error(self, tmp_path):
        origin, clone = make_clone(tmp_path)
        worker.restore(clone, "j", 0)


class TestTheWholeRun:
    def test_every_step_runs_and_its_file_is_pushed(self, tmp_path, monkeypatch):
        import sys
        origin, clone = make_clone(tmp_path)
        (clone / "data/experiments/jobs").mkdir(parents=True)
        steps = [{"name": "a", "args": []}, {"name": "b", "args": []}]
        (clone / "data/experiments/jobs/j.json").write_text(json.dumps({"steps": steps}))

        def command(job, step, shard, shards):
            out = worker.out_path(job, step, shard)
            return [sys.executable, "-c", f"open({str(out)!r}, 'a').write({step['name']!r} + '\\n')"]

        monkeypatch.setattr(worker, "train_command", command)
        monkeypatch.setattr(worker, "_git", lambda root, *args, token=None, check=True: _local(origin, root, *args, check=check))
        worker.run("j", 0, 1, push_every=0.1, token="t", root=clone)
        shown = git(origin, "show", f"{worker.branch_of('j', 0)}:{worker.RUNS}/j_b_0.jsonl")
        assert shown == "b\n"
        assert git(origin, "show", f"{worker.branch_of('j', 0)}:{worker.RUNS}/j_a_0.jsonl") == "a\n"


def _local(origin, root, *args, check=True):
    """git against a local origin: `origin` in a push is the bare repository's path."""
    args = [str(origin) if a == "origin" else a for a in args]
    return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True, text=True)


class TestWhatTheyAreFor:
    def test_a_long_step_is_pushed_while_it_runs(self, tmp_path, monkeypatch):
        import sys
        origin, clone = make_clone(tmp_path)
        (clone / "data/experiments/jobs").mkdir(parents=True)
        (clone / "data/experiments/jobs/j.json").write_text(json.dumps({"steps": [{"name": "a", "args": []}]}))
        monkeypatch.setattr(worker, "train_command", lambda *a: [sys.executable, "-c", "import time; time.sleep(1.5)"])
        calls = []
        monkeypatch.setattr(worker, "publish", lambda *a, **k: calls.append(1))
        worker.run("j", 0, 1, push_every=0.2, token="t", root=clone)
        assert len(calls) >= 4

    def test_a_notebook_started_again_replaces_its_branch(self, tmp_path):
        origin, clone = make_clone(tmp_path)
        (clone / worker.RUNS).mkdir(parents=True)
        (clone / worker.RUNS / "j_a_0.jsonl").write_text("one\n")
        worker.publish(clone, "j", 0, remote=str(origin))
        other = tmp_path / "other"
        subprocess.run(["git", "clone", "-q", str(origin), str(other)], check=True)
        (other / worker.RUNS).mkdir(parents=True)
        (other / worker.RUNS / "j_a_0.jsonl").write_text("two\n")
        assert worker.publish(other, "j", 0, remote=str(origin))


    def test_without_a_token_nothing_is_pushed_and_the_files_are_copied(self, tmp_path, monkeypatch):
        import sys
        origin, clone = make_clone(tmp_path)
        (clone / "data/experiments/jobs").mkdir(parents=True)
        (clone / "data/experiments/jobs/j.json").write_text(json.dumps({"steps": [{"name": "a", "args": []}]}))

        def command(job, step, shard, shards):
            return [sys.executable, "-c", f"open({str(worker.out_path(job, step, shard))!r}, 'a').write('a\\n')"]

        monkeypatch.setattr(worker, "train_command", command)
        monkeypatch.setattr(worker, "publish", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pushed")))
        worker.run("j", 0, 1, push_every=0.1, root=clone, copy_to=tmp_path / "out")
        assert (tmp_path / "out" / "j_a_0.jsonl").read_text() == "a\n"
