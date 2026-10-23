"""The notebooks the launcher makes: what they run, and where their output goes."""
import json

from pathlib import Path

from scripts import kaggle_launch as launch
from scripts import kaggle_worker as worker


def test_a_notebook_runs_its_shard_of_the_job_and_leaves_its_files_in_the_output():
    code = launch.script("gaps", 2, 5)
    assert "kaggle_worker.py gaps --shard 2 --shards 5 --copy-to /kaggle/working" in code
    assert "optional-dependencies" in code and launch.REPO in code and "PYTHONPATH=." in code
    assert "pip install -e" not in code
    assert f"git clone {launch.REPO} /tmp/ARC-AGI" in code and "cd /tmp/ARC-AGI" in code
    compile(code, "worker.py", "exec")


def test_the_metadata_names_the_notebook_and_asks_for_the_internet_and_privacy():
    meta = launch.metadata("me", "gaps_x", 1, gpu=True)
    assert meta["id"] == "me/arc-worker-gaps-x-1" and meta["is_private"] == "true"
    assert meta["enable_internet"] == "true" and meta["enable_gpu"] == "true"
    assert launch.metadata("me", "gaps", 1)["enable_gpu"] == "false"
    json.dumps(meta)
    assert launch.slug("a", 0) != launch.slug("a", 1)


def test_a_worker_without_a_token_copies_its_files_instead_of_pushing_them(tmp_path):
    (tmp_path / worker.RUNS).mkdir(parents=True)
    (tmp_path / worker.RUNS / "j_a_0.jsonl").write_text("x\n")
    worker.copy_out(tmp_path, tmp_path / "out")
    assert (tmp_path / "out" / "j_a_0.jsonl").read_text() == "x\n"
    worker.copy_out(tmp_path, None)


def test_setup_commands_run_after_the_install_and_before_the_worker():
    code = launch.script("j", 0, 1, setup=["pip install llama-cpp-python"])
    assert code.index("optional-dependencies") < code.index("llama-cpp-python") < code.index("kaggle_worker.py")
    compile(code, "worker.py", "exec")


def test_a_clone_left_in_a_notebooks_output_is_not_collected(tmp_path, monkeypatch):
    def fake(*args):
        folder = Path(args[args.index("-p") + 1])
        (folder / "ARC-AGI" / "data").mkdir(parents=True)
        (folder / "ARC-AGI" / "data" / "wave1.jsonl").write_text("old\n")
        (folder / "j_a_0.jsonl").write_text("new\n")
        (folder / "j_a_0").mkdir()
        (folder / "j_a_0" / "x.txt").write_text("x")
        (folder / "run.log").write_text("log")
        return type("R", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(launch, "kaggle", fake)
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    got = launch.collect("j", 1)
    assert sorted(p.relative_to(tmp_path / "runs").as_posix() for p in got) == ["j_a_0.jsonl", "j_a_0/x.txt"]


def test_the_gpu_install_script_is_valid_shell_and_checks_for_the_gpu_at_the_end():
    script = Path(launch.ROOT / "scripts" / "install_llama_gpu.sh")
    import subprocess
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert text.index("--only-binary=:all:") < text.index("DGGML_CUDA=on") < text.index("llama_supports_gpu_offload")
    assert "raise SystemExit(0 if ok else 3)" in text


def test_a_tpu_notebook_asks_for_the_tpu_and_not_the_gpu_and_installs_no_extras_when_given_none():
    meta = launch.metadata("me", "j", 0, tpu=True)
    assert meta["enable_tpu"] == "true" and meta["enable_gpu"] == "false"
    assert launch.metadata("me", "j", 0)["enable_tpu"] == "false"
    code = launch.script("j", 0, 1, extras="")
    compile(code, "worker.py", "exec")
    assert "if e]" in code


def test_deleting_a_jobs_notebooks_names_each_and_counts_one_already_gone_as_deleted(monkeypatch):
    calls = []

    class Result:
        def __init__(self, code, text):
            self.returncode, self.stdout, self.stderr = code, text, ""

    answers = iter([Result(0, "Kernel x deleted successfully"), Result(1, "404 not found")])
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "kaggle", lambda *args: calls.append(args) or next(answers))
    assert launch.delete("some_job", 2) is True
    assert calls == [("kernels", "delete", "-y", "me/arc-worker-some-job-0"),
                     ("kernels", "delete", "-y", "me/arc-worker-some-job-1")]


def test_a_notebook_that_fails_to_delete_makes_the_whole_delete_false(monkeypatch):
    class Result:
        returncode, stdout, stderr = 1, "", "timeout talking to kaggle"

    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "kaggle", lambda *args: Result())
    assert launch.delete("j", 1) is False


def test_a_job_that_names_models_has_them_attached_in_the_metadata(tmp_path, monkeypatch):
    jobs = tmp_path / "data" / "experiments" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "with.json").write_text('{"steps": [], "model_sources": ["keras/gemma4/keras/gemma4_instruct_31b/2"]}')
    (jobs / "without.json").write_text('{"steps": []}')
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    assert launch.model_sources("with") == ["keras/gemma4/keras/gemma4_instruct_31b/2"]
    assert launch.model_sources("without") == [] and launch.model_sources("missing") == []
    assert launch.metadata("me", "with", 0, tpu=True, models=launch.model_sources("with"))["model_sources"] == [
        "keras/gemma4/keras/gemma4_instruct_31b/2"]
    assert launch.metadata("me", "without", 0)["model_sources"] == []


def test_a_push_that_exits_zero_with_an_error_is_a_refusal():
    from types import SimpleNamespace
    quota = SimpleNamespace(returncode=0, stdout="Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.\n",
                            stderr="")
    done = SimpleNamespace(returncode=0, stdout="Kernel version 1 successfully pushed.  Please check progress at x\n",
                           stderr="")
    assert not launch.pushed(quota) and launch.pushed(done)


def test_start_raises_when_the_first_push_is_refused_and_not_when_a_later_one_is(monkeypatch):
    from types import SimpleNamespace
    import pytest
    answers = iter(["Kernel push error: no quota", "Kernel version 1 successfully pushed."])
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "script", lambda *a, **k: "")
    monkeypatch.setattr(launch, "kaggle", lambda *a: SimpleNamespace(returncode=0, stdout=next(answers), stderr=""))
    with pytest.raises(launch.PushRefused, match="no quota"):
        launch.start("j", 2, gpu=True)
    answers = iter(["Kernel version 1 successfully pushed.", "Kernel push error: no quota"])
    launch.start("j", 2, gpu=True)                      # the first went; the second's refusal is printed, not raised


def test_a_status_query_that_failed_for_the_service_s_reason_is_unknown_and_for_the_notebook_s_is_not():
    from types import SimpleNamespace

    def answer(stdout="", stderr=""):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    assert launch.shard_status(answer('me/x has status "KernelWorkerStatus.RUNNING"')) == "RUNNING"
    assert launch.shard_status(answer('me/x has status "KernelWorkerStatus.COMPLETE"')) == "COMPLETE"
    assert launch.shard_status(answer("429 Client Error: Too Many Requests for url: https://x")) == "UNKNOWN"
    assert launch.shard_status(answer("", "503 Server Error: Service Unavailable")) == "UNKNOWN"
    assert launch.shard_status(answer("", "HTTPSConnectionPool: Read timed out.")) == "UNKNOWN"
    assert launch.shard_status(answer("404 Client Error: Not Found for url: https://x")) == "?"
    assert launch.shard_status(answer()) == "?"


def test_a_download_that_gives_nothing_is_asked_again_and_a_log_is_kept_for_every_notebook(tmp_path, monkeypatch):
    calls = []

    def fake(*args):
        folder = Path(args[args.index("-p") + 1])
        calls.append(1)
        if len(calls) >= 2:
            (folder / "arc-worker-j-0.log").write_text("why it ended")
            (folder / "j_a_0.jsonl").write_text("new\n")
        return type("R", (), {"stdout": "", "stderr": ""})()

    waits = []
    monkeypatch.setattr(launch, "kaggle", fake)
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    got = launch.collect("j", 1, wait=7, sleep=waits.append)
    assert [p.name for p in got] == ["j_a_0.jsonl"] and len(calls) == 2 and waits == [7]
    assert (tmp_path / "runs" / "logs" / "arc-worker-j-0.log").read_text() == "why it ended"


def test_a_notebook_that_gave_only_a_log_is_collected_with_its_log_and_no_files(tmp_path, monkeypatch):
    def fake(*args):
        (Path(args[args.index("-p") + 1]) / "x.log").write_text("Traceback")
        return type("R", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(launch, "kaggle", fake)
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    assert launch.collect("j", 1) == []
    assert (tmp_path / "runs" / "logs" / "arc-worker-j-0.log").read_text() == "Traceback"


def test_a_download_that_never_gives_anything_raises_after_its_tries(tmp_path, monkeypatch):
    import pytest
    calls = []
    monkeypatch.setattr(launch, "kaggle", lambda *a: calls.append(1) or type("R", (), {"stdout": "", "stderr": ""})())
    monkeypatch.setattr(launch, "username", lambda: "me")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    with pytest.raises(launch.CollectFailed, match="nothing came back in 3 tries"):
        launch.collect("j", 1, sleep=lambda s: None)
    assert len(calls) == 3
