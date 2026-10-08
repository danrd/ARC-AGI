"""The notebooks the launcher makes: what they run, and where their output goes."""
import json

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
