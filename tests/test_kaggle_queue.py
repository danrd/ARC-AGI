"""The queue: a free notebook is given to the next job that fits, and a finished job is collected."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import kaggle_queue as queue  # noqa: E402
from kaggle_queue import Job, load, save, slots_used, tick  # noqa: E402


class Fake:
    """The notebooks: what each job's are doing, what was started, what was collected."""

    def __init__(self, **states):
        self.states, self.started, self.collected, self.deleted = states, [], [], []

    def status(self, job):
        return self.states.get(job.name, [])

    def start(self, job):
        self.started.append(job.name)
        self.states[job.name] = ["RUNNING"] * job.shards

    def collect(self, job):
        self.collected.append(job.name)
        return [f"{job.name}_0.jsonl"]

    def delete(self, job):
        self.deleted.append(job.name)
        return job.name != "stuck"

    def tick(self, jobs, capacity=None, delete=True):
        return tick(jobs, self.status, self.start, self.collect, capacity or {"cpu": 5, "gpu": 1},
                    delete_fn=self.delete if delete else None)


def test_a_queued_job_is_started_when_its_notebooks_are_free():
    fake = Fake()
    jobs = [Job("a", "cpu", 2), Job("b", "gpu", 1)]
    events = fake.tick(jobs)
    assert fake.started == ["a", "b"] and [j.status for j in jobs] == ["running", "running"]
    assert "a: started on 2 cpu notebook(s)" in events


def test_a_job_too_big_for_the_free_slots_waits_and_a_smaller_one_behind_it_goes_first():
    fake = Fake(busy=["RUNNING"] * 3)
    jobs = [Job("busy", "cpu", 3, "running"), Job("big", "cpu", 3), Job("small", "cpu", 2)]
    events = fake.tick(jobs)
    assert fake.started == ["small"] and jobs[1].status == "queued"
    assert any(e.startswith("big: waits, 2 free cpu") for e in events)


def test_the_resources_are_counted_apart():
    fake = Fake(g=["RUNNING"])
    jobs = [Job("g", "gpu", 1, "running"), Job("c", "cpu", 5), Job("g2", "gpu", 1)]
    fake.tick(jobs)
    assert fake.started == ["c"] and jobs[2].status == "queued"


def test_a_job_whose_notebooks_all_ended_is_collected_and_its_slots_given_on_in_the_same_pass():
    fake = Fake(a=["COMPLETE", "COMPLETE"])
    jobs = [Job("a", "cpu", 2, "running"), Job("next", "cpu", 5)]
    events = fake.tick(jobs)
    assert jobs[0].status == "done" and jobs[0].result == ["a_0.jsonl"] and fake.collected == ["a"]
    assert fake.started == ["next"] and "a: done, 1 files collected" in events


def test_a_job_with_a_notebook_in_error_is_failed_but_still_collected():
    fake = Fake(a=["COMPLETE", "ERROR"])
    jobs = [Job("a", "cpu", 2, "running")]
    fake.tick(jobs)
    assert jobs[0].status == "failed" and fake.collected == ["a"]


def test_a_job_with_a_notebook_still_running_is_not_collected_but_the_notebook_that_ended_is_free():
    fake = Fake(a=["COMPLETE", "RUNNING"])
    jobs = [Job("a", "cpu", 2, "running"), Job("b", "cpu", 4)]
    fake.tick(jobs)
    assert jobs[0].status == "running" and fake.collected == []
    assert fake.started == ["b"]                       # the four left after the one still running
    assert slots_used(jobs[:1], {"a": ["COMPLETE", "RUNNING"]}) == {"cpu": 1, "gpu": 0, "tpu": 0}
    assert Fake(a=["COMPLETE", "RUNNING"]).tick([Job("a", "cpu", 2, "running"), Job("c", "cpu", 5)])[0].startswith("c: waits")


def test_idle_notebooks_with_nothing_queued_are_said_so():
    events = Fake().tick([Job("a", "cpu", 2)])
    assert "idle with nothing queued: 3 cpu, 1 gpu" in events


def test_a_running_job_whose_status_is_unknown_is_left_as_it_is():
    fake = Fake()
    jobs = [Job("a", "cpu", 1, "running")]
    fake.tick(jobs)
    assert jobs[0].status == "running" and fake.collected == []


def test_the_queue_is_a_file_that_comes_back_as_it_was(tmp_path):
    path = tmp_path / "q.json"
    jobs = [Job("a", "gpu", 1, "queued", ["pip install x"], "rl", "why", []), Job("b", "cpu", 3, "done", result=["f"])]
    save(jobs, path)
    assert load(path) == jobs and load(tmp_path / "none.json") == []
    assert queue.CAPACITY == {"cpu": 5, "gpu": 1, "tpu": 1}


def test_the_notebooks_of_a_job_that_is_done_with_files_are_deleted_once():
    fake = Fake(a=["COMPLETE"])
    jobs = [Job("a", "cpu", 1, "running")]
    events = fake.tick(jobs)
    assert fake.deleted == ["a"] and jobs[0].cleaned and "a: 1 notebook(s) deleted from Kaggle" in events
    fake.tick(jobs)
    assert fake.deleted == ["a"]


def test_a_failed_job_and_a_job_with_no_files_keep_their_notebooks_for_the_logs():
    class NoFiles(Fake):
        def collect(self, job):
            return []

    failed = Fake(a=["ERROR"])
    jobs = [Job("a", "cpu", 1, "running")]
    failed.tick(jobs)
    empty = NoFiles(b=["COMPLETE"])
    other = [Job("b", "cpu", 1, "running")]
    empty.tick(other)
    assert failed.deleted == [] and empty.deleted == [] and not jobs[0].cleaned and not other[0].cleaned


def test_a_notebook_that_could_not_be_deleted_is_tried_again_at_the_next_tick():
    fake = Fake()
    jobs = [Job("stuck", "cpu", 1, "done", result=["f"])]
    fake.tick(jobs)
    fake.tick(jobs)
    assert fake.deleted == ["stuck", "stuck"] and not jobs[0].cleaned


def test_without_a_delete_function_nothing_is_deleted():
    fake = Fake()
    jobs = [Job("a", "cpu", 1, "done", result=["f"])]
    fake.tick(jobs, delete=False)
    assert fake.deleted == [] and not jobs[0].cleaned
