"""The judge with a fake AIL Cup: no network, no real test drives."""
import pytest

from autoresearch.ailcup_client import AilCupError
from autoresearch.judge import judge
from autoresearch.score import NotScorable

TIMES = {"fiit-oval": 98.05, "hairpin-valley": 125.01, "danube-chicane": 114.75}


class FakeClient:
    """Plays the site: each track gets a list of outcomes, used one per test drive."""

    def __init__(self, outcomes=None, code_ok=True):
        self.outcomes = {t: list(v) for t, v in (outcomes or {}).items()}
        self.code_ok = code_ok
        self.jobs, self.uploads = {}, 0

    def check_code(self, code):
        return {"ok": self.code_ok, "problems": [] if self.code_ok else ["import os not allowed"]}

    def upload(self, code, model_bytes, name, label):
        self.uploads += 1
        return 7

    def start_test_drive(self, code, laps, track, model_from):
        outcome = self.outcomes.get(track, ["ok"]).pop(0) if self.outcomes.get(track) else "ok"
        if outcome == "start_fails":
            raise AilCupError("HTTP 502")
        job = len(self.jobs) + 1
        self.jobs[job] = (track, outcome)
        return job

    def wait_run(self, job_id, wait_max_s):
        track, outcome = self.jobs[job_id]
        if outcome == "sim_error":
            return {"id": job_id, "status": "error", "error": "sim died"}
        if outcome == "crash":
            return {"id": job_id, "status": "done", "result": {"status": "dnf", "dnf_reason": "crash"}}
        return {"id": job_id, "status": "done", "result": {"status": "finished", "total_time": TIMES[track]}}


@pytest.fixture
def files(tmp_path):
    code, model = tmp_path / "c.py", tmp_path / "m.npz"
    code.write_text("x = 1")
    model.write_bytes(b"\0" * 10)
    return code, model


def test_all_tracks_finish_gives_the_baseline_and_uploads_once(files):
    client = FakeClient()
    v = judge(*files, "t", client)
    assert v.score.total_s == pytest.approx(337.81)
    assert client.uploads == 1


def test_tracks_run_one_after_another(files):
    class OneAtATime(FakeClient):
        running = None

        def start_test_drive(self, code, laps, track, model_from):
            assert self.running is None, "a second test drive started while one was running (E38)"
            self.running = super().start_test_drive(code, laps, track, model_from)
            return self.running

        def wait_run(self, job_id, wait_max_s):
            self.running = None
            return super().wait_run(job_id, wait_max_s)

    assert judge(*files, "t", OneAtATime()).score.total_s == pytest.approx(337.81)


def test_a_crash_is_scored_with_the_penalty(files):
    v = judge(*files, "t", FakeClient({"hairpin-valley": ["crash"]}))
    assert v.score.dnf_tracks == ("hairpin-valley",)
    assert v.score.per_track_s["hairpin-valley"] == 300.0


def test_platform_errors_are_retried_not_scored(files):
    client = FakeClient({"fiit-oval": ["sim_error", "ok"], "danube-chicane": ["start_fails", "ok"]})
    v = judge(*files, "t", client)
    assert v.score.total_s == pytest.approx(337.81)


def test_a_platform_that_keeps_failing_is_not_scorable(files):
    with pytest.raises(NotScorable, match="fiit-oval"):
        judge(*files, "t", FakeClient({"fiit-oval": ["sim_error"] * 5}))


def test_a_failed_code_check_is_a_dnf_everywhere_without_driving(files):
    client = FakeClient(code_ok=False)
    v = judge(*files, "t", client)
    assert not v.code_ok and v.score.total_s == 900.0
    assert client.jobs == {} and client.uploads == 0
