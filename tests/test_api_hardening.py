"""Policy Hardening API (M6 Requirement 6)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app
from robustlab.api.jobs import HardeningJob
from robustlab.llm.openrouter import OpenRouterCompiler
from test_suite_config import BASE


def _client(tmp_path, workers: int = 1):
    small = tmp_path / "small-suite.json"
    small.write_text(json.dumps({**BASE, "name": "api-small"}))
    settings = ServerSettings(root=REPO_ROOT, runs_dir=tmp_path / "runs", live_suite=small,
                              web_dist=tmp_path / "no-dist", hardening_workers=workers)  # fmt: skip
    return TestClient(create_app(settings, compiler=OpenRouterCompiler(api_key=None)))


@pytest.fixture()
def client(tmp_path):
    with _client(tmp_path) as c:
        yield c


def _wait(client, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/hardening/runs/{run_id}").json()
        if status["status"] in ("done", "failed", "incomplete"):
            return status
        time.sleep(0.1)
    raise AssertionError("hardening run did not finish")


def test_info_lists_episodes_and_precomputed_candidate(client) -> None:
    info = client.get("/api/hardening").json()
    assert info["sets"]["locked_test"]["status"].startswith("reserved for M7")
    ids = {e["id"] for e in info["episodes"]}
    assert not any(i.startswith("lt-") for i in ids)
    selectable = [e["id"] for e in info["episodes"] if e["selectable"]]
    assert len(selectable) == 8 and all(e in ids for e in selectable)
    preset = info["preset"]
    assert preset["result_origin"] == "precomputed" and preset["record"]["status"] == "candidate"
    assert info["defaults"]["popsize"] == 8 and info["limits"]["max_generations"] == [1, 30]


@pytest.mark.parametrize(
    "body",
    [{"generations": 0}, {"generations": 31}, {"seed": -1}, {"focus": "../x"},
     {"focus": "m3-150N-564"}, {"focus": "nominal-forward-1.0"}, {"popsize": 4}],
)  # fmt: skip
def test_invalid_requests_are_422(client, body) -> None:
    assert client.post("/api/hardening/runs", json=body).status_code == 422


def test_preset_replay_compares_identical_episode(client) -> None:
    replay = client.get("/api/hardening/runs/preset-v1/replay?episode=m2-100N-500").json()
    assert replay["mode"] == "hardening" and replay["candidate_status"] == "candidate"
    assert replay["labels"]["attacked"].startswith("Candidate")
    assert replay["labels"]["nominal"].startswith("Frozen baseline")
    assert replay["verification"]["all_match"] is True
    assert replay["cases"]["attacked"]["outcome"]["outcome"] == "survived"
    assert replay["cases"]["nominal"]["outcome"]["outcome"] == "fell"
    assert replay["push"]["world_force_n"] == [0.0, -100.0, 0.0]
    assert len(replay["reflex"]["activation"]) == replay["cases"]["attacked"]["frames"]
    assert replay["reflex"]["activation"][0] == 0.0 and max(replay["reflex"]["activation"]) > 0


def test_matched_ghost_and_refusals(client) -> None:
    matched = client.get(
        "/api/hardening/runs/preset-v1/replay?episode=m5-100N-567-mu0.3&ghost=matched"
    ).json()
    assert matched["ghost"]["kind"] == "matched" and matched["environment"]["mu"] == 0.3
    assert matched["cases"]["nominal"]["outcome"]["outcome"] == "survived"
    url = "/api/hardening/runs/preset-v1/replay?episode="
    assert client.get(url + "lt-m4-cmaes-s7").status_code == 404  # locked M7 episode
    assert client.get(url + "nominal-forward-1.0&ghost=matched").status_code == 404
    assert client.get(url + "nope").status_code == 404


def test_hardening_shares_the_busy_guard(client) -> None:
    jobs = client.app.state.jobs
    fake = HardeningJob(run_id="hd-29990101T000000Z-1", request={}, status="running")
    jobs.hardenings[fake.run_id] = fake
    try:
        assert client.post("/api/hardening/runs", json={}).status_code == 409
        assert client.post("/api/presets/live-stress/runs", json={}).status_code == 409
        assert client.post("/api/discovery/runs", json={"algorithm": "random"}).status_code == 409
    finally:
        del jobs.hardenings[fake.run_id]


def test_bad_ids_are_404(client) -> None:
    for run_id in ("hd-..", "ds-20261121T000000Z-random-1", "hd-20261121T000000Z-1"):
        assert client.get(f"/api/hardening/runs/{run_id}").status_code == 404
        assert (
            client.get(f"/api/hardening/runs/{run_id}/replay?episode=m2-100N-500").status_code
            == 404
        )
    assert client.post("/api/hardening/runs/hd-20261121T000000Z-1/cancel").status_code == 404


@pytest.mark.integration
def test_small_hardening_job_end_to_end(tmp_path) -> None:
    with _client(tmp_path, workers=1) as c:
        started = c.post("/api/hardening/runs",
                         json={"seed": 2, "generations": 1, "focus": "m2-100N-500"})  # fmt: skip
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        assert (
            run_id.endswith("-2")
            and "--focus m2-100N-500" in started.json()["request"]["reproduce"]
        )
        status = _wait(c, run_id)
        assert status["status"] == "done", status
        record = status["record"]
        assert record["status"] == "candidate" and record["request"]["run_id"] == run_id
        assert record["accounting"]["optimization_rollouts"] == 8 * 4
        assert (tmp_path / "runs/hardening" / run_id / "candidate.json").exists()
        replay = c.get(f"/api/hardening/runs/{run_id}/replay?episode=m2-100N-500").json()
        assert replay["verification"]["all_match"] is True
        runs = c.get("/api/hardening").json()["runs"]
        assert runs[0]["run_id"] == run_id and runs[0]["status"] == "done"
        assert c.post(f"/api/hardening/runs/{run_id}/cancel").status_code == 409


@pytest.mark.integration
def test_cancel_stops_a_hardening_job(tmp_path) -> None:
    with _client(tmp_path, workers=1) as c:
        run_id = c.post("/api/hardening/runs", json={"seed": 3, "generations": 30}).json()["run_id"]
        for _ in range(600):
            if c.get(f"/api/hardening/runs/{run_id}").json().get("stage") == "search":
                break
            time.sleep(0.05)
        assert c.post(f"/api/hardening/runs/{run_id}/cancel").status_code == 202
        status = _wait(c, run_id, timeout=180)
        assert status["record"]["result"]["stop_reason"] == "cancelled"
        assert status["record"]["result"]["generations_completed"] < 30
