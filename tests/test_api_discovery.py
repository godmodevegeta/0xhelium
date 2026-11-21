"""Failure Discovery API (M4 Requirement 7)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app
from robustlab.api.jobs import DiscoveryJob
from test_suite_config import BASE


@pytest.fixture()
def client(tmp_path):
    small = tmp_path / "small-suite.json"
    small.write_text(json.dumps({**BASE, "name": "api-small"}))
    settings = ServerSettings(
        root=REPO_ROOT,
        runs_dir=tmp_path / "runs",
        live_suite=small,
        web_dist=tmp_path / "no-dist",
        discovery_timeout_s=60.0,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def _wait(client, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/discovery/runs/{run_id}").json()
        if status["status"] in ("done", "failed", "incomplete"):
            return status
        time.sleep(0.05)
    raise AssertionError("discovery did not finish")


@pytest.mark.parametrize(
    "body",
    [
        {"algorithm": "gradient"},
        {"algorithm": "random", "budget": 12},
        {"algorithm": "random", "budget": 0},
        {"algorithm": "random", "budget": 264},
        {"algorithm": "cmaes", "seed": -1},
        {"algorithm": "cmaes", "shrink_budget": 129},
        {"algorithm": "cmaes", "extra": 1},
        {"budget": 8},
    ],
)
def test_invalid_discovery_requests_are_422(client, body) -> None:
    assert client.post("/api/discovery/runs", json=body).status_code == 422


def test_discovery_info_serves_precomputed_comparison(client) -> None:
    info = client.get("/api/discovery").json()
    assert info["bounds"]["force_n"] == [10.0, 150.0]
    assert info["bounds"]["start_substeps"] == [500, 755]
    comparison = info["comparison"]
    assert comparison["result_origin"] == "precomputed"
    assert comparison["protocol"]["budget"] == 64 and len(comparison["protocol"]["seeds"]) == 20
    assert set(comparison["aggregate"]) == {"random", "cmaes"}


def test_small_discovery_runs_and_records(client) -> None:
    started = client.post(
        "/api/discovery/runs",
        json={"algorithm": "cmaes", "budget": 8, "seed": 5, "shrink_budget": 0},
    )
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert run_id.startswith("ds-") and run_id.endswith("-cmaes-5")
    status = _wait(client, run_id)
    assert status["status"] == "done", status
    record = status["record"]
    assert record["accounting"]["search_rollouts"] == 8
    assert record["config"]["algorithm"] == "cmaes" and record["config"]["seed"] == 5
    assert record["request"]["reproduce"].startswith("uv run robustlab discover --algorithm cmaes")
    assert len(record["evaluations"]) == 8
    if record["final"] is None:  # an explicit no-failure outcome, never a substituted one
        assert client.get(f"/api/discovery/runs/{run_id}/replay").status_code == 404
    runs = client.get("/api/discovery").json()["runs"]
    assert any(r["run_id"] == run_id and r["status"] == "done" for r in runs)


def test_cancel_stops_a_running_search(client) -> None:
    started = client.post(
        "/api/discovery/runs",
        json={"algorithm": "random", "budget": 256, "seed": 1, "shrink_budget": 0},
    )
    run_id = started.json()["run_id"]
    for _ in range(200):  # wait until it is evaluating
        if client.get(f"/api/discovery/runs/{run_id}").json().get("completed", 0) > 0:
            break
        time.sleep(0.02)
    assert client.post(f"/api/discovery/runs/{run_id}/cancel").status_code == 202
    status = _wait(client, run_id)
    record = status["record"]
    assert record["search"]["stop_reason"] == "cancelled"
    assert record["accounting"]["search_rollouts"] < 256
    assert client.post(f"/api/discovery/runs/{run_id}/cancel").status_code == 409


def test_one_expensive_job_at_a_time(client) -> None:
    jobs = client.app.state.jobs
    fake = DiscoveryJob(run_id="ds-29990101T000000Z-random-1", request={}, status="running")
    jobs.discoveries[fake.run_id] = fake
    try:
        assert client.post("/api/discovery/runs", json={"algorithm": "random"}).status_code == 409
        assert client.post("/api/presets/live-stress/runs", json={}).status_code == 409
    finally:
        del jobs.discoveries[fake.run_id]


def test_bad_discovery_ids_are_404(client) -> None:
    for run_id in ("ds-..", "ls-20261121T000000Z-1", "ds-20261121T000000Z-random-1"):
        assert client.get(f"/api/discovery/runs/{run_id}").status_code == 404
        assert client.get(f"/api/discovery/runs/{run_id}/replay").status_code == 404
    assert client.post("/api/discovery/runs/ds-20261121T000000Z-random-1/cancel").status_code == 404
