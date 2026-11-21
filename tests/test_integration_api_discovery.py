"""A discovery run through the API ends in a replay that re-simulates the discovered failure
(M4 Requirements 7.5, 8.2)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app

pytestmark = pytest.mark.integration


def test_discovered_failure_replays_with_matching_digest(tmp_path) -> None:
    settings = ServerSettings(
        root=REPO_ROOT,
        runs_dir=tmp_path,
        live_suite=REPO_ROOT / "presets/live-stress-v1/suite.json",
        web_dist=tmp_path / "none",
    )
    with TestClient(create_app(settings)) as client:
        # Random Search, seed 0, finds its first failure at evaluation 29 (M4 comparison).
        body = {"algorithm": "random", "budget": 32, "seed": 0, "shrink_budget": 8}
        run_id = client.post("/api/discovery/runs", json=body).json()["run_id"]
        for _ in range(600):
            status = client.get(f"/api/discovery/runs/{run_id}").json()
            if status["status"] != "running" and status["status"] != "queued":
                break
            time.sleep(0.05)
        assert status["status"] == "done"
        record = status["record"]
        assert record["final"] is not None
        replay = client.get(f"/api/discovery/runs/{run_id}/replay?which=final").json()
        search = client.get(f"/api/discovery/runs/{run_id}/replay?which=search").json()
    assert replay["verification"]["all_match"], replay["verification"]
    assert replay["candidate"] == record["final"]["candidate"]
    assert replay["cases"]["attacked"]["outcome"]["outcome"] == "fell"
    assert replay["cases"]["nominal"]["outcome"]["outcome"] == "survived"
    assert replay["stance_at_start"] is not None
    assert search["verification"]["all_match"]
    assert search["candidate"] == record["confirmation"]["confirmed"]["candidate"]
