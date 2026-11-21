"""Re-simulated Hidden Phase replay reproduces the committed M2 record (M3 Requirement 2.5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("substep", [500, 501])  # verified failure, and its surviving neighbour
def test_hidden_phase_replay_matches_recorded_trial(tmp_path, substep) -> None:
    settings = ServerSettings(
        root=REPO_ROOT,
        runs_dir=tmp_path,
        live_suite=REPO_ROOT / "presets/live-stress-v1/suite.json",
        web_dist=tmp_path / "none",
    )
    with TestClient(create_app(settings)) as client:
        replay = client.get(f"/api/presets/hidden-phase/replay/{substep}").json()
    verification = replay["verification"]
    assert replay["source"] == "re-simulated" and verification["all_match"], verification
    expected = "fell" if substep == 500 else "survived"
    assert replay["cases"]["attacked"]["outcome"]["outcome"] == expected
    assert replay["cases"]["nominal"]["outcome"]["outcome"] == "survived"
    assert replay["frames"] == 251 and replay["push"]["start_s"] == substep * 0.004
    if substep == 500:
        assert replay["cases"]["attacked"]["outcome"]["fall_time_s"] == pytest.approx(2.504)
