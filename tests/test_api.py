"""Backend API tests (M3 Requirements 1-5) with a small live config so runs finish quickly."""

from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app
from robustlab.api.presets import PresetError, load_preset
from robustlab.experiments.suite import load_suite_config
from robustlab.physics.scene import build_scene
from robustlab.replay import kinematic_frames, visual_geoms
from test_suite_config import BASE


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    small = tmp / "small-suite.json"
    small.write_text(json.dumps({**BASE, "name": "api-small"}))
    settings = ServerSettings(
        root=REPO_ROOT, runs_dir=tmp / "runs", live_suite=small, web_dist=tmp / "no-dist"
    )
    with TestClient(create_app(settings)) as c:
        yield c


def _wait_done(client, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{run_id}").json()
        if status["status"] in ("done", "failed", "incomplete"):
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_presets_verify_and_live_suite_matches_m2() -> None:
    hidden = load_preset(REPO_ROOT, "presets/hidden-phase-v1")
    live = load_preset(REPO_ROOT, "presets/live-stress-v1")
    assert hidden.meta["verified_failure_substep"] == 500
    m2 = load_suite_config(REPO_ROOT / "examples/m2/sideways-shove-stratified-16.json").to_dict()
    stress = load_suite_config(live.path("suite.json")).to_dict()
    for d in (m2, stress):
        d.pop("name"), d.pop("description"), d["attack"].pop("force_n")
    assert m2 == stress
    assert load_suite_config(live.path("suite.json")).attack.force_n == 150.0


def test_tampered_preset_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "presets" / "hidden-phase-v1"
    shutil.copytree(REPO_ROOT / "presets/hidden-phase-v1", copy)
    report = json.loads((copy / "experiment.json").read_text())
    report["aggregates"]["attacked"]["physical_falls"] = 0
    (copy / "experiment.json").write_text(json.dumps(report))
    with pytest.raises(PresetError, match="SHA-256"):
        load_preset(tmp_path, "presets/hidden-phase-v1")


def test_hidden_phase_summary(client) -> None:
    body = client.get("/api/presets/hidden-phase").json()
    att = body["aggregates"]["attacked"]
    assert (att["physical_falls"], att["valid_trials"], att["numerical_failures"]) == (11, 256, 0)
    assert len(body["trials"]) == 256 and body["preset"]["result_origin"] == "precomputed"
    assert body["provenance"]["git"]["commit"].startswith("0b57e70")
    assert body["stance_finding"] == {
        "stance": "FR+RL", "fell": 11, "tested": 110, "all_failures_in_stance": True
    }  # fmt: skip
    assert sum(r["tested"] for r in body["stance_table"]) == 256
    assert body["failing_substeps"] == [500, 505, 611, 612, 613, 702, 708, 721, 723, 724, 725]
    assert body["verified_failure_substep"] in body["failing_substeps"]
    starts = [t["start_substep"] for t in body["trials"]]
    assert starts == list(range(500, 756))


def test_unknown_timing_and_ids_are_404(client) -> None:
    assert client.get("/api/presets/hidden-phase/replay/499").status_code == 404
    assert client.get("/api/presets/hidden-phase/replay/abc").status_code == 422
    for run_id in ("../etc", "ls-..%2F..", "ls-20261121T000000Z-1"):
        assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.post("/api/presets/live-stress/runs", json={"seed": -1}).status_code == 422
    assert client.post("/api/presets/live-stress/runs", json={"x": 1}).status_code == 422


def test_model_payload_shapes(client) -> None:
    body = client.get("/api/model").json()
    assert len(body["geoms"]) == 13 and body["geoms"][0]["body"] == "trunk"
    trunk = body["meshes"]["trunk"]
    verts = np.frombuffer(base64.b64decode(trunk["vertices"]), np.float32)
    faces = np.frombuffer(base64.b64decode(trunk["faces"]), np.int32)
    assert verts.size == 3 * trunk["n_vertices"] and faces.size == 3 * trunk["n_faces"]
    assert faces.max() < trunk["n_vertices"]


def test_kinematic_frames_match_mujoco() -> None:
    model = build_scene(REPO_ROOT / "assets/go1/walk.xml", "fall-contact").model
    rng = np.random.default_rng(3)
    qpos = np.tile(model.qpos0, (5, 1))
    qpos[:, 0:3] += rng.normal(0, 0.3, (5, 3))
    quat = rng.normal(size=(5, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 7:] += rng.normal(0, 0.3, (5, 12))
    pos, quats = kinematic_frames(model, qpos)
    data = mujoco.MjData(model)
    for f in range(5):
        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)
        for j, g in enumerate(visual_geoms(model)):
            np.testing.assert_allclose(pos[f, j], data.geom_xpos[g], atol=1e-12)
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, quats[f, j])
            np.testing.assert_allclose(mat, data.geom_xmat[g], atol=1e-9)


def test_live_run_end_to_end_and_busy_guard(client) -> None:
    started = client.post("/api/presets/live-stress/runs", json={"seed": 11})
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    busy = client.post("/api/presets/live-stress/runs", json={})
    assert busy.status_code in (202, 409)  # 409 unless the tiny run already finished
    status = _wait_done(client, run_id)
    assert status["status"] == "done", status
    summary = status["summary"]
    assert summary["sampling"]["seed"] == 11 and len(summary["trials"]) == 4
    att = summary["aggregates"]["attacked"]
    assert att["physical_falls"] + att["survived"] + att["numerical_failures"] == 4
    assert "--seed 11" in status["reproduce"]
    replay = client.get(f"/api/runs/{run_id}/trials/0/replay").json()
    assert replay["source"] == "recorded" and replay["frames"] == 51  # reset + 50 Ticks
    assert set(replay["cases"]) == {"nominal", "attacked"}
    pos = np.frombuffer(base64.b64decode(replay["cases"]["attacked"]["pos"]), np.float32)
    assert pos.size == 51 * 13 * 3
    assert replay["push"]["world_force_n"] == [0.0, -150.0, 0.0]
    assert client.get(f"/api/runs/{run_id}/trials/4/replay").status_code == 404
    runs = client.get("/api/presets/live-stress").json()["runs"]
    assert any(r["run_id"] == run_id and r["status"] == "done" for r in runs)
    if busy.status_code == 202:
        _wait_done(client, busy.json()["run_id"])


def test_busy_guard_rejects_second_run(client) -> None:
    jobs = client.app.state.jobs
    from robustlab.api.jobs import Job

    jobs.jobs["ls-29990101T000000Z-1"] = Job(run_id="ls-29990101T000000Z-1", seed=1, total=4,
                                              status="running")  # fmt: skip
    try:
        assert client.post("/api/presets/live-stress/runs", json={}).status_code == 409
    finally:
        del jobs.jobs["ls-29990101T000000Z-1"]


def test_core_modules_do_not_import_the_web_stack() -> None:
    import subprocess
    import sys

    probe = (
        "import sys, robustlab.cli, robustlab.experiments.suite, robustlab.replay; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('fastapi', 'uvicorn', 'starlette')))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
