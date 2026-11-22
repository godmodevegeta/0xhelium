"""Regression Gate API (M7 Requirement 6): the committed gate-v1 result, registry state, gate
status in the Hardening view, and re-simulated gate replays checked against the recorded result."""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app
from robustlab.api.gate import GATE_PRESET_DIR, GateService
from robustlab.api.presets import PresetError
from robustlab.llm.openrouter import OpenRouterCompiler
from test_suite_config import BASE

PRIMARY = "cand-ea7b9dbc0bbd"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gate-api")
    small = tmp / "small-suite.json"
    small.write_text(json.dumps({**BASE, "name": "api-small"}))
    settings = ServerSettings(root=REPO_ROOT, runs_dir=tmp / "runs", live_suite=small,
                              web_dist=tmp / "no-dist", hardening_workers=1)  # fmt: skip
    with TestClient(create_app(settings, compiler=OpenRouterCompiler(api_key=None))) as c:
        yield c


def test_gate_view_reports_decision_registry_and_every_candidate(client) -> None:
    g = client.get("/api/gate").json()
    assert g["available"] and g["result_origin"] == "precomputed"
    assert g["decision"] == {**g["decision"], "candidate_id": PRIMARY, "verdict": "promote"}
    active = json.loads((REPO_ROOT / "policies/registry/active.json").read_text())
    assert g["registry"]["active"] == active["active"] == f"v1-{PRIMARY}"
    by_id = {c["candidate_id"]: c for c in g["candidates"]}
    assert len(by_id) == 5 and sum(c["primary"] for c in by_id.values()) == 1
    primary = by_id[PRIMARY]
    assert primary["gate_status"]["status"] == "promoted"
    assert primary["gate_status"]["version"] == f"v1-{PRIMARY}"
    assert all(
        c["gate_status"]["status"] == "not_eligible" for c in by_id.values() if not c["primary"]
    )
    # individual regressions are reported, not hidden behind the aggregate
    conditions = primary["G3_regression"]["conditions"]
    listed = sum(len(c["regressions"]) for c in conditions)
    assert listed == primary["G3_regression"]["regressed"] > 0
    for c in conditions:
        for r in c["regressions"]:
            assert (
                r["baseline"]["outcome"] == "survived" and r["candidate"]["outcome"] != "survived"
            )
            assert r["id"].endswith(f"-{r['start_substep']}")
    assert len(g["holdout"]) == g["suites"]["holdout"] == 10
    assert all(set(row["candidates"]) == set(by_id) for row in g["holdout"])
    assert "outcomes" not in g  # the full per-episode result stays in the preset


def test_hardening_view_carries_gate_status(client) -> None:
    info = client.get("/api/hardening").json()
    status = info["preset"]["gate_status"]
    assert info["preset"]["record"]["candidate_id"] == PRIMARY and status["status"] == "promoted"
    assert "pending" not in info["semantics"]


@pytest.mark.parametrize("episode", ["lt-m4-cmaes-s7", "g-5kg-forward"])
def test_gate_replay_reproduces_recorded_digests(client, episode) -> None:
    r = client.get(f"/api/gate/replay?candidate={PRIMARY}&episode={episode}").json()
    assert r["mode"] == "gate" and r["gate_status"]["status"] == "promoted"
    v = r["verification"]
    assert v["all_match"] is True, v
    assert set(v["cases"]) == {"baseline", "candidate"}
    assert all("recorded_digest" in c for c in v["cases"].values())
    assert r["labels"]["attacked"].startswith(f"Candidate {PRIMARY[5:13]} (promoted")
    # measured server work: 2 display runs + 2 runs re-made exactly as recorded for the digests
    c = r["compute"]
    assert c["episodes"] == 4 and c["physics_steps"] == 5 * c["control_ticks"] > 0
    assert r["evidence_compute"]["rollouts"] == 8022


def test_gate_replay_refusals(client) -> None:
    base = "/api/gate/replay?candidate="
    assert client.get(f"{base}cand-000000000000&episode=lt-m4-cmaes-s7").status_code == 404
    assert client.get(f"{base}{PRIMARY}&episode=nope").status_code == 404
    assert client.get(f"{base}{PRIMARY}&episode=g-forward-10s&ghost=matched").status_code == 404
    assert client.get(f"{base}../x&episode=lt-m4-cmaes-s7").status_code == 422
    assert client.get(f"{base}{PRIMARY}&episode=lt-m4-cmaes-s7&ghost=other").status_code == 422


def test_missing_gate_result_is_reported_as_unavailable(tmp_path) -> None:
    service = GateService(tmp_path)
    assert service.view["available"] is False and service.status_for(PRIMARY) is None
    with pytest.raises(LookupError):
        service.replay_compute(PRIMARY, "lt-m4-cmaes-s7", "baseline")


def test_tampered_gate_result_fails_closed(tmp_path) -> None:
    target = tmp_path / GATE_PRESET_DIR
    target.mkdir(parents=True)
    shutil.copy(REPO_ROOT / GATE_PRESET_DIR / "preset.json", target / "preset.json")
    (target / "gate-run.log").write_text("edited\n")
    (target / "gate_result.json").write_text("{}")
    with pytest.raises(PresetError):
        GateService(tmp_path)
