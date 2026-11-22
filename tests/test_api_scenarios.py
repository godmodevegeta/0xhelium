"""Scenario Designer API (M5 Requirement 7): config, validate, compile (mocked), runs, replays."""

from __future__ import annotations

import copy
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from robustlab.api.app import ServerSettings, create_app
from robustlab.api.jobs import ScenarioJob
from robustlab.llm.openrouter import OpenRouterCompiler
from robustlab.scenarios.schema import nominal_dict
from test_openrouter import GOOD, KEY, answer
from test_suite_config import BASE

SCENARIO = {
    **nominal_dict("api-combined"),
    "horizon_s": 3.5,
    "environment": {"ground_friction_mu": 0.3},
    "payload": {"mass_kg": 2.0, "offset_xyz_m": [0.02, 0.0, 0.04]},
    "push": {"force_n": 100.0, "direction_deg": 270.0, "start_s": 2.0013, "duration_s": 0.1},
}


def _app(tmp_path, compiler):
    small = tmp_path / "small-suite.json"
    small.write_text(json.dumps({**BASE, "name": "api-small"}))
    settings = ServerSettings(root=REPO_ROOT, runs_dir=tmp_path / "runs", live_suite=small,
                              web_dist=tmp_path / "no-dist")  # fmt: skip
    return create_app(settings, compiler=compiler)


@pytest.fixture()
def client(tmp_path):
    with TestClient(_app(tmp_path, OpenRouterCompiler(api_key=None))) as c:
        yield c


def _mocked(tmp_path, content):
    handler = answer(content)
    compiler = OpenRouterCompiler(api_key=KEY, transport=httpx.MockTransport(handler))
    return TestClient(_app(tmp_path, compiler)), handler


def _wait(client, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/scenarios/runs/{run_id}").json()
        if status["status"] in ("done", "failed", "incomplete"):
            return status
        time.sleep(0.05)
    raise AssertionError("scenario run did not finish")


def test_config_lists_validated_presets_and_unconfigured_compiler(client) -> None:
    config = client.get("/api/scenarios/config").json()
    assert {p["id"] for p in config["presets"]} == {
        "nominal", "low-friction-0.2", "payload-5kg-top", "icy-heavy-shove"
    }  # fmt: skip
    assert config["compiler"]["configured"] is False
    assert config["nominal_mu"] == 0.6
    assert config["bounds"]["environment.ground_friction_mu"] == [0.05, 1.5]
    assert config["examples"] and config["suite_defaults"]["trials"] == 16


def test_validate_reports_errors_without_clamping(client) -> None:
    bad = copy.deepcopy(SCENARIO)
    bad["environment"]["ground_friction_mu"] = 9.0
    bad["payload"]["mass_kg"] = 40.0
    body = client.post("/api/scenarios/validate", json={"scenario": bad}).json()
    assert body["valid"] is False
    assert {e["field"] for e in body["errors"]} == {
        "environment.ground_friction_mu", "payload.mass_kg"
    }  # fmt: skip
    assert client.post("/api/scenarios/validate", json={"scenario": {}, "x": 1}).status_code == 422


def test_validate_preview_has_observed_friction_and_mass(client) -> None:
    body = client.post("/api/scenarios/validate", json={"scenario": SCENARIO}).json()
    assert body["valid"] is True
    assert body["scenario"]["push"]["start_s"] == 2.0
    assert [a["field"] for a in body["adjustments"]] == ["push.start_s"]
    friction = body["friction"]
    assert friction["requested_mu"] == 0.3 and friction["configured_mu"] == 0.3
    assert friction["observed_at_reset"]["sliding_mu_values"] == [0.3]
    assert set(friction["resolved_foot_floor"]) == {"FR", "FL", "RR", "RL"}
    payload = body["mass"]["payload"]
    assert payload["mass_kg"] == 2.0 and payload["positive_definite"] is True
    assert body["runnable"] == {"single": True, "suite": True, "suite_reason": None}
    assert body["primary_baseline"] == "matched"


def test_compile_without_key_is_503_and_rest_works(client) -> None:
    response = client.post("/api/scenarios/compile", json={"prompt": "icy floor"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "not_configured"
    assert client.post("/api/scenarios/validate", json={"scenario": SCENARIO}).json()["valid"]


def test_compile_valid_answer_gets_preview_and_audit(tmp_path) -> None:
    c, handler = _mocked(tmp_path, json.dumps(GOOD))
    with c:
        body = c.post("/api/scenarios/compile", json={"prompt": "wet tile shove"}).json()
        assert body["valid"] is True and body["compile_id"].startswith("cp-")
        assert body["preview"]["friction"]["observed_at_reset"]["sliding_mu_values"] == [0.25]
        assert any(a["field"] == "push.start_s" for a in body["preview"]["adjustments"])
        assert len(handler.requests) == 1
    audit = (tmp_path / "runs/llm/compiles.jsonl").read_text()
    record = json.loads(audit.splitlines()[0])
    assert record["status"] == "valid" and record["llm"]["model"] == "openai/gpt-4o-mini"
    assert KEY not in audit and KEY not in json.dumps(body)


def test_compile_out_of_range_answer_is_rejected_with_fields(tmp_path) -> None:
    c, _ = _mocked(tmp_path, json.dumps({**GOOD, "payload_mass_kg": 40.0}))
    with c:
        response = c.post("/api/scenarios/compile", json={"prompt": "40 kg anvil"})
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False and "preview" not in body
    assert [e["field"] for e in body["errors"]] == ["payload.mass_kg"]


def test_compile_garbage_is_502(tmp_path) -> None:
    c, _ = _mocked(tmp_path, "not json at all")
    with c:
        response = c.post("/api/scenarios/compile", json={"prompt": "anything"})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_model_output"


def test_single_run_from_compile_and_replays(tmp_path) -> None:
    c, _ = _mocked(tmp_path, json.dumps({**GOOD, "payload_mass_kg": 2.0}))
    with c:
        compiled = c.post("/api/scenarios/compile", json={"prompt": "wet tile shove"}).json()
        scenario = {**compiled["scenario"], "horizon_s": 3.0}  # edited after compile
        started = c.post(
            "/api/scenarios/runs",
            json={"scenario": scenario, "mode": "single", "compile_id": compiled["compile_id"]},
        )
        assert started.status_code == 202, started.text
        origin = started.json()["request"]["origin"]
        assert origin["kind"] == "openrouter" and origin["edited_after_compile"] is True
        status = _wait(c, started.json()["run_id"])
        assert status["status"] == "done", status
        result = status["result"]
        assert set(result["cases"]) == {"scenario", "matched", "original"}
        assert result["primary_baseline"] == "matched"
        run_id = status["run_id"]
        replay = c.get(f"/api/scenarios/runs/{run_id}/replay").json()
        assert replay["ghost"]["kind"] == "matched"
        assert replay["environment"] == {"mu": 0.25, "ghost_mu": 0.25, "nominal_mu": 0.6}
        assert replay["cases"]["attacked"]["payload_box"]["mass_kg"] == 2.0
        assert replay["cases"]["nominal"]["payload_box"] is not None
        assert replay["push"]["force_n"] == 90.0
        original = c.get(f"/api/scenarios/runs/{run_id}/replay?ghost=original").json()
        assert original["ghost"]["kind"] == "original"
        assert original["cases"]["nominal"]["payload_box"] is None
        assert original["environment"]["ghost_mu"] == 0.6
        assert c.get(f"/api/scenarios/runs/{run_id}/replay?trial=1").status_code == 404
        listed = c.get("/api/scenarios/config").json()["runs"]
        assert listed[0]["run_id"] == run_id and listed[0]["origin"] == "openrouter"


def test_no_push_single_compares_original_and_refuses_suite(client) -> None:
    scenario = {**nominal_dict("api-friction"), "horizon_s": 1.0,
                "environment": {"ground_friction_mu": 0.2}}  # fmt: skip
    refused = client.post("/api/scenarios/runs", json={"scenario": scenario, "mode": "suite"})
    assert refused.status_code == 422 and "needs a push" in refused.json()["detail"]
    started = client.post("/api/scenarios/runs", json={"scenario": scenario})
    status = _wait(client, started.json()["run_id"])
    assert set(status["result"]["cases"]) == {"scenario", "original"}
    run_id = status["run_id"]
    replay = client.get(f"/api/scenarios/runs/{run_id}/replay").json()
    assert replay["ghost"]["kind"] == "original" and replay["push"] is None
    assert replay["cases"]["attacked"]["payload_box"] is None
    assert client.get(f"/api/scenarios/runs/{run_id}/replay?ghost=matched").status_code == 404


def test_suite_varies_push_start_and_replays_trials(client) -> None:
    started = client.post(
        "/api/scenarios/runs",
        json={"scenario": SCENARIO, "mode": "suite", "trials": 2, "seed": 3},
    )
    assert started.status_code == 202, started.text
    request = started.json()["request"]
    assert request["suite"]["seed"] == 3 and "--suite --trials 2" in request["reproduce"]
    status = _wait(client, started.json()["run_id"])
    assert status["status"] == "done", status
    assert status["result"]["mode"] == "suite"
    assert len(status["summary"]["trials"]) == 2
    run_id = status["run_id"]
    replay = client.get(f"/api/scenarios/runs/{run_id}/replay?trial=1").json()
    assert replay["start_s"] == status["summary"]["trials"][1]["start_s"]
    assert replay["cases"]["attacked"]["payload_box"] is not None
    original = client.get(f"/api/scenarios/runs/{run_id}/replay?trial=1&ghost=original").json()
    assert original["cases"]["nominal"]["payload_box"] is None
    assert client.get(f"/api/scenarios/runs/{run_id}/replay?trial=2").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"scenario": SCENARIO, "mode": "batch"},
        {"scenario": SCENARIO, "mode": "suite", "trials": 33},
        {"scenario": SCENARIO, "mode": "suite", "trials": 2, "interval_s": [3.0, 2.0]},
        {"scenario": SCENARIO, "compile_id": "cp-20261121T000000Z-00000000"},
        {"scenario": SCENARIO, "compile_id": "../../etc"},
        {"scenario": {**SCENARIO, "payload": {"mass_kg": 6.0, "offset_xyz_m": [0, 0, 0]}}},
        {"scenario": SCENARIO, "extra": True},
    ],
)
def test_invalid_run_requests_are_422(client, body) -> None:
    assert client.post("/api/scenarios/runs", json=body).status_code == 422


def test_scenario_runs_share_the_busy_guard(client) -> None:
    jobs = client.app.state.jobs
    fake = ScenarioJob(run_id="sc-29990101T000000Z-00000000", request={}, total=1,
                       status="running")  # fmt: skip
    jobs.scenarios[fake.run_id] = fake
    try:
        assert client.post("/api/scenarios/runs", json={"scenario": SCENARIO}).status_code == 409
        assert client.post("/api/presets/live-stress/runs", json={}).status_code == 409
        assert client.post("/api/discovery/runs", json={"algorithm": "random"}).status_code == 409
    finally:
        del jobs.scenarios[fake.run_id]


def test_bad_scenario_ids_are_404(client) -> None:
    for run_id in ("sc-..", "ls-20261121T000000Z-1", "sc-20261121T000000Z-0000000g",
                   "sc-20261121T000000Z-00000000"):  # fmt: skip
        assert client.get(f"/api/scenarios/runs/{run_id}").status_code == 404
        assert client.get(f"/api/scenarios/runs/{run_id}/replay").status_code == 404
