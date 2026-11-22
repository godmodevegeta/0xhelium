"""Multi-axis physics evidence (M5 Requirements 2-5): nominal identity, friction contacts and
slip, payload walking, suite semantics, reproducibility."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.experiments.push import PushScenario
from robustlab.experiments.scenario_run import foot_slip, run_scenario_suite, run_single
from robustlab.experiments.stack import load_stack
from robustlab.experiments.trace import read_run
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner
from robustlab.physics.scenario_model import ScenarioModelCache
from robustlab.scenarios.schema import nominal_dict, validate

pytestmark = pytest.mark.integration
FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def stack():
    return load_stack(REPO_ROOT, POLICY_PATH, "fall-contact")


def scenario(**kw):
    d = nominal_dict(kw.pop("name", "t"))
    d["horizon_s"] = kw.pop("horizon_s", 5.0)
    d["command"] = kw.pop("command", [1.0, 0.0, 0.0])
    if "mu" in kw:
        d["environment"]["ground_friction_mu"] = kw.pop("mu")
    if "mass" in kw:
        d["payload"] = {"mass_kg": kw.pop("mass"), "offset_xyz_m": kw.pop("offset", [0, 0, 0])}
    d["push"] = kw.pop("push", None)
    return validate(d).scenario


def test_nominal_scenario_reproduces_m0_golden(stack) -> None:
    golden = np.load(FIXTURES / "m0_golden.npz")
    result = run_single(scenario(horizon_s=10.0), stack, None)
    assert result["model"]["reused_base_model"] and result["primary_baseline"] == "original"
    assert result["cases"]["scenario"]["digest"] == result["cases"]["original"]["digest"]
    runner = EpisodeRunner(stack.model, stack.ids, stack.policy, stack.actor)
    record = runner.run(EpisodeConfig((1.0, 0.0, 0.0), 10.0))
    np.testing.assert_array_equal(record.qpos, golden["forward_qpos"])


def _contact_ratio_episode(stack, mu: float):
    """Standing robot shoved sideways; every floor contact inspected at each Tick."""
    entry = ScenarioModelCache().get(stack, mu, None)
    runner = EpisodeRunner(entry.model, entry.ids, stack.policy, stack.actor)
    push = PushScenario(150.0, (0.0, -1.0, 0.0), 1.0, 0.3)
    runner.reset((0.0, 0.0, 0.0), push)
    floor = entry.ids.floor_geom
    frictions, ratios = set(), []
    force = np.zeros(6)
    for _ in range(100):  # 2 s
        runner.tick()
        d = runner.data
        for i in range(d.ncon):
            c = d.contact[i]
            if floor not in (c.geom1, c.geom2):
                continue
            frictions.add(round(float(c.friction[0]), 12))
            mujoco.mj_contactForce(entry.model, d, i, force)
            if force[0] > 1e-6:
                ratios.append(float(np.hypot(force[1], force[2]) / force[0]))
    record = runner.run(EpisodeConfig((0.0, 0.0, 0.0), 2.0, push=push))
    return frictions, np.array(ratios), record, entry


@pytest.mark.parametrize("mu", [0.1, 0.6])
def test_runtime_contact_friction_and_cone(stack, mu) -> None:
    frictions, ratios, _, _ = _contact_ratio_episode(stack, mu)
    assert frictions == {round(mu, 12)}  # every observed floor contact uses the configured value
    assert ratios.max() <= mu * (1 + 1e-6)  # tangential/normal never exceeds mu (pyramidal cone)
    if mu == 0.1:
        assert ratios.max() >= 0.95 * mu  # saturates at mu while the feet slide


def test_lower_friction_increases_slip_under_the_same_shove(stack) -> None:
    slips = {}
    for mu in (0.1, 0.6):
        _, _, record, entry = _contact_ratio_episode(stack, mu)
        slips[mu] = foot_slip(entry.model, record)["total_slip_distance_m"]
    assert slips[0.1] > 2 * slips[0.6], slips


def test_payload_walking_and_mass(stack) -> None:
    result = run_single(scenario(mass=5.0, offset=[0, 0, 0.05]), stack, None)
    mass = result["model"]["mass"]
    assert mass["total_mass_kg"] == pytest.approx(12.743448 + 5.0)
    assert mass["payload"]["positive_definite"]
    assert result["changed_from_original"][0]["parameter"] == "payload"
    assert result["cases"]["scenario"]["digest"] != result["cases"]["original"]["digest"]


def test_single_run_with_push_is_paired_and_reproducible(stack, tmp_path) -> None:
    push = {"force_n": 100.0, "direction_deg": 270.0, "start_s": 2.0, "duration_s": 0.1}
    s = scenario(mu=0.3, mass=3.0, offset=[0, 0, 0.04], push=push)
    a = run_single(s, stack, tmp_path / "a")
    b = run_single(s, stack, None)
    assert a["results_sha256"] == b["results_sha256"]
    assert a["primary_baseline"] == "matched" and a["paired"]["pairing_ok"]
    assert a["cases"]["matched"]["physics"]["payload_kg"] == 3.0
    assert a["cases"]["matched"]["physics"]["push"] is None
    assert a["cases"]["original"]["physics"]["ground_friction_mu"] == 0.6
    saved = json.loads((tmp_path / "a" / "scenario_result.json").read_text())
    assert saved["results_sha256"] == a["results_sha256"]
    for case in ("scenario", "matched", "original"):
        traj, manifest = read_run(tmp_path / "a" / case)
        assert manifest["case"] == case and manifest["scenario"]["name"] == "t"
        assert len(traj["qpos"]) == 250
    mods = read_run(tmp_path / "a" / "scenario")[1]["scene"]["modifications"]
    assert mods["friction"]["sliding_after"] == 0.3 and mods["payload"]["mass_kg"] == 3.0


def test_suite_varies_only_push_start(stack, tmp_path) -> None:
    push = {"force_n": 100.0, "direction_deg": 270.0, "start_s": 2.0, "duration_s": 0.1}
    s = scenario(mu=0.3, mass=3.0, push=push)
    result = run_scenario_suite(s, stack, tmp_path / "suite", trials=4, interval_s=(2.0, 2.064),
                                seed=1)  # fmt: skip
    report = json.loads((tmp_path / "suite" / "experiment.json").read_text())
    starts = [t["start_substep"] for t in report["trials"]]
    assert len(set(starts)) == 4 and all(500 <= k < 516 for k in starts)
    assert report["aggregates"]["nominal"]["distinct_trajectories"] == 1  # matched baseline
    assert report["aggregates"]["paired"]["pairing_ok_all"]
    assert result["primary_baseline"] == "matched" and result["original"]["outcome"] == "survived"
    with pytest.raises(ValueError, match="needs a push"):
        run_scenario_suite(scenario(mu=0.3), stack, tmp_path / "x")


def test_original_and_low_friction_differ_in_behaviour(stack) -> None:
    result = run_single(scenario(mu=0.2), stack, None)
    cases = result["cases"]
    assert (
        cases["scenario"]["slip"]["mean_slip_speed_mps"]
        > cases["original"]["slip"]["mean_slip_speed_mps"]
    )
    assert result["model"]["friction"]["observed_at_reset"]["sliding_mu_values"] == [0.2]
