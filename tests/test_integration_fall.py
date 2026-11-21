"""M1 closed-loop evidence: golden reproduction in both scenes, nominal walking in the
Fall_Contact_Scene, and a deterministic sideways shove (Requirements 1.3, 1.4, 7.4)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import REPO_ROOT, build_model
from robustlab.experiments.push import DIRECTIONS, PushScenario
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, evaluate
from robustlab.policy.actor import Actor

pytestmark = pytest.mark.integration

FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_COMMANDS = {"forward": (1.0, 0.0, 0.0), "zero": (0.0, 0.0, 0.0)}
SHOVE = PushScenario(force_n=100.0, direction=DIRECTIONS["right"], start_s=2.0, duration_s=0.1)


def _run(policy, scene, command, seconds, push=None):
    model, ids = build_model(policy, scene)
    record = EpisodeRunner(model, ids, policy, Actor(policy)).run(
        EpisodeConfig(command, seconds, push=push)
    )
    return record, evaluate(record)


@pytest.fixture(scope="module")
def golden():
    meta = json.loads((FIXTURES / "m0_golden.json").read_text())
    raw = (FIXTURES / "m0_golden.npz").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta["npz_sha256"]
    return np.load(FIXTURES / "m0_golden.npz")


@pytest.mark.parametrize("scene", ["feet-only", "fall-contact"])
@pytest.mark.parametrize("name", ["forward", "zero"])
def test_reproduces_m0_golden_trajectory(policy, golden, scene, name) -> None:
    record, out = _run(policy, scene, GOLDEN_COMMANDS[name], 10.0)
    for field in ("qpos", "qvel", "action", "contact"):
        reference = golden[f"{name}_{field}"]
        diff = np.max(np.abs(getattr(record, field).astype(float) - reference.astype(float)))
        print(f"\n{scene} {name} {field}: max abs diff vs M0 golden = {diff:.3e}")
        np.testing.assert_array_equal(getattr(record, field), reference, err_msg=field)
    assert out["fall"]["kind"] == "none" and out["passed"], out["failed_checks"]
    assert not record.nonfoot_contact.any()


def test_fall_contact_forward_walking_meets_m0_criteria(policy) -> None:
    record, out = _run(policy, "fall-contact", (1.0, 0.0, 0.0), 10.0)
    assert out["criteria_set"] == "forward" and out["passed"], out["failed_checks"]
    assert out["mean_forward_speed"] >= 0.7 and min(out["touchdowns"]) >= 5
    assert record.contact.any(axis=0).all()  # every foot touches down


def test_sideways_shove_is_measured_and_deterministic(policy) -> None:
    first, out = _run(policy, "fall-contact", (1.0, 0.0, 0.0), 5.0, SHOVE)
    second, _ = _run(policy, "fall-contact", (1.0, 0.0, 0.0), 5.0, SHOVE)
    for field in ("qpos", "qvel", "action", "contact", "nonfoot_contact", "push_impulse"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field), err_msg=field)
    assert first.fall.to_dict() == second.fall.to_dict()
    fall = out["fall"]
    print(
        f"\nsideways shove 100 N x 0.1 s toward -y at 2.0 s: {fall['kind']} {fall['reason']} "
        f"at {fall['time_s']} s; events { {k: v['time_s'] for k, v in fall['events'].items()} }"
    )
    # Whatever happens, it must happen after the shove starts: before it the run is nominal.
    nominal, _ = _run(policy, "fall-contact", (1.0, 0.0, 0.0), 5.0)
    before = first.time < SHOVE.start_s - 1e-9
    np.testing.assert_array_equal(first.qpos[before], nominal.qpos[before])
    if fall["kind"] != "none":
        assert fall["time_s"] >= SHOVE.start_s
    np.testing.assert_allclose(first.push_impulse.sum(axis=0), [0.0, -10.0, 0.0], atol=1e-9)
