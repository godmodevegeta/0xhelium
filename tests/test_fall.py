"""Fall_Monitor and FallResult tests (Requirements 3 and 4)."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from conftest import build_model
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, evaluate
from robustlab.physics.fall import (
    NUMERICAL_REASONS,
    PHYSICAL_REASONS,
    REASONS,
    FallEvent,
    FallMonitor,
    height_ok,
    make_result,
    nonfoot_floor_mask,
    tilt_ok,
)
from robustlab.policy.actor import Actor

SEED = 21
COS45 = math.cos(math.radians(45))


# Feature: fall-detection, Property 15: Primary reason is the earliest event with fixed tie order
def test_property_primary_reason() -> None:
    rng = np.random.default_rng(SEED)
    for case in range(300):
        chosen = [r for r in REASONS if rng.random() < 0.5]
        events = {
            r: FallEvent(r, int(rng.integers(0, 4)), 0.0, {}) for r in chosen
        }  # few distinct indices, so ties are common
        result = make_result(events, nonfoot_contact_detection=True)
        context = (
            f"seed={SEED} case={case} events={ {r: e.state_index for r, e in events.items()} }"
        )
        if not events:
            assert result.kind == "none" and result.reason is None, context
            assert not result.fell and not result.numerical_failure, context
            continue
        earliest = min(e.state_index for e in events.values())
        expected = next(r for r in REASONS if r in events and events[r].state_index == earliest)
        assert result.reason == expected, context
        assert result.kind == ("numerical" if expected in NUMERICAL_REASONS else "physical")
        assert result.fell == bool(PHYSICAL_REASONS & events.keys()), context
        assert result.numerical_failure == bool(NUMERICAL_REASONS & events.keys()), context


# Feature: fall-detection, Property 16: Tilt and height predicates at boundaries
def test_property_tilt_and_height_predicates() -> None:
    rng = np.random.default_rng(SEED)
    assert tilt_ok(COS45, COS45) and not tilt_ok(np.nextafter(COS45, -1), COS45)
    assert height_ok(0.18, 0.18) and not height_ok(np.nextafter(0.18, 0), 0.18)
    assert not tilt_ok(math.nan, COS45) and not height_ok(math.nan, 0.18)
    up = rng.uniform(-1, 1, 500)
    z = rng.uniform(0, 0.4, 500)
    np.testing.assert_array_equal(tilt_ok(up, COS45), [u >= COS45 for u in up])
    np.testing.assert_array_equal(height_ok(z, 0.18), [h >= 0.18 for h in z])


def _upside_down_on_floor(model) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, 0.04]  # trunk centre 4 cm above the floor
    data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # rotated 180 deg about x: legs point up
    mujoco.mj_forward(model, data)
    return data


def test_trunk_on_floor_is_nonfoot_contact(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    monitor = FallMonitor(model, ids, 45.0, 0.18)
    assert monitor.detection and int(nonfoot_floor_mask(model, ids).sum()) == 29
    data = _upside_down_on_floor(model)
    monitor.reset()
    mujoco.mj_step(model, data)
    assert monitor.check(data, 1) is True
    result = monitor.result()
    contact = result.events["nonfoot_contact"]
    assert contact.detail["body"] == "trunk" and contact.detail["dist_m"] < 0
    assert contact.state_index == 0  # contacts describe the state before the step
    assert {"tilt", "low_height"} <= result.events.keys()
    assert result.reason == "nonfoot_contact" and result.kind == "physical" and result.fell
    assert not result.numerical_failure


def test_feet_only_scene_reports_detection_unavailable(policy) -> None:
    model, ids = build_model(policy, "feet-only")
    monitor = FallMonitor(model, ids, 45.0, 0.18)
    data = _upside_down_on_floor(model)
    mujoco.mj_step(model, data)
    assert monitor.check(data, 1) is False
    result = monitor.result()
    assert not result.nonfoot_contact_detection
    assert "nonfoot_contact" not in result.events
    assert result.reason == "tilt"  # still caught by orientation


def test_nominal_standing_has_foot_contacts_and_no_fall(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    record = EpisodeRunner(model, ids, policy, Actor(policy)).run(EpisodeConfig((0, 0, 0), 1.0))
    assert record.contact.all(axis=1).any()  # all four feet down at some Tick
    assert not record.nonfoot_contact.any()
    assert record.fall.kind == "none" and record.fall.nonfoot_contact_detection


class NanActor:
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(12, np.nan, np.float32)


def test_numerical_failure_is_not_a_fall_and_truncates(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    record = EpisodeRunner(model, ids, policy, NanActor()).run(EpisodeConfig((0, 0, 0), 1.0))
    assert record.truncated and record.n_ticks == 1
    assert record.fall.kind == "numerical" and record.fall.numerical_failure
    assert record.fall.reason in NUMERICAL_REASONS
    outcome = evaluate(record)
    assert "numerical_failure" in outcome["failed_checks"] and not outcome["passed"]
    assert outcome["fall"]["kind"] == "numerical"


def test_nonfinite_state_detected(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    monitor = FallMonitor(model, ids, 45.0, 0.18)
    data = mujoco.MjData(model)
    data.qpos[:] = policy.default_qpos
    data.qpos[2] = np.nan
    monitor.check(data, 3)
    event = monitor.result().events["nonfinite_state"]
    assert event.state_index == 3 and "qpos" in event.detail["fields"]
    assert math.isclose(event.time_s, 3 * 0.004)
