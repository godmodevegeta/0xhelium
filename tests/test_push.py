"""PushScenario and push-application tests (Requirements 5 and 6)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from conftest import build_model
from robustlab.experiments.push import (
    DIRECTIONS,
    MAX_FORCE_N,
    PushError,
    PushScenario,
    heading_to_world,
    parse_direction,
)
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner
from robustlab.policy.actor import Actor

SEED = 33
DT = 0.004


def _push(**kw) -> PushScenario:
    base = {"force_n": 50.0, "direction": (0.0, -1.0, 0.0), "start_s": 0.1, "duration_s": 0.1}
    base.update(kw)
    return PushScenario(**base)


# Feature: fall-detection, Property 14: Invalid PushScenarios are rejected naming the field
def test_property_invalid_push_rejected() -> None:
    rng = np.random.default_rng(SEED)
    bad = {
        "force_n": [-1e-9, -5.0, MAX_FORCE_N + 1e-6, 1e9, math.nan, math.inf, "x"],
        "direction": [(0.0, 0.0, 0.0), (1.0, 0.0), (math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)],
        "start_s": [-DT, 0.003, 0.1005, math.nan],
        "duration_s": [-DT, 0.5 + DT, 0.002, math.inf],
    }
    cases = [(f, v) for f, values in bad.items() for v in values]
    for _ in range(100):  # random non-multiples of the physics step
        t = float(rng.uniform(0, 0.5))
        if abs(round(t / DT) * DT - t) > 1e-6:
            cases.append((str(rng.choice(["start_s", "duration_s"])), t))
    for field, value in cases:
        with pytest.raises(PushError) as info:
            _push(**{field: value})
        assert info.value.field == field, (field, value)


def test_valid_push_window_and_impulse() -> None:
    push = _push(force_n=100.0, start_s=2.0, duration_s=0.1)
    assert push.window() == (500, 525)
    assert math.isclose(push.impulse_ns, 10.0)
    assert _push(duration_s=0.0).window() == (25, 25)
    np.testing.assert_allclose(push.world_force(0.0), [0.0, -100.0, 0.0])
    np.testing.assert_allclose(_push(direction=(0, 3, 4)).unit_direction, [0, 0.6, 0.8])


def test_parse_direction() -> None:
    assert parse_direction("Right") == (0.0, -1.0, 0.0) == DIRECTIONS["right"]
    assert parse_direction("left") == (0.0, 1.0, 0.0)
    assert parse_direction("1,0,-0.5") == (1.0, 0.0, -0.5)
    for text in ("sideways", "1,0", "a,b,c"):
        with pytest.raises(PushError):
            parse_direction(text)


# Feature: fall-detection, Property 12: Heading-to-world rotation
def test_property_heading_to_world() -> None:
    rng = np.random.default_rng(SEED)
    np.testing.assert_allclose(heading_to_world([1, 0, 0], math.pi / 2), [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(heading_to_world([0, -1, 0], math.pi), [0, 1, 0], atol=1e-15)
    for case in range(200):
        yaw = float(rng.uniform(-math.pi, math.pi))
        v = rng.normal(size=3)
        w = heading_to_world(v, yaw)
        rot = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        np.testing.assert_allclose(w, rot @ v, atol=1e-12, err_msg=f"case={case}")
        assert math.isclose(np.linalg.norm(w), np.linalg.norm(v), rel_tol=1e-12)
        assert w[2] == v[2]


class ZeroActor:
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(12, np.float32)


# Feature: fall-detection, Property 13: Push acts on exactly the window substeps
def test_property_push_window_substeps(policy) -> None:
    rng = np.random.default_rng(SEED)
    model, ids = build_model(policy, "fall-contact")
    trunk = ids.trunk_body
    seen: list[np.ndarray] = []

    def on_control(_model, data):  # called once inside every mj_step, after xfrc is set
        seen.append(data.xfrc_applied.copy())

    mujoco.set_mjcb_control(on_control)
    try:
        for case in range(100):
            horizon_ticks = int(rng.integers(5, 25))
            n_sub = horizon_ticks * 5
            first = int(rng.integers(0, n_sub))
            length = int(rng.integers(0, min(n_sub - first, 125) + 1))
            push = PushScenario(
                float(rng.uniform(0, MAX_FORCE_N)),
                tuple(rng.normal(size=3)),
                first * DT,
                length * DT,
            )
            seen.clear()
            runner = EpisodeRunner(model, ids, policy, ZeroActor())
            record = runner.run(EpisodeConfig((0, 0, 0), horizon_ticks * 0.02, push=push))
            context = f"seed={SEED} case={case} window=[{first}, {first + length})"
            # The first callback is mj_forward inside reset(); the rest are the n_sub mj_steps.
            assert len(seen) == n_sub + 1, context
            assert not seen[0].any(), context
            force = push.world_force(0.0)
            for i, xfrc in enumerate(seen[1:]):
                expected = force if first <= i < first + length else np.zeros(3)
                np.testing.assert_array_equal(xfrc[trunk, :3], expected, err_msg=f"{context} i={i}")
                assert not xfrc[trunk, 3:].any(), context
                others = np.delete(xfrc, trunk, axis=0)
                assert not others.any(), context
            np.testing.assert_allclose(
                record.push_impulse.sum(axis=0), force * length * DT, atol=1e-9, err_msg=context
            )
            assert record.push["active_substeps"] == [first, first + length], context
    finally:
        mujoco.set_mjcb_control(None)


def test_reset_clears_external_force(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    runner = EpisodeRunner(model, ids, policy, ZeroActor())
    runner.reset((0, 0, 0), _push(start_s=0.0, duration_s=0.02))
    runner.tick()
    runner.data.xfrc_applied[:] = 7.0
    runner.reset((0, 0, 0))
    assert not runner.data.xfrc_applied.any()
    assert runner.push_window == (0, 0) and runner.substep == 0


def _qpos(policy, push, seconds=1.0, command=(0.5, 0, 0)):
    model, ids = build_model(policy, "fall-contact")
    runner = EpisodeRunner(model, ids, policy, Actor(policy))
    return runner.run(EpisodeConfig(command, seconds, push=push))


def test_zero_force_and_zero_duration_equal_nominal(policy) -> None:
    nominal = _qpos(policy, None)
    for push in (_push(force_n=0.0, start_s=0.2), _push(force_n=150.0, duration_s=0.0)):
        record = _qpos(policy, push)
        np.testing.assert_array_equal(record.qpos, nominal.qpos)
        np.testing.assert_array_equal(record.qvel, nominal.qvel)
        np.testing.assert_array_equal(record.action, nominal.action)
        assert not record.push_impulse.any()


@pytest.mark.parametrize(("name", "sign"), [("right", -1.0), ("left", 1.0)])
def test_push_moves_trunk_in_commanded_direction(policy, name, sign) -> None:
    push = _push(force_n=80.0, direction=DIRECTIONS[name], start_s=0.2, duration_s=0.1)
    nominal = _qpos(policy, None, seconds=0.4, command=(0, 0, 0))
    pushed = _qpos(policy, push, seconds=0.4, command=(0, 0, 0))
    before = pushed.time < 0.2 - 1e-9
    np.testing.assert_array_equal(pushed.qpos[before], nominal.qpos[before])  # nothing before
    end = int(np.flatnonzero(np.isclose(pushed.time, 0.3))[0])  # Tick ending at the push end
    dvy = pushed.qvel[end, 1] - nominal.qvel[end, 1]
    assert sign * dvy > 0.3, dvy  # 8 N*s on 12.7 kg is about 0.6 m/s before resistance
    assert abs(pushed.qvel[end, 0] - nominal.qvel[end, 0]) < abs(dvy)


def test_push_after_horizon_and_wrong_dt_rejected(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    runner = EpisodeRunner(model, ids, policy, ZeroActor())
    with pytest.raises(ValueError, match="horizon"):
        runner.run(EpisodeConfig((0, 0, 0), 0.2, push=_push(start_s=0.16, duration_s=0.08)))
    with pytest.raises(ValueError, match="sim_dt"):
        runner.reset((0, 0, 0), PushScenario(10.0, (0, 1, 0), 0.0, 0.0, sim_dt=0.002))
