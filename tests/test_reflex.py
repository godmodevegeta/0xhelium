"""Residual reflex (M6 Requirements 1-2): identity, bounds, dead-zone, conventions, parameters."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import build_model
from robustlab.adapt.harden import decode
from robustlab.adapt.reflex import (
    N_PARAMS,
    PARAMS,
    RATE_LIMIT,
    RESIDUAL_CAP,
    SIGNALS,
    Calibration,
    Reflex,
    ReflexActor,
    ReflexError,
    check_action_bounds,
    envelope,
    features,
    joint_bases,
    signal_indices,
    smoothstep,
    validate_params,
)
from robustlab.policy.actor import Actor
from robustlab.policy.observation import ObservationBuilder

CAL = Calibration((0.016, 0.005, 0.0, -0.002, 0.001), (0.04, 0.09, 2.7, 1.07, 0.30), {"test": 1})
LO = np.array([p.lo for p in PARAMS])
HI = np.array([p.hi for p in PARAMS])

unit = st.floats(-1.0, 1.0, allow_nan=False)
obs_value = st.floats(-20.0, 20.0, allow_nan=False, width=32)


def obs_strategy(n: int = 48):
    return st.lists(obs_value, min_size=n, max_size=n).map(lambda v: np.asarray(v, np.float32))


@pytest.fixture(scope="module")
def actor(policy):
    return Actor(policy)


class FixedActor:
    """A base actor returning a fixed action buffer (to check identity without the network)."""

    def __init__(self, action) -> None:
        self.action = np.asarray(action, np.float32)

    def act(self, obs):
        return self.action


def test_zero_vector_is_valid_and_means_no_residual(policy, actor) -> None:
    params = validate_params(np.zeros(N_PARAMS))
    wrapped = ReflexActor(actor, policy, CAL, params)
    assert wrapped.zero
    obs = np.full(48, 3.0, np.float32)
    expected = Actor(policy).act(obs).copy()
    out = wrapped.act(obs)
    assert out is actor._action  # the base actor's own buffer, untouched
    np.testing.assert_array_equal(out, expected)


@given(st.lists(unit, min_size=N_PARAMS, max_size=N_PARAMS), st.booleans(), obs_strategy())
def test_property_zero_gains_or_amplitude_is_exact_identity(policy, x, kill_amplitude, obs):
    params = np.array(decode(x))
    if kill_amplitude:
        params[9] = 0.0
    else:
        params[:7] = 0.0
    base = FixedActor(np.linspace(-1, 1, 12))
    wrapped = ReflexActor(base, policy, CAL, params)
    out = wrapped.act(obs)
    assert out is base.action and wrapped.activations == [0.0]


@given(st.lists(unit, min_size=N_PARAMS, max_size=N_PARAMS),
       st.lists(obs_strategy(), min_size=1, max_size=12),
       st.lists(st.floats(-1, 1, allow_nan=False, width=32), min_size=12, max_size=12))  # fmt: skip
def test_property_residual_and_action_bounds(policy, x, observations, base_action):
    params = decode(x)
    reflex = Reflex(policy, CAL, params)
    wrapped = ReflexActor(FixedActor(base_action), policy, CAL, params)
    previous = np.zeros(12)
    for obs in observations:
        r = reflex.step(obs).astype(np.float64)
        assert np.isfinite(r).all()
        assert np.all(np.abs(r) <= RESIDUAL_CAP + 1e-7)
        assert np.all(np.abs(r - previous) <= RATE_LIMIT + 1e-6)
        assert 0.0 <= reflex.activation <= 1.0
        previous = r
        a = wrapped.act(obs)
        assert np.isfinite(a).all() and np.all(a >= -1) and np.all(a <= 1)


def test_nonfinite_observation_gives_zero_residual(policy) -> None:
    params = decode(np.full(N_PARAMS, 0.5))
    reflex = Reflex(policy, CAL, params)
    obs = np.zeros(48, np.float32)
    obs[1] = 5.0  # large lateral velocity: active
    assert np.any(reflex.step(obs))
    obs[6] = np.nan
    assert not np.any(reflex.step(obs)) and reflex.activation == 0.0


@pytest.mark.parametrize(
    "values",
    [np.zeros(9), np.zeros(11), [math.nan] + [0.0] * 9, [3.0] + [0.0] * 9,
     [0.0] * 9 + [0.31], [0.0] * 7 + [-0.1, 0.0, 0.0], "abc"],
)  # fmt: skip
def test_invalid_parameters_are_rejected(values) -> None:
    with pytest.raises(ReflexError):
        validate_params(values)


@given(st.lists(unit, min_size=N_PARAMS, max_size=N_PARAMS))
def test_property_decode_stays_in_bounds(x) -> None:
    p = decode(x)
    assert np.all(p >= LO) and np.all(p <= HI)


def test_decode_origin_is_zero_gain() -> None:
    p = decode(np.zeros(N_PARAMS))
    assert not np.any(p[:7]) and p[7] == 1.0 and p[8] == 1.0 and p[9] == pytest.approx(0.15)


def test_smoothstep_is_continuous_and_bounded() -> None:
    xs = np.linspace(-1, 2, 3001)
    ys = np.array([smoothstep(x) for x in xs])
    assert ys.min() == 0.0 and ys.max() == 1.0
    assert np.all(np.diff(ys) >= 0) and np.max(np.abs(np.diff(ys))) < 0.002


@given(st.lists(st.lists(st.floats(-5, 5, allow_nan=False), min_size=5, max_size=5),
                min_size=2, max_size=60), st.floats(0.0, 2.0))  # fmt: skip
def test_property_calibration_ticks_never_activate(policy, rows, margin) -> None:
    samples = np.asarray(rows)
    samples[0] += 1e-3  # half-widths must be > 0
    samples = samples.astype(np.float32).astype(np.float64)  # observations are float32
    cal = envelope(samples, {"test": True})
    params = np.array(decode(np.full(N_PARAMS, 0.3)))
    params[7] = margin
    idx = signal_indices(policy)
    reflex = Reflex(policy, cal, params)
    for row in samples:
        obs = np.zeros(48, np.float32)
        obs[[idx["gravity_x"], idx["gravity_y"], idx["gyro_x"], idx["gyro_y"]]] = row[:4]
        obs[idx["linvel_y"]] = row[4]
        assert reflex.abnormality(obs.astype(np.float64)) <= 1.0 + 1e-6
        reflex.step(obs)
        assert reflex.activation == 0.0


def test_activation_ramp(policy) -> None:
    params = np.array(decode(np.zeros(N_PARAMS)))
    params[0] = 1.0  # nonzero gain so the reflex is live; margin 1, width 1 -> ramp m in [2, 3.25]
    reflex = Reflex(policy, CAL, params)
    idx = signal_indices(policy)
    values = []
    for m in (0.5, 1.9, 1.99, 2.5, 3.3, 5.0):
        obs = np.zeros(48, np.float32)
        obs[idx["linvel_y"]] = CAL.center[4] + m * CAL.half_width[4]
        for i, name in enumerate(("gravity_x", "gravity_y", "gyro_x", "gyro_y")):
            obs[idx[name]] = CAL.center[i]
        reflex.reset()
        reflex.step(obs)
        values.append(reflex.activation)
    assert values[:3] == [0.0, 0.0, 0.0] and 0 < values[3] < 1 and values[4] == values[5] == 1.0


def test_signal_indices_follow_policy_fields(policy) -> None:
    idx = signal_indices(policy)
    assert idx == {"gravity_x": 6, "gravity_y": 7, "gyro_x": 3, "gyro_y": 4, "linvel_y": 1,
                   "command_vy": 46}  # fmt: skip
    assert SIGNALS == ("gravity_x", "gravity_y", "gyro_x", "gyro_y", "lateral_velocity_error")


def test_features_match_mujoco_sensors_and_gravity_sign(policy) -> None:
    model, ids = build_model(policy, "fall-contact")
    data = mujoco.MjData(model)
    data.qpos[:] = policy.default_qpos
    roll = 0.2  # rotate the trunk about +x: left side up
    data.qpos[3:7] = [math.cos(roll / 2), math.sin(roll / 2), 0, 0]
    data.qvel[3] = 0.7  # roll rate
    data.qvel[1] = -0.4  # world y velocity
    mujoco.mj_forward(model, data)
    obs = ObservationBuilder(model, ids, policy).build(
        data, np.zeros(12, np.float32), np.array([1.0, 0.2, 0.0], np.float32),
        np.zeros(48, np.float32))  # fmt: skip
    f = features(obs, signal_indices(policy))
    assert f[1] == pytest.approx(-math.sin(roll), abs=1e-5)  # gravity_y = -sin(roll)
    assert f[2] == pytest.approx(0.7, abs=1e-5)  # gyro x = roll rate
    gyro = data.sensordata[ids.gyro_adr : ids.gyro_adr + 3]
    vel = data.sensordata[ids.velocimeter_adr : ids.velocimeter_adr + 3]
    assert f[2] == pytest.approx(gyro[0]) and f[3] == pytest.approx(gyro[1])
    assert f[4] == pytest.approx(vel[1] - 0.2, abs=1e-6)


def _feet_body_frame(model, data, ids, qjoint):
    data.qpos[7:] = qjoint
    mujoco.mj_kinematics(model, data)
    rot = data.xmat[ids.trunk_body].reshape(3, 3)
    base = data.xpos[ids.trunk_body]
    return np.array([rot.T @ (data.geom_xpos[g] - base) for g in ids.foot_geoms])


def test_joint_bases_against_mujoco_kinematics(policy) -> None:
    """shift: every foot +y; roll: left feet down, right feet up; pitch: front down, rear up;
    leg extension keeps the foot roughly under the hip (|dx| small)."""
    model, ids = build_model(policy, "fall-contact")
    data = mujoco.MjData(model)
    data.qpos[:] = policy.default_qpos
    bases = joint_bases(policy.joint_names)
    feet0 = _feet_body_frame(model, data, ids, policy.default_pose)
    delta = 0.1  # action units = 0.05 rad
    left = np.array([False, True, False, True])  # FR, FL, RR, RL
    front = np.array([True, True, False, False])
    for name, basis in bases.items():
        q = policy.default_pose + policy.action_scale * delta * basis
        moved = _feet_body_frame(model, data, ids, q) - feet0
        if name == "shift":
            assert np.all(moved[:, 1] > 0.01)
        else:
            up_side = left if name == "roll" else front
            assert np.all(moved[up_side, 2] < -0.005) and np.all(moved[~up_side, 2] > 0.005)
            assert np.all(np.abs(moved[:, 0]) < 0.3 * np.abs(moved[:, 2]))
    assert bases["shift"].tolist() == [1, 0, 0] * 4


def test_trained_action_range_is_inside_ctrlrange(policy) -> None:
    model, _ = build_model(policy, "fall-contact")
    check_action_bounds(model, policy)
    narrow = model
    saved = narrow.actuator_ctrlrange.copy()
    try:
        narrow.actuator_ctrlrange[2, 1] = -1.5  # calf upper limit below default + 0.5
        with pytest.raises(ReflexError):
            check_action_bounds(narrow, policy)
    finally:
        narrow.actuator_ctrlrange[:] = saved


def test_summary_reports_activation_and_joints(policy) -> None:
    params = decode(np.full(N_PARAMS, 0.5))
    wrapped = ReflexActor(FixedActor(np.zeros(12)), policy, CAL, params)
    quiet = np.zeros(48, np.float32)
    loud = quiet.copy()
    loud[1] = 3.0
    for obs in (quiet, quiet, loud, loud, loud):
        wrapped.act(obs)
    s = wrapped.summary()
    assert s["ticks"] == 5 and s["active_ticks"] == 3 and s["first_activation_s"] == 0.04
    assert s["affected_joints"] and s["max_abs_residual"] <= RESIDUAL_CAP
    assert wrapped.summary(upto_tick=1)["active_ticks"] == 0
