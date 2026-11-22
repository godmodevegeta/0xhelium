"""Bounded residual reflex around the frozen go1-walk-s0 actor (M6).

`a = clip(a_base + r, -1, 1)` in the policy's action units (`ctrl = default_pose + 0.5 * a`; the
trained range `a ∈ [-1, 1]` maps inside every actuator ctrlrange, see `check_action_bounds`).

The residual reads only the actor observation (offsets from the policy's declared fields):

    f = (gravity_x, gravity_y, gyro_x, gyro_y, linvel_y - command_vy),   d = f - centre

It is gated by a calibrated dead-zone, shaped by three fixed joint bases and limited in size and
rate:

    m      = max_i |d_i| / half_width_i
    A      = smoothstep(clip((m - 1 - margin) / (0.25 + width), 0, 1))
    target = clip(amplitude * A * (tanh(u_shift) B_shift + tanh(u_roll) B_roll
                                   + tanh(u_pitch) B_pitch), -0.30, 0.30)
    r_t    = r_{t-1} + clip(target - r_{t-1}, -0.05, 0.05)

with u_shift = k0 d4 + k1 d1 + k2 d2, u_roll = k3 d1 + k4 d2, u_pitch = k5 d0 + k6 d3.

Bases (policy joint order FR, FL, RR, RL x hip, thigh, calf; verified against MuJoCo kinematics in
the tests): `shift` +1 on every hip moves every foot toward +y (left); `roll` extends the left legs
and shortens the right legs; `pitch` extends the front legs and shortens the rear legs. A leg
"extension" is calf +1 with thigh -0.534, the measured ratio that keeps the foot under the hip.

With every gain zero, or amplitude zero, `ReflexActor.act` returns the base actor's output
unchanged, so episodes are bit-identical to the frozen policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1
SIGNALS = ("gravity_x", "gravity_y", "gyro_x", "gyro_y", "lateral_velocity_error")
SIGNAL_UNITS = ("unit gravity", "unit gravity", "rad/s", "rad/s", "m/s")
RESIDUAL_CAP = 0.30  # action units per joint (0.15 rad of joint target)
RATE_LIMIT = 0.05  # action units per joint per 20 ms tick
WIDTH_FLOOR = 0.25
LEG_EXTENSION = {"hip": 0.0, "thigh": -0.534, "calf": 1.0}
BASIS_NAMES = ("shift", "roll", "pitch")


class ReflexError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    unit: str
    doc: str


PARAMS = (
    ParamSpec("shift_from_lateral_velocity", -2.0, 2.0, "action/(m/s)",
              "lateral-velocity error -> shift basis (feet toward +y)"),
    ParamSpec("shift_from_roll", -10.0, 10.0, "action/unit gravity_y",
              "gravity_y deviation -> shift basis"),
    ParamSpec("shift_from_roll_rate", -1.0, 1.0, "action/(rad/s)",
              "roll-rate deviation -> shift basis"),
    ParamSpec("roll_from_roll", -10.0, 10.0, "action/unit gravity_y",
              "gravity_y deviation -> roll basis (left legs extend)"),
    ParamSpec("roll_from_roll_rate", -1.0, 1.0, "action/(rad/s)",
              "roll-rate deviation -> roll basis"),
    ParamSpec("pitch_from_pitch", -10.0, 10.0, "action/unit gravity_x",
              "gravity_x deviation -> pitch basis (front legs extend)"),
    ParamSpec("pitch_from_pitch_rate", -1.0, 1.0, "action/(rad/s)",
              "pitch-rate deviation -> pitch basis"),
    ParamSpec("deadzone_margin", 0.0, 2.0, "envelope units",
              "activation starts at 1 + margin calibrated half-widths"),
    ParamSpec("ramp_width", 0.0, 2.0, "envelope units",
              "activation reaches 1 over 0.25 + width half-widths"),
    ParamSpec("amplitude", 0.0, RESIDUAL_CAP, "action units",
              "largest contribution of each basis"),
)  # fmt: skip
PARAM_NAMES = tuple(p.name for p in PARAMS)
N_PARAMS = len(PARAMS)
N_GAINS = 7


def schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "robustlab-reflex",
        "action": "a = clip(a_base + r, -1, 1); ctrl = default_pose + 0.5 a",
        "signals": [{"name": n, "unit": u} for n, u in zip(SIGNALS, SIGNAL_UNITS, strict=True)],
        "parameters": [p.__dict__ for p in PARAMS],
        "bases": {
            "shift": "+1 on every hip: feet move toward +y (left)",
            "roll": "left legs extend, right legs shorten",
            "pitch": "front legs extend, rear legs shorten",
            "leg_extension": LEG_EXTENSION,
        },
        "residual_cap": RESIDUAL_CAP,
        "rate_limit_per_tick": RATE_LIMIT,
        "width_floor": WIDTH_FLOOR,
        "activation": "smoothstep(clip((max_i |d_i|/e_i - 1 - margin) / (0.25 + width), 0, 1))",
    }


def validate_params(values) -> np.ndarray:
    """Return a float64 copy of a 10-value parameter vector, or raise ReflexError."""
    try:
        array = np.array(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ReflexError("params", f"not numeric ({exc})") from exc
    if array.shape != (N_PARAMS,):
        raise ReflexError("params", f"expected {N_PARAMS} values, got {array.size}")
    for spec, value in zip(PARAMS, array, strict=True):
        if not math.isfinite(value):
            raise ReflexError(spec.name, "must be finite")
        if not spec.lo <= value <= spec.hi:
            raise ReflexError(spec.name, f"{value:g} outside [{spec.lo:g}, {spec.hi:g}]")
    array.setflags(write=False)
    return array


def params_dict(params: np.ndarray) -> dict[str, float]:
    return {name: float(v) for name, v in zip(PARAM_NAMES, params, strict=True)}


def is_zero_residual(params: np.ndarray) -> bool:
    return bool(not np.any(params[:N_GAINS]) or params[9] == 0.0)


def joint_bases(joint_names) -> dict[str, np.ndarray]:
    """Basis vectors in policy joint order, built from names like `FL_calf_joint`."""
    bases = {name: np.zeros(len(joint_names), np.float32) for name in BASIS_NAMES}
    for i, name in enumerate(joint_names):
        leg, joint, suffix = name.split("_")
        if suffix != "joint" or leg not in ("FR", "FL", "RR", "RL") or joint not in LEG_EXTENSION:
            raise ReflexError("joint_names", f"unexpected joint name {name!r}")
        left = 1.0 if leg[1] == "L" else -1.0
        front = 1.0 if leg[0] == "F" else -1.0
        if joint == "hip":
            bases["shift"][i] = 1.0
        bases["roll"][i] = left * LEG_EXTENSION[joint]
        bases["pitch"][i] = front * LEG_EXTENSION[joint]
    for b in bases.values():
        b.setflags(write=False)
    return bases


def signal_indices(policy) -> dict[str, int]:
    """Observation indices of the reflex inputs, from the policy's declared field offsets."""
    offsets = {name: offset for name, offset, _ in policy.fields}
    return {
        "gravity_x": offsets["gravity"] + 0,
        "gravity_y": offsets["gravity"] + 1,
        "gyro_x": offsets["gyro"] + 0,
        "gyro_y": offsets["gyro"] + 1,
        "linvel_y": offsets["local_linear_velocity"] + 1,
        "command_vy": offsets["command"] + 1,
    }


def features(obs: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """f = (gravity_x, gravity_y, gyro_x, gyro_y, linvel_y - command_vy) as float64."""
    o = obs.astype(np.float64, copy=False)
    return np.array([o[idx["gravity_x"]], o[idx["gravity_y"]], o[idx["gyro_x"]],
                     o[idx["gyro_y"]], o[idx["linvel_y"]] - o[idx["command_vy"]]])  # fmt: skip


def check_action_bounds(model, policy) -> None:
    """The trained action range a ∈ [-1, 1] must map inside every actuator ctrlrange."""
    lo = policy.default_pose - policy.action_scale
    hi = policy.default_pose + policy.action_scale
    rng = model.actuator_ctrlrange
    if not (
        np.all(model.actuator_ctrllimited) and np.all(lo >= rng[:, 0]) and np.all(hi <= rng[:, 1])
    ):
        raise ReflexError("action", "default_pose ± action_scale leaves an actuator ctrlrange")


@dataclass(frozen=True)
class Calibration:
    """Nominal envelope of the reflex signals (centre and half-width per signal)."""

    center: tuple[float, ...]
    half_width: tuple[float, ...]
    source: dict

    def __post_init__(self) -> None:
        if len(self.center) != len(SIGNALS) or len(self.half_width) != len(SIGNALS):
            raise ReflexError("calibration", f"expected {len(SIGNALS)} values per field")
        for c, e in zip(self.center, self.half_width, strict=True):
            if not (math.isfinite(c) and math.isfinite(e) and e > 0):
                raise ReflexError("calibration", "centre must be finite, half-width finite > 0")

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "kind": "robustlab-reflex-calibration",
                "signals": list(SIGNALS), "center": list(self.center),
                "half_width": list(self.half_width), "source": self.source}  # fmt: skip

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, doc: dict) -> Calibration:
        if doc.get("kind") != "robustlab-reflex-calibration" or list(
            doc.get("signals", [])
        ) != list(SIGNALS):
            raise ReflexError("calibration", "not a robustlab-reflex-calibration for these signals")
        return cls(tuple(map(float, doc["center"])), tuple(map(float, doc["half_width"])),
                   dict(doc.get("source", {})))  # fmt: skip


def envelope(samples: np.ndarray, source: dict) -> Calibration:
    """Centre = per-signal median; half-width = largest |sample - median| over all calibration
    ticks [N, 5], so every calibration tick has abnormality m <= 1."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != len(SIGNALS) or not np.isfinite(samples).all():
        raise ReflexError("calibration", "samples must be a finite [N, 5] array")
    center = np.median(samples, axis=0)
    half = np.abs(samples - center).max(axis=0)
    return Calibration(tuple(center.tolist()), tuple(half.tolist()),
                       {**source, "ticks": int(samples.shape[0])})  # fmt: skip


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class Reflex:
    """Stateful residual computation (state: the previous residual, for the rate limit)."""

    def __init__(self, policy, calibration: Calibration, params) -> None:
        self.params = validate_params(params)
        self.calibration = calibration
        self.idx = signal_indices(policy)
        self.bases = joint_bases(policy.joint_names)
        self.center = np.asarray(calibration.center, np.float64)
        self.inv_half = 1.0 / np.asarray(calibration.half_width, np.float64)
        self.zero = is_zero_residual(self.params)
        self.r = np.zeros(len(policy.joint_names), np.float32)
        self.activation = 0.0

    def reset(self) -> None:
        self.r[:] = 0
        self.activation = 0.0

    def abnormality(self, obs: np.ndarray) -> float:
        d = features(obs, self.idx) - self.center
        return float(np.max(np.abs(d) * self.inv_half))

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Update and return the residual for one observation (an internal buffer)."""
        k = self.params
        if self.zero:
            self.activation = 0.0
            return self.r
        d = features(obs, self.idx) - self.center
        if not np.isfinite(d).all():
            self.r[:] = 0
            self.activation = 0.0
            return self.r
        m = float(np.max(np.abs(d) * self.inv_half))
        a = smoothstep((m - 1.0 - k[7]) / (WIDTH_FLOOR + k[8]))
        self.activation = a
        if a > 0.0:
            u_shift = k[0] * d[4] + k[1] * d[1] + k[2] * d[2]
            u_roll = k[3] * d[1] + k[4] * d[2]
            u_pitch = k[5] * d[0] + k[6] * d[3]
            target = (k[9] * a) * (
                math.tanh(u_shift) * self.bases["shift"].astype(np.float64)
                + math.tanh(u_roll) * self.bases["roll"]
                + math.tanh(u_pitch) * self.bases["pitch"]
            )
            np.clip(target, -RESIDUAL_CAP, RESIDUAL_CAP, out=target)
        else:
            target = np.zeros(self.r.shape)
        step = np.clip(target - self.r, -RATE_LIMIT, RATE_LIMIT)
        np.add(self.r, step, out=self.r, casting="same_kind")
        np.clip(self.r, -RESIDUAL_CAP, RESIDUAL_CAP, out=self.r)
        return self.r


class ReflexActor:
    """Frozen base actor + Reflex. Records activation and residual per call (= per Tick)."""

    def __init__(self, base_actor, policy, calibration: Calibration, params) -> None:
        self.base = base_actor
        self.reflex = Reflex(policy, calibration, params)
        self.policy = policy
        self._action = np.empty(len(policy.joint_names), np.float32)
        self.activations: list[float] = []
        self.residuals: list[np.ndarray] = []

    @property
    def zero(self) -> bool:
        return self.reflex.zero

    def reset(self) -> None:
        self.reflex.reset()
        self.activations = []
        self.residuals = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        base = self.base.act(obs)
        r = self.reflex.step(obs)
        self.activations.append(self.reflex.activation)
        self.residuals.append(r.copy())
        if not r.any():
            return base  # exact identity: the base actor's own values
        np.add(base, r, out=self._action)
        return np.clip(self._action, -1.0, 1.0, out=self._action)

    def summary(self, upto_tick: int | None = None, ctrl_dt: float = 0.02) -> dict:
        """Activation and residual statistics over Ticks [0, upto_tick] (inclusive)."""
        n = len(self.activations) if upto_tick is None else min(len(self.activations),
                                                                 upto_tick + 1)  # fmt: skip
        act = np.asarray(self.activations[:n], np.float64)
        res = np.asarray(self.residuals[:n], np.float64).reshape(n, -1)
        active = np.flatnonzero(act > 0)
        peak = np.abs(res).max(axis=0) if n else np.zeros(len(self.policy.joint_names))
        return {
            "ticks": int(n),
            "active_ticks": int(active.size),
            "activation_fraction": float(active.size / n) if n else 0.0,
            "first_activation_s": float(active[0] * ctrl_dt) if active.size else None,
            "mean_abs_residual": float(np.abs(res).mean()) if n else 0.0,
            "max_abs_residual": float(peak.max()) if n else 0.0,
            "peak_residual_by_joint": {
                j: float(p) for j, p in zip(self.policy.joint_names, peak, strict=True)
            },
            "affected_joints": [
                j for j, p in zip(self.policy.joint_names, peak, strict=True) if p > 1e-6
            ],
            "time_note": "activation at Tick k is computed from the observation before Tick k "
            "(time k * 0.02 s)",
        }
