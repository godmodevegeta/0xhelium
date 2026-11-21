"""Closed-loop episode runner for the pretrained Go1 walking policy, and its outcome evaluation.

One Tick (control period 0.02 s):

1. `action = actor.act(obs_t)`
2. `xfrc_applied = 0`; `ctrl = default_pose + 0.5 * action` (float32 arithmetic, no clipping;
   MuJoCo's actuator ctrlrange still applies)
3. `mj_step(model, data, nstep=5)` with that ctrl held for all 5 substeps
4. `obs_last_action = last_action`; `last_action = action`
5. `obs_{t+1} = build(data, obs_last_action, command)` without `mj_forward`

So observation `o[t+1]` carries action `a[t-1]`, as in training. Nothing in an episode is random;
`seed` is recorded in the manifest for later milestones but does not change M0 results.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from robustlab.physics.model import ResolvedIds
from robustlab.policy.actor import Policy
from robustlab.policy.observation import ObservationBuilder

WARNING_NAMES = tuple(
    name.removeprefix("mjWARN_") for name in mujoco.mjtWarning.__members__ if name != "mjNWARNING"
)


class ActorLike(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeConfig:
    command: tuple[float, float, float]
    seconds: float
    seed: int = 0


@dataclass(frozen=True)
class WalkCriteria:
    """Walking criteria fixed before any M0 run (Requirement 10). Written to every manifest."""

    tilt_deg: float = 45.0
    tilt_compare_deg: float = 60.0
    min_trunk_z: float = 0.18
    window_start_s: float = 2.0
    window_end_s: float = 10.0
    min_forward_speed: float = 0.7
    min_touchdowns: int = 5
    zero_cmd_max_disp: float = 0.3
    forward_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    zero_command: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class EpisodeRecord:
    """Per-Tick arrays, row k is the state after Tick k (time = (k + 1) * ctrl_dt)."""

    ctrl_dt: float
    sim_dt: float
    n_substeps: int
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    time: np.ndarray  # [T]
    qpos: np.ndarray  # [T, 19]
    qvel: np.ndarray  # [T, 18]
    action: np.ndarray  # [T, 12] action applied during Tick k
    command: np.ndarray  # [T, 3]
    contact: np.ndarray  # [T, 4] bool, foot-floor contact in FR, FL, RR, RL order
    up_z: np.ndarray  # [T]
    trunk_z: np.ndarray  # [T]
    finite: np.ndarray  # [T] bool, qpos, qvel and action all finite
    warning_total: np.ndarray  # [T] cumulative MuJoCo warning count
    tick_ns: np.ndarray  # [T] wall time of each Tick
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def n_ticks(self) -> int:
        return int(self.time.size)


def up_z_from_quat(quat: np.ndarray) -> np.ndarray:
    """World z component of the body z axis for wxyz quaternion(s): 1 - 2(x^2 + y^2)."""
    quat = np.asarray(quat, dtype=np.float64)
    norm2 = np.sum(quat * quat, axis=-1)
    return 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2) / norm2


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = (quat[..., i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class EpisodeRunner:
    """Owns one MjData and runs Ticks of the policy on it."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: ResolvedIds,
        policy: Policy,
        actor: ActorLike,
    ) -> None:
        self.model = model
        self.ids = ids
        self.policy = policy
        self.actor = actor
        self.data = mujoco.MjData(model)
        self.builder = ObservationBuilder(model, ids, policy)
        self.n_substeps = policy.n_substeps
        self.default_pose = policy.default_pose
        self.action_scale = np.float32(policy.action_scale)
        self.command = np.zeros(3, np.float32)
        self.last_action = np.zeros(model.nu, np.float32)
        self.obs_last_action = np.zeros(model.nu, np.float32)
        self.obs = np.zeros(policy.obs_dim, np.float32)
        self._target = np.empty(model.nu, np.float32)
        self._feet = np.asarray(ids.foot_geoms)
        self._floor = ids.floor_geom

    def reset(self, command: tuple[float, float, float]) -> np.ndarray:
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self.policy.default_qpos
        data.qvel[:] = self.policy.default_qvel
        data.ctrl[:] = data.qpos[7:]
        self.command[:] = command
        self.last_action[:] = 0
        self.obs_last_action[:] = 0
        mujoco.mj_forward(model, data)
        return self.builder.build(data, self.obs_last_action, self.command, self.obs)

    def tick(self) -> np.ndarray:
        """Run one control period from the current observation; returns the applied action
        (an internal buffer of the actor, copied by callers that keep it)."""
        data = self.data
        action = self.actor.act(self.obs)
        np.multiply(self.action_scale, action, out=self._target, dtype=np.float32)
        np.add(self.default_pose, self._target, out=self._target)
        data.xfrc_applied[:] = 0
        data.ctrl[:] = self._target
        mujoco.mj_step(self.model, data, nstep=self.n_substeps)
        self.obs_last_action[:] = self.last_action
        self.builder.build(data, self.obs_last_action, self.command, self.obs)
        self.last_action[:] = action
        return action

    def foot_contacts(self, out: np.ndarray) -> np.ndarray:
        """Foot-floor contact flags from the contacts of the last substep."""
        out[:] = False
        geoms = self.data.contact.geom[: self.data.ncon]
        if geoms.size:
            for i, foot in enumerate(self._feet):
                pair = ((geoms[:, 0] == foot) & (geoms[:, 1] == self._floor)) | (
                    (geoms[:, 1] == foot) & (geoms[:, 0] == self._floor)
                )
                out[i] = bool(pair.any())
        return out

    def warning_total(self) -> int:
        return int(sum(self.data.warning[i].number for i in range(len(WARNING_NAMES))))

    def run(self, config: EpisodeConfig) -> EpisodeRecord:
        ctrl_dt = self.policy.control_dt
        n_ticks = round(config.seconds / ctrl_dt)
        if n_ticks < 1 or not math.isclose(n_ticks * ctrl_dt, config.seconds, abs_tol=1e-9):
            raise ValueError(f"seconds must be a positive multiple of {ctrl_dt}")
        model, data = self.model, self.data
        rec = EpisodeRecord(
            ctrl_dt=ctrl_dt,
            sim_dt=float(model.opt.timestep),
            n_substeps=self.n_substeps,
            initial_qpos=np.empty(model.nq),
            initial_qvel=np.empty(model.nv),
            time=np.empty(n_ticks),
            qpos=np.empty((n_ticks, model.nq)),
            qvel=np.empty((n_ticks, model.nv)),
            action=np.empty((n_ticks, model.nu), np.float32),
            command=np.empty((n_ticks, 3), np.float32),
            contact=np.zeros((n_ticks, 4), bool),
            up_z=np.empty(n_ticks),
            trunk_z=np.empty(n_ticks),
            finite=np.empty(n_ticks, bool),
            warning_total=np.empty(n_ticks, np.int64),
            tick_ns=np.empty(n_ticks, np.int64),
        )
        self.reset(config.command)
        rec.initial_qpos[:] = data.qpos
        rec.initial_qvel[:] = data.qvel
        clock = time.perf_counter_ns
        for k in range(n_ticks):
            start = clock()
            action = self.tick()
            rec.tick_ns[k] = clock() - start
            # Recording is outside the timed region.
            rec.time[k] = data.time
            rec.qpos[k] = data.qpos
            rec.qvel[k] = data.qvel
            rec.action[k] = action
            rec.command[k] = self.command
            self.foot_contacts(rec.contact[k])
            rec.warning_total[k] = self.warning_total()
        rec.up_z[:] = up_z_from_quat(rec.qpos[:, 3:7])
        rec.trunk_z[:] = rec.qpos[:, 2]
        rec.finite[:] = (
            np.isfinite(rec.qpos).all(axis=1)
            & np.isfinite(rec.qvel).all(axis=1)
            & np.isfinite(rec.action).all(axis=1)
        )
        rec.warnings = {
            name: int(data.warning[i].number)
            for i, name in enumerate(WARNING_NAMES)
            if data.warning[i].number
        }
        return rec


def upright_checks(
    up_z: np.ndarray,
    trunk_z: np.ndarray,
    finite: np.ndarray,
    warning_total: np.ndarray,
    time_s: np.ndarray,
    criteria: WalkCriteria,
) -> dict:
    """The upright predicate (Requirement 10.1) and the 60-degree comparison (10.2).

    Non-finite `up_z` or `trunk_z` values compare False, so they also fail the predicate."""
    cos_tilt = math.cos(math.radians(criteria.tilt_deg))
    cos_compare = math.cos(math.radians(criteria.tilt_compare_deg))
    with np.errstate(invalid="ignore"):
        tilt_ok = up_z >= cos_tilt
        height_ok = trunk_z >= criteria.min_trunk_z
        compare_ok = up_z >= cos_compare
    warnings_ok = warning_total == 0
    tick_ok = tilt_ok & height_ok & finite & warnings_ok
    bad = np.flatnonzero(~tick_ok)
    return {
        "upright_45": bool(tick_ok.all()),
        "upright_60": bool(compare_ok.all()),
        "tilt_ok": bool(tilt_ok.all()),
        "height_ok": bool(height_ok.all()),
        "finite": bool(finite.all()),
        "warnings_ok": bool(warnings_ok.all()),
        "first_violation_time": float(time_s[bad[0]]) if bad.size else None,
        "min_up_z": float(np.min(up_z)) if up_z.size else None,
        "max_tilt_deg": float(np.degrees(np.arccos(np.clip(np.min(up_z), -1, 1))))
        if up_z.size
        else None,
        "min_trunk_z": float(np.min(trunk_z)) if trunk_z.size else None,
    }


_TIME_EPS = 1e-9


def window_mask(time_s: np.ndarray, start: float, end: float) -> np.ndarray:
    """Ticks whose time lies in [start, end] (with a small float tolerance)."""
    return (time_s >= start - _TIME_EPS) & (time_s <= end + _TIME_EPS)


def mean_forward_speed(
    time_s: np.ndarray, qpos: np.ndarray, start: float, end: float
) -> float | None:
    """Mean body-frame forward speed over [start, end].

    Each Tick-to-Tick trunk displacement in the ground plane is projected onto the robot's heading
    (yaw) at the middle of that interval; the sum is divided by the elapsed time. Interval k counts
    if both of its end Ticks lie in the window. Returns None if the window holds < 2 Ticks."""
    inside = np.flatnonzero(window_mask(time_s, start, end))
    if inside.size < 2:
        return None
    first, last = inside[0], inside[-1]
    xy = qpos[first : last + 1, 0:2]
    yaw = yaw_from_quat(qpos[first : last + 1, 3:7])
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    mid = heading[:-1] + heading[1:]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    forward = np.sum(np.diff(xy, axis=0) * mid, axis=1)
    return float(forward.sum() / (time_s[last] - time_s[first]))


def touchdowns(time_s: np.ndarray, contact: np.ndarray, start: float, end: float) -> list[int]:
    """Per foot, false->true contact transitions whose later Tick time lies in [start, end]."""
    rises = ~contact[:-1] & contact[1:]
    later = window_mask(time_s[1:], start, end)
    return [int(n) for n in (rises & later[:, None]).sum(axis=0)]


def duty_factor(time_s: np.ndarray, contact: np.ndarray, start: float, end: float) -> list[float]:
    mask = window_mask(time_s, start, end)
    if not mask.any():
        return [0.0] * contact.shape[1]
    return [float(x) for x in contact[mask].mean(axis=0)]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def criteria_set_for(command, horizon_s: float, criteria: WalkCriteria) -> str:
    """Which fixed criteria apply: 'forward' for the Forward_Episode command with a horizon that
    covers the measurement window, 'zero' for the zero command, else 'upright' only."""
    cmd = tuple(float(c) for c in command)
    if cmd == criteria.forward_command and horizon_s >= criteria.window_end_s - _TIME_EPS:
        return "forward"
    if cmd == criteria.zero_command:
        return "zero"
    return "upright"


def evaluate(record: EpisodeRecord, criteria: WalkCriteria = WalkCriteria()) -> dict:
    """Outcome of one episode against the fixed criteria (Requirement 10)."""
    outcome = upright_checks(
        record.up_z,
        record.trunk_z,
        record.finite,
        record.warning_total,
        record.time,
        criteria,
    )
    outcome["warnings"] = dict(record.warnings)

    # Progress metrics use the trajectory up to the last Tick before the first violation, so
    # motion after a fall or a non-finite state is never credited.
    last = record.n_ticks - 1
    if outcome["first_violation_time"] is not None:
        last = int(np.flatnonzero(record.time >= outcome["first_violation_time"] - _TIME_EPS)[0])
        last -= 1
    start_xy = record.initial_qpos[0:2]
    yaw0 = float(yaw_from_quat(record.initial_qpos[3:7]))
    heading0 = np.array([math.cos(yaw0), math.sin(yaw0)])
    if last >= 0:
        valid = slice(0, last + 1)
        rel = record.qpos[valid, 0:2] - start_xy
        dist = np.linalg.norm(rel, axis=1)
        outcome["forward_distance_m"] = float(rel[-1] @ heading0)
        outcome["final_horizontal_disp_m"] = float(dist[-1])
        outcome["horizontal_disp_m"] = float(dist.max())  # max over the valid episode
        yaw_end = float(yaw_from_quat(record.qpos[last, 3:7]))
        outcome["yaw_drift_deg"] = math.degrees(_wrap_angle(yaw_end - yaw0))
        outcome["progress_until_s"] = float(record.time[last])
    else:
        outcome.update(
            forward_distance_m=0.0,
            final_horizontal_disp_m=0.0,
            horizontal_disp_m=0.0,
            yaw_drift_deg=0.0,
            progress_until_s=0.0,
        )

    w0, w1 = criteria.window_start_s, criteria.window_end_s
    outcome["window_s"] = [w0, w1]
    outcome["mean_forward_speed"] = mean_forward_speed(record.time, record.qpos, w0, w1)
    outcome["mean_speed"] = (
        outcome["forward_distance_m"] / outcome["progress_until_s"]
        if outcome["progress_until_s"] > 0
        else 0.0
    )
    outcome["touchdowns"] = touchdowns(record.time, record.contact, w0, w1)
    outcome["duty_factor"] = duty_factor(record.time, record.contact, w0, w1)

    horizon_s = float(record.n_ticks * record.ctrl_dt)
    criteria_set = criteria_set_for(record.command[0], horizon_s, criteria)
    outcome["criteria_set"] = criteria_set
    failed = [] if outcome["upright_45"] else ["upright_45"]
    if criteria_set == "forward":
        speed = outcome["mean_forward_speed"]
        if speed is None or speed < criteria.min_forward_speed:
            failed.append("mean_forward_speed")
        if min(outcome["touchdowns"]) < criteria.min_touchdowns:
            failed.append("touchdowns")
    elif criteria_set == "zero" and not (outcome["horizontal_disp_m"] < criteria.zero_cmd_max_disp):
        failed.append("horizontal_disp")
    outcome["passed"] = not failed
    outcome["failed_checks"] = failed
    return outcome


def latency_summary(tick_ns: np.ndarray, ctrl_dt: float, warmup: int = 50) -> dict:
    """Tick latency statistics after discarding `warmup` Ticks, against a 25 ms target."""
    target_ms = 25.0
    measured = np.asarray(tick_ns[warmup:], dtype=np.float64) / 1e6
    if measured.size == 0:
        return {"warmup_ticks": warmup, "measured_ticks": 0, "target_ms": target_ms}
    total_s = measured.sum() / 1e3
    return {
        "warmup_ticks": warmup,
        "measured_ticks": int(measured.size),
        "median_ms": float(np.median(measured)),
        "p95_ms": float(np.percentile(measured, 95)),
        "max_ms": float(measured.max()),
        "mean_ms": float(measured.mean()),
        "target_ms": target_ms,
        "median_within_target": bool(np.median(measured) <= target_ms),
        "p95_within_target": bool(np.percentile(measured, 95) <= target_ms),
        "max_within_target": bool(measured.max() <= target_ms),
        "real_time_factor": float(measured.size * ctrl_dt / total_s) if total_s > 0 else None,
    }


def criteria_dict(criteria: WalkCriteria) -> dict:
    return asdict(criteria)
