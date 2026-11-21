"""Authoritative fall and numerical-failure detection, checked after every physics substep.

A state index k is the simulation state after k substeps since reset (time k * sim_dt).

After `mj_step`, `qpos`, `qvel` and `qacc` describe state k, but `data.contact` was computed by the
collision pass at the start of that step, i.e. it describes state k - 1. Contact events are therefore
stamped with state index k - 1; state checks with k.

Reasons, in tie-break order (the earliest state index wins; equal indices use this order):

- `nonfinite_state` (numerical): any non-finite value in qpos, qvel or qacc
- `mujoco_warning` (numerical): any MuJoCo warning counter non-zero (MuJoCo may reset the state
  itself after e.g. BADQACC, so the trajectory is no longer physical)
- `nonfoot_contact` (physical): a non-foot robot geom touching the floor with dist <= 0
- `tilt` (physical): trunk up_z < cos(tilt threshold)
- `low_height` (physical): trunk height < minimum
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robustlab.physics.model import ResolvedIds

REASONS = ("nonfinite_state", "mujoco_warning", "nonfoot_contact", "tilt", "low_height")
NUMERICAL_REASONS = frozenset({"nonfinite_state", "mujoco_warning"})
PHYSICAL_REASONS = frozenset({"nonfoot_contact", "tilt", "low_height"})
WARNING_NAMES = tuple(
    name.removeprefix("mjWARN_") for name in mujoco.mjtWarning.__members__ if name != "mjNWARNING"
)


def tilt_ok(up_z, cos_tilt: float):
    """True where the trunk is within the tilt threshold. NaN compares False (a violation)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(up_z) >= cos_tilt


def height_ok(trunk_z, min_z: float):
    """True where the trunk is at or above the minimum height. NaN compares False."""
    with np.errstate(invalid="ignore"):
        return np.asarray(trunk_z) >= min_z


def up_z_of(quat: np.ndarray) -> float:
    """World z component of the body z axis for one wxyz quaternion (NaN if degenerate)."""
    norm2 = float(np.dot(quat, quat))
    if not (norm2 > 0 and math.isfinite(norm2)):
        return math.nan
    return 1.0 - 2.0 * float(quat[1] ** 2 + quat[2] ** 2) / norm2


@dataclass(frozen=True)
class FallEvent:
    reason: str
    state_index: int
    time_s: float
    detail: dict

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "state_index": self.state_index,
            "time_s": self.time_s,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FallResult:
    fell: bool  # a physical fall occurred (at any time)
    numerical_failure: bool  # a numerical event occurred (at any time)
    kind: str  # "none" | "physical" | "numerical": class of the primary (earliest) event
    reason: str | None
    time_s: float | None
    state_index: int | None
    events: dict[str, FallEvent]
    nonfoot_contact_detection: bool
    nonfoot_contact_substeps: int
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fell": self.fell,
            "numerical_failure": self.numerical_failure,
            "kind": self.kind,
            "reason": self.reason,
            "time_s": self.time_s,
            "state_index": self.state_index,
            "events": {name: event.to_dict() for name, event in self.events.items()},
            "nonfoot_contact_detection": self.nonfoot_contact_detection,
            "nonfoot_contact_substeps": self.nonfoot_contact_substeps,
            "criteria": dict(self.criteria),
            "reason_order": list(REASONS),
            "state_index_note": "state k is the state after k substeps since reset (t = k*sim_dt)",
        }


def primary_event(events: dict[str, FallEvent]) -> FallEvent | None:
    """Earliest event; ties broken by REASONS order."""
    if not events:
        return None
    return min(events.values(), key=lambda e: (e.state_index, REASONS.index(e.reason)))


def make_result(
    events: dict[str, FallEvent],
    *,
    nonfoot_contact_detection: bool,
    nonfoot_contact_substeps: int = 0,
    criteria: dict | None = None,
) -> FallResult:
    first = primary_event(events)
    if first is None:
        kind = "none"
    else:
        kind = "numerical" if first.reason in NUMERICAL_REASONS else "physical"
    return FallResult(
        fell=any(name in PHYSICAL_REASONS for name in events),
        numerical_failure=any(name in NUMERICAL_REASONS for name in events),
        kind=kind,
        reason=None if first is None else first.reason,
        time_s=None if first is None else first.time_s,
        state_index=None if first is None else first.state_index,
        events=dict(events),
        nonfoot_contact_detection=nonfoot_contact_detection,
        nonfoot_contact_substeps=nonfoot_contact_substeps,
        criteria=dict(criteria or {}),
    )


def nonfoot_floor_mask(model: mujoco.MjModel, ids: ResolvedIds) -> np.ndarray:
    """Robot geoms (not on the world body, not feet) that can collide with the floor."""
    floor = ids.floor_geom
    can_touch = (model.geom_contype & model.geom_conaffinity[floor]) | (
        model.geom_conaffinity & model.geom_contype[floor]
    )
    mask = (can_touch != 0) & (model.geom_bodyid != 0)
    mask[list(ids.foot_geoms)] = False
    mask[floor] = False
    return mask


class FallMonitor:
    """Call `reset()` after the episode reset, then `check(data, k)` after the k-th `mj_step`."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: ResolvedIds,
        tilt_deg: float,
        min_trunk_z: float,
    ) -> None:
        self.model = model
        self.sim_dt = float(model.opt.timestep)
        self.floor = ids.floor_geom
        self.tilt_deg = float(tilt_deg)
        self.min_trunk_z = float(min_trunk_z)
        self.cos_tilt = math.cos(math.radians(tilt_deg))
        self.nonfoot = nonfoot_floor_mask(model, ids)
        self.detection = bool(self.nonfoot.any())
        self.events: dict[str, FallEvent] = {}
        self.nonfoot_substeps = 0

    def reset(self) -> None:
        self.events = {}
        self.nonfoot_substeps = 0

    @property
    def numerical_failure(self) -> bool:
        return any(name in NUMERICAL_REASONS for name in self.events)

    def _record(self, reason: str, state_index: int, detail: dict) -> None:
        if reason not in self.events:
            self.events[reason] = FallEvent(reason, state_index, state_index * self.sim_dt, detail)

    def check(self, data: mujoco.MjData, substep: int) -> bool:
        """Inspect the state after substep `substep` (1-based). Returns whether a non-foot floor
        contact was present in that step's collision pass."""
        qpos = data.qpos
        if not (
            np.isfinite(qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            self._record("nonfinite_state", substep, {"fields": _nonfinite_fields(data)})
        counts = data.warning.number
        if counts.any():
            self._record(
                "mujoco_warning",
                substep,
                {WARNING_NAMES[i]: int(counts[i]) for i in np.flatnonzero(counts)},
            )

        contact_now = False
        ncon = data.ncon
        if self.detection and ncon:
            geom = data.contact.geom[:ncon]
            dist = data.contact.dist[:ncon]
            other = np.where(
                geom[:, 0] == self.floor,
                geom[:, 1],
                np.where(geom[:, 1] == self.floor, geom[:, 0], -1),
            )
            hit = (other >= 0) & self.nonfoot[np.maximum(other, 0)] & (dist <= 0)
            if hit.any():
                contact_now = True
                self.nonfoot_substeps += 1
                if "nonfoot_contact" not in self.events:
                    first = int(np.flatnonzero(hit)[np.argmin(dist[hit])])
                    gid = int(other[first])
                    body = self.model.body(int(self.model.geom_bodyid[gid])).name
                    self._record(
                        "nonfoot_contact",
                        substep - 1,
                        {"geom_id": gid, "body": body, "dist_m": float(dist[first])},
                    )

        up_z = up_z_of(qpos[3:7])
        if not tilt_ok(up_z, self.cos_tilt):
            tilt = (
                math.degrees(math.acos(max(-1.0, min(1.0, up_z)))) if math.isfinite(up_z) else None
            )
            self._record("tilt", substep, {"up_z": _finite_or_none(up_z), "tilt_deg": tilt})
        if not height_ok(qpos[2], self.min_trunk_z):
            self._record("low_height", substep, {"trunk_z_m": _finite_or_none(float(qpos[2]))})
        return contact_now

    def result(self) -> FallResult:
        return make_result(
            self.events,
            nonfoot_contact_detection=self.detection,
            nonfoot_contact_substeps=self.nonfoot_substeps,
            criteria={"tilt_deg": self.tilt_deg, "min_trunk_z": self.min_trunk_z},
        )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _nonfinite_fields(data: mujoco.MjData) -> list[str]:
    return [name for name in ("qpos", "qvel", "qacc") if not np.isfinite(getattr(data, name)).all()]
