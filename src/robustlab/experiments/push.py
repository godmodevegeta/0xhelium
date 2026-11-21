"""A single timed shove on the trunk, specified in the robot's initial-heading frame.

Initial-heading frame: fixed at reset. +x points along the trunk's yaw at reset (forward), +y to the
robot's left, +z up (world z). `direction` is where the force points: `right` = (0, -1, 0) is a push
toward the robot's right, i.e. a shove from its left side.

The force is constant, applied at the trunk centre of mass through `xfrc_applied`, with zero torque,
on every physics substep i (0-based, counted from reset) with first <= i < end, where
first = start_s / sim_dt and end = (start_s + duration_s) / sim_dt. Both times must be exact
multiples of sim_dt so the interval is precise; the delivered impulse is force_n * duration_s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_FORCE_N = 150.0  # tech.md go1 force envelope
MAX_DURATION_S = 0.5
TIME_TOLERANCE_S = 1e-9
DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}
FRAME_NOTE = (
    "initial-heading frame: +x trunk forward at reset, +y left, +z up; direction is where the "
    "force points (right = shove from the robot's left)"
)


class PushError(ValueError):
    """An invalid PushScenario field."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"push.{field}: {reason}")
        self.field = field
        self.reason = reason


def _finite(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise PushError(name, "not a number") from exc
    if not math.isfinite(value):
        raise PushError(name, "must be finite")
    return value


def steps_of(seconds: float, sim_dt: float, name: str) -> int:
    """`seconds` as a whole number of substeps; raises PushError if it is not a multiple."""
    steps = round(seconds / sim_dt)
    if abs(steps * sim_dt - seconds) > TIME_TOLERANCE_S:
        raise PushError(
            name,
            f"{seconds} s is not a multiple of the {sim_dt} s physics step "
            f"(nearest: {steps * sim_dt:.6g} s)",
        )
    return steps


@dataclass(frozen=True)
class PushScenario:
    force_n: float
    direction: tuple[float, float, float]
    start_s: float
    duration_s: float
    sim_dt: float = 0.004

    def __post_init__(self) -> None:
        force = _finite(self.force_n, "force_n")
        if not 0.0 <= force <= MAX_FORCE_N:
            raise PushError("force_n", f"{force} N outside [0, {MAX_FORCE_N}] N")
        if len(self.direction) != 3:
            raise PushError("direction", "expected 3 components (x, y, z)")
        direction = tuple(_finite(c, "direction") for c in self.direction)
        if math.hypot(*direction) == 0.0:
            raise PushError("direction", "must be non-zero")
        start = _finite(self.start_s, "start_s")
        if start < 0.0:
            raise PushError("start_s", "must be >= 0")
        duration = _finite(self.duration_s, "duration_s")
        if not 0.0 <= duration <= MAX_DURATION_S:
            raise PushError("duration_s", f"{duration} s outside [0, {MAX_DURATION_S}] s")
        steps_of(start, self.sim_dt, "start_s")
        steps_of(duration, self.sim_dt, "duration_s")
        object.__setattr__(self, "force_n", force)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "duration_s", duration)

    @property
    def unit_direction(self) -> np.ndarray:
        d = np.asarray(self.direction, np.float64)
        return d / np.linalg.norm(d)

    def window(self) -> tuple[int, int]:
        """Active substeps [first, end), 0-based from reset."""
        first = steps_of(self.start_s, self.sim_dt, "start_s")
        return first, first + steps_of(self.duration_s, self.sim_dt, "duration_s")

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def impulse_ns(self) -> float:
        return self.force_n * self.duration_s

    def world_force(self, initial_yaw: float) -> np.ndarray:
        return self.force_n * heading_to_world(self.unit_direction, initial_yaw)

    def describe(self, initial_yaw: float) -> dict:
        first, end = self.window()
        return {
            "force_n": self.force_n,
            "direction_heading_frame": list(self.direction),
            "unit_direction_heading_frame": self.unit_direction.tolist(),
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "end_s": self.end_s,
            "frame": FRAME_NOTE,
            "initial_yaw_rad": float(initial_yaw),
            "world_force_n": self.world_force(initial_yaw).tolist(),
            "body": "trunk (force at centre of mass, zero torque)",
            "active_substeps": [first, end],
            "active_substep_note": "0-based substep i active when first <= i < end",
            "impulse_ns": self.impulse_ns,
        }


def heading_to_world(vector: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a heading-frame vector into world coordinates (rotation by `yaw` about world z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = (float(v) for v in vector)
    return np.array([c * x - s * y, s * x + c * y, z])


def parse_direction(text: str) -> tuple[float, float, float]:
    """`forward|backward|left|right` or `x,y,z` in the initial-heading frame."""
    name = text.strip().lower()
    if name in DIRECTIONS:
        return DIRECTIONS[name]
    parts = name.split(",")
    if len(parts) != 3:
        raise PushError("direction", f"expected one of {sorted(DIRECTIONS)} or x,y,z, got {text!r}")
    return tuple(_finite(p, "direction") for p in parts)  # type: ignore[return-value]
