"""Typed multi-axis scenario schema (M5): push, ground friction, trunk payload, command, horizon.

Every numeric leaf is a *Param*: a fixed number, or `{"range": [lo, hi]}` reserved for future
search. The M5 executor runs fixed scenarios only (`mode="fixed"` rejects ranges).

Validation never clamps. Out-of-range, non-finite, wrongly typed and unknown fields are errors with
their field path, and all errors are reported together. Three deterministic normalizations are
applied and disclosed as adjustments: push start and duration are rounded to the 4 ms physics step,
direction is taken modulo 360 degrees, and a zero-force or zero-duration push becomes "no push".
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1
KIND = "robustlab-scenario"
SIM_DT = 0.004
CTRL_DT = 0.02
SCENES = ("fall-contact", "feet-only")
NOMINAL_FRICTION_MU = 0.6  # the trained floor friction (walk.xml), resolved for every floor contact

BOUNDS = {
    "environment.ground_friction_mu": (0.05, 1.50),
    "payload.mass_kg": (0.0, 5.0),
    "payload.offset_xyz_m[0]": (-0.10, 0.10),
    "payload.offset_xyz_m[1]": (-0.08, 0.08),
    "payload.offset_xyz_m[2]": (0.0, 0.08),
    "push.force_n": (0.0, 150.0),
    "push.duration_s": (0.0, 0.5),
    "horizon_s": (CTRL_DT, 60.0),
}
DEFAULT_COMMAND_RANGES = ((-1.5, 1.5), (-0.8, 0.8), (-1.2, 1.2))  # go1-walk-s0 trained ranges


class ScenarioError(ValueError):
    def __init__(self, errors: list[dict]) -> None:
        super().__init__("; ".join(f"{e['field']}: {e['reason']}" for e in errors))
        self.errors = errors


@dataclass(frozen=True)
class Push:
    force_n: float
    direction_deg: float
    start_s: float
    duration_s: float

    @property
    def start_substep(self) -> int:
        return round(self.start_s / SIM_DT)

    @property
    def duration_substeps(self) -> int:
        return round(self.duration_s / SIM_DT)

    @property
    def direction_vector(self) -> tuple[float, float, float]:
        """Unit vector in the initial-heading frame. Components are rounded to 15 decimals so
        cardinal directions are exact (270 deg -> (0, -1, 0), as in the M1/M2 presets)."""
        a = math.radians(self.direction_deg)
        return (round(math.cos(a), 15) + 0.0, round(math.sin(a), 15) + 0.0, 0.0)

    @property
    def impulse_ns(self) -> float:
        return self.force_n * self.duration_s


@dataclass(frozen=True)
class Scenario:
    name: str
    command: tuple[float, float, float]
    horizon_s: float
    scene: str
    ground_friction_mu: float | None
    payload_mass_kg: float
    payload_offset_xyz_m: tuple[float, float, float]
    push: Push | None

    @property
    def is_nominal_physics(self) -> bool:
        """Trained friction and no payload (the original model is used unchanged)."""
        return self.ground_friction_mu is None and self.payload_mass_kg == 0.0

    def effective_mu(self) -> float:
        return NOMINAL_FRICTION_MU if self.ground_friction_mu is None else self.ground_friction_mu

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "name": self.name,
            "command": list(self.command),
            "horizon_s": self.horizon_s,
            "scene": self.scene,
            "environment": {"ground_friction_mu": self.ground_friction_mu},
            "payload": {
                "mass_kg": self.payload_mass_kg,
                "offset_xyz_m": list(self.payload_offset_xyz_m),
            },
            "push": None if self.push is None else asdict(self.push),
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def changed_from_nominal(self) -> list[dict]:
        out = []
        if self.ground_friction_mu is not None and self.ground_friction_mu != NOMINAL_FRICTION_MU:
            out.append({"parameter": "ground_friction_mu", "nominal": NOMINAL_FRICTION_MU,
                        "value": self.ground_friction_mu})  # fmt: skip
        if self.payload_mass_kg > 0:
            out.append({"parameter": "payload", "nominal": "none",
                        "value": {"mass_kg": self.payload_mass_kg,
                                  "offset_xyz_m": list(self.payload_offset_xyz_m)}})  # fmt: skip
        if self.push is not None:
            out.append({"parameter": "push", "nominal": "none", "value": asdict(self.push)})
        return out


@dataclass(frozen=True)
class Validated:
    scenario: Scenario
    adjustments: list[dict]

    def to_dict(self) -> dict:
        return {"scenario": self.scenario.to_dict(), "adjustments": self.adjustments,
                "scenario_sha256": self.scenario.sha256()}  # fmt: skip


class _Validator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.errors: list[dict] = []
        self.adjustments: list[dict] = []

    def error(self, field: str, reason: str) -> None:
        self.errors.append({"field": field, "reason": reason})

    def keys(self, obj, field: str, required: set[str], optional: set[str] = frozenset()) -> bool:
        if not isinstance(obj, dict):
            self.error(field or "scenario", "expected an object")
            return False
        prefix = f"{field}." if field else ""
        for key in sorted(required - obj.keys()):
            self.error(prefix + key, "missing")
        for key in sorted(obj.keys() - required - optional):
            self.error(prefix + key, "unknown field")
        return True

    def number(self, value, field: str) -> float | None:
        """A fixed number, or a range Param (rejected in fixed mode)."""
        if isinstance(value, dict):
            if (
                set(value) != {"range"}
                or not isinstance(value["range"], list)
                or len(value["range"]) != 2
            ):
                self.error(field, 'a Param object must be {"range": [lo, hi]}')
                return None
            if self.mode == "fixed":
                self.error(field, "ranges are reserved for search and are not executable in M5")
                return None
            self.error(field, "range execution is not implemented")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(field, "expected a number")
            return None
        if not math.isfinite(value):
            self.error(field, "must be finite")
            return None
        return float(value)

    def bounded(self, value, field: str, bounds_key: str | None = None) -> float | None:
        number = self.number(value, field)
        if number is None:
            return None
        lo, hi = BOUNDS[bounds_key or field]
        if not lo <= number <= hi:
            self.error(field, f"{number:g} is outside [{lo:g}, {hi:g}] (values are not clamped)")
            return None
        return number

    def quantize_time(self, value: float, field: str) -> float:
        steps = round(value / SIM_DT)
        applied = round(steps * SIM_DT, 6)
        if abs(applied - value) > 1e-9:
            self.adjustments.append({"field": field, "requested": value, "applied": applied,
                                     "reason": "rounded to the 4 ms physics step"})  # fmt: skip
        return applied


def validate(
    raw,
    *,
    mode: str = "fixed",
    command_ranges=DEFAULT_COMMAND_RANGES,
) -> Validated:
    """Validate a scenario dict. Raises ScenarioError listing every problem."""
    v = _Validator(mode)
    top = {"schema_version", "kind", "name", "command", "horizon_s", "scene", "environment",
           "payload", "push"}  # fmt: skip
    if not v.keys(raw, "", top):
        raise ScenarioError(v.errors)
    if raw.get("schema_version") != SCHEMA_VERSION:
        v.error("schema_version", f"expected {SCHEMA_VERSION}")
    if raw.get("kind") != KIND:
        v.error("kind", f"expected {KIND!r}")
    name = raw.get("name")
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 64
        or not all(c.isalnum() or c in "-_. " for c in name)
    ):
        v.error("name", "1-64 characters: letters, digits, space, - _ .")
    scene = raw.get("scene")
    if scene not in SCENES:
        v.error("scene", f"expected one of {SCENES}")

    command = raw.get("command")
    cmd: list[float] = []
    if not isinstance(command, list) or len(command) != 3:
        v.error("command", "expected [vx, vy, yaw_rate]")
    else:
        for i, (value, (lo, hi)) in enumerate(zip(command, command_ranges)):
            n = v.number(value, f"command[{i}]")
            if n is not None and not lo <= n <= hi:
                v.error(
                    f"command[{i}]", f"{n:g} is outside the policy's trained range [{lo:g}, {hi:g}]"
                )
            elif n is not None:
                cmd.append(n)

    horizon = v.bounded(raw.get("horizon_s"), "horizon_s")
    if horizon is not None and abs(round(horizon / CTRL_DT) * CTRL_DT - horizon) > 1e-9:
        v.error("horizon_s", "must be a multiple of the 0.02 s control period")
        horizon = None

    env = raw.get("environment")
    mu = None
    if (
        v.keys(env, "environment", {"ground_friction_mu"})
        and env.get("ground_friction_mu") is not None
    ):
        mu = v.bounded(env.get("ground_friction_mu"), "environment.ground_friction_mu")

    payload = raw.get("payload")
    mass, offset = 0.0, [0.0, 0.0, 0.0]
    if v.keys(payload, "payload", {"mass_kg", "offset_xyz_m"}):
        mass = v.bounded(payload.get("mass_kg"), "payload.mass_kg")
        off = payload.get("offset_xyz_m")
        if not isinstance(off, list) or len(off) != 3:
            v.error("payload.offset_xyz_m", "expected [x, y, z] in metres")
        else:
            offset = [v.bounded(x, f"payload.offset_xyz_m[{i}]") for i, x in enumerate(off)]

    push_raw = raw.get("push")
    push = None
    if push_raw is not None and v.keys(push_raw, "push",
                                       {"force_n", "direction_deg", "start_s", "duration_s"}):  # fmt: skip
        force = v.bounded(push_raw.get("force_n"), "push.force_n")
        direction = v.number(push_raw.get("direction_deg"), "push.direction_deg")
        start = v.number(push_raw.get("start_s"), "push.start_s")
        duration = v.bounded(push_raw.get("duration_s"), "push.duration_s")
        if direction is not None:
            wrapped = round(direction % 360.0, 6) % 360.0
            if wrapped != direction:
                v.adjustments.append({"field": "push.direction_deg", "requested": direction,
                                      "applied": wrapped, "reason": "taken modulo 360 degrees"})  # fmt: skip
            direction = wrapped
        if start is not None and start < 0:
            v.error("push.start_s", f"{start:g} is negative")
            start = None
        if start is not None:
            start = v.quantize_time(start, "push.start_s")
        if duration is not None:
            duration = v.quantize_time(duration, "push.duration_s")
        if (
            None not in (start, duration)
            and horizon is not None
            and start + duration > horizon + 1e-9
        ):
            v.error(
                "push.start_s",
                f"push ends at {start + duration:g} s, after the {horizon:g} s horizon",
            )
        if None not in (force, direction, start, duration):
            if force == 0 or duration == 0:
                v.adjustments.append({"field": "push", "requested": push_raw, "applied": None,
                                      "reason": "zero force or zero duration: no push"})  # fmt: skip
            else:
                push = Push(force, direction, start, duration)

    if v.errors:
        raise ScenarioError(v.errors)
    scenario = Scenario(
        name=name,
        command=tuple(cmd),  # type: ignore[arg-type]
        horizon_s=horizon,
        scene=scene,
        ground_friction_mu=mu,
        payload_mass_kg=mass,
        payload_offset_xyz_m=tuple(offset),  # type: ignore[arg-type]
        push=push,
    )
    return Validated(scenario, v.adjustments)


def json_schema() -> dict:
    """JSON Schema of the fixed-value scenario document (documentation and the web form)."""
    num = {"type": "number"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "robustlab scenario v1 (fixed values)",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "kind",
            "name",
            "command",
            "horizon_s",
            "scene",
            "environment",
            "payload",
            "push",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "kind": {"const": KIND},
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
            "command": {"type": "array", "items": num, "minItems": 3, "maxItems": 3},
            "horizon_s": {"type": "number", "minimum": CTRL_DT, "maximum": 60},
            "scene": {"enum": list(SCENES)},
            "environment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ground_friction_mu"],
                "properties": {
                    "ground_friction_mu": {
                        "type": ["number", "null"],
                        "minimum": 0.05,
                        "maximum": 1.5,
                    }
                },
            },
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mass_kg", "offset_xyz_m"],
                "properties": {
                    "mass_kg": {"type": "number", "minimum": 0, "maximum": 5},
                    "offset_xyz_m": {"type": "array", "items": num, "minItems": 3, "maxItems": 3},
                },
            },
            "push": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["force_n", "direction_deg", "start_s", "duration_s"],
                "properties": {
                    "force_n": {"type": "number", "minimum": 0, "maximum": 150},
                    "direction_deg": num,
                    "start_s": {"type": "number", "minimum": 0},
                    "duration_s": {"type": "number", "minimum": 0, "maximum": 0.5},
                },
            },
        },
    }


def nominal_dict(name: str = "nominal") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "name": name,
        "command": [1.0, 0.0, 0.0],
        "horizon_s": 5.0,
        "scene": "fall-contact",
        "environment": {"ground_friction_mu": None},
        "payload": {"mass_kg": 0.0, "offset_xyz_m": [0.0, 0.0, 0.0]},
        "push": None,
    }
