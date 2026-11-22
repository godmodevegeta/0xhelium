"""Hardening data sets: training, validation preview and the locked M7 test set.

An episode spec is one deterministic episode: command, horizon, floor friction, payload and an
optional push. Confirmed failures store the exact heading-frame direction vector their source
milestone used (M2/M3 `right` = (0, -1, 0); M4 (cos, sin, 0) of the recorded angle), so the
baseline episode is the recorded one.

The locked test set is listed (with its SHA-256) for M7 and is refused by every M6 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robustlab.experiments.push import PushScenario
from robustlab.physics.scene import Payload

SETS_KIND = "robustlab-hardening-sets"
SCHEMA_VERSION = 1
SIM_DT = 0.004
USABLE_SPLITS = ("train", "validation_preview")
LOCKED_SPLIT = "locked_test"


class SetsError(ValueError):
    pass


@dataclass(frozen=True)
class PushSpec:
    force_n: float
    direction: tuple[float, float, float]
    start_substep: int
    duration_substeps: int

    @property
    def start_s(self) -> float:
        return self.start_substep * SIM_DT

    @property
    def duration_s(self) -> float:
        return self.duration_substeps * SIM_DT

    @property
    def direction_deg(self) -> float:
        return math.degrees(math.atan2(self.direction[1], self.direction[0])) % 360.0

    def scenario(self) -> PushScenario:
        return PushScenario(self.force_n, self.direction, round(self.start_s, 6),
                            round(self.duration_s, 6))  # fmt: skip

    def to_dict(self) -> dict:
        return {"force_n": self.force_n, "direction": list(self.direction),
                "direction_deg": round(self.direction_deg, 3), "start_substep": self.start_substep,
                "duration_substeps": self.duration_substeps, "start_s": round(self.start_s, 6),
                "duration_s": round(self.duration_s, 6)}  # fmt: skip


@dataclass(frozen=True)
class EpisodeSpec:
    id: str
    kind: str  # "attack" | "nominal"
    split: str
    source: str
    command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    horizon_s: float = 5.0
    ground_friction_mu: float | None = None
    payload_kg: float = 0.0
    payload_offset_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    push: PushSpec | None = None
    confirmed_failure: bool = False  # recorded as a baseline fall by an earlier milestone
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def payload(self) -> Payload | None:
        return None if self.payload_kg <= 0 else Payload(self.payload_kg, self.payload_offset_xyz_m)

    @property
    def gait_phase_substep(self) -> int | None:
        """Push start modulo the nominal gait period (0.44 s = 110 substeps)."""
        return None if self.push is None else self.push.start_substep % 110

    def physics_key(self) -> tuple:
        return (self.ground_friction_mu, self.payload_kg, tuple(self.payload_offset_xyz_m))

    def reference_key(self) -> tuple:
        """Episodes sharing this key share the same no-push reference run."""
        return (*self.physics_key(), tuple(self.command), self.horizon_s)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "split": self.split, "source": self.source,
            "command": list(self.command), "horizon_s": self.horizon_s,
            "ground_friction_mu": self.ground_friction_mu, "payload_kg": self.payload_kg,
            "payload_offset_xyz_m": list(self.payload_offset_xyz_m),
            "push": None if self.push is None else self.push.to_dict(),
            "confirmed_failure": self.confirmed_failure, "tags": list(self.tags),
            "gait_phase_substep": self.gait_phase_substep,
        }  # fmt: skip


def _push(doc) -> PushSpec | None:
    if doc is None:
        return None
    direction = tuple(float(v) for v in doc["direction"])
    if len(direction) != 3 or math.hypot(*direction) == 0:
        raise SetsError(f"bad direction {doc['direction']!r}")
    push = PushSpec(float(doc["force_n"]), direction, int(doc["start_substep"]),
                    int(doc["duration_substeps"]))  # fmt: skip
    push.scenario()  # validates force, duration and timing
    return push


def _episode(doc: dict, split: str) -> EpisodeSpec:
    return EpisodeSpec(
        id=str(doc["id"]),
        kind=str(doc["kind"]),
        split=split,
        source=str(doc.get("source", "")),
        command=tuple(float(v) for v in doc.get("command", (1.0, 0.0, 0.0))),
        horizon_s=float(doc.get("horizon_s", 5.0)),
        ground_friction_mu=None if doc.get("ground_friction_mu") is None
        else float(doc["ground_friction_mu"]),
        payload_kg=float(doc.get("payload_kg", 0.0)),
        payload_offset_xyz_m=tuple(float(v) for v in doc.get("payload_offset_xyz_m", (0, 0, 0))),
        push=_push(doc.get("push")),
        confirmed_failure=bool(doc.get("confirmed_failure", False)),
        tags=tuple(doc.get("tags", ())),
    )  # fmt: skip


def variants(
    base: EpisodeSpec, rule: dict, rng: np.random.Generator, count: int
) -> list[EpisodeSpec]:
    """Seeded nearby variants of a training attack (not confirmed failures)."""
    out = []
    p = base.push
    for i in range(count):
        scale = rng.uniform(1 - rule["force_rel"], 1 + rule["force_rel"])
        turn = math.radians(rng.uniform(-rule["direction_deg"], rule["direction_deg"]))
        shift = int(rng.integers(-rule["start_substeps"], rule["start_substeps"] + 1))
        dur = int(rng.integers(-rule["duration_substeps"], rule["duration_substeps"] + 1))
        c, s = math.cos(turn), math.sin(turn)
        x, y, z = p.direction
        push = PushSpec(
            force_n=round(min(150.0, p.force_n * scale), 1),
            direction=(c * x - s * y, s * x + c * y, z),
            start_substep=p.start_substep + shift,
            duration_substeps=max(10, p.duration_substeps + dur),
        )
        out.append(EpisodeSpec(id=f"{base.id}-v{i}", kind="attack", split=base.split,
                               source=f"seeded variant of {base.id}", command=base.command,
                               horizon_s=base.horizon_s,
                               ground_friction_mu=base.ground_friction_mu,
                               payload_kg=base.payload_kg,
                               payload_offset_xyz_m=base.payload_offset_xyz_m, push=push,
                               confirmed_failure=False, tags=("variant",)))  # fmt: skip
    return out


@dataclass(frozen=True)
class HardeningSets:
    name: str
    sha256: str
    path: str
    gait_period_s: float
    splits: dict[str, tuple[EpisodeSpec, ...]]
    locked_sha256: str
    locked_count: int
    notes: tuple[str, ...]

    def split(self, name: str) -> tuple[EpisodeSpec, ...]:
        if name == LOCKED_SPLIT:
            raise SetsError("the locked test set is reserved for M7 and cannot be evaluated in M6")
        if name not in self.splits:
            raise SetsError(f"unknown split {name!r}; expected one of {USABLE_SPLITS}")
        return self.splits[name]

    def summary(self) -> dict:
        return {
            "name": self.name, "path": self.path, "sha256": self.sha256,
            "gait_period_s": self.gait_period_s,
            "splits": {k: {"episodes": len(v), "attacks": sum(e.kind == "attack" for e in v),
                           "nominal": sum(e.kind == "nominal" for e in v),
                           "ids": [e.id for e in v]} for k, v in self.splits.items()},
            "locked_test": {"episodes": self.locked_count, "sha256": self.locked_sha256,
                            "status": "reserved for M7; not evaluated in M6"},
            "notes": list(self.notes),
        }  # fmt: skip


def locked_digest(episodes: list) -> str:
    return hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest()


def load_sets(path: Path, root: Path | None = None) -> HardeningSets:
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("kind") != SETS_KIND or doc.get("schema_version") != SCHEMA_VERSION:
        raise SetsError(f"{path}: not a {SETS_KIND} v{SCHEMA_VERSION} file")
    locked = doc[LOCKED_SPLIT]
    if locked_digest(locked["episodes"]) != locked["sha256"]:
        raise SetsError("locked test set does not match its recorded SHA-256")
    splits: dict[str, tuple[EpisodeSpec, ...]] = {}
    for name in USABLE_SPLITS:
        section = doc[name]
        episodes = [_episode(e, name) for e in section["episodes"]]
        rule = section.get("variants")
        if rule:
            rng = np.random.Generator(np.random.PCG64(rule["seed"]))
            extra = []
            for e in episodes:
                if e.kind == "attack" and e.confirmed_failure:
                    extra += variants(e, rule["jitter"], rng, rule["per_attack"])
            episodes += extra
        splits[name] = tuple(episodes)
    ids = [e.id for v in splits.values() for e in v] + [e["id"] for e in locked["episodes"]]
    if len(ids) != len(set(ids)):
        raise SetsError("episode ids must be unique across splits")
    shown = path.relative_to(root).as_posix() if root and path.is_relative_to(root) else str(path)
    return HardeningSets(name=doc["name"], sha256=hashlib.sha256(raw).hexdigest(), path=shown,
                         gait_period_s=float(doc["gait_period_s"]), splits=splits,
                         locked_sha256=locked["sha256"], locked_count=len(locked["episodes"]),
                         notes=tuple(doc.get("notes", ())))  # fmt: skip
