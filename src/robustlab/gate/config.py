"""Load and verify the pre-registered gate configuration and build its episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from robustlab.adapt.reflex import Calibration, validate_params
from robustlab.adapt.sets import (
    LOCKED_SPLIT,
    EpisodeSpec,
    PushSpec,
    episode_from_dict,
    locked_digest,
)

GATE_KIND = "robustlab-gate"
SCHEMA_VERSION = 1


class GateConfigError(ValueError):
    pass


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    record_path: str
    record_sha256: str
    seed: int
    vector: tuple[float, ...]
    primary: bool


@dataclass(frozen=True)
class Gate:
    name: str
    path: str
    sha256: str
    doc: dict
    calibration: Calibration
    candidates: tuple[Candidate, ...]
    primary_id: str
    holdout: tuple[EpisodeSpec, ...]
    nominal: tuple[EpisodeSpec, ...]
    gauntlet: tuple[EpisodeSpec, ...]
    known_failures: tuple[EpisodeSpec, ...]

    @property
    def criteria(self) -> dict:
        return self.doc["criteria"]

    def suite_ids(self) -> dict[str, list[str]]:
        return {name: [e.id for e in getattr(self, name)]
                for name in ("holdout", "nominal", "gauntlet", "known_failures")}  # fmt: skip


def _check(path: Path, expected: str, what: str) -> None:
    actual = file_sha(path)
    if actual != expected:
        raise GateConfigError(f"{what} {path}: SHA-256 {actual} != recorded {expected}")


def load_gate(path: Path, root: Path) -> Gate:
    """Verify every pinned input and build the episode suites. Raises GateConfigError."""
    root = Path(root)
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("kind") != GATE_KIND or doc.get("schema_version") != SCHEMA_VERSION:
        raise GateConfigError(f"{path}: not a {GATE_KIND} v{SCHEMA_VERSION} file")
    _check(root / doc["base_policy"]["path"], doc["base_policy"]["sha256"], "base policy")
    cal_path = root / doc["calibration"]["path"]
    _check(cal_path, doc["calibration"]["file_sha256"], "calibration")
    calibration = Calibration.from_dict(json.loads(cal_path.read_text()))
    if calibration.sha256() != doc["calibration"]["calibration_sha256"]:
        raise GateConfigError("calibration content hash mismatch")
    sets_path = root / doc["locked_set"]["path"]
    _check(sets_path, doc["locked_set"]["file_sha256"], "hardening sets")
    sets_doc = json.loads(sets_path.read_text())
    locked = sets_doc[LOCKED_SPLIT]
    if locked_digest(locked["episodes"]) != doc["locked_set"]["locked_sha256"]:
        raise GateConfigError("locked test set does not match its recorded SHA-256")
    by_id = {e["id"]: e for e in locked["episodes"]}

    primary_id = doc["primary"]["candidate_id"]
    candidates = []
    for c in doc["candidates"]:
        record_path = root / c["record"]
        _check(record_path, c["record_sha256"], "candidate record")
        record = json.loads(record_path.read_text())
        if record.get("status") != "candidate" or record["candidate_id"] != c["candidate_id"]:
            raise GateConfigError(
                f"{c['record']}: not the registered candidate {c['candidate_id']}"
            )
        if record["calibration"]["sha256"] != calibration.sha256():
            raise GateConfigError(f"{c['candidate_id']}: made with another calibration")
        if record["base_policy"]["sha256"] != doc["base_policy"]["sha256"]:
            raise GateConfigError(f"{c['candidate_id']}: made for another base policy")
        vector = tuple(float(v) for v in validate_params(record["reflex"]["vector"]))
        candidates.append(Candidate(c["candidate_id"], c["record"], c["record_sha256"], int(c["seed"]),
                                    vector, c["candidate_id"] == primary_id))  # fmt: skip
    if sum(c.primary for c in candidates) != 1:
        raise GateConfigError("exactly one registered candidate must be the primary")

    suites = doc["suites"]
    holdout = tuple(
        episode_from_dict(by_id[i], "gate_holdout") for i in suites["holdout_attacks"]["ids"]
    )
    nominal = tuple(
        [episode_from_dict(by_id[i], "gate_nominal") for i in suites["nominal"]["locked_ids"]]
        + [episode_from_dict(e, "gate_nominal") for e in suites["nominal"]["episodes"]]
    )
    g = suites["gauntlet"]
    lo, hi = g["starts_substeps"]
    gauntlet = tuple(
        EpisodeSpec(
            id=f"gx-{cond['id']}-{k}", kind="attack", split="gate_gauntlet",
            source=f"gauntlet {cond['id']} start {k}",
            ground_friction_mu=cond.get("ground_friction_mu"), payload_kg=cond.get("payload_kg", 0.0),
            payload_offset_xyz_m=tuple(cond.get("payload_offset_xyz_m", (0.0, 0.0, 0.0))),
            push=PushSpec(cond["force_n"], tuple(g["direction"]), k, g["duration_substeps"]),
            tags=(cond["id"],),
        )
        for cond in g["conditions"]
        for k in range(lo, hi)
    )  # fmt: skip
    known = tuple(
        episode_from_dict(e, "gate_known_failures") for e in suites["known_failures"]["episodes"]
    )
    if any(e.kind != "attack" for e in holdout) or any(e.kind != "nominal" for e in nominal):
        raise GateConfigError("suite kinds do not match")
    shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    return Gate(doc["name"], shown, file_sha(path), doc, calibration, tuple(candidates), primary_id,
                holdout, nominal, gauntlet, known)  # fmt: skip
