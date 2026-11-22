"""Policy registry: the frozen baseline (v0), immutable promoted versions, rejection records and
an atomically replaced `active.json` pointer. Only `apply_decision` (called by the gate) writes
versions or moves the pointer."""

from __future__ import annotations

import json
import time
from pathlib import Path

from robustlab.experiments.trace import _atomic_write_json

REGISTRY_DIR = "policies/registry"
BASELINE_VERSION = "v0-baseline"


class RegistryError(RuntimeError):
    pass


def _write_once(path: Path, doc: dict) -> None:
    if path.exists():
        raise RegistryError(f"{path} exists; registry entries are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, doc)


def registry_dir(root: Path) -> Path:
    return Path(root) / REGISTRY_DIR


def read_active(root: Path) -> dict | None:
    path = registry_dir(root) / "active.json"
    return json.loads(path.read_text()) if path.exists() else None


def ensure_baseline(root: Path, policy_path: str, policy_sha256: str) -> dict:
    """Create v0 (the frozen policy, no reflex) and point `active.json` at it if absent."""
    reg = registry_dir(root)
    version = reg / "versions" / f"{BASELINE_VERSION}.json"
    if not version.exists():
        _write_once(version, {"schema_version": 1, "kind": "robustlab-policy-version",
                              "version": BASELINE_VERSION, "parent": None,
                              "policy": {"path": policy_path, "sha256": policy_sha256},
                              "reflex": None, "source": "frozen pretrained go1-walk-s0 (M0)",
                              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})  # fmt: skip
    if read_active(root) is None:
        _atomic_write_json(reg / "active.json", {"schema_version": 1, "active": BASELINE_VERSION,
                                                 "history": [{"version": BASELINE_VERSION,
                                                              "reason": "initial frozen baseline"}]})  # fmt: skip
    return read_active(root)


def apply_decision(root: Path, gate_name: str, result_sha256: str, decision: dict, candidate: dict,
                   policy_path: str, policy_sha256: str) -> dict:  # fmt: skip
    """Promote (new immutable version + pointer move) or record a rejection. Returns the entry."""
    reg = registry_dir(root)
    active = ensure_baseline(root, policy_path, policy_sha256)
    entry = {"schema_version": 1, "gate": gate_name, "gate_result_sha256": result_sha256,
             "candidate_id": candidate["candidate_id"], "candidate_record": candidate["record"],
             "candidate_record_sha256": candidate["record_sha256"], "decision": decision["verdict"],
             "failed_criteria": decision["failed_criteria"], "statement": decision["statement"],
             "parent": active["active"],
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    if decision["verdict"] == "promote":
        number = len(list((reg / "versions").glob("v*.json")))
        version = f"v{number}-{candidate['candidate_id']}"
        _write_once(reg / "versions" / f"{version}.json", {
            **entry, "kind": "robustlab-policy-version", "version": version,
            "policy": {"path": policy_path, "sha256": policy_sha256, "weights": "frozen, unmodified"},
            "reflex": {"record": candidate["record"], "record_sha256": candidate["record_sha256"]},
        })  # fmt: skip
        history = [*active["history"], {"version": version, "reason": f"promoted by {gate_name}"}]
        _atomic_write_json(
            reg / "active.json", {"schema_version": 1, "active": version, "history": history}
        )
        return {**entry, "version": version}
    _write_once(reg / "rejected" / f"{candidate['candidate_id']}-{gate_name}.json",
                {**entry, "kind": "robustlab-policy-rejection", "active_unchanged": active["active"]})  # fmt: skip
    return entry
