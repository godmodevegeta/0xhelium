"""Write and read one M0 run: `trajectory.json` (50 Hz) and `manifest.json`.

Each file is written to a temporary file in the run directory, fsynced and renamed into place with
`os.replace`. The trajectory is written first and the manifest last, so a run directory with a
`manifest.json` is complete. The manifest records the trajectory's SHA-256.

Stdlib + NumPy only; callers pass library versions in.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robustlab.experiments.walk import EpisodeRecord

SCHEMA_VERSION = 1
TRAJECTORY_NAME = "trajectory.json"
MANIFEST_NAME = "manifest.json"
JOINT_ORDER_NOTE = "qpos[7:], qvel[6:], action: FR, FL, RR, RL legs x (hip, thigh, calf)"

MANIFEST_REQUIRED_KEYS = (
    "schema_version",
    "run_id",
    "created_utc",
    "assets",
    "policy",
    "sources",
    "versions",
    "platform",
    "git",
    "ctrl_dt",
    "sim_dt",
    "n_substeps",
    "command",
    "horizon_s",
    "n_ticks",
    "initial_state",
    "seed",
    "criteria",
    "outcome",
    "latency",
    "timing",
    "trajectory",
)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    trajectory: Path
    manifest: Path
    trajectory_write_s: float
    manifest_write_s: float


def _jsonable(array: np.ndarray) -> list:
    values = array.tolist()
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        # JSON has no NaN/inf; store them as null (the record's `finite` flags mark the Ticks).
        def clean(v):
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v if math.isfinite(v) else None

        values = clean(values)
    return values


def trajectory_dict(record: EpisodeRecord) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rate_hz": round(1.0 / record.ctrl_dt),
        "ctrl_dt": record.ctrl_dt,
        "row_semantics": "row k is the state after Tick k; action[k] was applied during Tick k",
        "joint_order": JOINT_ORDER_NOTE,
        "foot_order": ["FR", "FL", "RR", "RL"],
        "qpos_layout": "[0:3] trunk position (m, world), [3:7] trunk quat wxyz, [7:19] joints",
        "initial_qpos": _jsonable(record.initial_qpos),
        "initial_qvel": _jsonable(record.initial_qvel),
        "time": _jsonable(record.time),
        "qpos": _jsonable(record.qpos),
        "qvel": _jsonable(record.qvel),
        "action": _jsonable(record.action),
        "command": _jsonable(record.command),
        "contact": record.contact.tolist(),
        "up_z": _jsonable(record.up_z),
        "trunk_z": _jsonable(record.trunk_z),
        "finite": record.finite.tolist(),
        "warning_total": record.warning_total.tolist(),
    }


def _atomic_write_json(path: Path, payload: Any) -> tuple[float, str]:
    """Write JSON atomically. Returns (seconds, sha256 of the bytes written)."""
    start = time.perf_counter()
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600; run evidence is meant to be shared
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return time.perf_counter() - start, hashlib.sha256(data).hexdigest()


def write_run(
    run_dir: Path,
    record: EpisodeRecord,
    manifest: dict[str, Any],
    *,
    overwrite: bool = False,
) -> RunPaths:
    """Write `trajectory.json` then `manifest.json` into `run_dir`.

    `manifest` is completed in place with the trajectory hash and JSON write times. Raises
    `FileExistsError` if the run already has a manifest and `overwrite` is false."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / TRAJECTORY_NAME
    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass overwrite=True to replace the run")

    trajectory_s, trajectory_sha = _atomic_write_json(trajectory_path, trajectory_dict(record))
    manifest["trajectory"] = {"file": TRAJECTORY_NAME, "sha256": trajectory_sha}
    manifest.setdefault("timing", {})["trajectory_write_s"] = trajectory_s
    manifest_s, _ = _atomic_write_json(manifest_path, manifest)
    return RunPaths(run_dir, trajectory_path, manifest_path, trajectory_s, manifest_s)


def read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = Path(run_dir)
    trajectory = json.loads((run_dir / TRAJECTORY_NAME).read_text())
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text())
    for name, doc in ((TRAJECTORY_NAME, trajectory), (MANIFEST_NAME, manifest)):
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{name}: unsupported schema_version {doc.get('schema_version')!r}")
    return trajectory, manifest


def git_revision(repo_root: Path) -> dict[str, Any]:
    """Current commit and whether tracked files differ from it; 'unknown' without git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def build_manifest(
    *,
    run_id: str,
    record: EpisodeRecord,
    command: tuple[float, float, float],
    seconds: float,
    seed: int,
    criteria: dict[str, Any],
    outcome: dict[str, Any],
    latency: dict[str, Any],
    load_s: float,
    policy_info: dict[str, Any],
    lock: dict[str, Any],
    provenance: dict[str, Any],
    library_versions: dict[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": {entry["path"]: entry["sha256"] for entry in lock["files"]},
        "policy": policy_info,
        "sources": {
            "weeble": {"repo": lock["source"]["repo"], "commit": lock["source"]["commit"]},
            "playground_commit": provenance.get("playground_commit"),
            "menagerie_commit": provenance.get("menagerie_commit"),
            "model_export_mujoco": provenance.get("mujoco_version"),
        },
        "versions": {"python": platform.python_version(), **library_versions},
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_executable": Path(sys.executable).name,
        },
        "git": git_revision(repo_root),
        "ctrl_dt": record.ctrl_dt,
        "sim_dt": record.sim_dt,
        "n_substeps": record.n_substeps,
        "command": list(command),
        "command_frame": "robot body frame: vx forward (m/s), vy left (m/s), yaw rate (rad/s)",
        "horizon_s": seconds,
        "n_ticks": record.n_ticks,
        "initial_state": {
            "qpos": record.initial_qpos.tolist(),
            "qvel": record.initial_qvel.tolist(),
            "source": "policy default_qpos / default_qvel, ctrl = qpos[7:], last_action = 0",
        },
        "seed": seed,
        "seed_note": "M0 episodes contain no randomness; the seed does not change results",
        "perturbations": "none (xfrc_applied cleared every Tick; no noise, pushes or resets)",
        "criteria": criteria,
        "outcome": outcome,
        "latency": latency,
        "timing": {"model_and_policy_load_s": load_s},
    }
