"""Load the verified scene + policy once and build M1-format run manifests for its episodes.

Shared by `robustlab walk` and `robustlab experiment`, so both record the same provenance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robustlab.assets import load_lock, locked_sha256
from robustlab.experiments.trace import build_manifest
from robustlab.experiments.walk import (
    EpisodeRecord,
    WalkCriteria,
    criteria_dict,
    latency_summary,
)
from robustlab.physics.model import ResolvedIds, apply_model_parameters, check_model, resolve_ids
from robustlab.physics.scene import SceneInfo, build_scene
from robustlab.policy.actor import Actor, Policy, load_policy

MODEL_XML_RELPATH = "assets/go1/walk.xml"
PROVENANCE_RELPATH = "assets/go1/walk.provenance.json"


@dataclass
class Stack:
    root: Path
    policy_path: Path
    scene_info: SceneInfo
    ids: ResolvedIds
    policy: Policy
    actor: Actor
    load_s: float
    lock: dict
    provenance: dict

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene_info.model

    @property
    def policy_info(self) -> dict[str, Any]:
        return {
            "path": self.policy_path.as_posix(),
            "sha256": self.policy.sha256,
            "run_id": self.policy.run_id,
            "env_name": self.policy.env_name,
            "trained_with": self.policy.versions,
        }

    @property
    def library_versions(self) -> dict[str, str]:
        return {"numpy": np.__version__, "mujoco": mujoco.__version__}

    def command_error(self, command) -> str | None:
        """Why `command` is outside the policy's trained range, or None."""
        for value, (lo, hi), name in zip(command, self.policy.command_ranges, ("vx", "vy", "yaw")):
            if not lo <= value <= hi:
                return f"command {name}={value} outside trained range [{lo}, {hi}]"
        return None

    def manifest_for(
        self,
        record: EpisodeRecord,
        *,
        run_id: str,
        command,
        seconds: float,
        seed: int,
        outcome: dict,
        criteria: WalkCriteria,
    ) -> dict[str, Any]:
        return build_manifest(
            run_id=run_id,
            record=record,
            command=tuple(command),
            seconds=seconds,
            seed=seed,
            criteria=criteria_dict(criteria),
            outcome=outcome,
            latency=latency_summary(record.tick_ns, record.ctrl_dt),
            load_s=self.load_s,
            policy_info=self.policy_info,
            lock=self.lock,
            provenance=self.provenance,
            library_versions=self.library_versions,
            repo_root=self.root,
            scene=self.scene_info.describe(),
            push=record.push,
        )


def load_stack(root: Path, policy_path: Path, scene: str = "fall-contact") -> Stack:
    """Load and cross-check scene + policy against the asset lock. Raises PolicyError,
    ModelError, AssetError, SceneError or ValueError."""
    root = Path(root)
    policy_path = Path(policy_path)
    start = time.perf_counter()
    resolved = policy_path if policy_path.is_absolute() else root / policy_path
    policy_rel = resolved.resolve().relative_to(root.resolve()).as_posix()
    policy = load_policy(resolved, locked_sha256(root, policy_rel))
    scene_info = build_scene(root / MODEL_XML_RELPATH, scene)
    apply_model_parameters(scene_info.model, policy.model_parameters)
    check_model(scene_info.model, policy)
    ids = resolve_ids(scene_info.model)
    actor = Actor(policy)
    load_s = time.perf_counter() - start
    return Stack(
        root=root,
        policy_path=policy_path,
        scene_info=scene_info,
        ids=ids,
        policy=policy,
        actor=actor,
        load_s=load_s,
        lock=load_lock(root),
        provenance=json.loads((root / PROVENANCE_RELPATH).read_text()),
    )
