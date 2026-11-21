"""Compiled models for multi-axis scenarios (friction, payload), with physics reports and an LRU
cache. A cached model is read-only after it is built; every episode uses its own MjData."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robustlab.physics.model import (
    ResolvedIds,
    apply_model_parameters,
    check_model,
    resolve_ids,
)
from robustlab.physics.scene import (
    PAYLOAD_BODY,
    PAYLOAD_GEOM,
    PAYLOAD_MODEL_VERSION,
    SCENE_TRANSFORM_VERSION,
    Payload,
    SceneError,
    SceneInfo,
    build_scene,
)

CACHE_SIZE = 8
RESOLUTION_RULE = (
    "MuJoCo contact parameters: the geom with the higher priority supplies friction and condim; "
    "equal priority takes the element-wise maximum friction. Floor priority 1 > foot priority 0, "
    "so every floor contact uses the floor's friction."
)


def signature(scene: str, base_xml_sha256: str, mu: float | None, payload: Payload | None) -> tuple:
    mass = 0.0 if payload is None else float(payload.mass_kg)
    offset = (
        (0.0, 0.0, 0.0) if payload is None or mass == 0 else tuple(map(float, payload.offset_xyz_m))
    )
    return (scene, SCENE_TRANSFORM_VERSION, base_xml_sha256, PAYLOAD_MODEL_VERSION,
            None if mu is None else float(mu), mass, *offset)  # fmt: skip


@dataclass(frozen=True)
class ScenarioModel:
    scene_info: SceneInfo
    ids: ResolvedIds
    signature: tuple
    base: bool  # True: the unmodified base model object (no compilation)

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene_info.model


def _check_contact_rules(model: mujoco.MjModel) -> None:
    if model.npair:
        raise SceneError("explicit <pair> contacts present; floor friction alone would not apply")
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE:
        raise SceneError("contact override flag set; geom friction would be ignored")


def resolved_friction(model: mujoco.MjModel, geom_a: int, geom_b: int) -> list[float]:
    pa, pb = int(model.geom_priority[geom_a]), int(model.geom_priority[geom_b])
    if pa > pb:
        return [float(x) for x in model.geom_friction[geom_a]]
    if pb > pa:
        return [float(x) for x in model.geom_friction[geom_b]]
    return [float(x) for x in np.maximum(model.geom_friction[geom_a], model.geom_friction[geom_b])]


def observed_floor_contacts(model, data, floor: int) -> list[dict]:
    out = []
    for c in data.contact[: data.ncon]:
        if floor not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor else c.geom1
        out.append({"geom": model.geom(int(other)).name or model.body(int(model.geom_bodyid[other])).name,
                    "dim": int(c.dim), "friction": [float(x) for x in c.friction]})  # fmt: skip
    return out


def friction_report(
    model: mujoco.MjModel, ids: ResolvedIds, policy, requested: float | None
) -> dict:
    floor = ids.floor_geom
    data = mujoco.MjData(model)
    data.qpos[:] = policy.default_qpos
    data.qvel[:] = policy.default_qvel
    mujoco.mj_forward(model, data)
    contacts = observed_floor_contacts(model, data, floor)
    sliding = sorted({round(c["friction"][0], 12) for c in contacts})
    return {
        "requested_mu": requested,
        "nominal_mu": 0.6,
        "configured_floor_friction": [float(x) for x in model.geom_friction[floor]],
        "configured_mu": float(model.geom_friction[floor, 0]),
        "resolved_foot_floor": {
            model.geom(int(g)).name: resolved_friction(model, floor, int(g)) for g in ids.foot_geoms
        },
        "observed_at_reset": {
            "label": "observed in MuJoCo contacts at the reset pose (mj_forward), before running",
            "contacts": len(contacts),
            "sliding_mu_values": sliding,
            "dims": sorted({c["dim"] for c in contacts}),
        },
        "priorities": {"floor": int(model.geom_priority[floor]),
                       "feet": sorted({int(model.geom_priority[g]) for g in ids.foot_geoms})},
        "condim": {"floor": int(model.geom_condim[floor]),
                   "feet": sorted({int(model.geom_condim[g]) for g in ids.foot_geoms})},
        "rule": RESOLUTION_RULE,
        "applies_to": "all floor contacts (feet and, in fall-contact, fallen body parts)",
    }  # fmt: skip


def _quat_to_mat(quat) -> np.ndarray:
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def mass_report(model: mujoco.MjModel, ids: ResolvedIds) -> dict:
    """Total mass, and for a payload the composite trunk+payload rigid body in trunk axes."""
    total = float(model.body_mass.sum())
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    report = {"total_mass_kg": total, "payload": None}
    if pid < 0:
        return report
    trunk = ids.trunk_body
    # Everything in the trunk body frame. Trunk: CoM body_ipos, principal axes body_iquat.
    # Payload: body frame at body_pos/body_quat inside the trunk frame, then its own ipos/iquat.
    r_payload = _quat_to_mat(model.body_quat[pid])
    frames = (
        (trunk, np.asarray(model.body_ipos[trunk]), _quat_to_mat(model.body_iquat[trunk])),
        (
            pid,
            np.asarray(model.body_pos[pid]) + r_payload @ model.body_ipos[pid],
            r_payload @ _quat_to_mat(model.body_iquat[pid]),
        ),
    )
    parts = []
    for body, com, rot in frames:
        inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T
        parts.append((float(model.body_mass[body]), com.astype(float), inertia))
    m_total = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m_total
    inertia = np.zeros((3, 3))
    for m, c, i_c in parts:
        d = c - com
        inertia += i_c + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    eig = np.linalg.eigvalsh(inertia)
    report["payload"] = {
        "body_id": int(pid),
        "geom_id": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)),
        "mass_kg": float(model.body_mass[pid]),
        "trunk_mass_kg": float(model.body_mass[trunk]),
        "composite_mass_kg": float(m_total),
        "composite_com_trunk_frame_m": com.tolist(),
        "composite_inertia_kgm2": inertia.tolist(),
        "composite_inertia_eigenvalues": eig.tolist(),
        "positive_definite": bool(eig.min() > 0 and np.allclose(inertia, inertia.T)),
    }
    return report


class ScenarioModelCache:
    """LRU cache of compiled scenario models keyed by `signature(...)`."""

    def __init__(self, maxsize: int = CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, ScenarioModel] = OrderedDict()
        self.builds = 0
        self.hits = 0

    def get(self, base_stack, mu: float | None, payload: Payload | None) -> ScenarioModel:
        base_info = base_stack.scene_info
        key = signature(base_info.name, base_info.base_xml_sha256, mu, payload)
        if mu is None and key[5] == 0.0:  # nominal physics: the original model, never recompiled
            return ScenarioModel(base_info, base_stack.ids, key, base=True)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
        info = build_scene(Path(base_stack.root) / "assets/go1/walk.xml", base_info.name,
                           friction_mu=mu, payload=payload)  # fmt: skip
        model = info.model
        _check_contact_rules(model)
        apply_model_parameters(model, base_stack.policy.model_parameters)
        check_model(model, base_stack.policy)
        entry = ScenarioModel(info, resolve_ids(model), key, base=False)
        with self._lock:
            self.builds += 1
            self._entries[key] = entry
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
        return entry

    def stats(self) -> dict:
        return {"size": len(self._entries), "maxsize": self.maxsize, "builds": self.builds,
                "hits": self.hits}  # fmt: skip


MODEL_CACHE = ScenarioModelCache()
