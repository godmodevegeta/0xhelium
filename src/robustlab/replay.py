"""Renderer-neutral replay data from recorded trajectories.

Poses come from MuJoCo forward kinematics (`mj_kinematics`) applied to recorded `qpos`; nothing is
integrated. Meshes are exported from the compiled model, whose mesh frames are the frames
`geom_xpos` / `geom_xmat` refer to (MuJoCo recentres meshes at compile time, so the raw STL
coordinates would not line up).
"""

from __future__ import annotations

import base64
from typing import Any

import mujoco
import numpy as np

from robustlab.physics.scene import PAYLOAD_GEOM

VISUAL_GROUP = 2
FOOT_ORDER = ("FR", "FL", "RR", "RL")


def _b64(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def visual_geoms(model: mujoco.MjModel) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_group[g] == VISUAL_GROUP]


def model_payload(model: mujoco.MjModel) -> dict[str, Any]:
    """Visual geoms in replay order and their meshes (float32 vertices, int32 faces, base64)."""
    geoms = visual_geoms(model)
    mesh_ids = sorted({int(model.geom_dataid[g]) for g in geoms})
    meshes = {}
    for m in mesh_ids:
        v0, nv = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        f0, nf = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        meshes[model.mesh(m).name] = {
            "vertices": _b64(model.mesh_vert[v0 : v0 + nv], np.float32),
            "faces": _b64(model.mesh_face[f0 : f0 + nf], np.int32),
            "n_vertices": nv,
            "n_faces": nf,
        }
    return {
        "frame": "MuJoCo world frame, z up, metres; quaternions wxyz",
        "geoms": [
            {
                "body": model.body(int(model.geom_bodyid[g])).name,
                "mesh": model.mesh(int(model.geom_dataid[g])).name,
                "rgba": [float(c) for c in model.geom_rgba[g]],
            }
            for g in geoms
        ],
        "meshes": meshes,
    }


def kinematic_frames(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World positions [F, G, 3] and wxyz quaternions [F, G, 4] of the visual geoms for each qpos
    row, computed by `mj_kinematics` on a scratch MjData."""
    geoms = visual_geoms(model)
    data = mujoco.MjData(model)
    frames = qpos.shape[0]
    pos = np.empty((frames, len(geoms), 3))
    quat = np.empty((frames, len(geoms), 4))
    for f in range(frames):
        data.qpos[:] = qpos[f]
        mujoco.mj_kinematics(model, data)
        for j, g in enumerate(geoms):
            pos[f, j] = data.geom_xpos[g]
            mujoco.mju_mat2Quat(quat[f, j], data.geom_xmat[g])
    return pos, quat


def payload_box_frames(model: mujoco.MjModel, qpos: np.ndarray) -> dict | None:
    """World pose of the welded payload box (M5) for each qpos row, or None if the model has no
    payload. The box is drawn from the same kinematics as the robot, not placed by the browser."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)
    if gid < 0:
        return None
    data = mujoco.MjData(model)
    pos = np.empty((qpos.shape[0], 3))
    quat = np.empty((qpos.shape[0], 4))
    for f in range(qpos.shape[0]):
        data.qpos[:] = qpos[f]
        mujoco.mj_kinematics(model, data)
        pos[f] = data.geom_xpos[gid]
        mujoco.mju_mat2Quat(quat[f], data.geom_xmat[gid])
    body = int(model.geom_bodyid[gid])
    return {
        "half_size_m": [float(x) for x in model.geom_size[gid]],
        "mass_kg": float(model.body_mass[body]),
        "pos": _b64(pos, np.float32),
        "quat": _b64(quat, np.float32),
    }


def case_payload(model: mujoco.MjModel, trajectory: dict[str, Any], outcome: dict) -> dict:
    """Replay data for one episode from an M1-format trajectory dict (frame 0 = reset state)."""
    qpos = np.vstack([np.asarray(trajectory["initial_qpos"]), np.asarray(trajectory["qpos"])])
    if not np.isfinite(qpos).all():  # non-finite rows (numerical failure) are not drawable
        last_good = int(np.flatnonzero(~np.isfinite(qpos).all(axis=1))[0])
        qpos = qpos[:last_good]
    pos, quat = kinematic_frames(model, qpos)
    payload_box = payload_box_frames(model, qpos)
    n = qpos.shape[0]
    up_z = [1.0] + list(trajectory["up_z"])[: n - 1]
    trunk_z = [float(qpos[0, 2])] + list(trajectory["trunk_z"])[: n - 1]
    nonfoot = [False] + list(trajectory.get("nonfoot_contact", [False] * (n - 1)))[: n - 1]
    feet = [[True] * 4] + [list(c) for c in trajectory["contact"]][: n - 1]
    return {
        "frames": n,
        "pos": _b64(pos, np.float32),
        "quat": _b64(quat, np.float32),
        "up_z": [round(float(u), 5) for u in up_z],
        "trunk_z": [round(float(z), 5) for z in trunk_z],
        "nonfoot": [bool(x) for x in nonfoot],
        "feet": feet,
        "outcome": outcome,
        "payload_box": payload_box,
    }


def replay_payload(
    model: mujoco.MjModel,
    nominal: tuple[dict, dict],
    attacked: tuple[dict, dict],
    push: dict | None,
    *,
    source: str,
    verification: dict | None = None,
    extra: dict | None = None,
    case_models: dict[str, mujoco.MjModel] | None = None,
) -> dict[str, Any]:
    """Combine two episodes (trajectory dict, outcome summary) into one synchronized replay.
    `case_models` overrides `model` per case ("nominal" / "attacked"), e.g. a scenario model
    with a payload; visual geoms are identical across those models (the payload is not visual
    group 2), so one mesh set serves both."""
    ctrl_dt = float(attacked[0]["ctrl_dt"])
    models = {"nominal": model, "attacked": model, **(case_models or {})}
    for name, m in models.items():
        if len(visual_geoms(m)) != len(visual_geoms(model)):
            raise ValueError(f"{name} model has a different visual geom set")
    cases = {
        "nominal": case_payload(models["nominal"], *nominal),
        "attacked": case_payload(models["attacked"], *attacked),
    }
    frames = max(c["frames"] for c in cases.values())
    return {
        "fps": round(1.0 / ctrl_dt),
        "ctrl_dt": ctrl_dt,
        "frames": frames,
        "time": [round(i * ctrl_dt, 6) for i in range(frames)],
        "geom_count": len(visual_geoms(model)),
        "foot_order": list(FOOT_ORDER),
        "cases": cases,
        "push": None
        if push is None
        else {
            "start_s": push["start_s"],
            "end_s": push["end_s"],
            "world_force_n": push["world_force_n"],
            "force_n": push["force_n"],
        },
        "source": source,
        "verification": verification,
        **(extra or {}),
    }
