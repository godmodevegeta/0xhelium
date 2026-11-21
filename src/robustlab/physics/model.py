"""Load weeble's exported Go1 `walk.xml`, apply the policy's training-time model parameters and
resolve the named MuJoCo IDs used by the control loop.

The policy JSON does not list joint names. Its joint order is implicit: `qpos[7:]`, `qvel[6:]`,
`default_pose`, the action vector and `ctrl` all share one order, the Playground Go1 convention of
legs in `feet_geoms` order (FR, FL, RR, RL) and joints (hip, thigh, calf) within each leg.
`check_model` derives that order and verifies that actuator `i` drives the joint stored at
`qpos[7 + i]` / `qvel[6 + i]`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

NQ, NV, NU = 19, 18, 12
SIM_DT = 0.004
CTRL_DT = 0.02
N_SUBSTEPS = 5
LEG_JOINTS = ("hip", "thigh", "calf")

# MuJoCo names in walk.xml; check_model confirms they equal the names the policy declares.
TRUNK_BODY = "trunk"
IMU_SITE = "imu"
FOOT_GEOMS = ("FR", "FL", "RR", "RL")
FLOOR_GEOM = "floor"
GYRO_SENSOR = "gyro"
VELOCIMETER_SENSOR = "local_linvel"


class ModelError(Exception):
    """The MuJoCo model does not match the policy's model contract."""


@dataclass(frozen=True)
class ResolvedIds:
    trunk_body: int
    imu_site: int
    foot_geoms: tuple[int, int, int, int]
    floor_geom: int
    velocimeter_adr: int
    gyro_adr: int


def load_model(xml_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _require_id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str, kind: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj, name)
    if obj_id < 0:
        raise ModelError(f"{kind} {name!r} not found in model")
    return int(obj_id)


def _sensor_adr(model: mujoco.MjModel, name: str, dim: int) -> int:
    sensor_id = _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name, "sensor")
    if int(model.sensor_dim[sensor_id]) != dim:
        raise ModelError(f"sensor {name!r} has dim {model.sensor_dim[sensor_id]}, expected {dim}")
    return int(model.sensor_adr[sensor_id])


def resolve_ids(model: mujoco.MjModel) -> ResolvedIds:
    """Resolve every named object the loop needs; raises `ModelError` naming a missing one."""
    geom = mujoco.mjtObj.mjOBJ_GEOM
    feet = tuple(_require_id(model, geom, name, "geom") for name in FOOT_GEOMS)
    return ResolvedIds(
        trunk_body=_require_id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY, "body"),
        imu_site=_require_id(model, mujoco.mjtObj.mjOBJ_SITE, IMU_SITE, "site"),
        foot_geoms=feet,  # type: ignore[arg-type]
        floor_geom=_require_id(model, geom, FLOOR_GEOM, "geom"),
        velocimeter_adr=_sensor_adr(model, VELOCIMETER_SENSOR, 3),
        gyro_adr=_sensor_adr(model, GYRO_SENSOR, 3),
    )


def apply_model_parameters(model: mujoco.MjModel, params: Mapping[str, Any]) -> None:
    """Assign the policy's `model_parameters` arrays and `opt` fields. Pure assignment, so
    applying twice equals applying once. Unknown names or wrong shapes raise `ModelError`."""
    arrays = params.get("arrays", {})
    options = params.get("options", {})
    for name, value in arrays.items():
        target = getattr(model, name, None)
        if not isinstance(target, np.ndarray):
            raise ModelError(f"model_parameters.arrays.{name}: not a model array")
        source = np.asarray(value, dtype=target.dtype)
        if source.shape != target.shape:
            raise ModelError(
                f"model_parameters.arrays.{name}: shape {source.shape}, model has {target.shape}"
            )
        target[...] = source
    for name, value in options.items():
        if not hasattr(model.opt, name):
            raise ModelError(f"model_parameters.options.{name}: not an MjOption field")
        setattr(model.opt, name, value)


def policy_joint_order(feet: Sequence[str]) -> tuple[str, ...]:
    """Joint names in policy order: legs in `feet` order, then hip, thigh, calf."""
    return tuple(f"{leg}_{joint}_joint" for leg in feet for joint in LEG_JOINTS)


def actuated_joint_order(model: mujoco.MjModel) -> tuple[str, ...]:
    """Names of the joints driven by each actuator, in actuator order."""
    names = []
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ModelError(f"actuator {i} does not use a joint transmission")
        names.append(model.joint(int(model.actuator_trnid[i, 0])).name)
    return tuple(names)


def check_model(model: mujoco.MjModel, policy: Mapping[str, Any]) -> None:
    """Assert the model matches the policy's model contract; raises `ModelError`.

    `policy` is the raw policy JSON mapping (switched to the validated `Policy` in Task 4.2).
    """
    facts = {"nq": (model.nq, NQ), "nv": (model.nv, NV), "nu": (model.nu, NU)}
    for field, (actual, expected) in facts.items():
        if actual != expected:
            raise ModelError(f"{field} is {actual}, expected {expected}")
    if model.opt.timestep != SIM_DT:
        raise ModelError(f"timestep is {model.opt.timestep}, expected {SIM_DT}")

    declared = {
        "n_substeps": (policy["n_substeps"], N_SUBSTEPS),
        "simulation_dt": (policy["simulation_dt"], SIM_DT),
        "control_dt": (policy["control_dt"], CTRL_DT),
        "force_body": (policy["force_body"], TRUNK_BODY),
        "floor_geom": (policy["floor_geom"], FLOOR_GEOM),
        "feet_geoms": (tuple(policy["feet_geoms"]), FOOT_GEOMS),
        "sensors.gravity_site": (policy["sensors"]["gravity_site"], IMU_SITE),
        "sensors.gyro": (policy["sensors"]["gyro"], GYRO_SENSOR),
        "sensors.local_linvel": (policy["sensors"]["local_linvel"], VELOCIMETER_SENSOR),
    }
    for field, (actual, expected) in declared.items():
        if actual != expected:
            raise ModelError(f"policy {field} is {actual!r}, expected {expected!r}")
    if not np.isclose(policy["simulation_dt"] * policy["n_substeps"], policy["control_dt"]):
        raise ModelError("policy simulation_dt * n_substeps != control_dt")

    expected_order = policy_joint_order(policy["feet_geoms"])
    actual_order = actuated_joint_order(model)
    if actual_order != expected_order:
        raise ModelError(f"actuator joint order {actual_order} != policy order {expected_order}")
    for i, name in enumerate(expected_order):
        joint = model.joint(name)
        if int(joint.qposadr[0]) != 7 + i or int(joint.dofadr[0]) != 6 + i:
            raise ModelError(f"joint {name} is not at qpos[{7 + i}] / qvel[{6 + i}]")
