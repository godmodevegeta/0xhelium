"""Assemble the 48-value actor observation for go1-walk-s0 from MuJoCo state.

Fields, in the policy's `observation.fields` order (validated by `parse_policy`):

| offset | size | field | source |
| --- | --- | --- | --- |
| 0 | 3 | local_linear_velocity | `local_linvel` velocimeter sensor at the IMU site |
| 3 | 3 | gyro | `gyro` sensor |
| 6 | 3 | gravity | `R_imu.T @ [0, 0, -1]`, i.e. minus row 2 of the IMU site rotation |
| 9 | 12 | joint_position | `qpos[7:] - default_pose` |
| 21 | 12 | joint_velocity | `qvel[6:]` |
| 33 | 12 | last_action | the delayed action supplied by the caller |
| 45 | 3 | command | `[vx, vy, yaw_rate]` in the robot frame |

`build` must be called right after `mj_step` with no extra `mj_forward`: the policy was trained on
the sensor values and site frames that `mj_step` leaves from its last substep.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robustlab.physics.model import ResolvedIds
from robustlab.policy.actor import Policy


class ObservationBuilder:
    def __init__(self, model: mujoco.MjModel, ids: ResolvedIds, policy: Policy) -> None:
        offsets = {name: (offset, size) for name, offset, size in policy.fields}

        def window(name: str) -> slice:
            offset, size = offsets[name]
            return slice(offset, offset + size)

        self._linvel = window("local_linear_velocity")
        self._gyro = window("gyro")
        self._gravity = window("gravity")
        self._joint_pos = window("joint_position")
        self._joint_vel = window("joint_velocity")
        self._last_action = window("last_action")
        self._command = window("command")
        self._velocimeter = slice(ids.velocimeter_adr, ids.velocimeter_adr + 3)
        self._gyro_sensor = slice(ids.gyro_adr, ids.gyro_adr + 3)
        self._imu = ids.imu_site
        self._default_pose = policy.default_pose
        self.size = policy.obs_dim

    def build(
        self,
        data: mujoco.MjData,
        last_action: np.ndarray,
        command: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        """Write the observation into `out` (float32, shape (48,)) and return it."""
        out[self._linvel] = data.sensordata[self._velocimeter]
        out[self._gyro] = data.sensordata[self._gyro_sensor]
        # site_xmat is a row-major 3x3; R.T @ [0, 0, -1] = -(row 2 of R).
        np.negative(data.site_xmat[self._imu, 6:9], out=out[self._gravity], dtype=np.float32)
        np.subtract(data.qpos[7:], self._default_pose, out=out[self._joint_pos], dtype=np.float32)
        out[self._joint_vel] = data.qvel[6:]
        out[self._last_action] = last_action
        out[self._command] = command
        return out
