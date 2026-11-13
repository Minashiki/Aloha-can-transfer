"""Swappable low-level controllers. M1 tracks a Cartesian target with position actuators."""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

from aloha_pp.kin.ik import DlsIk, IkResult
from aloha_pp.robot import ARM_JOINTS, GRIPPER_ACTUATOR


class LowLevelController(Protocol):
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
        """Apply one 20 Hz action. ``action`` is xyz, quat wxyz, gripper opening, in the base frame."""


class PositionIkController:
    """Damped least-squares IK into the position actuators, from the current arm q."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        *,
        damping: float = 0.05,
        max_iters: int = 40,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
        rot_weight: float = 0.4,
    ) -> None:
        self.ik = DlsIk(
            model,
            site_name,
            damping=damping,
            max_iters=max_iters,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            rot_weight=rot_weight,
        )
        self.arm_act = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS],
            dtype=int,
        )
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        self.grip_range = model.actuator_ctrlrange[self.grip_act].copy()
        self.last_result: IkResult | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model, data
        self.last_result = None

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
        del model
        action = np.asarray(action, dtype=float).reshape(8)
        quat = action[3:7]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError("action quaternion has zero length")
        quat = quat / norm
        q_init = self.ik.arm_qpos(data)
        result = self.ik.solve(action[:3], quat, q_init)
        q_cmd = result.q if result.success else q_init
        data.ctrl[self.arm_act] = q_cmd
        grip = float(np.clip(action[7], self.grip_range[0], self.grip_range[1]))
        data.ctrl[self.grip_act] = grip
        self.last_result = result
