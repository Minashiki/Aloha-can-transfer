"""Arm joint indexing shared by the M3 controllers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aloha_pp.robot import ARM_JOINTS, GRIPPER_ACTUATOR


@dataclass(frozen=True)
class ArmIndex:
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    act_ids: np.ndarray
    grip_act: int
    force_lo: np.ndarray
    force_hi: np.ndarray
    damping: np.ndarray


def arm_index(model: mujoco.MjModel) -> ArmIndex:
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
        dtype=int,
    )
    act_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS],
        dtype=int,
    )
    if np.any(joint_ids < 0) or np.any(act_ids < 0):
        raise ValueError("arm joints or actuators are missing from the model")
    dof = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    return ArmIndex(
        joint_ids=joint_ids,
        qpos_adr=np.asarray(model.jnt_qposadr[joint_ids], dtype=int),
        dof_adr=dof,
        act_ids=act_ids,
        grip_act=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR),
        force_lo=model.jnt_actfrcrange[joint_ids, 0].copy(),
        force_hi=model.jnt_actfrcrange[joint_ids, 1].copy(),
        damping=model.dof_damping[dof].copy(),
    )


def dense_mass(model: mujoco.MjModel, data: mujoco.MjData, dof_adr: np.ndarray, buffer: np.ndarray) -> np.ndarray:
    """6×6 arm block of ``mj_fullM``. ``data.M`` must already match ``data.qpos``."""
    mujoco.mj_fullM(model, data, buffer)
    return buffer[np.ix_(dof_adr, dof_adr)].copy()
