"""Runtime switch for MuJoCo gravcomp + actuatorgravcomp on the position model."""

from __future__ import annotations

import mujoco

from aloha_pp.robot import ARM_JOINTS


def apply_gravity_compensation(model: mujoco.MjModel, enabled: bool) -> None:
    """Turn body ``gravcomp`` and joint ``actuatorgravcomp`` on or off.

    Default XML leaves both at zero. M3's B1 baseline enables them so the
    compensation force is applied through the actuators and therefore counts
    against ``actuatorfrcrange``. Only ``left/`` robot bodies are affected;
    the can is not.
    """
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("left/"):
            model.body_gravcomp[body_id] = 1.0 if enabled else 0.0
    model.flg_gravcomp = 1 if enabled else 0
    for joint_name in ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        model.jnt_actgravcomp[joint_id] = 1 if enabled else 0
