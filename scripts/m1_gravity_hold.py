"""Motor-actuator hold with ctrl = qfrc_bias. Reports drift and gravity/limit ratios.

Run: conda run -n AIControl python scripts/m1_gravity_hold.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

from aloha_pp.kin.ik import DlsIk, top_down_quat
from aloha_pp.robot import ARM_JOINTS, GRIPPER_ACTUATOR, scene_xml
from aloha_pp.sim.env import load_config


def main() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
    data = mujoco.MjData(model)
    ik = DlsIk(model, cfg["tcp_site"], damping=0.05, max_iters=40, pos_tol=1e-3, rot_tol=1e-2)
    home = np.asarray(cfg["home_q"], dtype=float)
    poses = {"home": home}
    ik_failures = 0
    for label, xy in (("above_left", (0.30, 0.125)), ("above_right", (0.30, -0.125))):
        yaw = float(np.arctan2(xy[1], xy[0]))
        result = ik.solve(np.array([xy[0], xy[1], 0.16]), top_down_quat(yaw, 0), home)
        if not result.success:
            ik_failures += 1
        poses[label] = result.q
    print(f"ik_failures {ik_failures}")
    if ik_failures:
        raise RuntimeError(f"IK failed for {ik_failures} gravity pose(s)")

    act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS
    ]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS
    ]
    dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    qpos_ids = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in ARM_JOINTS
    ]
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    steps = int(round(2.0 / model.opt.timestep))
    worst = 0.0
    for label, q in poses.items():
        mujoco.mj_resetData(model, data)
        for adr, value in zip(qpos_ids, q):
            data.qpos[adr] = value
        data.ctrl[grip] = float(cfg["gripper_open"])
        mujoco.mj_forward(model, data)
        bias = np.array([data.qfrc_bias[dof] for dof in dof_ids])
        print(f"pose {label}")
        for name, torque, joint_id in zip(ARM_JOINTS, bias, joint_ids):
            limit = float(model.jnt_actfrcrange[joint_id, 1])
            print(f"  {name}: gravity {torque:.4f} Nm  ratio {abs(torque) / limit:.4f}")
        q0 = data.qpos[qpos_ids].copy()
        max_drift = np.zeros(6)
        for _ in range(steps):
            for act, dof in zip(act_ids, dof_ids):
                data.ctrl[act] = data.qfrc_bias[dof]
            mujoco.mj_step(model, data)
            max_drift = np.maximum(max_drift, np.abs(data.qpos[qpos_ids] - q0))
        worst = max(worst, float(max_drift.max()))
        print("  drift " + " ".join(f"{value:.6f}" for value in max_drift))
    print(f"max_drift_rad {worst:.6e} limit 0.01")
    if worst >= 0.01:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
