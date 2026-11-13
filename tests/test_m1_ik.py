"""DLS IK from the ready configuration into both bins."""

from __future__ import annotations

import mujoco
import numpy as np

from aloha_pp.kin.ik import DlsIk, top_down_quat
from aloha_pp.robot import scene_xml
from aloha_pp.sim.env import load_config


def test_ik_holds_current_pose_and_reaches_bin_samples() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    ik = DlsIk(
        model,
        cfg["tcp_site"],
        damping=float(cfg["ik"]["damping"]),
        max_iters=int(cfg["ik"]["max_iters"]),
        pos_tol=float(cfg["ik"]["pos_tol"]),
        rot_tol=float(cfg["ik"]["rot_tol"]),
        rot_weight=float(cfg["ik"]["rot_weight"]),
    )
    home = np.asarray(cfg["home_q"], dtype=float)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    pos, quat = ik.site_pose_in_base(data)
    held = ik.solve(pos, quat, home)
    assert held.success
    assert held.pos_err < 1e-3
    assert held.rot_err < 1e-2

    rng = np.random.default_rng(0)
    regions = (
        (np.array([0.26, 0.07]), np.array([0.34, 0.18])),
        (np.array([0.26, -0.18]), np.array([0.34, -0.07])),
    )
    pos_errs = []
    rot_errs = []
    successes = 0
    n = 200
    for i in range(n):
        low, high = regions[i % 2]
        xy = rng.uniform(low, high)
        yaw = float(np.arctan2(xy[1], xy[0]))
        target = np.array([xy[0], xy[1], 0.05])
        result = ik.solve(target, top_down_quat(yaw, 0), home)
        successes += int(result.success)
        pos_errs.append(result.pos_err)
        rot_errs.append(result.rot_err)
    pos_errs = np.asarray(pos_errs)
    rot_errs = np.asarray(rot_errs)
    print(
        f"ik samples {n} success {successes / n:.3f} "
        f"pos mean {pos_errs.mean():.4e} max {pos_errs.max():.4e} "
        f"rot mean {rot_errs.mean():.4e} max {rot_errs.max():.4e}"
    )
    assert successes == n
    assert pos_errs.max() <= float(cfg["ik"]["pos_tol"])
    assert rot_errs.max() <= float(cfg["ik"]["rot_tol"])
