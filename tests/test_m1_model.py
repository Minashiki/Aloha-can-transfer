"""M1 model parameters for the motor and position scenes."""

from __future__ import annotations

import mujoco
import numpy as np

from aloha_pp.robot import ARM_JOINTS, scene_xml
from aloha_pp.sim.gravity import apply_gravity_compensation

# Expected joint parameters, written here so they stay independent of robot.py.
ARM_ORDER = ARM_JOINTS
TORQUE_LIMITS = {
    "left/waist": 10.0,
    "left/shoulder": 20.0,
    "left/elbow": 20.0,
    "left/forearm_roll": 10.0,
    "left/wrist_angle": 10.0,
    "left/wrist_rotate": 4.0,
}
DAMPING = {
    "left/waist": 1.0,
    "left/shoulder": 2.0,
    "left/elbow": 2.0,
    "left/forearm_roll": 0.8,
    "left/wrist_angle": 0.8,
    "left/wrist_rotate": 0.3,
}
KP = {
    "left/waist": 43.0,
    "left/shoulder": 265.0,
    "left/elbow": 227.0,
    "left/forearm_roll": 78.0,
    "left/wrist_angle": 37.0,
    "left/wrist_rotate": 10.4,
}
KV = {
    "left/waist": 5.76,
    "left/shoulder": 20.0,
    "left/elbow": 18.49,
    "left/forearm_roll": 6.78,
    "left/wrist_angle": 6.28,
    "left/wrist_rotate": 1.2,
}


def _load(version: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_xml(version)))


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _act_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def test_shared_physics_and_limits() -> None:
    for version in ("motor", "position"):
        model = _load(version)
        assert model.neq == 1
        assert model.opt.timestep == 0.002
        assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        for name in ARM_ORDER:
            joint_id = _joint_id(model, name)
            dof = int(model.jnt_dofadr[joint_id])
            assert model.dof_damping[dof] == DAMPING[name]
            assert model.jnt_actfrclimited[joint_id]
            np.testing.assert_allclose(
                model.jnt_actfrcrange[joint_id],
                [-TORQUE_LIMITS[name], TORQUE_LIMITS[name]],
            )
        # Menagerie armature / frictionloss kept on the two joints that had them.
        shoulder = _joint_id(model, "left/shoulder")
        elbow = _joint_id(model, "left/elbow")
        assert model.dof_armature[int(model.jnt_dofadr[shoulder])] == 0.395
        assert model.dof_frictionloss[int(model.jnt_dofadr[shoulder])] == 2.0
        assert model.dof_armature[int(model.jnt_dofadr[elbow])] == 0.383
        assert model.dof_frictionloss[int(model.jnt_dofadr[elbow])] == 1.15
        grip = _act_id(model, "left/gripper")
        assert model.actuator_gainprm[grip, 0] == 2000.0
        np.testing.assert_allclose(model.actuator_ctrlrange[grip], [0.002, 0.037])
        np.testing.assert_allclose(model.actuator_forcerange[grip], [-20.0, 20.0])
        assert model.actuator_forcelimited[grip]


def test_motor_actuators() -> None:
    model = _load("motor")
    for name in ARM_ORDER:
        act = _act_id(model, name)
        assert model.actuator_gear[act, 0] == 1.0
        np.testing.assert_allclose(
            model.actuator_ctrlrange[act],
            [-TORQUE_LIMITS[name], TORQUE_LIMITS[name]],
        )
        assert int(model.actuator_trntype[act]) != int(mujoco.mjtTrn.mjTRN_UNDEFINED)


def test_position_gains_and_gravcomp_switch() -> None:
    model = _load("position")
    data = mujoco.MjData(model)
    for name in ARM_ORDER:
        act = _act_id(model, name)
        assert model.actuator_gainprm[act, 0] == KP[name]
        assert model.actuator_biasprm[act, 2] == -KV[name]
    assert np.all(model.body_gravcomp == 0.0)
    assert not np.any(model.jnt_actgravcomp)
    apply_gravity_compensation(model, True)
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("left/"):
            assert model.body_gravcomp[body_id] == 1.0
    for name in ARM_ORDER:
        assert model.jnt_actgravcomp[_joint_id(model, name)]
    assert model.flg_gravcomp
    home = np.array([0.0, 0.06131763, 0.55376181, 0.0, 0.95550113, 0.0])
    for name, value in zip(ARM_ORDER, home):
        joint_id = _joint_id(model, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        data.ctrl[_act_id(model, name)] = value
    mujoco.mj_forward(model, data)
    bias = np.array(
        [data.qfrc_bias[int(model.jnt_dofadr[_joint_id(model, name)])] for name in ARM_ORDER]
    )
    applied = np.array(
        [data.qfrc_actuator[int(model.jnt_dofadr[_joint_id(model, name)])] for name in ARM_ORDER]
    )
    np.testing.assert_allclose(applied, bias, atol=1e-4)
    apply_gravity_compensation(model, False)
    assert np.all(model.body_gravcomp == 0.0)
    assert not np.any(model.jnt_actgravcomp)


def test_tcp_site_is_pad_midpoint_open_and_on_can() -> None:
    """Pad midpoint is the mean of each finger's three spheres, then their midpoint.

    The inner opening is the closest sphere-pair distance minus both radii.
    """
    model = _load("position")
    data = mujoco.MjData(model)
    lefts = ("left/left_g0", "left/left_g1", "left/left_g2")
    rights = ("left/right_g0", "left/right_g1", "left/right_g2")

    def geom_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    radius = float(model.geom_size[geom_id(lefts[0])][0])
    can_diameter = 2.0 * float(model.geom_size[geom_id("can_collision")][0])
    left_adr = int(model.jnt_qposadr[_joint_id(model, "left/left_finger")])
    right_adr = int(model.jnt_qposadr[_joint_id(model, "left/right_finger")])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/tcp")

    def gap_and_distance(opening: float) -> tuple[float, float]:
        data.qpos[left_adr] = opening
        data.qpos[right_adr] = opening
        mujoco.mj_forward(model, data)
        left = np.mean([data.geom_xpos[geom_id(name)] for name in lefts], axis=0)
        right = np.mean([data.geom_xpos[geom_id(name)] for name in rights], axis=0)
        midpoint = 0.5 * (left + right)
        closest = min(
            np.linalg.norm(data.geom_xpos[geom_id(a)] - data.geom_xpos[geom_id(b)])
            for a in lefts
            for b in rights
        )
        return float(closest - 2.0 * radius), float(np.linalg.norm(data.site_xpos[site] - midpoint))

    open_gap, open_distance = gap_and_distance(0.037)
    assert open_gap > can_diameter
    assert open_distance < 1e-3
    low, high = 0.002, 0.037
    for _ in range(40):
        middle = 0.5 * (low + high)
        gap, _distance = gap_and_distance(middle)
        if gap < can_diameter:
            low = middle
        else:
            high = middle
    closed_gap, closed_distance = gap_and_distance(0.5 * (low + high))
    assert abs(closed_gap - can_diameter) < 1e-4
    assert closed_distance < 1e-3
