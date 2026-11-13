"""RNEA, mass matrix, gravity and Coriolis checks against MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np

from aloha_pp.dyn.rnea import (
    CHRISTOFFEL_STEP,
    composite_inertia,
    coriolis_force,
    coriolis_matrix,
    gravity_torque,
    mass_matrix,
    mass_matrix_energy,
    rnea,
    skew_residual,
)
from aloha_pp.kin.fk import ArmChain
from aloha_pp.robot import robot_xml

SEED = 20261113
ATOL = 1e-8


def _load() -> tuple[mujoco.MjModel, mujoco.MjData, ArmChain]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml()))
    return model, mujoco.MjData(model), ArmChain(model, finger_q=0.037)


def _prepare(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chain: ArmChain,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> None:
    idx = chain.arm_dof
    data.qpos[:] = chain.pack_q(q)
    data.qvel[:] = 0.0
    data.qvel[idx] = qd
    data.qacc[:] = 0.0
    mujoco.mj_fwdPosition(model, data)
    data.qvel[:] = 0.0
    data.qvel[idx] = qd
    data.qacc[:] = 0.0
    data.qacc[idx] = qdd
    mujoco.mj_fwdVelocity(model, data)
    data.qacc[:] = 0.0
    data.qacc[idx] = qdd


def test_dynamics_match_mujoco_and_mass_matrix_properties() -> None:
    model, data, chain = _load()
    rng = np.random.default_rng(SEED)
    gravity = model.opt.gravity.copy()
    try:
        for opening in (0.037, 0.015):
            chain.set_finger_q(opening)
            for _ in range(10):
                q = rng.uniform(chain.arm_range[:, 0], chain.arm_range[:, 1])
                qd = rng.uniform(-2.0, 2.0, size=6)
                qdd = rng.uniform(-10.0, 10.0, size=6)
                _prepare(model, data, chain, q, qd, qdd)
                dense = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, data, dense)
                block = dense[np.ix_(chain.arm_dof, chain.arm_dof)]
                got_m = mass_matrix(chain, q)
                energy = mass_matrix_energy(chain, q)
                np.testing.assert_allclose(got_m, block, atol=ATOL, rtol=0.0)
                np.testing.assert_allclose(energy, got_m, atol=ATOL, rtol=0.0)
                np.testing.assert_allclose(got_m, got_m.T, atol=1e-10, rtol=0.0)
                assert float(np.linalg.eigvalsh(got_m).min()) > 1e-8

                rne_tau = np.zeros(model.nv)
                mujoco.mj_rne(model, data, 1, rne_tau)
                np.testing.assert_allclose(
                    rnea(chain, q, qd, qdd, armature=False),
                    rne_tau[chain.arm_dof],
                    atol=ATOL,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    rnea(chain, q, qd, qdd, armature=True),
                    rne_tau[chain.arm_dof] + chain.arm_armature * qdd,
                    atol=ATOL,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    data.qfrc_passive[chain.arm_dof],
                    -chain.arm_damping * qd,
                    atol=ATOL,
                    rtol=0.0,
                )

                data.qvel[:] = 0.0
                mujoco.mj_fwdVelocity(model, data)
                np.testing.assert_allclose(
                    gravity_torque(chain, q),
                    data.qfrc_bias[chain.arm_dof],
                    atol=ATOL,
                    rtol=0.0,
                )

                model.opt.gravity[:] = 0.0
                data.qvel[:] = 0.0
                data.qvel[chain.arm_dof] = qd
                mujoco.mj_fwdVelocity(model, data)
                model.opt.gravity[:] = gravity
                np.testing.assert_allclose(
                    coriolis_force(chain, q, qd),
                    data.qfrc_bias[chain.arm_dof],
                    atol=ATOL,
                    rtol=0.0,
                )
    finally:
        model.opt.gravity[:] = gravity


def test_christoffel_matches_rnea_and_is_skew() -> None:
    model, _, chain = _load()
    rng = np.random.default_rng(SEED + 1)
    for _ in range(5):
        q = rng.uniform(chain.arm_range[:, 0], chain.arm_range[:, 1])
        qd = rng.uniform(-2.0, 2.0, size=6)
        coriolis, mdot = coriolis_matrix(chain, q, qd, step=CHRISTOFFEL_STEP)
        np.testing.assert_allclose(coriolis @ qd, coriolis_force(chain, q, qd), atol=ATOL, rtol=0.0)
        assert np.max(np.abs(skew_residual(mdot, coriolis))) < ATOL


def test_composite_inertia_is_pose_invariant_and_merges_fingers() -> None:
    chain = ArmChain(_load()[0], finger_q=0.037)
    q = np.array([0.3, -0.4, 0.6, -0.2, 0.5, 0.1])
    at_zero = composite_inertia(chain, np.zeros(6))
    at_q = composite_inertia(chain, q)
    gripper = None
    for row_zero, row_q in zip(at_zero, at_q):
        np.testing.assert_allclose(row_zero["com"], row_q["com"], atol=1e-12)
        np.testing.assert_allclose(row_zero["inertia"], row_q["inertia"], atol=1e-12)
        if len(row_zero["bodies"]) == 1:
            body = chain._body_id(str(row_zero["link"]))
            np.testing.assert_allclose(row_zero["com"], chain.body_ipos[body], atol=1e-12)
            np.testing.assert_allclose(row_zero["inertia"], chain.I_com_body[body], atol=1e-12)
        else:
            gripper = row_zero
    assert gripper is not None
    assert "left/left_finger_link" in gripper["bodies"]
    assert "left/right_finger_link" in gripper["bodies"]
    assert "left/gripper_base" in gripper["bodies"]
    chain.set_finger_q(0.015)
    opened = {row["link"]: row for row in composite_inertia(chain)}
    # Finger travel changes the gripper inertia and leaves the other links alone.
    assert abs(float(opened["left/gripper_link"]["mass"]) - float(gripper["mass"])) < 1e-12
    assert np.linalg.norm(opened["left/gripper_link"]["inertia"] - gripper["inertia"]) > 1e-6
    assert np.linalg.norm(opened["left/shoulder_link"]["inertia"] - at_zero[0]["inertia"]) < 1e-12
