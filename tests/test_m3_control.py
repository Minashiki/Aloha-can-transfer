"""Torque laws, actuator limits, and the per-step state contract."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from aloha_pp.control.arm import arm_index, dense_mass
from aloha_pp.control.laws import TorqueLaw
from aloha_pp.control.refresh import refresh_state
from aloha_pp.control.tracking import TrackingController
from aloha_pp.robot import repo_root, scene_xml
from aloha_pp.sim.env import AlohaCanEnv, load_config


def _m3() -> dict:
    path = repo_root() / "configs" / "m3_track.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _motor_home():
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
    data = mujoco.MjData(model)
    arm = arm_index(model)
    data.qpos[arm.qpos_adr] = np.asarray(cfg["home_q"], dtype=float)
    data.qvel[:] = 0.0
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/left_finger")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/right_finger")
    opening = float(cfg["gripper_open"])
    data.qpos[int(model.jnt_qposadr[left])] = opening
    data.qpos[int(model.jnt_qposadr[right])] = opening
    return cfg, model, data, arm


def _controller(model, kind: str) -> TrackingController:
    cfg = load_config()
    m3 = _m3()
    return TrackingController(
        model,
        kind=kind,
        site_name=cfg["tcp_site"],
        v_max=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
        control_dt=0.05,
        n_substeps=25,
        gripper_open=float(cfg["gripper_open"]),
        ik_damping=0.05,
        ik_max_iters=40,
        ik_pos_tol=1e-3,
        ik_rot_tol=1e-2,
        ik_rot_weight=0.4,
        kp=np.asarray(m3["kp"], dtype=float),
        kd=np.asarray(m3["kd"], dtype=float),
        ki=np.asarray(m3["ki"], dtype=float),
        tt=0.05,
    )


def test_gains_follow_mass_diagonal_at_home() -> None:
    _cfg, model, data, arm = _motor_home()
    refresh_state(model, data)
    dense = np.zeros((model.nv, model.nv))
    mass = dense_mass(model, data, arm.dof_adr, dense)
    m_diag = np.diag(mass)
    m3 = _m3()
    omega = float(m3["omega"])
    zeta = float(m3["zeta"])
    np.testing.assert_allclose(m_diag, np.asarray(m3["m_diag"]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m3["kp"]), m_diag * omega**2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m3["kd"]), 2.0 * zeta * m_diag * omega, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m3["ki"]), np.asarray(m3["kp"]) * omega * float(m3["ki_factor"]), rtol=0, atol=1e-4)


def test_c3_at_rest_matches_bias_and_pd_sign_and_clip() -> None:
    _cfg, model, data, arm = _motor_home()
    c3 = _controller(model, "C3")
    c3.reset(model, data)
    c3.compute(model, data)
    np.testing.assert_allclose(c3.sample.tau_cmd, data.qfrc_bias[arm.dof_adr], atol=1e-8)
    assert not np.any(c3.sample.saturated)

    c1 = _controller(model, "C1")
    c1.reset(model, data)
    # Unit positive error on every joint, held as a constant reference (no segment).
    # Small positive error: PD torque has the same sign and is not yet clipped
    # on the wrist, whose Kp is below the force limit.
    c1.stream.q = data.qpos[arm.qpos_adr] + 0.01
    c1.stream.v[:] = 0.0
    c1.compute(model, data)
    assert np.all(c1.sample.tau_cmd > 0.0)
    np.testing.assert_allclose(c1.sample.tau_cmd, np.asarray(_m3()["kp"]) * 0.01, atol=1e-8)

    # Large error: every joint hits actuatorfrcrange, sign still follows the error.
    c1.stream.q = data.qpos[arm.qpos_adr] + 20.0
    c1.compute(model, data)
    assert np.all(c1.sample.tau_cmd > 0.0)
    np.testing.assert_allclose(data.ctrl[arm.act_ids], arm.force_hi)
    assert np.all(c1.sample.saturated)
    c1.stream.q = data.qpos[arm.qpos_adr] - 20.0
    c1.compute(model, data)
    assert np.all(c1.sample.tau_cmd < 0.0)
    np.testing.assert_allclose(data.ctrl[arm.act_ids], arm.force_lo)


def test_c4_conditional_integration_stops_at_saturation() -> None:
    _cfg, model, _data, arm = _motor_home()
    m3 = _m3()
    law = TorqueLaw(
        model,
        arm.dof_adr,
        kind="C4",
        kp=np.asarray(m3["kp"], dtype=float),
        kd=np.asarray(m3["kd"], dtype=float),
        ki=np.asarray(m3["ki"], dtype=float),
        damping=arm.damping,
        force_lo=arm.force_lo,
        force_hi=arm.force_hi,
        tt=0.05,
        dt=0.002,
    )
    error = np.zeros(6)
    error[1] = 0.05  # shoulder: Kp*e is inside the 20 Nm rail, so the integral is what saturates
    pd = law.kp * error
    assert float(pd[1]) < float(arm.force_hi[1])
    steps = int(2.0 / law.dt)
    for _ in range(steps):
        law._integrate(pd, error)
    frozen = law.integral.copy()
    for _ in range(steps):
        law._integrate(pd, error)
    np.testing.assert_allclose(law.integral, frozen, atol=1e-9)
    assert np.all(np.abs(law.ki * law.integral) <= arm.force_hi + 1e-6)
    naive = error[1] * 2.0
    assert law.integral[1] < naive * 0.5

    # Reversing the error is allowed to integrate, so the wound-up state comes back down.
    error[1] = -0.05
    pd = law.kp * error
    before = float(law.integral[1])
    for _ in range(steps):
        law._integrate(pd, error)
    assert law.integral[1] < before


def test_c2_absolute_damping_drops_the_velocity_feedforward() -> None:
    _cfg, model, data, arm = _motor_home()
    m3 = _m3()
    law = TorqueLaw(
        model,
        arm.dof_adr,
        kind="C2",
        kp=np.asarray(m3["menagerie_kp"], dtype=float),
        kd=np.asarray(m3["menagerie_kv"], dtype=float),
        ki=np.zeros(6),
        damping=arm.damping,
        force_lo=arm.force_lo,
        force_hi=arm.force_hi,
        tt=0.05,
        dt=0.002,
        vel_feedback="absolute",
    )
    refresh_state(model, data)
    q = data.qpos[arm.qpos_adr].copy()
    qd = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    data.qvel[arm.dof_adr] = qd
    q_d = q.copy()
    qd_d = np.ones(6)
    tau = law(data, q, qd, q_d, qd_d, np.zeros(6))
    gravity = law.gravity(data)
    expected = -np.asarray(m3["menagerie_kv"]) * qd + gravity
    np.testing.assert_allclose(tau, expected, atol=1e-8)
    # The error-velocity law would have added kv * qd_d on the shoulder.
    assert abs(float(tau[1] - expected[1])) < 1e-8
    assert abs(float(m3["menagerie_kv"][1] * qd_d[1])) > 1.0


def test_compute_refreshes_bias_to_the_current_state() -> None:
    _cfg, model, data, arm = _motor_home()
    controller = _controller(model, "C3")
    controller.reset(model, data)
    data.qvel[arm.dof_adr] = np.array([0.3, -0.2, 0.15, 0.0, 0.1, -0.05])
    refresh_state(model, data)
    stale = data.qfrc_bias.copy()
    data.qvel[arm.dof_adr] = data.qvel[arm.dof_adr] + 0.25
    # No forward yet: qfrc_bias still describes the previous velocity.
    np.testing.assert_allclose(data.qfrc_bias, stale)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    data.qfrc_bias[:] = 99.0
    controller.stream.q = qpos[arm.qpos_adr]
    controller.stream.v[:] = 0.0
    controller.compute(model, data)
    fresh = mujoco.MjData(model)
    fresh.qpos[:] = qpos
    fresh.qvel[:] = qvel
    mujoco.mj_forward(model, fresh)
    np.testing.assert_allclose(controller.sample.used_bias, fresh.qfrc_bias[arm.dof_adr], atol=1e-9)
    np.testing.assert_allclose(controller.sample.used_q, qpos[arm.qpos_adr], atol=1e-12)
    np.testing.assert_allclose(controller.sample.used_qd, qvel[arm.dof_adr], atol=1e-12)
    dense = np.zeros((model.nv, model.nv))
    expected = dense_mass(model, fresh, arm.dof_adr, dense)
    np.testing.assert_allclose(controller.law.used_mass, expected, atol=1e-8)
    assert not np.allclose(controller.sample.used_bias, np.full(6, 99.0))


def test_env_steps_substep_controller_and_leaves_position_ik_alone() -> None:
    env = AlohaCanEnv()
    try:
        calls = {"compute": 0}

        def compute(*_args, **_kwargs):
            calls["compute"] += 1

        env.controller.compute = compute
        env.reset(seed=0, render=False)
        pos, quat = env.tcp_pose()
        action = np.concatenate((pos, quat, [float(env.cfg["gripper_open"])]))
        env.step(action, render=False)
        assert calls["compute"] == 0

        class Counter:
            def __init__(self) -> None:
                self.targets = 0
                self.computes = 0
                self.applied = 0

            def reset(self, model, data) -> None:
                del model, data

            def set_target(self, model, data, action) -> None:
                del model, data, action
                self.targets += 1

            def compute(self, model, data) -> None:
                del model, data
                self.computes += 1

            def read_applied(self, model, data) -> None:
                del model, data
                self.applied += 1

        counter = Counter()
        env.controller = counter
        env.reset(seed=0, render=False)
        assert counter.targets == 0
        assert counter.computes == 50
        counter.computes = 0
        env.step(action, render=False)
        assert counter.targets == 1
        assert counter.computes == env.n_substeps
        assert counter.applied == env.n_substeps
    finally:
        env.close()


def test_each_physics_step_uses_that_steps_state() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["actuator"] = "motor"
    cfg["gravity_compensation"] = False
    env = AlohaCanEnv(cfg)
    try:
        controller = _controller(env.model, "C3")
        env.controller = controller
        env.reset(seed=0, mode="fixed_1", render=False)
        pos, quat = env.tcp_pose()
        # A nearby target so the arm actually moves and the pre/post states differ.
        # Lookahead spends the first command filling its one-tick buffer, so the
        # motion, and the changing bias, are on the second command.
        action = np.concatenate((pos + np.array([0.02, -0.01, 0.01]), quat, [float(cfg["gripper_open"])]))
        env.step(action, render=False)
        snapshots = []

        def on_substep(current: AlohaCanEnv) -> None:
            del current
            # debug_* is the state compute() refreshed. The live data here is
            # already one mj_step later, so it must not be used as the reference.
            snapshots.append(
                (controller.debug_qpos.copy(), controller.debug_qvel.copy(), controller.sample.used_bias.copy())
            )

        env.step(action, on_substep=on_substep, render=False)
        assert len(snapshots) == env.n_substeps
        biases = np.stack([item[2] for item in snapshots])
        assert np.max(np.abs(np.diff(biases, axis=0))) > 1e-6
        fresh = mujoco.MjData(env.model)
        arm = controller.arm
        for qpos, qvel, used_bias in snapshots:
            fresh.qpos[:] = qpos
            fresh.qvel[:] = qvel
            mujoco.mj_forward(env.model, fresh)
            np.testing.assert_allclose(used_bias, fresh.qfrc_bias[arm.dof_adr], atol=1e-9)
    finally:
        env.close()
