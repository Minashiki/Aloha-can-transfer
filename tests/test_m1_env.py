"""Observation shapes for reset and step. Privileged state stays out of the policy obs."""

from __future__ import annotations

import mujoco
import numpy as np

from aloha_pp.kin.ik import quat_to_mat
from aloha_pp.sim.env import AlohaCanEnv
from aloha_pp.sim.success import in_target


def _check_obs(obs: dict) -> None:
    assert obs["agentview"].shape == (96, 96, 3)
    assert obs["agentview"].dtype == np.uint8
    assert obs["wrist_cam_left"].shape == (96, 96, 3)
    assert obs["wrist_cam_left"].dtype == np.uint8
    assert obs["tcp_pose"].shape == (7,)
    assert obs["gripper_opening"].shape == (1,)
    assert obs["q"].shape == (6,)
    assert obs["qvel"].shape == (6,)
    assert "can_pose" not in obs
    assert int(obs["agentview"].min()) != int(obs["agentview"].max())
    assert int(obs["wrist_cam_left"].min()) != int(obs["wrist_cam_left"].max())


def test_reset_and_step_observation_shapes() -> None:
    env = AlohaCanEnv()
    try:
        for mode in ("fixed_1", "fixed_2", "random"):
            obs = env.reset(seed=1, mode=mode)
            _check_obs(obs)
            privileged = env.get_privileged()
            assert privileged["can_pose"].shape == (7,)
            assert privileged["can_linvel"].shape == (3,)
            action = np.concatenate((obs["tcp_pose"], obs["gripper_opening"]))
            obs, success, info = env.step(action)
            _check_obs(obs)
            assert success is False
            assert info["ik_success"] is True
            assert info["ik_pos_err"] < 1e-3
            assert info["ik_rot_err"] < 1e-2
    finally:
        env.close()


def test_context_manager_closes_renderer() -> None:
    with AlohaCanEnv() as env:
        env.reset(seed=0)
        env.render("agentview")
        assert env._renderer is not None
    assert env._renderer is None


def test_target_cell_follows_mode() -> None:
    expected = {
        "fixed_1": np.array([0.336, -0.080, 0.03604]),
        "random": np.array([0.336, -0.080, 0.03604]),
        "fixed_2": np.array([0.264, -0.170, 0.03604]),
    }
    half = np.array([0.0324, 0.0414, 0.02])
    with AlohaCanEnv() as env:
        for mode, center in expected.items():
            env.reset(seed=0, mode=mode)
            np.testing.assert_allclose(env.tracker.center, center)
            np.testing.assert_allclose(env.tracker.half_extents, half)
        for mode in ("fixed_1", "fixed_2"):
            place = np.asarray(env.cfg["modes"][mode]["place_pos"], dtype=float)
            assert in_target(place, expected[mode], half)


def test_can_center_matches_collision_geom_at_two_orientations() -> None:
    with AlohaCanEnv() as env:
        env.reset(seed=0, mode="fixed_1")
        orientations = (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0]),
        )
        for quat in orientations:
            quat = quat / np.linalg.norm(quat)
            env.data.qpos[env.can_qpos_adr : env.can_qpos_adr + 3] = np.array([0.25, 0.10, 0.08])
            env.data.qpos[env.can_qpos_adr + 3 : env.can_qpos_adr + 7] = quat
            center_world = env._base_to_world(env.can_center_base())
            direct = env.data.geom_xpos[env.can_geom].copy()
            np.testing.assert_allclose(center_world, direct, atol=1e-12)
            assert np.linalg.norm(direct - env.data.xpos[env.can_body]) > 1e-4


def test_motion_obs_matches_independent_fk() -> None:
    with AlohaCanEnv() as env:
        obs = env.reset(seed=0, mode="fixed_1")
        q_reset = obs["q"].copy()
        quat = obs["tcp_pose"][3:7].copy()
        grip = float(obs["gripper_opening"][0])
        start = obs["tcp_pose"][:3].copy()
        for index in range(6):
            pos = start + np.array([0.015 * (index + 1), -0.012 * (index + 1), 0.025])
            action = np.concatenate((pos, quat, [grip]))
            obs, _success, _info = env.step(action)
        assert np.linalg.norm(obs["q"] - q_reset) > 1e-3

        fresh = mujoco.MjData(env.model)
        fresh.qpos[:] = env.data.qpos
        mujoco.mj_forward(env.model, fresh)
        site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, env.cfg["tcp_site"])
        base = env.base_id
        base_mat = fresh.xmat[base].reshape(3, 3)
        tcp = base_mat.T @ (fresh.site_xpos[site] - fresh.xpos[base])
        np.testing.assert_allclose(obs["tcp_pose"][:3], tcp, atol=1e-9)

        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, fresh.site_xmat[site])
        base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(base_quat, fresh.xmat[base])
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, base_quat)
        expected_quat = np.zeros(4)
        mujoco.mju_mulQuat(expected_quat, inv, site_quat)
        assert abs(abs(float(np.dot(expected_quat, obs["tcp_pose"][3:7]))) - 1.0) < 1e-9

        privileged = env.get_privileged()
        adr = env.can_qpos_adr
        body_pos = env.data.qpos[adr : adr + 3]
        body_quat = env.data.qpos[adr + 3 : adr + 7]
        world = body_pos + quat_to_mat(body_quat) @ env.model.geom_pos[env.can_geom]
        expected = base_mat.T @ (world - fresh.xpos[base])
        np.testing.assert_allclose(privileged["can_pose"][:3], expected, atol=1e-9)
        np.testing.assert_allclose(fresh.geom_xpos[env.can_geom], world, atol=1e-9)
