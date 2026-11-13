"""FK, Jacobian and wrist-axis checks against MuJoCo."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from aloha_pp.kin.fk import TCP_SITE, ArmChain
from aloha_pp.kin.spatial import rotation_angle
from aloha_pp.robot import robot_xml

SEED = 20261113
ATOL = 1e-8


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(robot_xml()))


def _sync(model: mujoco.MjModel, data: mujoco.MjData, chain: ArmChain, q: np.ndarray) -> None:
    data.qpos[:] = chain.pack_q(q)
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def test_fk_and_jacobian_match_mujoco() -> None:
    model = _model()
    data = mujoco.MjData(model)
    chain = ArmChain(model, finger_q=0.037)
    rng = np.random.default_rng(SEED)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for opening in (0.037, 0.015):
        chain.set_finger_q(opening)
        for _ in range(10):
            q = rng.uniform(chain.arm_range[:, 0], chain.arm_range[:, 1])
            _sync(model, data, chain, q)
            world, in_base = chain.poses(q)
            state = chain.forward(q)
            for body in range(model.nbody):
                assert np.linalg.norm(state.xpos[body] - data.xpos[body]) < ATOL
                assert rotation_angle(state.xmat[body], data.xmat[body].reshape(3, 3)) < ATOL
            base_rot = data.xmat[chain.base_id].reshape(3, 3)
            base_pos = data.xpos[chain.base_id]
            for name, body in chain.link_ids:
                got = in_base[name]
                ref_pos = base_rot.T @ (data.xpos[body] - base_pos)
                ref_rot = base_rot.T @ data.xmat[body].reshape(3, 3)
                assert np.linalg.norm(got.pos - ref_pos) < ATOL
                assert rotation_angle(got.rot, ref_rot) < ATOL
            tcp_pos = base_rot.T @ (data.site_xpos[site] - base_pos)
            tcp_rot = base_rot.T @ data.site_xmat[site].reshape(3, 3)
            assert np.linalg.norm(in_base[TCP_SITE].pos - tcp_pos) < ATOL
            assert rotation_angle(in_base[TCP_SITE].rot, tcp_rot) < ATOL
            assert np.linalg.norm(world[TCP_SITE].pos - data.site_xpos[site]) < ATOL
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            jac = chain.tcp_jacobian(q, state)
            ref = np.vstack((jacp[:, chain.arm_dof], jacr[:, chain.arm_dof]))
            np.testing.assert_allclose(jac, ref, atol=ATOL, rtol=0.0)


def test_wrist_axes_are_collinear_only_at_zero_wrist_angle() -> None:
    chain = ArmChain(_model(), finger_q=0.037)
    q = np.array([0.4, -0.5, 0.8, 1.2, 0.0, -0.7])
    aligned = chain.wrist_alignment(q)
    assert aligned["cross_norm"] < 1e-12
    assert aligned["line_distance"] < 1e-12
    q[3] = -1.5
    still = chain.wrist_alignment(q)
    assert still["cross_norm"] < 1e-12
    q[4] = 0.4
    tilted = chain.wrist_alignment(q)
    assert tilted["cross_norm"] > 0.2
    span = np.linspace(chain.arm_range[4, 0], chain.arm_range[4, 1], 81)
    cross = []
    for angle in span:
        q[4] = angle
        cross.append(chain.wrist_alignment(q)["cross_norm"])
    step = float(span[1] - span[0])
    assert abs(float(span[int(np.argmin(cross))])) <= 0.5 * step + 1e-9


def test_library_does_not_call_mujoco_solvers() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "mj_kinematics",
        "mj_forward",
        "mj_rne",
        "mj_fullM",
        "mj_jac",
        "mj_fwd",
        "mj_inverse",
        "mj_comPos",
        "mj_comVel",
        "mj_crb",
        "mj_makeM",
        "mj_step",
        "mju_",
    )
    paths = [
        root / "src/aloha_pp/kin/spatial.py",
        root / "src/aloha_pp/kin/fk.py",
        root / "src/aloha_pp/dyn/rnea.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} contains {name}"
    assert "import mujoco" not in (root / "src/aloha_pp/kin/spatial.py").read_text(encoding="utf-8")
    assert "import mujoco" not in (root / "src/aloha_pp/dyn/rnea.py").read_text(encoding="utf-8")
