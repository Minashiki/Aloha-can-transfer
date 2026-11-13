"""Damped least-squares inverse kinematics from ``mj_jacSite``."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aloha_pp.robot import ARM_JOINTS, BASE_BODY


@dataclass(frozen=True)
class IkResult:
    success: bool
    q: np.ndarray
    pos_err: float
    rot_err: float
    iterations: int


class DlsIk:
    """Iterative DLS IK in the robot base frame.

    The target position and quaternion (wxyz) are expressed in ``base_body``.
    Iteration starts from ``q_init`` and the result is clipped to the joint range.
    Only the six arm joints are solved; the gripper is left unchanged.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        *,
        damping: float = 0.05,
        max_iters: int = 80,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        rot_weight: float = 0.4,
        max_dq: float = 0.8,
        base_body: str = BASE_BODY,
    ) -> None:
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if max_iters < 1:
            raise ValueError("max_iters must be positive")
        self.model = model
        self.damping = float(damping)
        self.max_iters = int(max_iters)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self.rot_weight = float(rot_weight)
        self.max_dq = float(max_dq)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if self.site_id < 0:
            raise ValueError(f"site not found: {site_name}")
        if self.base_id < 0:
            raise ValueError(f"body not found: {base_body}")
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            missing = [name for name, jid in zip(ARM_JOINTS, self.joint_ids) if jid < 0]
            raise ValueError(f"joints not found: {missing}")
        self.qpos_adr = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=int).copy()
        self.dof_adr = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=int).copy()
        self.q_lo = model.jnt_range[self.joint_ids, 0].copy()
        self.q_hi = model.jnt_range[self.joint_ids, 1].copy()
        self._data = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._eye = np.eye(6)

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.q_lo, self.q_hi)

    def arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def site_pose_in_base(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Current TCP position and quaternion (wxyz) in the base frame.

        ``data`` must already have kinematics computed. This does not write qpos.
        """
        return self._site_pose_base(data)

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray,
    ) -> IkResult:
        q = self.clip(q_init)
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = np.asarray(target_quat, dtype=float)
        target_quat = target_quat / np.linalg.norm(target_quat)
        pos_world, quat_world = self._target_in_world(target_pos, target_quat, q)

        pos_err = np.inf
        rot_err = np.inf
        iterations = 0
        for iterations in range(1, self.max_iters + 1):
            err, jac, pos_err, rot_err = self._error_and_jac(q, pos_world, quat_world)
            if pos_err <= self.pos_tol and rot_err <= self.rot_tol:
                return IkResult(True, q, float(pos_err), float(rot_err), iterations)
            dq = self._dls_step(jac, err)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > self.max_dq:
                dq *= self.max_dq / step_norm
            improved = False
            step = 1.0
            base_cost = pos_err + rot_err
            for _ in range(8):
                q_try = self.clip(q + step * dq)
                _, _, pos_try, rot_try = self._error_and_jac(q_try, pos_world, quat_world)
                if pos_try + rot_try < base_cost:
                    q = q_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                q = self.clip(q + 0.05 * dq)
        err, _, pos_err, rot_err = self._error_and_jac(q, pos_world, quat_world)
        success = bool(pos_err <= self.pos_tol and rot_err <= self.rot_tol)
        return IkResult(success, q, float(pos_err), float(rot_err), iterations)

    def _write_q(self, q: np.ndarray, data: mujoco.MjData | None = None) -> mujoco.MjData:
        data = self._data if data is None else data
        data.qpos[self.qpos_adr] = q
        data.qvel[:] = 0.0
        # mj_jacSite reads the motion dofs that mj_comPos maps into the world frame.
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        return data

    def _target_in_world(
        self, pos_base: np.ndarray, quat_base: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self._write_q(q)
        base_pos = data.xpos[self.base_id].copy()
        base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(base_quat, data.xmat[self.base_id])
        base_mat = data.xmat[self.base_id].reshape(3, 3).copy()
        pos_world = base_pos + base_mat @ pos_base
        quat_world = np.zeros(4)
        mujoco.mju_mulQuat(quat_world, base_quat, quat_base)
        return pos_world, quat_world

    def _site_pose_base(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        base_pos = data.xpos[self.base_id]
        base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(base_quat, data.xmat[self.base_id])
        base_mat = data.xmat[self.base_id].reshape(3, 3)
        pos = base_mat.T @ (data.site_xpos[self.site_id] - base_pos)
        quat_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_inv, base_quat)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, quat_inv, self._site_quat(data))
        return pos, quat

    def _site_quat(self, data: mujoco.MjData) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        return quat

    def _error_and_jac(
        self, q: np.ndarray, pos_world: np.ndarray, quat_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        data = self._write_q(q)
        pos_err_v = pos_world - data.site_xpos[self.site_id]
        # mju_subQuat returns the angular velocity in the current site frame.
        # mj_jacSite's rotational rows are in the world frame.
        rot_body = np.zeros(3)
        mujoco.mju_subQuat(rot_body, quat_world, self._site_quat(data))
        rot_err_v = data.site_xmat[self.site_id].reshape(3, 3) @ rot_body
        self._jacp.fill(0.0)
        self._jacr.fill(0.0)
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        jac = np.vstack(
            (
                self._jacp[:, self.dof_adr],
                self.rot_weight * self._jacr[:, self.dof_adr],
            )
        )
        err = np.concatenate((pos_err_v, self.rot_weight * rot_err_v))
        return err, jac, float(np.linalg.norm(pos_err_v)), float(np.linalg.norm(rot_err_v))

    def _dls_step(self, jac: np.ndarray, err: np.ndarray) -> np.ndarray:
        a = jac @ jac.T + (self.damping ** 2) * self._eye
        return jac.T @ np.linalg.solve(a, err)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def top_down_quat(yaw: float, approach_axis: int = 0) -> np.ndarray:
    """Site quaternion whose ``approach_axis`` points down, spun by ``yaw``.

    ``approach_axis`` is 0, 1, or 2 for the site x, y, or z axis.
    ``yaw`` rotates the gripper about the world vertical.
    """
    approach = np.array([0.0, 0.0, -1.0])
    # Opening / secondary axis stays horizontal.
    secondary = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    third = np.cross(approach, secondary)
    columns = [None, None, None]
    columns[approach_axis] = approach
    rest = [i for i in range(3) if i != approach_axis]
    columns[rest[0]] = secondary
    columns[rest[1]] = third
    mat = np.column_stack(columns)
    # Keep a right-handed frame.
    if np.linalg.det(mat) < 0.0:
        columns[rest[1]] = -third
        mat = np.column_stack(columns)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat
