"""Per-physics-step tracking controller shared by B0, B1, and C1–C4.

``set_target`` runs one DLS solve per 20 Hz Cartesian command and opens a
quintic segment. ``compute`` runs once per physics step: it refreshes MuJoCo
so ``qfrc_bias`` and ``M`` match the current ``qpos`` / ``qvel``, samples the
segment, and writes either a position setpoint (B0, B1) or a clipped torque
(C1–C4). The gripper stays on its position actuator.

``read_applied`` must run after ``mj_step`` and before the next ``mj_forward``.
``mj_step`` leaves ``qfrc_actuator`` equal to the force it just applied;
``mj_forward`` replaces that field with the force of the new state.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aloha_pp.control.arm import ArmIndex, arm_index
from aloha_pp.control.laws import TorqueLaw
from aloha_pp.control.quintic import SCHEMES, JointStream
from aloha_pp.control.refresh import refresh_state
from aloha_pp.kin.ik import DlsIk, IkResult
from aloha_pp.robot import ARM_JOINTS

POSITION_KINDS = ("B0", "B1")
TORQUE_KINDS = ("C1", "C2", "C3", "C4")


@dataclass
class StepSample:
    q_d: np.ndarray
    qd_d: np.ndarray
    qdd_d: np.ndarray
    tau_cmd: np.ndarray
    tau_applied: np.ndarray
    saturated: np.ndarray
    vel_exceeded: np.ndarray
    integral: np.ndarray
    ik_pos_err: float
    ik_rot_err: float
    ik_success: bool
    tcp_qd_pos: np.ndarray
    tcp_qd_quat: np.ndarray
    used_q: np.ndarray
    used_qd: np.ndarray
    used_bias: np.ndarray


class TrackingController:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        kind: str,
        site_name: str,
        v_max: np.ndarray,
        control_dt: float,
        n_substeps: int,
        gripper_open: float,
        ik_damping: float,
        ik_max_iters: int,
        ik_pos_tol: float,
        ik_rot_tol: float,
        ik_rot_weight: float,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        ki: np.ndarray | None = None,
        tt: float = 0.05,
        scheme: str = "lookahead",
        a_max: np.ndarray | None = None,
        vel_feedback: str = "error",
    ) -> None:
        if kind not in POSITION_KINDS + TORQUE_KINDS:
            raise ValueError(f"unknown controller {kind}")
        if scheme not in SCHEMES:
            raise ValueError(f"unknown reference scheme {scheme}")
        self.kind = kind
        self.scheme = scheme
        self.vel_feedback = vel_feedback
        self.model = model
        self.arm = arm_index(model)
        self.position_mode = kind in POSITION_KINDS
        self.gripper_open = float(gripper_open)
        self.dt = float(control_dt) / int(n_substeps)
        self.ik = DlsIk(
            model,
            site_name,
            damping=ik_damping,
            max_iters=ik_max_iters,
            pos_tol=ik_pos_tol,
            rot_tol=ik_rot_tol,
            rot_weight=ik_rot_weight,
        )
        self.stream = JointStream(
            np.zeros(6),
            control_dt=control_dt,
            n_substeps=n_substeps,
            v_max=np.asarray(v_max, dtype=float),
            scheme=scheme,
            a_max=None if a_max is None else np.asarray(a_max, dtype=float),
        )
        self.home_q = np.zeros(6)
        self.home_pos = np.zeros(3)
        self.home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._seed = np.zeros(6)
        self.last_result: IkResult | None = None
        self.sample: StepSample | None = None
        self._fk = mujoco.MjData(model)
        self._ik_pos = 0.0
        self._ik_rot = 0.0
        self._ik_ok = True
        zeros = np.zeros(6)
        if self.position_mode:
            self.kp_pos = np.array([model.actuator_gainprm[act, 0] for act in self.arm.act_ids])
            self.kv_pos = np.array([-model.actuator_biasprm[act, 2] for act in self.arm.act_ids])
            self.law = None
        else:
            if kp is None or kd is None or ki is None:
                raise ValueError("torque controllers need kp, kd, and ki")
            self.kp_pos = None
            self.kv_pos = None
            self.law = TorqueLaw(
                model,
                self.arm.dof_adr,
                kind=kind,
                kp=np.asarray(kp, dtype=float),
                kd=np.asarray(kd, dtype=float),
                ki=np.asarray(ki if kind == "C4" else zeros, dtype=float),
                damping=self.arm.damping,
                force_lo=self.arm.force_lo,
                force_hi=self.arm.force_hi,
                tt=tt,
                dt=self.dt,
                vel_feedback=vel_feedback,
            )
            # C4 needs a nonzero Ki even though the constructor above replaced it
            # for other kinds. Set C4's Ki explicitly.
            if kind == "C4":
                self.law.ki = np.asarray(ki, dtype=float).copy()

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        self.home_q = data.qpos[self.arm.qpos_adr].copy()
        self.stream.reset(self.home_q)
        self._seed = self.home_q.copy()
        self.home_pos, self.home_quat = self._tcp_of(self.home_q)
        self.last_result = None
        self.sample = None
        self._ik_pos = 0.0
        self._ik_rot = 0.0
        self._ik_ok = True
        if self.law is not None:
            self.law.reset()

    def set_target(self, model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
        del model, data
        action = np.asarray(action, dtype=float).reshape(8)
        quat = action[3:7]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError("action quaternion has zero length")
        quat = quat / norm
        pos = action[:3]
        result = self._solve(pos, quat)
        q_cmd = result.q if result.success else self._seed
        self._seed = np.asarray(q_cmd, dtype=float).copy()
        self.stream.push(self._seed)
        self.last_result = result
        self._ik_pos = float(result.pos_err)
        self._ik_rot = float(result.rot_err)
        self._ik_ok = bool(result.success)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Torque or position command from the current, refreshed state."""
        refresh_state(model, data)
        q_d, qd_d, qdd_d, vel_flag = self.stream.advance()
        q = data.qpos[self.arm.qpos_adr].copy()
        qd = data.qvel[self.arm.dof_adr].copy()
        if self.position_mode:
            data.ctrl[self.arm.act_ids] = q_d
            tau_cmd = self.kp_pos * (q_d - q) - self.kv_pos * qd + data.qfrc_gravcomp[self.arm.dof_adr]
            integral = np.zeros(6)
        else:
            assert self.law is not None
            tau_cmd = self.law(data, q, qd, q_d, qd_d, qdd_d)
            data.ctrl[self.arm.act_ids] = np.clip(tau_cmd, self.arm.force_lo, self.arm.force_hi)
            integral = self.law.integral.copy()
        data.ctrl[self.arm.grip_act] = self.gripper_open
        saturated = (tau_cmd < self.arm.force_lo - 1e-8) | (tau_cmd > self.arm.force_hi + 1e-8)
        tcp_pos, tcp_quat = self._tcp_of(q_d)
        bias = data.qfrc_bias[self.arm.dof_adr].copy()
        self.debug_qpos = data.qpos.copy()
        self.debug_qvel = data.qvel.copy()
        self.sample = StepSample(
            q_d=q_d,
            qd_d=qd_d,
            qdd_d=qdd_d,
            tau_cmd=np.asarray(tau_cmd, dtype=float).copy(),
            tau_applied=np.full(6, np.nan),
            saturated=saturated,
            vel_exceeded=np.asarray(vel_flag, dtype=bool).copy(),
            integral=integral,
            ik_pos_err=self._ik_pos,
            ik_rot_err=self._ik_rot,
            ik_success=self._ik_ok,
            tcp_qd_pos=tcp_pos,
            tcp_qd_quat=tcp_quat,
            used_q=q,
            used_qd=qd,
            used_bias=bias,
        )

    def read_applied(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Copy the joint actuator force produced by the ``mj_step`` just taken."""
        del model
        if self.sample is None:
            raise RuntimeError("read_applied called before compute")
        self.sample.tau_applied = data.qfrc_actuator[self.arm.dof_adr].copy()

    def _solve(self, pos: np.ndarray, quat: np.ndarray) -> IkResult:
        if _same_home(pos, quat, self.home_pos, self.home_quat):
            return IkResult(True, self.home_q.copy(), 0.0, 0.0, 0)
        return self.ik.solve(pos, quat, self._seed)

    def _tcp_of(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._fk.qpos[self.arm.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self._fk)
        return self.ik.site_pose_in_base(self._fk)


def _same_home(pos: np.ndarray, quat: np.ndarray, home_pos: np.ndarray, home_quat: np.ndarray) -> bool:
    if float(np.linalg.norm(pos - home_pos)) > 1e-10:
        return False
    return abs(abs(float(np.dot(quat, home_quat))) - 1.0) < 1e-10


def velocity_limits(cfg: dict) -> np.ndarray:
    table = cfg["velocity_limit"]
    return np.array([float(table[name]) for name in ARM_JOINTS], dtype=float)
