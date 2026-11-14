"""Joint-space torque laws for C1–C4.

C1 is PD. C2 adds g(q), the bias MuJoCo reports at zero velocity. C3 adds the
arm mass matrix from ``mj_fullM``, the full velocity bias ``qfrc_bias``
(g + C qdot, damping excluded), and the joint-damping feedforward B qdot_d.
C4 is C2 plus an integral with conditional integration: once a joint's
unsaturated torque is outside ``actuatorfrcrange`` and the error points further
outward, that joint stops integrating. The integral torque is also clamped to
the force range, so a saturated contact cannot wind the state without bound.

Gains are diagonal. The same Kp and Kd are used for C1–C4; Ki is zero except
for C4. Menagerie comparison is the same C2 law with different numbers.
``vel_feedback="absolute"`` replaces ``Kd (qd_d - qd)`` with ``-Kd qd``, which
is the position actuator's damping term. That variant is only used for the
C2-versus-B1 check.
"""

from __future__ import annotations

import mujoco
import numpy as np

from aloha_pp.control.arm import dense_mass


class TorqueLaw:
    def __init__(
        self,
        model: mujoco.MjModel,
        dof_adr: np.ndarray,
        *,
        kind: str,
        kp: np.ndarray,
        kd: np.ndarray,
        ki: np.ndarray,
        damping: np.ndarray,
        force_lo: np.ndarray,
        force_hi: np.ndarray,
        tt: float,
        dt: float,
        vel_feedback: str = "error",
    ) -> None:
        self.model = model
        self.dof_adr = np.asarray(dof_adr, dtype=int)
        self.kind = kind
        self.kp = np.asarray(kp, dtype=float).copy()
        self.kd = np.asarray(kd, dtype=float).copy()
        self.ki = np.asarray(ki, dtype=float).copy()
        self.damping = np.asarray(damping, dtype=float).copy()
        self.force_lo = np.asarray(force_lo, dtype=float).copy()
        self.force_hi = np.asarray(force_hi, dtype=float).copy()
        if vel_feedback not in ("error", "absolute"):
            raise ValueError(f"unknown velocity feedback {vel_feedback}")
        self.tt = float(tt)
        self.dt = float(dt)
        self.vel_feedback = vel_feedback
        self.integral = np.zeros(6)
        self._dense = np.zeros((model.nv, model.nv))
        self._grav = mujoco.MjData(model)
        self.used_bias = np.zeros(6)
        self.used_mass = np.zeros((6, 6))

    def reset(self) -> None:
        self.integral[:] = 0.0

    def gravity(self, data: mujoco.MjData) -> np.ndarray:
        """g(q): ``qfrc_bias`` with every velocity set to zero on a scratch state."""
        self._grav.qpos[:] = data.qpos
        self._grav.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._grav)
        return self._grav.qfrc_bias[self.dof_adr].copy()

    def __call__(
        self,
        data: mujoco.MjData,
        q: np.ndarray,
        qd: np.ndarray,
        q_d: np.ndarray,
        qd_d: np.ndarray,
        qdd_d: np.ndarray,
    ) -> np.ndarray:
        error = q_d - q
        edot = qd_d - qd
        pd = self.kp * error + self.kd * edot
        if self.kind == "C1":
            self.used_bias[:] = 0.0
            return pd
        if self.kind == "C2":
            gravity = self.gravity(data)
            self.used_bias[:] = gravity
            if self.vel_feedback == "absolute":
                # Position-actuator damping: -kv qdot, with no qdot_d term.
                pd = self.kp * error - self.kd * qd
            return pd + gravity
        if self.kind == "C3":
            bias = data.qfrc_bias[self.dof_adr].copy()
            mass = dense_mass(self.model, data, self.dof_adr, self._dense)
            self.used_bias[:] = bias
            self.used_mass[:] = mass
            return mass @ qdd_d + pd + bias + self.damping * qd_d
        if self.kind == "C4":
            gravity = self.gravity(data)
            self.used_bias[:] = gravity
            return self._integrate(pd + gravity, error)
        raise ValueError(f"unknown torque law {self.kind}")

    def _integrate(self, pd_g: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Conditional integration. Returns the unsaturated torque (clip is outside)."""
        proposed = self.integral + self.dt * error
        # Integral torque itself cannot exceed the actuator range.
        proposed = np.clip(proposed, self.force_lo / self.ki, self.force_hi / self.ki)
        raw = pd_g + self.ki * proposed
        pushing_out = ((raw > self.force_hi) & (error > 0.0)) | ((raw < self.force_lo) & (error < 0.0))
        self.integral = np.where(pushing_out, self.integral, proposed)
        return pd_g + self.ki * self.integral
