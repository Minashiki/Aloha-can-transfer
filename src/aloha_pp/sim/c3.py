"""Motor-actuator C3 environment used by M4, M5, and M6.

``make_c3_env`` is the shared assembly. M4's runner and M5's dataset
generator both call these functions (M5 through the wrappers in
``scripts/m4_run.py``). The controller is constructed with the lookahead
stream; passing ``next_action`` to ``env.step`` switches that stream to the
50 ms preview used when the demonstrations were recorded.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.control.gains import resolve_gains
from aloha_pp.control.tracking import TrackingController, velocity_limits
from aloha_pp.sim.env import AlohaCanEnv


def motor_cfg(m1: dict) -> dict:
    cfg = dict(m1)
    cfg["actuator"] = "motor"
    cfg["gravity_compensation"] = False
    return cfg


def c3_gains(m3: dict, m4: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp, kd, ki, _omega = resolve_gains(
        np.asarray(m3["m_diag"], dtype=float),
        omega=float(m4["omega"]),
        zeta=float(m4["zeta"]),
        ki_factor=float(m4["ki_factor"]),
        omega_joints=m4.get("omega_joints"),
        reference=m3,
    )
    return kp, kd, ki


def c3_controller(env: AlohaCanEnv, m3: dict, kp, kd, ki) -> TrackingController:
    return TrackingController(
        env.model,
        kind="C3",
        site_name=env.cfg["tcp_site"],
        v_max=velocity_limits(env.cfg),
        control_dt=1.0 / float(env.cfg["control_hz"]),
        n_substeps=env.n_substeps,
        gripper_open=float(env.cfg["gripper_open"]),
        ik_damping=float(m3["ik_damping"]),
        ik_max_iters=int(m3["ik_max_iters"]),
        ik_pos_tol=float(m3["ik_pos_tol"]),
        ik_rot_tol=float(m3["ik_rot_tol"]),
        ik_rot_weight=float(m3["ik_rot_weight"]),
        kp=kp,
        kd=kd,
        ki=ki,
        tt=1.0 / float(m3["omega"]),
        scheme="lookahead",
        a_max=np.asarray(m3["accel_limit"], dtype=float),
    )


def make_c3_env(m1: dict, m3: dict, m4: dict) -> AlohaCanEnv:
    """ALOHA can task with the motor model and the M4/M5 C3 controller."""
    env = AlohaCanEnv(motor_cfg(m1))
    env.controller = c3_controller(env, m3, *c3_gains(m3, m4))
    return env


def reset_controller_after_settle(env: AlohaCanEnv) -> None:
    """Reset the controller again after ``env.reset`` has finished settling.

    ``AlohaCanEnv.reset`` resets the controller before the 50-step settle, so
    ``home_q`` and the reference stream still hold the pre-settle joints.
    M5 captures the settled state and then calls ``controller.reset`` once
    more. M6 eval does the same. This does not change ``env.reset`` itself.
    """
    env.controller.reset(env.model, env.data)
