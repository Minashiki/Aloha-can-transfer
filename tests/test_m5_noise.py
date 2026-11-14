"""Zero noise matches the M4 planner. A fixed seed repeats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aloha_pp.control.gains import resolve_gains
from aloha_pp.control.tracking import TrackingController, velocity_limits
from aloha_pp.planning.noise import NoisyPickPlacePlanner, zero_noise
from aloha_pp.planning.planner import PickPlacePlanner
from aloha_pp.sim.env import AlohaCanEnv, load_config


def _load(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _motor_env(m1: dict, m3: dict, m4: dict) -> AlohaCanEnv:
    cfg = dict(m1)
    cfg["actuator"] = "motor"
    cfg["gravity_compensation"] = False
    env = AlohaCanEnv(cfg)
    kp, kd, ki, _omega = resolve_gains(
        np.asarray(m3["m_diag"], dtype=float),
        omega=float(m4["omega"]),
        zeta=float(m4["zeta"]),
        ki_factor=float(m4["ki_factor"]),
        omega_joints=m4.get("omega_joints"),
        reference=m3,
    )
    env.controller = TrackingController(
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
    return env


def _actions(planner) -> np.ndarray:
    return np.stack([tick.action() for tick in planner.schedule])


def test_zero_noise_matches_planner_tick_for_tick() -> None:
    m1 = load_config()
    m3 = _load("m3_track.yaml")
    m4 = _load("m4_task.yaml")
    env_a = _motor_env(m1, m3, m4)
    env_b = _motor_env(m1, m3, m4)
    try:
        env_a.reset(seed=2, mode="fixed_1", render=False)
        env_b.reset(seed=2, mode="fixed_1", render=False)
        plain = PickPlacePlanner(m1, m4, m3, "medium")
        noisy = NoisyPickPlacePlanner(m1, m4, m3, "medium", zero_noise(), np.random.default_rng(0))
        plain.reset(env_a, mode="fixed_1", log=lambda *_a, **_k: None)
        noisy.reset(env_b, mode="fixed_1", log=lambda *_a, **_k: None)
        np.testing.assert_allclose(_actions(plain), _actions(noisy))
        plain.offset = np.array([0.001, -0.002, 0.0004])
        noisy.offset = plain.offset.copy()
        plain._rebuild_carry()
        noisy._rebuild_carry()
        np.testing.assert_allclose(_actions(plain), _actions(noisy))
    finally:
        env_a.close()
        env_b.close()


def test_fixed_seed_repeats() -> None:
    m1 = load_config()
    m3 = _load("m3_track.yaml")
    m4 = _load("m4_task.yaml")
    noise = _load("m5_noise.yaml")
    env = _motor_env(m1, m3, m4)
    try:
        schedules = []
        for _ in range(2):
            env.reset(seed=3, mode="random", render=False)
            planner = NoisyPickPlacePlanner(
                m1, m4, m3, "medium", noise, np.random.default_rng(11)
            )
            planner.reset(env, mode="random", log=lambda *_a, **_k: None)
            schedules.append(_actions(planner))
        np.testing.assert_allclose(schedules[0], schedules[1])
        env.reset(seed=3, mode="random", render=False)
        other = NoisyPickPlacePlanner(m1, m4, m3, "medium", noise, np.random.default_rng(12))
        other.reset(env, mode="random", log=lambda *_a, **_k: None)
        other_actions = _actions(other)
        assert other_actions.shape != schedules[0].shape or not np.allclose(other_actions, schedules[0])
    finally:
        env.close()
