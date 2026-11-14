"""Future-target reference, gripper command, and default-omega gains."""

from __future__ import annotations

import numpy as np
import yaml

from aloha_pp.control.gains import resolve_gains
from aloha_pp.control.quintic import JointStream
from aloha_pp.control.tracking import TrackingController
from aloha_pp.robot import repo_root
from aloha_pp.sim.env import AlohaCanEnv, load_config


def _play(stream: JointStream):
    last = None
    for _ in range(stream.n_substeps):
        last = stream.advance()
    assert last is not None
    return last[0], last[1], last[2]


def _stream() -> JointStream:
    return JointStream(np.zeros(6), control_dt=0.05, n_substeps=4, v_max=np.full(6, 50.0))


def _knot(value: float) -> np.ndarray:
    q = np.zeros(6)
    q[0] = value
    return q


def test_wait_push_matches_m3_step_for_step() -> None:
    knots = [_knot(0.1 * k) for k in range(1, 6)]
    left = _stream()
    right = _stream()
    for knot in knots:
        left.push(knot)
        right.push(knot, q_future=None)
        for _ in range(4):
            lq, lv, la, lf = left.advance()
            rq, rv, ra, rf = right.advance()
            np.testing.assert_allclose(lq, rq, atol=0.0)
            np.testing.assert_allclose(lv, rv, atol=0.0)
            np.testing.assert_allclose(la, ra, atol=0.0)
            np.testing.assert_array_equal(lf, rf)


def test_preview_arrives_next_tick_with_central_difference() -> None:
    duration = 0.05
    q0 = np.zeros(6)
    q1 = _knot(0.20)
    q2 = _knot(0.50)
    q3 = _knot(0.70)
    stream = JointStream(q0, control_dt=duration, n_substeps=4, v_max=np.full(6, 50.0))
    stream.push(q1, q_future=q2)
    end_q, end_v, end_a = _play(stream)
    np.testing.assert_allclose(end_q, q1, atol=1e-8)
    np.testing.assert_allclose(end_v, (q2 - q0) / (2.0 * duration), atol=1e-8)
    np.testing.assert_allclose(end_a, (q2 - 2.0 * q1 + q0) / duration**2, atol=1e-6)
    # The first command is the knot reached at the end of this tick: (0+1) T.
    assert stream.held_segment is False
    stream.push(q2, q_future=q3)
    np.testing.assert_allclose(stream._q0, end_q, atol=1e-12)
    np.testing.assert_allclose(stream._v0, end_v, atol=1e-12)
    end_q, end_v, end_a = _play(stream)
    np.testing.assert_allclose(end_q, q2, atol=1e-8)
    np.testing.assert_allclose(end_v, (q3 - q1) / (2.0 * duration), atol=1e-8)
    np.testing.assert_allclose(end_a, (q3 - 2.0 * q2 + q1) / duration**2, atol=1e-6)


def _scalar_stream(v_max: float = 50.0, n_substeps: int = 25) -> JointStream:
    return JointStream(np.zeros(1), control_dt=0.05, n_substeps=n_substeps, v_max=np.array([v_max]))


def _trace(stream: JointStream, target: float, future: float | None):
    command = np.array([target])
    if future is None:
        stream.push(command)
    else:
        stream.push(command, q_future=np.array([future]))
    samples = []
    for _ in range(stream.n_substeps):
        samples.append(stream.advance())
    return samples


def test_preview_after_wait_drops_cached_knot_and_returns_to_one_tick() -> None:
    """Case A: a few wait-mode ticks, then a future. The cached knot is discarded."""
    stream = _scalar_stream(n_substeps=4)
    ends = []
    for k in range(3):
        samples = _trace(stream, 0.02 * (k + 1), None)
        ends.append(float(samples[-1][0][0]))
        assert stream.dropped_knot is False
    assert stream.seen_preview is False
    cached = stream._pending.copy()
    samples = _trace(stream, 0.08, 0.10)
    assert stream.dropped_knot is True
    assert stream._pending is None
    end_q, end_v, end_a, _flag = samples[-1]
    np.testing.assert_allclose(end_q, [0.08], atol=1e-8)
    # Last played knot is 0.04. The discarded 0.06 is not a sample of the difference.
    np.testing.assert_allclose(end_v, [(0.10 - 0.04) / 0.10], atol=1e-8)
    np.testing.assert_allclose(end_a, [(0.10 - 2.0 * 0.08 + 0.04) / 0.05**2], atol=1e-6)
    assert abs(float(cached[0]) - 0.06) < 1e-12
    assert abs(float(end_v[0]) - (0.06 - 0.04) / 0.10) > 1e-3
    np.testing.assert_allclose(ends, [0.0, 0.02, 0.04], atol=1e-8)
    # The new segment starts where the previous one ended, and 0.08 is reached
    # at the end of this tick, so the lag after the switch is one tick.
    np.testing.assert_allclose(stream._q0, [0.04], atol=1e-8)


def test_missing_preview_tick_stays_monotonic() -> None:
    """Case B: one omitted future in the middle of a ramp does not reverse qdot."""
    stream = _scalar_stream()
    positions = []
    speeds = []
    for k in range(10):
        future = None if k == 5 else 0.02 * (k + 2)
        samples = _trace(stream, 0.02 * (k + 1), future)
        for q, qd, _qdd, _flag in samples:
            positions.append(float(q[0]))
            speeds.append(float(qd[0]))
    assert np.all(np.diff(positions) >= -1e-9)
    assert np.min(speeds) >= -1e-9
    # The omitted tick still ends on its own target, one tick after the command.
    # Tick k=5 is the 6th command (target 0.12). 5 full ticks precede it.
    index = 6 * stream.n_substeps - 1
    assert abs(positions[index] - 0.12) < 1e-8


def test_repeated_missing_future_stops_on_the_target() -> None:
    """After preview, repeating the last target with no future settles on it."""
    stream = _scalar_stream(n_substeps=25)
    for k in range(8):
        _trace(stream, 0.02 * (k + 1), 0.02 * (k + 2))
    first = _trace(stream, 0.18, None)
    q = np.array([sample[0][0] for sample in first])
    end_q, end_v, end_a, _flag = first[-1]
    np.testing.assert_allclose(end_q, [0.18], atol=1e-8)
    np.testing.assert_allclose(end_v, [(0.18 - 0.16) / 0.10], atol=1e-8)
    np.testing.assert_allclose(end_a, [(0.16 - 0.18) / 0.05**2], atol=1e-6)
    first_overshoot = float(np.max(q) - 0.18)
    second = _trace(stream, 0.18, None)
    third = _trace(stream, 0.18, None)
    for samples in (second, third):
        end_q, end_v, end_a, _flag = samples[-1]
        np.testing.assert_allclose(end_q, [0.18], atol=1e-8)
        np.testing.assert_allclose(end_v, [0.0], atol=1e-8)
        np.testing.assert_allclose(end_a, [0.0], atol=1e-6)
    second_q = np.array([sample[0][0] for sample in second])
    settle_overshoot = float(np.max(second_q) - 0.18)
    held = np.array([sample[0][0] for sample in third])
    np.testing.assert_allclose(held, 0.18, atol=1e-8)
    # The arriving segment stays at or under the target. The next tick, whose
    # endpoint velocity and acceleration are zero, overshoots by this amount
    # and then sits on the target. Locked so a different stop rule moves it.
    assert abs(first_overshoot) < 1e-12
    assert abs(settle_overshoot - 0.001650139136) < 1e-12


def test_announced_future_is_replaced_by_the_next_command() -> None:
    duration = 0.05
    stream = JointStream(np.zeros(6), control_dt=duration, n_substeps=4, v_max=np.full(6, 50.0))
    announced = _knot(0.30)
    played = _knot(0.10)
    actual = _knot(0.12)
    nxt = _knot(0.50)
    stream.push(played, q_future=announced)
    prev_q, prev_v, _prev_a = _play(stream)
    stream.push(actual, q_future=nxt)
    np.testing.assert_allclose(stream._q0, prev_q, atol=1e-12)
    np.testing.assert_allclose(stream._v0, prev_v, atol=1e-12)
    end_q, end_v, end_a = _play(stream)
    np.testing.assert_allclose(end_q, actual, atol=1e-8)
    np.testing.assert_allclose(end_v, (nxt - played) / (2.0 * duration), atol=1e-8)
    np.testing.assert_allclose(end_a, (nxt - 2.0 * actual + played) / duration**2, atol=1e-6)
    used_announced = (announced - played) / (2.0 * duration)
    assert np.max(np.abs(end_v - used_announced)) > 1e-3


def test_limited_segment_is_flagged_and_need_not_arrive() -> None:
    stream = _scalar_stream(v_max=0.2, n_substeps=8)
    samples = _trace(stream, 1.0, 2.0)
    flags = np.array([bool(sample[3][0]) for sample in samples])
    assert np.any(flags)
    assert abs(float(samples[-1][0][0]) - 1.0) > 1e-3
    assert stream.dropped_knot is False


def test_reset_clears_preview_state() -> None:
    stream = _scalar_stream(n_substeps=4)
    _trace(stream, 0.2, 0.4)
    assert stream.seen_preview is True
    stream.reset(np.zeros(1))
    assert stream.seen_preview is False
    assert stream.dropped_knot is False
    assert stream._pending is None
    _trace(stream, 0.2, None)
    assert stream.held_segment is True
    assert stream._pending is not None
    assert stream.seen_preview is False


def test_gripper_follows_action_and_clips() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["actuator"] = "motor"
    m3 = yaml.safe_load((repo_root() / "configs" / "m3_track.yaml").read_text(encoding="utf-8"))
    env = AlohaCanEnv(cfg)
    controller = TrackingController(
        env.model,
        kind="C3",
        site_name=cfg["tcp_site"],
        v_max=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
        control_dt=0.05,
        n_substeps=25,
        gripper_open=float(cfg["gripper_open"]),
        ik_damping=float(m3["ik_damping"]),
        ik_max_iters=int(m3["ik_max_iters"]),
        ik_pos_tol=float(m3["ik_pos_tol"]),
        ik_rot_tol=float(m3["ik_rot_tol"]),
        ik_rot_weight=float(m3["ik_rot_weight"]),
        kp=np.asarray(m3["kp"], dtype=float),
        kd=np.asarray(m3["kd"], dtype=float),
        ki=np.asarray(m3["ki"], dtype=float),
    )
    try:
        env.controller = controller
        env.reset(seed=0, mode="fixed_1", render=False)
        pos, quat = env.tcp_pose()
        for requested, expected in ((0.01, 0.01), (1.0, 0.037), (-1.0, 0.002), (float(cfg["gripper_open"]), 0.037)):
            action = np.concatenate((pos, quat, [requested]))
            env.step(action, render=False)
            assert controller._grip_cmd == expected
            assert float(env.data.ctrl[controller.arm.grip_act]) == expected
    finally:
        env.close()


def test_default_omega_matches_m3_gains_exactly() -> None:
    m3 = yaml.safe_load((repo_root() / "configs" / "m3_track.yaml").read_text(encoding="utf-8"))
    m4 = yaml.safe_load((repo_root() / "configs" / "m4_task.yaml").read_text(encoding="utf-8"))
    assert m4["omega_joints"] is None
    kp, kd, ki, used = resolve_gains(
        np.asarray(m3["m_diag"]),
        omega=float(m4["omega"]),
        zeta=float(m4["zeta"]),
        ki_factor=float(m4["ki_factor"]),
        omega_joints=m4["omega_joints"],
        reference=m3,
    )
    np.testing.assert_array_equal(kp, np.asarray(m3["kp"], dtype=float))
    np.testing.assert_array_equal(kd, np.asarray(m3["kd"], dtype=float))
    np.testing.assert_array_equal(ki, np.asarray(m3["ki"], dtype=float))
    np.testing.assert_allclose(used, np.full(6, float(m3["omega"])))
    kp2, kd2, ki2, _used2 = resolve_gains(
        np.asarray(m3["m_diag"]),
        omega=float(m3["omega"]),
        zeta=float(m3["zeta"]),
        ki_factor=float(m3["ki_factor"]),
        omega_joints=np.full(6, float(m3["omega"])),
        reference=m3,
    )
    np.testing.assert_array_equal(kp2, kp)
    np.testing.assert_array_equal(kd2, kd)
    np.testing.assert_array_equal(ki2, ki)
    custom, _, _, _used = resolve_gains(
        np.asarray(m3["m_diag"]),
        omega=10.0,
        zeta=1.0,
        ki_factor=0.1,
        reference=m3,
    )
    np.testing.assert_allclose(custom, np.asarray(m3["m_diag"]) * 100.0, atol=1e-12)
    assert not np.allclose(custom, np.asarray(m3["kp"]))
