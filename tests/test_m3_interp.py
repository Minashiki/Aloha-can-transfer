"""Joint reference: endpoints, C1, speed clamp, stretch, stop and reverse."""

from __future__ import annotations

import numpy as np

from aloha_pp.control.quintic import (
    JointStream,
    eval_quintic,
    plan_segment,
    simulate_stream,
    stretch_joint_trajectory,
)


def _play(stream: JointStream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last = None
    for _ in range(stream.n_substeps):
        last = stream.advance()
    assert last is not None
    return last[0], last[1], last[2]


def test_quintic_matches_nonzero_endpoint_acceleration() -> None:
    q0 = np.zeros(6)
    q1 = np.array([0.2, -0.1, 0.15, 0.05, -0.08, 0.02])
    v0 = np.array([0.1, -0.2, 0.0, 0.3, 0.0, -0.1])
    v1 = np.array([0.4, 0.1, -0.2, 0.0, 0.2, 0.05])
    a0 = np.array([1.5, -2.0, 0.5, 0.0, 3.0, -1.0])
    a1 = np.array([-1.0, 0.5, 2.0, -0.4, 0.0, 0.8])
    s = np.array([0.0, 1.0])
    q, qd, qdd = eval_quintic(q0, v0, a0, q1, v1, a1, 0.05, s)
    np.testing.assert_allclose(q[0], q0, atol=1e-12)
    np.testing.assert_allclose(q[1], q1, atol=1e-12)
    np.testing.assert_allclose(qd[0], v0, atol=1e-10)
    np.testing.assert_allclose(qd[1], v1, atol=1e-10)
    np.testing.assert_allclose(qdd[0], a0, atol=1e-8)
    np.testing.assert_allclose(qdd[1], a1, atol=1e-8)
    assert np.any(np.abs(a0) > 0.1)
    dense = np.linspace(0.0, 1.0, 401)
    _q, qd_dense, qdd_dense = eval_quintic(q0, v0, a0, q1, v1, a1, 0.05, dense)
    dt = 0.05 * (dense[1] - dense[0])
    fd = np.diff(_q, axis=0) / dt
    np.testing.assert_allclose(0.5 * (qd_dense[:-1] + qd_dense[1:]), fd, atol=1e-3)
    assert np.max(np.abs(qdd_dense)) < 1e5


def test_lookahead_reproduces_constant_acceleration() -> None:
    """Knots of q = t^2. Central derivatives are exact, so qdd stays at 2."""
    duration = 0.05
    knots = np.array([(k * duration) ** 2 for k in range(8)])
    stream = JointStream(np.zeros(6), control_dt=duration, n_substeps=25, v_max=np.full(6, 100.0))
    qdd_rows = []
    for knot in knots:
        target = np.zeros(6)
        target[0] = knot
        stream.push(target)
        for _ in range(25):
            _q, _qd, qdd, flag = stream.advance()
            assert not np.any(flag)
            qdd_rows.append(qdd[0])
    # The segment opened by the 4th push (index 3) has central acceleration 2
    # at both ends. Samples 3*25 .. 4*25 are that segment.
    segment = np.asarray(qdd_rows[75:100])
    np.testing.assert_allclose(segment, np.full(25, 2.0), atol=1e-6)


def test_stream_hits_endpoints_and_is_c1() -> None:
    q0 = np.zeros(6)
    q1 = np.array([0.2, -0.1, 0.05, 0.0, 0.1, -0.05])
    q2 = np.array([0.35, -0.05, 0.1, 0.05, 0.0, 0.02])
    q3 = np.array([0.4, 0.0, 0.05, 0.02, -0.05, 0.01])
    stream = JointStream(q0, control_dt=0.05, n_substeps=25, v_max=np.full(6, 30.0))
    stream.push(q1)
    held, held_v, _held_a = _play(stream)
    np.testing.assert_allclose(held, q0, atol=1e-9)
    np.testing.assert_allclose(held_v, np.zeros(6), atol=1e-9)
    stream.push(q2)
    end_q, end_v, end_a = _play(stream)
    np.testing.assert_allclose(end_q, q1, atol=1e-8)
    stream.push(q3)
    np.testing.assert_allclose(stream._q0, end_q, atol=1e-12)
    np.testing.assert_allclose(stream._v0, end_v, atol=1e-12)
    np.testing.assert_allclose(stream._a0, end_a, atol=1e-10)
    q_at_0, qd_at_0, qdd_at_0 = eval_quintic(
        stream._q0, stream._v0, stream._a0, stream._q1, stream._v1, stream._a1, 0.05, np.array([0.0])
    )
    np.testing.assert_allclose(q_at_0[0], end_q, atol=1e-12)
    np.testing.assert_allclose(qd_at_0[0], end_v, atol=1e-12)
    np.testing.assert_allclose(qdd_at_0[0], end_a, atol=1e-10)


def test_online_speed_clamp_flags_and_limits() -> None:
    vmax = np.full(6, 0.5)
    far = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stream = JointStream(np.zeros(6), control_dt=0.05, n_substeps=25, v_max=vmax, scheme="lookahead")
    stream.push(far)
    flag = stream.push(far)
    assert flag[0]
    speeds = []
    prev = np.zeros(6)
    for _ in range(25):
        q, qd, qdd, step_flag = stream.advance()
        assert step_flag[0]
        assert np.all(np.isfinite(qdd))
        assert np.all(np.abs(q - prev) <= 0.5 * 0.002 + 1e-8)
        prev = q
        speeds.append(qd.copy())
    assert np.max(np.abs(np.stack(speeds))) <= 0.5 * (1.0 + 1e-4)
    assert abs(float(q[0]) - 2.0) > 0.5

    causal = JointStream(np.zeros(6), control_dt=0.05, n_substeps=25, v_max=vmax, scheme="causal")
    assert causal.push(far)[0]
    for _ in range(25):
        q, qd, _qdd, step_flag = causal.advance()
        assert step_flag[0]
        assert np.max(np.abs(qd)) <= 0.5 * (1.0 + 1e-4)
    assert abs(float(q[0]) - 2.0) > 0.5


def test_stretch_increases_duration_until_within_limit() -> None:
    knots = np.zeros((5, 6))
    knots[:, 0] = np.linspace(0.0, 2.0, 5)
    original = (knots.shape[0] - 1) * 0.05
    stretched, duration = stretch_joint_trajectory(
        knots, control_dt=0.05, n_substeps=25, v_max=np.full(6, 1.0)
    )
    assert duration > original + 1e-9
    speeds, _acc, flags = simulate_stream(
        stretched[1:], stretched[0], control_dt=0.05, n_substeps=25, v_max=np.full(6, 1.0)
    )
    assert np.max(np.abs(speeds) / 1.0) <= 1.0 + 1e-4
    assert not np.any(flags)


def test_limit_speed_can_brake_and_reverse() -> None:
    """The reported coast: v0 = v_max and the target is behind the joint."""
    q = np.array([0.0])
    v = np.array([1.0])
    a = np.array([0.0])
    target = np.array([-1.0])
    vmax = np.array([1.0])
    positions = [0.0]
    saw_reverse = False
    prev_q, prev_v = q.copy(), v.copy()
    for _ in range(30):
        q1, v1, a1, a0_used, _flag = plan_segment(q, v, target, 0.05, vmax, a0=a)
        samples = np.linspace(0.0, 1.0, 401)
        qq, qd, _qdd = eval_quintic(q, v, a0_used, q1, v1, a1, 0.05, samples)
        np.testing.assert_allclose(qq[0], prev_q, atol=1e-12)
        np.testing.assert_allclose(qd[0], prev_v, atol=1e-10)
        np.testing.assert_allclose(qq[-1], q1, atol=1e-12)
        np.testing.assert_allclose(qd[-1], v1, atol=1e-10)
        assert np.max(np.abs(qd)) <= 1.0 * (1.0 + 1e-4)
        if float(v1[0]) < 0.0:
            saw_reverse = True
        prev_q, prev_v = q1.copy(), v1.copy()
        q, v, a = q1, v1, a1
        positions.append(float(q[0]))
    # Three coasts at v = 1 would be 0.05, 0.10, 0.15. Braking must beat that.
    assert positions[1] < 0.04
    assert saw_reverse
    assert positions[-1] < positions[1]


def test_stop_from_the_speed_limit() -> None:
    q = np.array([0.0])
    v = np.array([1.0])
    a = np.array([0.0])
    target = np.array([0.0])
    for _ in range(30):
        q1, v1, a1, a0_used, _flag = plan_segment(q, v, target, 0.05, np.array([1.0]), a0=a)
        _qq, qd, _qdd = eval_quintic(q, v, a0_used, q1, v1, a1, 0.05, np.linspace(0.0, 1.0, 201))
        assert np.max(np.abs(qd)) <= 1.0 * (1.0 + 1e-4)
        np.testing.assert_allclose(qd[0], v, atol=1e-10)
        q, v, a = q1, v1, a1
    assert abs(float(v[0])) < 0.05
    assert abs(float(q[0])) < 0.05


def test_stream_reverse_near_the_speed_limit() -> None:
    vmax = np.full(6, 1.0)
    stream = JointStream(np.zeros(6), control_dt=0.05, n_substeps=25, v_max=vmax, scheme="lookahead")
    # Spacing of 0.045 rad/tick asks for 0.9 rad/s, near the limit of 1.
    forward = [0.045 * k for k in range(1, 12)]
    backward = [forward[-1] - 0.045 * k for k in range(1, 25)]
    speeds = []
    for knot in forward + backward:
        target = np.zeros(6)
        target[0] = knot
        stream.push(target)
        for _ in range(25):
            _q, qd, qdd, _flag = stream.advance()
            assert np.all(np.isfinite(qdd))
            assert abs(float(qd[0])) <= 1.0 * (1.0 + 1e-4)
            speeds.append(float(qd[0]))
    assert max(speeds) > 0.5
    assert min(speeds) < -0.2
