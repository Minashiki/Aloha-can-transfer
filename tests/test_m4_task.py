"""Planner schedule, failure labels, and the task metrics on made-up logs."""

from __future__ import annotations

import numpy as np

from aloha_pp.planning.metrics import (
    classify_episode,
    placement_errors,
    qdd_ripple,
    release_split,
    signals_from_log,
    slip_peak,
    tilt_angle,
    wilson_interval,
    wrist_peaks,
)
from aloha_pp.planning.schedule import (
    APPROACH,
    CARRY,
    CLOSE,
    DESCEND,
    DESCEND_PLACE,
    DWELL_CLOSED,
    DWELL_OPEN,
    LIFT,
    OPEN,
    PREGRASP,
    RETREAT,
    RETURN,
    WAIT_CLOSE,
    WAIT_OPEN,
    WAIT_SUCCESS,
    build_schedule,
    grip_wait_ticks,
    state_runs,
)
from aloha_pp.kin.ik import top_down_quat


def _waypoints():
    poses = []
    for index in range(8):
        pos = np.array([0.01 * index, 0.0, 0.1])
        poses.append((pos, top_down_quat(0.0, 0)))
    return poses


def _schedule(motion):
    return build_schedule(
        _waypoints(),
        np.asarray(motion, dtype=int),
        gripper_open=0.037,
        gripper_close=0.015,
        pregrasp_hold_ticks=2,
        wait_close_ticks=6,
        grip_ramp_ticks=12,
        dwell_closed_ticks=8,
        wait_open_ticks=6,
        dwell_open_ticks=4,
        success_wait_ticks=10,
    )


def test_wait_before_grip_covers_lag_and_settle() -> None:
    ticks = grip_wait_ticks(1, 0.25, 0.05)
    assert ticks == 6
    assert ticks * 0.05 >= 0.05 + 0.25 - 1e-12


def test_state_order_and_motion_scale() -> None:
    fast = _schedule([2, 3, 4, 5, 6, 7, 8])
    runs = state_runs(fast)
    assert [state for state, _count in runs] == [
        APPROACH,
        PREGRASP,
        DESCEND,
        WAIT_CLOSE,
        CLOSE,
        DWELL_CLOSED,
        LIFT,
        CARRY,
        DESCEND_PLACE,
        WAIT_OPEN,
        OPEN,
        DWELL_OPEN,
        RETREAT,
        RETURN,
        WAIT_SUCCESS,
    ]
    counts = dict(runs)
    assert counts[WAIT_CLOSE] == 6
    assert counts[WAIT_OPEN] == 6
    assert counts[CLOSE] == 12
    assert counts[OPEN] == 12
    assert counts[DWELL_CLOSED] == 8
    assert fast[0].grip == 0.037
    assert abs(fast[counts[APPROACH] + counts[PREGRASP] + counts[DESCEND] + counts[WAIT_CLOSE] + 11].grip - 0.015) < 1e-12
    slow_motion = np.array([2, 3, 4, 5, 6, 7, 8]) * 4
    slow = _schedule(slow_motion)
    slow_counts = dict(state_runs(slow))
    for state in (APPROACH, DESCEND, LIFT, CARRY, DESCEND_PLACE, RETREAT, RETURN):
        assert slow_counts[state] == 4 * counts[state]
    for state in (PREGRASP, WAIT_CLOSE, CLOSE, DWELL_CLOSED, WAIT_OPEN, OPEN, DWELL_OPEN, WAIT_SUCCESS):
        assert slow_counts[state] == counts[state]


def test_classify_priority_on_records() -> None:
    base = {
        "success": False,
        "ik_failed": False,
        "grasped": True,
        "dropped": False,
        "released_at_end": True,
        "in_cell": True,
        "still": True,
        "timed_out": True,
    }
    assert classify_episode({**base, "success": True, "ik_failed": True}) is None
    assert classify_episode({**base, "ik_failed": True, "grasped": False}) == "ik_fail"
    assert classify_episode({**base, "grasped": False}) == "not_grasped"
    assert classify_episode({**base, "dropped": True}) == "dropped"
    assert classify_episode({**base, "released_at_end": False}) == "not_released"
    assert classify_episode({**base, "in_cell": False}) == "outside"
    assert classify_episode({**base, "still": False}) == "not_still"
    assert classify_episode(base) == "timeout"
    assert classify_episode({**base, "timed_out": False}) == "other"


def _log(n, state, contact, can, vel, ik=True):
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    return {
        "planner_state": np.full(n, state, dtype=int),
        "finger_contact": np.repeat(np.asarray(contact, dtype=bool).reshape(1, 2), n, axis=0),
        "can_pos": np.repeat(np.asarray(can, dtype=float).reshape(1, 3), n, axis=0),
        "can_linvel": np.repeat(np.asarray(vel, dtype=float).reshape(1, 3), n, axis=0),
        "ik_success": np.ones(n, dtype=bool) if ik else np.zeros(n, dtype=bool),
    }


def test_signals_from_a_constructed_log() -> None:
    dwell = _log(5, DWELL_CLOSED, [True, True], [0.3, 0.1, 0.04], [0.0, 0.0, 0.0])
    carry = _log(12, CARRY, [False, False], [0.3, 0.1, 0.0], [0.0, 0.0, 0.0])
    arrays = {key: np.concatenate((dwell[key], carry[key])) for key in dwell}
    meta = {
        "success": False,
        "target_center": [0.336, -0.08, 0.036],
        "target_half_extents": [0.03, 0.04, 0.02],
        "vel_thresh": 0.01,
        "drop_z": 0.03,
        "timed_out": True,
    }
    record = signals_from_log(arrays, meta)
    assert record["grasped"] is True
    assert record["dropped"] is True
    assert classify_episode(record) == "dropped"


def _carry_case(contact, steps: int, z: float) -> dict:
    dwell = _log(5, DWELL_CLOSED, [True, True], [0.3, 0.1, 0.04], [0.0, 0.0, 0.0])
    carry = _log(steps, CARRY, contact, [0.3, 0.1, z], [0.0, 0.0, 0.0])
    arrays = {key: np.concatenate((dwell[key], carry[key])) for key in dwell}
    meta = {
        "success": False,
        "target_center": [0.336, -0.08, 0.036],
        "target_half_extents": [0.03, 0.04, 0.02],
        "vel_thresh": 0.01,
        "drop_z": 0.03,
        "timed_out": True,
    }
    return signals_from_log(arrays, meta)


def test_one_finger_off_is_not_a_drop_and_both_off_is() -> None:
    one = _carry_case([True, False], 20, 0.20)
    assert one["dropped"] is False
    short = _carry_case([False, False], 9, 0.20)
    assert short["dropped"] is False
    both = _carry_case([False, False], 10, 0.20)
    assert both["dropped"] is True
    assert classify_episode(both) == "dropped"


def test_release_split_uses_open_start_and_actual_detach() -> None:
    n_open = 4
    n_touch = 3
    n_free = 5
    def block(state, n, contact, tcp, can, q):
        return {
            "planner_state": np.full(n, state, dtype=int),
            "finger_contact": np.repeat(np.asarray(contact, dtype=bool).reshape(1, 2), n, axis=0),
            "tcp_pos": np.repeat(np.asarray(tcp, dtype=float).reshape(1, 3), n, axis=0),
            "tcp_ref_pos": np.repeat(np.array([[0.0, 0.0, 0.1]]), n, axis=0),
            "can_pos": np.repeat(np.asarray(can, dtype=float).reshape(1, 3), n, axis=0),
            "q": np.repeat(np.asarray(q, dtype=float).reshape(1, 6), n, axis=0),
            "q_d": np.zeros((n, 6)),
        }
    from aloha_pp.planning.schedule import OPEN
    parts = [
        block(0, 2, [False, False], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros(6)),
        block(4, 2, [True, True], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros(6)),
        block(OPEN, n_touch, [True, True], [0.004, -0.001, 0.103], [0.336, -0.076, 0.046], np.array([0, -0.006, -0.004, 0, 0, 0])),
        block(OPEN, n_free, [False, False], [0.004, -0.001, 0.103], [0.340, -0.078, 0.040], np.array([0, -0.007, -0.005, 0, 0, 0])),
    ]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    # Can moves during the free part so the displacement is not a single point.
    free = arrays["can_pos"].copy()
    start = 2 + 2 + n_touch
    for i, step in enumerate(range(start, start + n_free)):
        free[step, 0] = 0.340 + 0.001 * i
        free[step, 1] = -0.078
    arrays["can_pos"] = free
    meta = {"target_center": [0.336, -0.080, 0.036], "place_tcp": [0.0, 0.0, 0.10]}
    split = release_split(arrays, meta)
    assert abs(split["open_tcp_x"] - 0.004) < 1e-12
    assert abs(split["open_tcp_y"] - -0.001) < 1e-12
    assert abs(split["open_tcp_z"] - 0.003) < 1e-12
    assert abs(split["open_can_xy"] - np.hypot(0.0, 0.004)) < 1e-12
    assert abs(split["open_can_z"] - 0.010) < 1e-12
    assert abs(split["open_shoulder_err"] - -0.006) < 1e-12
    assert abs(split["open_elbow_err"] - -0.004) < 1e-12
    assert abs(split["detach_shoulder_err"] - -0.007) < 1e-12
    # First both-off sample is still at x=0.340; the last free sample is 0.001*(n_free-1) further.
    end_x = 0.340 + 0.001 * (n_free - 1)
    disp = np.hypot(end_x - 0.340, 0.0)
    assert abs(split["detach_can_xy_disp"] - disp) < 1e-12


def test_metric_numbers_on_constructed_samples() -> None:
    low, high = wilson_interval(50, 100, z=1.96)
    # Hand evaluation of the Wilson formula at p = 0.5, n = 100, z = 1.96.
    assert abs(low - 0.40382982859014716) < 1e-9
    assert abs(high - 0.5961701714098528) < 1e-9
    errors = placement_errors(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.4, 3.5]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert abs(errors["xy"] - np.hypot(0.1, 0.4)) < 1e-12
    assert abs(errors["xyz"] - np.linalg.norm([0.1, 0.4, 0.5])) < 1e-12
    assert abs(errors["place"] - np.linalg.norm([1.0, 2.0, 3.0])) < 1e-12
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    assert tilt_angle(upright) < 1e-12
    # 90 degrees about x: body z goes to -y, so the axis is horizontal.
    side = np.zeros(4)
    # w = cos(pi/4), x = sin(pi/4)
    side[0] = np.cos(np.pi / 4)
    side[1] = np.sin(np.pi / 4)
    assert abs(tilt_angle(side) - np.pi / 2) < 1e-9
    can = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.004, 0.0, 0.0]])
    tcp = np.zeros((3, 3))
    quat = np.tile(upright, (3, 1))
    assert abs(slip_peak(can, tcp, quat, np.array([True, True, True])) - 0.004) < 1e-12
    q_err = np.zeros((4, 6))
    q_err[:, 4] = [0.1, -0.4, 0.2, 0.0]
    q_err[:, 5] = 0.05
    tau = np.zeros((4, 6))
    tau[:, 4] = 2.0
    tau[:, 5] = -1.0
    limits = np.array([10.0, 20.0, 20.0, 10.0, 10.0, 4.0])
    peaks = wrist_peaks(q_err, tau, limits, None)
    assert abs(peaks["wrist_angle_err"] - 0.4) < 1e-12
    assert abs(peaks["wrist_rotate_err"] - 0.05) < 1e-12
    assert abs(peaks["wrist_angle_tau"] - 2.0) < 1e-12
    assert abs(peaks["wrist_angle_tau_ratio"] - 0.2) < 1e-12
    assert abs(peaks["wrist_rotate_tau_ratio"] - 0.25) < 1e-12
    qdd = np.zeros((50, 6))
    qdd[:, 1] = 2.0
    ripple = qdd_ripple(qdd, joint=1, window=25)
    assert abs(ripple["qdd_rms"] - 2.0) < 1e-12
    assert abs(ripple["qdd_rms_ma25"] - 2.0) < 1e-12
