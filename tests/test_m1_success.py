"""Positive and negative cases for the placement hold counter."""

from __future__ import annotations

import numpy as np

from aloha_pp.sim.success import SuccessTracker, in_target, is_settled

CENTER = np.array([0.336, -0.08, 0.036])
HALF = np.array([0.0324, 0.0414, 0.02])
HOLD = 10


def test_in_target_and_velocity_predicates() -> None:
    assert in_target(CENTER, CENTER, HALF)
    assert not in_target(CENTER + np.array([0.05, 0.0, 0.0]), CENTER, HALF)
    assert is_settled(np.array([0.001, 0.0, 0.0]), 0.01)
    assert not is_settled(np.array([0.02, 0.0, 0.0]), 0.01)


def test_success_after_consecutive_hold() -> None:
    tracker = SuccessTracker(CENTER, HALF, vel_thresh=0.01, hold_steps=HOLD)
    vel = np.zeros(3)
    assert tracker.update(CENTER, vel) is False
    for _ in range(HOLD - 2):
        assert tracker.update(CENTER, vel) is False
    assert tracker.update(CENTER, vel) is True


def test_release_gate_in_cell_released_and_outside() -> None:
    tracker = SuccessTracker(CENTER, HALF, vel_thresh=0.01, hold_steps=HOLD, require_release=True)
    still = np.zeros(3)
    outside = CENTER + np.array([0.0, 0.1, 0.0])
    for _ in range(HOLD + 2):
        assert tracker.update(CENTER, still, released=False) is False
    assert tracker.count == 0
    tracker.reset()
    for _ in range(HOLD - 1):
        assert tracker.update(CENTER, still, released=True) is False
    assert tracker.update(CENTER, still, released=True) is True
    tracker.reset()
    for _ in range(HOLD + 2):
        assert tracker.update(outside, still, released=True) is False
    assert tracker.count == 0


def test_release_can_be_disabled() -> None:
    tracker = SuccessTracker(CENTER, HALF, vel_thresh=0.01, hold_steps=1, require_release=False)
    assert tracker.update(CENTER, np.zeros(3), released=False) is True


def test_rigid_tcp_motion_has_zero_slip() -> None:
    from aloha_pp.kin.ik import quat_to_mat, top_down_quat
    from aloha_pp.sim.success import tcp_local_offset

    offset_local = np.array([0.0, 0.012, -0.02])
    samples = []
    for yaw in np.linspace(0.0, 1.4, 5):
        rotation = quat_to_mat(top_down_quat(float(yaw), 0))
        for shift in np.linspace(0.0, 0.2, 4):
            tcp = np.array([0.28 + shift, -0.05 * shift, 0.12 + 0.01 * shift])
            can = tcp + rotation @ offset_local
            samples.append(tcp_local_offset(can, tcp, rotation))
    stacked = np.stack(samples)
    slip = np.linalg.norm(stacked - stacked[0], axis=1)
    assert float(slip.max()) < 1e-12
    stacked[-1] = stacked[-1] + np.array([0.006, 0.0, 0.0])
    slip = np.linalg.norm(stacked - stacked[0], axis=1)
    assert float(slip.max()) > 0.005


def test_success_rejects_position_velocity_and_broken_hold() -> None:
    tracker = SuccessTracker(CENTER, HALF, vel_thresh=0.01, hold_steps=HOLD)
    outside = CENTER + np.array([0.0, 0.1, 0.0])
    fast = np.array([0.05, 0.0, 0.0])
    for _ in range(HOLD + 2):
        assert tracker.update(outside, np.zeros(3)) is False
    assert tracker.count == 0
    for _ in range(HOLD + 2):
        assert tracker.update(CENTER, fast) is False
    assert tracker.count == 0
    for _ in range(HOLD - 1):
        tracker.update(CENTER, np.zeros(3))
    assert tracker.count == HOLD - 1
    assert tracker.update(CENTER, fast) is False
    assert tracker.count == 0
