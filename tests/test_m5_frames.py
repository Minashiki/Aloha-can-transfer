"""Rigid map from the robosuite can target cell onto fixed_1."""

from __future__ import annotations

import numpy as np

from aloha_pp.data.frames import (
    BIN1_POS,
    FIXED_1_CENTER,
    TABLE_FULL_SIZE,
    base_to_robo,
    can_target_center,
    map_can_xy,
    rigid_transform,
    robo_to_base,
)


def test_rotation_is_proper_and_round_trips() -> None:
    rotation, translation = rigid_transform()
    assert rotation.shape == (3, 3)
    assert abs(float(np.linalg.det(rotation)) - 1.0) < 1e-12
    points = np.array(
        [
            can_target_center(),
            BIN1_POS,
            BIN1_POS + np.array([0.02, -0.03, 0.04]),
            np.array([0.0, 0.0, 0.8]),
        ]
    )
    back = base_to_robo(robo_to_base(points))
    np.testing.assert_allclose(back, points, atol=1e-12)
    # A right-handed triad stays right-handed.
    triad = rotation @ np.eye(3)
    assert np.linalg.det(triad) > 0.0
    assert translation.shape == (3,)


def test_can_target_lands_on_fixed_1() -> None:
    mapped = robo_to_base(can_target_center())
    # The map is built to hit the cell center. 1e-9 m is the roundoff budget.
    np.testing.assert_allclose(mapped, FIXED_1_CENTER, atol=1e-9)
    # bin id 3 is the +x +y quadrant: positive offset from bin 2.
    offset = can_target_center() - np.array([0.1, 0.28, 0.8])
    assert offset[0] == TABLE_FULL_SIZE[0] / 4.0
    assert offset[1] == TABLE_FULL_SIZE[1] / 4.0


def test_can_xy_stays_inside_random_can() -> None:
    low = np.array([0.26, 0.07])
    high = np.array([0.34, 0.18])
    center, clamped = map_can_xy(BIN1_POS[:2], low, high)
    assert clamped is False
    np.testing.assert_allclose(center, 0.5 * (low + high))
    corner, clamped = map_can_xy(BIN1_POS[:2] + TABLE_FULL_SIZE[:2] / 2.0, low, high)
    assert clamped is False
    np.testing.assert_allclose(corner, high)
    outside, clamped = map_can_xy(BIN1_POS[:2] + np.array([1.0, 1.0]), low, high)
    assert clamped is True
    assert np.all(outside <= high + 1e-12)
    assert np.all(outside >= low - 1e-12)
