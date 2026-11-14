"""Rigid map from robosuite PickPlaceCan into this project's base frame.

The numbers are taken from ``third_party/robosuite`` (v1.5.1) and are copied
here so runtime never imports robosuite.

Sources
-------
* ``robosuite/environments/manipulation/pick_place.py``
  ``PickPlace.__init__`` defaults: ``bin1_pos=(0.1, -0.25, 0.8)`` (start bin),
  ``bin2_pos=(0.1, 0.28, 0.8)`` (goal bin),
  ``table_full_size=(0.39, 0.49, 0.82)``.
* Same file, ``object_to_id``: ``"can"`` is index 3. ``_setup_references``
  sets ``bin_id = i`` while enumerating ``self.objects``, and ``_construct_objects``
  builds Milk, Bread, Cereal, Can in that order, so the can's bin id is 3.
  For bin id 3 the ``+x,+y`` quadrant center is

  ``x = bin2_x + table_full_size_x / 4``,
  ``y = bin2_y + table_full_size_y / 4``,
  ``z = bin2_z``.

  That matches the scene comment in ``assets/scene/pick_place_can.xml``
  ("Robosuite Can target is the +x,+y cell of bin2") and this project's
  ``fixed_1`` cell, which is the right bin's ``+X+Y`` cell.
* ``robosuite/models/robots/manipulators/panda_robot.py``:
  ``base_xpos_offset["bins"] = (-0.5, -0.1, 0)``. The Panda base faces ``+X``
  with ``+Z`` up, the same axis sense as ``left/base_link`` (``+X`` forward,
  ``+Y`` left, ``+Z`` up).

The two goal cells are the same quadrant of their bins, so the rotation is
the identity. A mirror would be required only to also swap which side of the
robot the start bin sits on; this map does not try to do that. Bin centers
are 0.53 m apart in robosuite and 0.25 m apart here, so one rigid map cannot
send both bin centers to both of ours. The can's initial position uses a
separate in-bin fraction, not this rigid map. Det(R) = +1.
"""

from __future__ import annotations

import numpy as np

# pick_place.py defaults.
BIN1_POS = np.array([0.1, -0.25, 0.8])
BIN2_POS = np.array([0.1, 0.28, 0.8])
TABLE_FULL_SIZE = np.array([0.39, 0.49, 0.82])
CAN_OBJECT_INDEX = 3

# configs/m1_pick_place.yaml target_cells.fixed_1.center.
FIXED_1_CENTER = np.array([0.336, -0.080, 0.03604])


def can_target_center() -> np.ndarray:
    """World-frame center of the can's goal cell, from ``_setup_references``."""
    bin_id = CAN_OBJECT_INDEX
    bin_x_low = float(BIN2_POS[0])
    bin_y_low = float(BIN2_POS[1])
    if bin_id == 0 or bin_id == 2:
        bin_x_low -= float(TABLE_FULL_SIZE[0]) / 2.0
    if bin_id < 2:
        bin_y_low -= float(TABLE_FULL_SIZE[1]) / 2.0
    bin_x_low += float(TABLE_FULL_SIZE[0]) / 4.0
    bin_y_low += float(TABLE_FULL_SIZE[1]) / 4.0
    return np.array([bin_x_low, bin_y_low, float(BIN2_POS[2])])


def rigid_transform() -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` with ``p_base = R @ p_robo + t``.

    ``R`` is identity, so ``+X/+Y/+Z`` agree. ``t`` sends the can target cell
    center onto ``fixed_1``.
    """
    rotation = np.eye(3)
    translation = FIXED_1_CENTER - rotation @ can_target_center()
    return rotation, translation


def robo_to_base(points: np.ndarray) -> np.ndarray:
    rotation, translation = rigid_transform()
    pts = np.asarray(points, dtype=float)
    return (rotation @ pts.reshape(-1, 3).T).T.reshape(pts.shape) + translation


def base_to_robo(points: np.ndarray) -> np.ndarray:
    rotation, translation = rigid_transform()
    pts = np.asarray(points, dtype=float)
    shifted = pts.reshape(-1, 3) - translation
    return (rotation.T @ shifted.T).T.reshape(pts.shape)


def map_can_xy(
    can_xy: np.ndarray,
    xy_low: np.ndarray,
    xy_high: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Map a robosuite can xy into ``random_can`` by its fraction inside bin 1.

    The fraction uses ``table_full_size / 2`` as the bin half-extent (the
    same lengths ``_setup_references`` uses for the cells). The destination
    rectangle is ``random_can``, whose center is the left-bin center. Values
    outside the source bin are clamped, and the bool reports that clamp.
    """
    low = np.asarray(xy_low, dtype=float).reshape(2)
    high = np.asarray(xy_high, dtype=float).reshape(2)
    half = TABLE_FULL_SIZE[:2] / 2.0
    frac = (np.asarray(can_xy, dtype=float).reshape(2) - BIN1_POS[:2]) / half
    clamped = bool(np.any(np.abs(frac) > 1.0 + 1e-8))
    frac = np.clip(frac, -1.0, 1.0)
    center = 0.5 * (low + high)
    dest_half = 0.5 * (high - low)
    return center + frac * dest_half, clamped
