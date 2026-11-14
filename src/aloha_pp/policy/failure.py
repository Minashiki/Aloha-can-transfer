"""Task-process failure reasons for a learned policy.

M4 and M5 keep their own classifiers. This one does not treat IK failure as
the episode reason: a single unreachable command leaves the previous joint
solution in place, and the arm can still grasp. IK misses are counted
separately, once per control step, for the current target and the preview.

A grasp attempt is one contiguous run of closed-gripper physics steps. The
label looks at every attempt, not only the first:

- ``not_grasped``: no attempt has both fingers on the can.
- ``not_lifted``: some attempt grasps, but none raises the can by
  ``lift_thresh`` above that attempt's own starting height. "Ever lifted"
  means any attempt, not the first one.
- ``dropped_after_lift``: an attempt that did lift then, still inside that
  same closed run, loses contact for ``lost_steps`` or falls by ``drop_z``.
  Opening the gripper after the run is a release, not a drop.
- ``not_released`` / ``outside`` / ``not_still`` / ``timeout`` use the final
  physics step, in that order, as M5 does.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.sim.success import in_target, is_settled


def _consecutive(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _closed_bursts(closed: np.ndarray) -> list[tuple[int, int]]:
    bursts = []
    index = 0
    n = int(closed.shape[0])
    while index < n:
        if not bool(closed[index]):
            index += 1
            continue
        end = index
        while end < n and bool(closed[end]):
            end += 1
        bursts.append((index, end))
        index = end
    return bursts


def classify_physics(
    *,
    success: bool,
    contact: np.ndarray,
    closed: np.ndarray,
    can: np.ndarray,
    vel: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    vel_thresh: float,
    drop_z: float,
    lift_thresh: float,
    lost_steps: int,
) -> str | None:
    """One task-process reason, or ``None`` when the episode succeeded."""
    if success:
        return None
    contact = np.asarray(contact, dtype=bool)
    closed = np.asarray(closed, dtype=bool)
    can = np.asarray(can, dtype=float)
    both = contact[:, 0] & contact[:, 1]
    neither = ~(contact[:, 0] | contact[:, 1])
    grasped = False
    lifted = False
    dropped = False
    for start, end in _closed_bursts(closed):
        window = np.zeros(len(closed), dtype=bool)
        window[start:end] = True
        if bool(np.any(both & window)):
            grasped = True
        ref_z = float(can[start, 2])
        rise = can[:, 2] - ref_z
        if float(np.max(rise[window])) < float(lift_thresh):
            continue
        lifted = True
        running = np.maximum.accumulate(np.where(window, rise, -np.inf))
        after = window & (running >= float(lift_thresh))
        lost = _consecutive(after & neither) >= int(lost_steps)
        fell = bool(np.any(after & (can[:, 2] < ref_z - float(drop_z))))
        dropped = dropped or lost or fell
    if not grasped:
        return "not_grasped"
    if not lifted:
        return "not_lifted"
    if dropped:
        return "dropped_after_lift"
    released = not bool(contact[-1, 0] or contact[-1, 1])
    cell = in_target(can[-1], center, half_extents)
    still = is_settled(vel[-1], float(vel_thresh))
    if not released:
        return "not_released"
    if not cell:
        return "outside"
    if not still:
        return "not_still"
    return "timeout"
