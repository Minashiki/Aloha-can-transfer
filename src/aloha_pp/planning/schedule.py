"""Open-loop pick-and-place schedule: motion ticks scale, waits do not.

Waypoints are home, above the can, grasp, lifted, above the cell, the drop,
the retreat, and home again. Gripper ramps and the pre-grasp / pre-release
waits keep the same tick counts at every speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aloha_pp.control.reference import quintic_blend, slerp

APPROACH = 0
PREGRASP = 1
DESCEND = 2
WAIT_CLOSE = 3
CLOSE = 4
DWELL_CLOSED = 5
LIFT = 6
CARRY = 7
DESCEND_PLACE = 8
WAIT_OPEN = 9
OPEN = 10
DWELL_OPEN = 11
RETREAT = 12
RETURN = 13
WAIT_SUCCESS = 14

STATE_NAMES = (
    "approach",
    "pregrasp",
    "descend",
    "wait_close",
    "close",
    "dwell_closed",
    "lift",
    "carry",
    "descend_place",
    "wait_open",
    "open",
    "dwell_open",
    "retreat",
    "return",
    "wait_success",
)

# After the fingers have closed, until the release ramp starts.
HOLD_CAN = (DWELL_CLOSED, LIFT, CARRY, DESCEND_PLACE, WAIT_OPEN)
# Transport, while a fall is still the planner's fault rather than a release.
CARRY_STATES = (LIFT, CARRY, DESCEND_PLACE)

MOTION_STATES = (APPROACH, DESCEND, LIFT, CARRY, DESCEND_PLACE, RETREAT, RETURN)
N_MOTION = 7


@dataclass
class Tick:
    state: int
    pos: np.ndarray
    quat: np.ndarray
    grip: float
    span: tuple | None = None

    def action(self) -> np.ndarray:
        return np.concatenate((self.pos, self.quat, np.array([self.grip])))

    def dense(self, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
        """Cartesian reference at the end of each physics step of this tick."""
        if self.span is None:
            pos = np.repeat(self.pos.reshape(1, 3), n_sub, axis=0)
            quat = np.repeat(self.quat.reshape(1, 4), n_sub, axis=0)
            return pos, quat
        p0, q0, p1, q1, k, n_ticks = self.span
        pos = np.empty((n_sub, 3))
        quat = np.empty((n_sub, 4))
        for i in range(n_sub):
            u = (k + (i + 1) / n_sub) / n_ticks
            blend = quintic_blend(u)
            pos[i] = p0 + blend * (p1 - p0)
            quat[i] = slerp(q0, q1, blend)
        return pos, quat


def grip_wait_ticks(lag_ticks: int, settle_s: float, control_dt: float) -> int:
    """Ticks to wait at a grasp or place pose before the gripper moves.

    The first ``lag_ticks`` cover the reference lag. The rest are ``settle_s``,
    rounded to the control period, so the arm can sit on the waypoint.
    """
    settle = int(round(float(settle_s) / float(control_dt)))
    if settle < 0:
        raise ValueError("settle_s must be non-negative")
    return int(lag_ticks) + settle


def seconds_to_ticks(seconds: float, control_dt: float, *, minimum: int = 1) -> int:
    ticks = int(round(float(seconds) / float(control_dt)))
    return max(int(minimum), ticks)


def _hold(state: int, pose: tuple[np.ndarray, np.ndarray], grip: float, n: int) -> list[Tick]:
    pos, quat = pose
    return [Tick(state, pos.copy(), quat.copy(), float(grip)) for _ in range(int(n))]


def _motion(
    state: int,
    start: tuple[np.ndarray, np.ndarray],
    end: tuple[np.ndarray, np.ndarray],
    n: int,
    grip: float,
) -> list[Tick]:
    p0, q0 = start
    p1, q1 = end
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    ticks = []
    for k in range(int(n)):
        blend = quintic_blend((k + 1) / int(n))
        pos = p0 + blend * (p1 - p0)
        quat = slerp(q0, q1, blend)
        ticks.append(Tick(state, pos, quat, float(grip), span=(p0, q0, p1, q1, k, int(n))))
    return ticks


def _ramp(state: int, pose: tuple[np.ndarray, np.ndarray], start: float, end: float, n: int) -> list[Tick]:
    """``n`` ticks from just after ``start`` through ``end``."""
    pos, quat = pose
    levels = np.linspace(float(start), float(end), int(n) + 1)[1:]
    return [Tick(state, pos.copy(), quat.copy(), float(level)) for level in levels]


def build_schedule(
    waypoints: list[tuple[np.ndarray, np.ndarray]],
    motion_ticks: np.ndarray,
    *,
    gripper_open: float,
    gripper_close: float,
    pregrasp_hold_ticks: int,
    wait_close_ticks: int,
    grip_ramp_ticks: int,
    dwell_closed_ticks: int,
    wait_open_ticks: int,
    dwell_open_ticks: int,
    success_wait_ticks: int,
) -> list[Tick]:
    """One control tick per entry. ``motion_ticks`` has one count per moving segment."""
    if len(waypoints) != 8:
        raise ValueError("expected 8 waypoints")
    counts = np.asarray(motion_ticks, dtype=int).reshape(N_MOTION)
    if np.any(counts < 1):
        raise ValueError("every motion segment needs at least one tick")
    w = waypoints
    ticks: list[Tick] = []
    ticks += _motion(APPROACH, w[0], w[1], counts[0], gripper_open)
    ticks += _hold(PREGRASP, w[1], gripper_open, pregrasp_hold_ticks)
    ticks += _motion(DESCEND, w[1], w[2], counts[1], gripper_open)
    ticks += _hold(WAIT_CLOSE, w[2], gripper_open, wait_close_ticks)
    ticks += _ramp(CLOSE, w[2], gripper_open, gripper_close, grip_ramp_ticks)
    ticks += _hold(DWELL_CLOSED, w[2], gripper_close, dwell_closed_ticks)
    ticks += _motion(LIFT, w[2], w[3], counts[2], gripper_close)
    ticks += _motion(CARRY, w[3], w[4], counts[3], gripper_close)
    ticks += _motion(DESCEND_PLACE, w[4], w[5], counts[4], gripper_close)
    ticks += _hold(WAIT_OPEN, w[5], gripper_close, wait_open_ticks)
    ticks += _ramp(OPEN, w[5], gripper_close, gripper_open, grip_ramp_ticks)
    ticks += _hold(DWELL_OPEN, w[5], gripper_open, dwell_open_ticks)
    ticks += _motion(RETREAT, w[5], w[6], counts[5], gripper_open)
    ticks += _motion(RETURN, w[6], w[7], counts[6], gripper_open)
    ticks += _hold(WAIT_SUCCESS, w[7], gripper_open, success_wait_ticks)
    return ticks


def state_runs(ticks: list[Tick]) -> list[tuple[int, int]]:
    """``(state, count)`` in order, merging consecutive ticks of the same state."""
    if not ticks:
        return []
    runs = [[ticks[0].state, 1]]
    for tick in ticks[1:]:
        if tick.state == runs[-1][0]:
            runs[-1][1] += 1
        else:
            runs.append([tick.state, 1])
    return [(state, count) for state, count in runs]
