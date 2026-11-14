"""Task metrics and the single failure reason for one episode.

The classifier reads a flat record so it can be checked on made-up logs.
``signals_from_log`` builds that record from the arrays the runner writes.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.kin.ik import quat_to_mat
from aloha_pp.planning.schedule import CARRY_STATES, DWELL_CLOSED, HOLD_CAN, OPEN
from aloha_pp.sim.success import in_target

WRIST_ANGLE = 4
WRIST_ROTATE = 5

FAILURES = (
    "ik_fail",
    "not_grasped",
    "dropped",
    "not_released",
    "outside",
    "not_still",
    "timeout",
    "other",
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(low, high)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = float(successes) / float(n)
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = float(z) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return float(center - margin), float(center + margin)


def placement_errors(can: np.ndarray, center: np.ndarray, place_pos: np.ndarray) -> dict[str, float]:
    can = np.asarray(can, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    place_pos = np.asarray(place_pos, dtype=float).reshape(3)
    delta = can - center
    return {
        "xy": float(np.hypot(delta[0], delta[1])),
        "xyz": float(np.linalg.norm(delta)),
        "place": float(np.linalg.norm(can - place_pos)),
    }


def tilt_angle(quat: np.ndarray) -> float:
    """Angle of the can axis from vertical, in radians, using ``|cos|`` so a cylinder flip is upright."""
    mat = quat_to_mat(np.asarray(quat, dtype=float).reshape(4))
    cosine = abs(float(mat[:, 2][2]))
    return float(np.arccos(np.clip(cosine, 0.0, 1.0)))


def classify_episode(record: dict) -> str | None:
    """One reason, or ``None`` when the episode succeeded.

    Order: IK, missed grasp, drop during carry, fingers still touching, can
    outside the cell, can still moving, timeout, then anything else.
    """
    if record.get("success"):
        return None
    if record.get("ik_failed"):
        return "ik_fail"
    if not record.get("grasped"):
        return "not_grasped"
    if record.get("dropped"):
        return "dropped"
    if not record.get("released_at_end"):
        return "not_released"
    if not record.get("in_cell"):
        return "outside"
    if not record.get("still"):
        return "not_still"
    if record.get("timed_out"):
        return "timeout"
    return "other"


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


def signals_from_log(arrays: dict[str, np.ndarray], meta: dict) -> dict:
    """Booleans the classifier needs, taken from one episode log."""
    state = np.asarray(arrays["planner_state"])
    contact = np.asarray(arrays["finger_contact"], dtype=bool)
    if contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError("finger_contact must have shape (N, 2)")
    both = contact[:, 0] & contact[:, 1]
    dwell = state == DWELL_CLOSED
    grasped = bool(np.any(both & dwell)) if np.any(dwell) else False
    can = np.asarray(arrays["can_pos"], dtype=float)
    if np.any(dwell):
        grasp_z = float(np.median(can[dwell, 2]))
    else:
        grasp_z = float(can[0, 2])
    drop_z = float(meta.get("drop_z", 0.03))
    cell_z = float(np.asarray(meta["target_center"], dtype=float)[2])
    carry = np.isin(state, CARRY_STATES)
    high = can[:, 2] > cell_z + 0.02
    # Both fingers off the can. One finger losing contact is not a drop.
    neither = ~(contact[:, 0] | contact[:, 1])
    lost = carry & high & neither
    fell = carry & (can[:, 2] < grasp_z - drop_z)
    dropped = bool(grasped and (_consecutive(lost) >= 10 or np.any(fell)))
    center = np.asarray(meta["target_center"], dtype=float)
    half = np.asarray(meta["target_half_extents"], dtype=float)
    final = can[-1]
    vel = np.asarray(arrays["can_linvel"], dtype=float)[-1]
    thresh = float(meta["vel_thresh"])
    released = not bool(contact[-1, 0] or contact[-1, 1])
    ik = np.asarray(arrays["ik_success"], dtype=bool)
    return {
        "success": bool(meta.get("success", False)),
        "ik_failed": not bool(np.all(ik)),
        "grasped": grasped,
        "dropped": dropped,
        "released_at_end": released,
        "in_cell": bool(in_target(final, center, half)),
        "still": bool(np.linalg.norm(vel) < thresh),
        "timed_out": bool(meta.get("timed_out", False)),
    }


def hold_mask(state: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(state), HOLD_CAN)


def slip_peak(can: np.ndarray, tcp: np.ndarray, quat: np.ndarray, mask: np.ndarray) -> float:
    """Peak change of the can-in-TCP vector over the held-can samples."""
    if not np.any(mask):
        return float("nan")
    can = np.asarray(can, dtype=float)[mask]
    tcp = np.asarray(tcp, dtype=float)[mask]
    quat = np.asarray(quat, dtype=float)[mask]
    local = np.empty_like(can)
    for i in range(can.shape[0]):
        local[i] = quat_to_mat(quat[i]).T @ (can[i] - tcp[i])
    return float(np.max(np.linalg.norm(local - local[0], axis=1)))


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(err, dtype=float) ** 2)))


def _maxabs(err: np.ndarray) -> float:
    return float(np.max(np.abs(err)))


def wrist_peaks(q_err: np.ndarray, tau: np.ndarray, limits: np.ndarray, mask: np.ndarray | None) -> dict:
    if mask is None:
        mask = np.ones(q_err.shape[0], dtype=bool)
    if not np.any(mask):
        nan = float("nan")
        return {
            "wrist_angle_err": nan,
            "wrist_rotate_err": nan,
            "wrist_angle_tau": nan,
            "wrist_rotate_tau": nan,
            "wrist_angle_tau_ratio": nan,
            "wrist_rotate_tau_ratio": nan,
        }
    q_sel = np.asarray(q_err)[mask]
    tau_sel = np.asarray(tau)[mask]
    limits = np.asarray(limits, dtype=float)
    out = {}
    for name, index in (("wrist_angle", WRIST_ANGLE), ("wrist_rotate", WRIST_ROTATE)):
        out[f"{name}_err"] = float(np.max(np.abs(q_sel[:, index])))
        peak = float(np.max(np.abs(tau_sel[:, index])))
        out[f"{name}_tau"] = peak
        out[f"{name}_tau_ratio"] = peak / float(limits[index])
    return out


def qdd_ripple(qdd: np.ndarray, joint: int = 1, window: int = 25) -> dict[str, float]:
    """Shoulder ``q̈_d`` RMS, and the RMS after a trailing ``window``-step mean.

    Twenty-five physics steps are one 20 Hz period. M3's review used this pair
    on the shoulder command acceleration.
    """
    series = np.asarray(qdd, dtype=float)[:, joint]
    rms = float(np.sqrt(np.mean(series**2)))
    if series.shape[0] < window:
        smooth = rms
    else:
        cumulative = np.cumsum(series)
        windowed = cumulative[window - 1 :] - np.concatenate((np.array([0.0]), cumulative[:-window]))
        smooth = float(np.sqrt(np.mean((windowed / window) ** 2)))
    return {"qdd_rms": rms, "qdd_rms_ma25": smooth}


_RELEASE_FIELDS = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_xy",
    "can_xy",
    "can_z",
    "shoulder_err",
    "elbow_err",
)


def release_split(arrays: dict[str, np.ndarray], meta: dict) -> dict[str, float]:
    """Errors at the open-ramp start and at the first step both fingers are off.

    ``detach_can_xy_disp`` is the length of the can's xy displacement from that
    release step to the last sample, not the difference of two error magnitudes.
    TCP error is relative to the place waypoint when ``meta["place_tcp"]`` is
    present, and otherwise to ``tcp_ref_pos`` at the same step.
    """
    nan = float("nan")
    out = {f"{prefix}_{field}": nan for prefix in ("open", "detach") for field in _RELEASE_FIELDS}
    out["detach_can_xy_disp"] = nan
    if "planner_state" not in arrays or "finger_contact" not in arrays:
        return out
    state = np.asarray(arrays["planner_state"])
    contact = np.asarray(arrays["finger_contact"], dtype=bool)
    opens = np.flatnonzero(state == OPEN)
    both = contact[:, 0] & contact[:, 1]
    neither = ~(contact[:, 0] | contact[:, 1])
    grasped = np.maximum.accumulate(both.astype(np.int8)) > 0
    detached = np.flatnonzero(grasped & neither)
    _fill_release(out, "open", int(opens[0]) if opens.size else None, arrays, meta)
    detach = int(detached[0]) if detached.size else None
    _fill_release(out, "detach", detach, arrays, meta)
    if detach is not None:
        delta = np.asarray(arrays["can_pos"][-1, :2], dtype=float) - np.asarray(arrays["can_pos"][detach, :2], dtype=float)
        out["detach_can_xy_disp"] = float(np.linalg.norm(delta))
    return out


def _fill_release(out: dict, prefix: str, index: int | None, arrays: dict, meta: dict) -> None:
    if index is None:
        return
    tcp = np.asarray(arrays["tcp_pos"][index], dtype=float)
    if meta.get("place_tcp") is not None:
        waypoint = np.asarray(meta["place_tcp"], dtype=float).reshape(3)
    else:
        waypoint = np.asarray(arrays["tcp_ref_pos"][index], dtype=float)
    delta = tcp - waypoint
    can = np.asarray(arrays["can_pos"][index], dtype=float)
    center = np.asarray(meta["target_center"], dtype=float).reshape(3)
    can_delta = can - center
    q_err = np.asarray(arrays["q"][index], dtype=float) - np.asarray(arrays["q_d"][index], dtype=float)
    out[f"{prefix}_tcp_x"] = float(delta[0])
    out[f"{prefix}_tcp_y"] = float(delta[1])
    out[f"{prefix}_tcp_z"] = float(delta[2])
    out[f"{prefix}_tcp_xy"] = float(np.hypot(delta[0], delta[1]))
    out[f"{prefix}_can_xy"] = float(np.hypot(can_delta[0], can_delta[1]))
    out[f"{prefix}_can_z"] = float(can_delta[2])
    out[f"{prefix}_shoulder_err"] = float(q_err[1])
    out[f"{prefix}_elbow_err"] = float(q_err[2])
