"""Numpy spatial algebra used by the hand-written kinematics and dynamics.

Quaternions are MuJoCo order (w, x, y, z). A rotation matrix maps body
coordinates into the parent (or world) frame: ``v_parent = R @ v_body``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < _EPS:
        raise ValueError("quaternion has zero length")
    return quat / norm


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix for a unit quaternion (w, x, y, z)."""
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation matrix. w is non-negative."""
    r = np.asarray(rot, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        )
    else:
        diag = np.diag(r)
        axis = int(np.argmax(diag))
        if axis == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            quat = np.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            )
        elif axis == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            quat = np.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            quat = np.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
            )
    quat = quat_normalize(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation about a (not necessarily unit) axis."""
    norm = np.linalg.norm(axis)
    if norm < _EPS:
        raise ValueError("rotation axis has zero length")
    x, y, z = np.asarray(axis, dtype=float) / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    v = 1.0 - c
    return np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )


def rotation_angle(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle in radians between two rotation matrices.

    ``atan2(sin, cos)`` keeps a true zero when the matrices agree to roundoff.
    ``arccos`` alone turns that roundoff into an angle near 1e-8.
    """
    rel = np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)
    vee = 0.5 * np.array(
        [
            rel[2, 1] - rel[1, 2],
            rel[0, 2] - rel[2, 0],
            rel[1, 0] - rel[0, 1],
        ]
    )
    sine = float(np.linalg.norm(vee))
    cosine = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arctan2(sine, cosine))


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vec, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
