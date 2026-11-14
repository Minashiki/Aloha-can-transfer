"""Quaternion hemisphere and the float32 action passed to the environment."""

from __future__ import annotations

import numpy as np


def canonicalize_quaternions(quat: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Flip each quaternion so ``w >= 0``.

    The rule is stateless: it does not look at the previous frame, so ``q`` and
    ``-q`` become the same float32 value. ``continuity`` is always 0 and is
    kept only so older logs still have the key. The flip uses float32.
    """
    flat = np.array(quat, dtype=np.float32, copy=True)
    shape = flat.shape
    if shape[-1] != 4:
        raise ValueError(f"expected (..., 4) wxyz, got {shape}")
    flat = flat.reshape(-1, 4)
    negative = flat[:, 0] < np.float32(0.0)
    flat[negative] *= np.float32(-1.0)
    return flat.reshape(shape), {"hemisphere": int(np.sum(negative)), "continuity": 0}


def pose_state(tcp: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """``tcp_pose`` (..., 7) and gripper, with the quaternion on the ``w >= 0`` hemisphere.

    This is the state both the training windows and ``DiffusionPolicy`` build
    before normalization.
    """
    pose = np.array(tcp, dtype=np.float32, copy=True)
    if pose.shape[-1] != 7:
        raise ValueError(f"expected tcp (..., 7), got {pose.shape}")
    quat, _info = canonicalize_quaternions(pose[..., 3:7])
    pose[..., 3:7] = quat
    opening = np.asarray(grip, dtype=np.float32).reshape(pose.shape[:-1] + (1,))
    return np.concatenate([pose, opening], axis=-1).astype(np.float32, copy=False)


def to_env_action(action: np.ndarray) -> np.ndarray:
    """Float32 unit quaternion (``w >= 0``), then float64 for the controller.

    Position and gripper are the float32 values promoted to float64, matching
    the precision of the stored demonstrations.
    """
    raw = np.asarray(action, dtype=np.float32).reshape(8).copy()
    q = raw[3:7]
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = q / np.float32(norm)
    if q[0] < 0.0:
        q = -q
    raw[3:7] = q
    return raw.astype(np.float64)
