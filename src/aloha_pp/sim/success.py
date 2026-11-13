"""Placement success: inside the target cell, nearly stopped, and released."""

from __future__ import annotations

import numpy as np


def in_target(pos: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> bool:
    return bool(np.all(np.abs(np.asarray(pos) - np.asarray(center)) <= np.asarray(half_extents)))


def is_settled(linvel: np.ndarray, vel_thresh: float) -> bool:
    return bool(np.linalg.norm(np.asarray(linvel)) < vel_thresh)


def tcp_local_offset(can_pos: np.ndarray, tcp_pos: np.ndarray, tcp_rot: np.ndarray) -> np.ndarray:
    """Can center in the TCP frame: ``R_tcp.T @ (p_can - p_tcp)``."""
    return np.asarray(tcp_rot, dtype=float).T @ (
        np.asarray(can_pos, dtype=float) - np.asarray(tcp_pos, dtype=float)
    )


class SuccessTracker:
    """Counts consecutive physics steps that satisfy the placement condition.

    ``require_release`` (default on) also demands that the fingers are not
    touching the can. The caller passes that contact result each update.
    """

    def __init__(
        self,
        center: np.ndarray,
        half_extents: np.ndarray,
        vel_thresh: float,
        hold_steps: int,
        require_release: bool = True,
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)
        self.vel_thresh = float(vel_thresh)
        self.hold_steps = int(hold_steps)
        self.require_release = bool(require_release)
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, pos: np.ndarray, linvel: np.ndarray, released: bool = True) -> bool:
        placed = in_target(pos, self.center, self.half_extents) and is_settled(linvel, self.vel_thresh)
        if self.require_release and not released:
            placed = False
        if placed:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.hold_steps
