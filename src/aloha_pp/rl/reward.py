"""Sparse success reward, and the optional once-per-episode stage bonus.

Stages use privileged state. Each one pays at most once per episode, so
leaving and re-entering a stage does not pay again. They are off unless the
config enables them. With them off, an RL step's reward is exactly 0 or 1.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.rl.spec import STAGE_NAMES


def td_target(
    reward: "torch.Tensor",
    terminated: "torch.Tensor",
    gamma: float,
    next_q: "torch.Tensor",
) -> "torch.Tensor":
    """Bootstrap unless the transition is terminal.

    Timeout is truncated, not terminal, so it is stored with ``terminated`` 0
    and still bootstraps. Success is terminal and does not.
    """
    return reward + float(gamma) * (1.0 - terminated) * next_q


class StageTracker:
    """Ordered bonuses: grasp contact, lift, above the cell, release into the cell.

    Lift uses the highest can z after the grasp contact. Above-cell and
    release use the pose at the end of the RL step. Above-cell and release
    require the lift bonus to have been paid already, so a can that was never
    grasped cannot collect the later terms.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        rewards: dict[str, float],
        lift_thresh: float,
        cell_center: np.ndarray,
        cell_half: np.ndarray,
    ) -> None:
        unknown = set(rewards) - set(STAGE_NAMES)
        if unknown:
            raise KeyError(f"unknown stage rewards {sorted(unknown)}")
        missing = [name for name in STAGE_NAMES if name not in rewards]
        if missing:
            raise KeyError(f"stage rewards missing {missing}")
        self.enabled = bool(enabled)
        self.rewards = {name: float(rewards[name]) for name in STAGE_NAMES}
        self.lift_thresh = float(lift_thresh)
        self.center = np.asarray(cell_center, dtype=np.float64).reshape(3)
        self.half = np.asarray(cell_half, dtype=np.float64).reshape(3)
        self.paid = {name: False for name in STAGE_NAMES}
        self.contact_z: float | None = None

    def reset(self) -> None:
        self.paid = {name: False for name in STAGE_NAMES}
        self.contact_z = None

    def update(
        self,
        *,
        grasped_contact: bool,
        contact_z: float | None,
        max_z_after_contact: float | None,
        end_pos: np.ndarray,
        end_released: bool,
    ) -> float:
        """Bonus for this RL step. Zero when the switch is off, and then no flag changes."""
        if not self.enabled:
            return 0.0
        bonus = 0.0
        pos = np.asarray(end_pos, dtype=np.float64).reshape(3)
        if grasped_contact and not self.paid["both_contact"]:
            if contact_z is None:
                raise ValueError("the first grasp contact needs the can height at contact")
            self.paid["both_contact"] = True
            self.contact_z = float(contact_z)
            bonus += self.rewards["both_contact"]
        if (
            self.paid["both_contact"]
            and not self.paid["lifted"]
            and max_z_after_contact is not None
            and self.contact_z is not None
            and float(max_z_after_contact) >= self.contact_z + self.lift_thresh
        ):
            self.paid["lifted"] = True
            bonus += self.rewards["lifted"]
        above_xy = bool(np.all(np.abs(pos[:2] - self.center[:2]) <= self.half[:2]))
        above_z = float(pos[2]) >= float(self.center[2]) + self.lift_thresh
        if self.paid["lifted"] and not self.paid["above_cell"] and above_xy and above_z:
            self.paid["above_cell"] = True
            bonus += self.rewards["above_cell"]
        inside = bool(np.all(np.abs(pos - self.center) <= self.half))
        if self.paid["lifted"] and not self.paid["released_in_cell"] and inside and bool(end_released):
            self.paid["released_in_cell"] = True
            bonus += self.rewards["released_in_cell"]
        return float(bonus)


def step_reward(success: bool, bonus: float, *, staged: bool) -> float:
    """Success pays 1. The stage bonus is added only when the switch is on."""
    reward = 1.0 if success else 0.0
    if staged:
        reward += float(bonus)
    return float(reward)
