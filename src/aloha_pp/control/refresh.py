"""Make MuJoCo force and inertia fields match the current state.

``mj_step`` integrates ``qpos`` and ``qvel``, then leaves ``qfrc_bias`` and the
sparse inertia (``MjData.M``, the array historically called ``qM``) at the
pre-step state. ``mj_forward`` runs the position and velocity passes again, so
after it returns those fields describe the q and qdot that are stored now.
Torque code calls this before reading them. A second forward on the same state
is idempotent.
"""

from __future__ import annotations

import mujoco


def refresh_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Recompute kinematics, ``qfrc_bias``, and ``M`` from the current state."""
    mujoco.mj_forward(model, data)
