"""C3 gains from a scalar or per-joint natural frequency.

The default is the single ``omega`` in ``configs/m3_track.yaml``. When every
joint uses that value, the arrays below are the yaml entries themselves, so
the M4 controller starts from the same numbers as M3. A different per-joint
``omega`` uses ``Kp = M_ii ω²``, ``Kd = 2 ζ M_ii ω``, ``Ki = Kp ω ki_factor``.
M4 does not enable a non-uniform ``omega``.
"""

from __future__ import annotations

import numpy as np


def resolve_gains(
    m_diag: np.ndarray,
    *,
    omega: float,
    zeta: float,
    ki_factor: float,
    omega_joints: np.ndarray | list | None = None,
    reference: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``kp, kd, ki, omega_per_joint``.

    ``reference`` is the M3 gain block. It is returned unchanged when every
    joint frequency equals ``reference["omega"]``.
    """
    if omega_joints is None:
        used = np.full(6, float(omega))
    else:
        used = np.asarray(omega_joints, dtype=float).reshape(6).copy()
    if reference is not None and np.allclose(used, float(reference["omega"])):
        kp = np.asarray(reference["kp"], dtype=float).copy()
        kd = np.asarray(reference["kd"], dtype=float).copy()
        ki = np.asarray(reference["ki"], dtype=float).copy()
        return kp, kd, ki, used
    mass = np.asarray(m_diag, dtype=float).reshape(6)
    kp = mass * used**2
    kd = 2.0 * float(zeta) * mass * used
    ki = kp * used * float(ki_factor)
    return kp, kd, ki, used
