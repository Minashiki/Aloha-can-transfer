"""One npz plus one json per tracking run. Analysis reads these and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Per physics step. Extra fields are allowed; these are required.
REQUIRED_ARRAYS = (
    "t",
    "q",
    "qd",
    "q_d",
    "qd_d",
    "qdd_d",
    "tau_cmd",
    "tau_applied",
    "saturated",
    "vel_exceeded",
    "tcp_pos",
    "tcp_quat",
    "tcp_ref_pos",
    "tcp_ref_quat",
    "tcp_qd_pos",
    "tcp_qd_quat",
    "ik_pos_err",
    "ik_rot_err",
    "phase",
    "integral",
    "can_pos",
    "can_quat",
    "finger_can_contact",
)

REQUIRED_META = (
    "controller",
    "mode",
    "speed",
    "actuator",
    "gravity_compensation",
    "kp",
    "kd",
    "ki",
    "torque_limit",
    "velocity_limit",
    "motion_time",
    "hold_s",
    "git_commit",
    "ik_pos_tol",
    "ik_rot_tol",
)


def save_run(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise KeyError(f"log is missing arrays {missing}")
    meta_missing = [key for key in REQUIRED_META if key not in meta]
    if meta_missing:
        raise KeyError(f"log is missing metadata {meta_missing}")
    np.savez_compressed(path, **{key: np.asarray(arrays[key]) for key in arrays})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_run(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    with np.load(path) as blob:
        arrays = {key: blob[key] for key in blob.files}
    missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
    if missing:
        raise KeyError(f"{path} is missing arrays {missing}")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    meta_missing = [key for key in REQUIRED_META if key not in meta]
    if meta_missing:
        raise KeyError(f"{path} is missing metadata {meta_missing}")
    return arrays, meta
