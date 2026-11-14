"""Log schema and the read-only summary."""

from __future__ import annotations

import numpy as np
import pytest

from aloha_pp.control.analyze import summarize_run
from aloha_pp.control.logs import REQUIRED_ARRAYS, load_run, save_run
from aloha_pp.control.reference import PHASE_HOLD


def _meta() -> dict:
    return {
        "name": "min",
        "controller": "C3",
        "mode": "fixed_1",
        "speed": "fast",
        "actuator": "motor",
        "gravity_compensation": False,
        "kp": [1.0] * 6,
        "kd": [1.0] * 6,
        "ki": [0.0] * 6,
        "torque_limit": [10.0, 20.0, 20.0, 10.0, 10.0, 4.0],
        "velocity_limit": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        "motion_time": 1.0,
        "hold_s": 0.5,
        "git_commit": "abc",
        "ik_pos_tol": 1e-4,
        "ik_rot_tol": 1e-3,
    }


def _arrays(n: int = 8) -> dict[str, np.ndarray]:
    q_d = np.zeros((n, 6))
    q = np.zeros((n, 6))
    q[:, 0] = 0.1
    tcp = np.zeros((n, 3))
    tcp[:, 0] = 0.2
    tcp_ref = tcp.copy()
    tcp_ref[:, 0] += 0.01
    tcp_qd = tcp.copy()
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    return {
        "t": np.arange(n) * 0.002,
        "q": q,
        "qd": np.zeros((n, 6)),
        "q_d": q_d,
        "qd_d": np.zeros((n, 6)),
        "qdd_d": np.zeros((n, 6)),
        "tau_cmd": np.ones((n, 6)),
        "tau_applied": np.ones((n, 6)),
        "saturated": np.zeros((n, 6), dtype=bool),
        "vel_exceeded": np.zeros((n, 6), dtype=bool),
        "tcp_pos": tcp,
        "tcp_quat": quat,
        "tcp_ref_pos": tcp_ref,
        "tcp_ref_quat": quat,
        "tcp_qd_pos": tcp_qd,
        "tcp_qd_quat": quat,
        "ik_pos_err": np.full(n, 1e-5),
        "ik_rot_err": np.full(n, 1e-4),
        "phase": np.full(n, PHASE_HOLD, dtype=np.int8),
        "integral": np.zeros((n, 6)),
        "can_pos": np.tile(np.array([0.3, 0.12, 0.036]), (n, 1)),
        "can_quat": quat.copy(),
        "finger_can_contact": np.zeros(n, dtype=bool),
    }


def test_roundtrip_and_summary(tmp_path) -> None:
    path = tmp_path / "min.npz"
    save_run(path, _arrays(), _meta())
    arrays, meta = load_run(path)
    for key in REQUIRED_ARRAYS:
        assert key in arrays
    row = summarize_run(arrays, meta)
    np.testing.assert_allclose(row["q_rmse"], 0.1 / np.sqrt(6.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(row["q_max"], 0.1, atol=1e-12)
    np.testing.assert_allclose(row["tcp_ref_rmse"], 0.01, atol=1e-12)
    np.testing.assert_allclose(row["tcp_fk_rmse"], 0.0, atol=1e-12)
    np.testing.assert_allclose(row["steady_q_mae"], 0.1 / 6.0, atol=1e-12)
    assert row["sat_ratio"] == 0.0
    assert row["vel_ratio"] == 0.0
    assert row["tau_peak_ratio"] == pytest.approx(0.25)
    assert row["can_disp_max"] == pytest.approx(0.0)
    assert row["can_contact_steps"] == 0


def test_missing_field_is_rejected(tmp_path) -> None:
    arrays = _arrays()
    del arrays["ik_pos_err"]
    with pytest.raises(KeyError):
        save_run(tmp_path / "bad.npz", arrays, _meta())
