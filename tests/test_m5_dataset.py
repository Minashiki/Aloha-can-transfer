"""Dataset round trip, and the checker catching a bad quaternion and a blank image."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from aloha_pp.data.dataset import DatasetWriter, length_stats, load_episode, validate_dataset


def _episode(image: np.ndarray, quat: np.ndarray) -> dict:
    steps = 4
    actions = np.zeros((steps, 8), dtype=np.float32)
    actions[:, 3:7] = quat
    actions[:, 7] = 0.037
    obs = {
        "agentview": np.broadcast_to(image, (steps, 96, 96, 3)).copy(),
        "wrist_cam_left": np.broadcast_to(image, (steps, 96, 96, 3)).copy(),
        "tcp_pose": np.zeros((steps, 7), dtype=np.float32),
        "gripper_opening": np.full((steps, 1), 0.037, dtype=np.float32),
        "q": np.zeros((steps, 6), dtype=np.float32),
        "qvel": np.zeros((steps, 6), dtype=np.float32),
    }
    obs["tcp_pose"][:, 3] = 1.0
    return {
        "obs": obs,
        "can_pose": np.zeros((steps, 7), dtype=np.float32),
        "actions": actions,
        "source": "retarget",
        "source_id": 3,
        "slowdown": 1.5,
        "success_step": steps - 1,
        "qpos0": np.zeros(8),
        "qvel0": np.zeros(8),
        "ctrl0": np.zeros(7),
        "next_actions": actions.copy(),
        "phase": np.zeros(steps, dtype=np.int8),
        "time0": 0.1,
        "qacc_warmstart0": np.zeros(4),
    }


def _root(writer: DatasetWriter) -> None:
    writer.set_attrs(
        {
            "control_hz": 20,
            "action_convention": "base wxyz absolute lead-1",
            "reference_mode": "preview",
            "git_commit": "abc",
            "git_dirty": "true",
            "file_sha256": "{}",
        }
    )


def test_round_trip_and_checker(tmp_path: Path) -> None:
    good = tmp_path / "good.hdf5"
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[0, 0, 0] = 255
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with DatasetWriter(good) as writer:
        _root(writer)
        writer.add_episode(_episode(image, quat))
    problems = validate_dataset(good)
    assert problems == []
    loaded = load_episode(good, "demo_0000")
    assert loaded["source"] == "retarget"
    assert loaded["source_id"] == 3
    assert loaded["actions"].shape == (4, 8)
    assert loaded["obs"]["agentview"].shape == (4, 96, 96, 3)
    assert loaded["obs"]["agentview"].dtype == np.uint8
    stats = length_stats(good)
    assert stats["n"] == 1
    assert stats["min"] == 4.0

    bad = tmp_path / "bad.hdf5"
    blank = np.zeros((96, 96, 3), dtype=np.uint8)
    broken = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with DatasetWriter(bad) as writer:
        _root(writer)
        writer.add_episode(_episode(blank, broken))
    problems = validate_dataset(bad)
    assert any("quaternion" in item for item in problems)
    assert any("constant image" in item for item in problems)

    # A file that is not a dataset at all.
    empty = tmp_path / "empty.hdf5"
    with h5py.File(empty, "w"):
        pass
    assert validate_dataset(empty)
