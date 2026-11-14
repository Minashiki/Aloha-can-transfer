"""HDF5 dataset of successful pick-and-place demonstrations.

See ``README.md`` in this directory for the layout. Observations at index t
are the state before ``actions[t]``. ``actions[t]`` is the absolute TCP pose
the controller should be at on the next tick (the M4 one-tick lead).
``next_actions[t]`` is the preview actually passed to ``env.step`` on that
tick, including the last tick (it is not recomputed at replay). Episodes are
stored only through the control tick on which the success flag becomes true.
``phase[t]`` labels that same stored action.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OBS_KEYS = ("agentview", "wrist_cam_left", "tcp_pose", "gripper_opening", "q", "qvel")
IMAGE_SHAPE = (96, 96, 3)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


class DatasetWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._data = self._file.create_group("data")
        self.count = 0

    def set_attrs(self, attrs: dict) -> None:
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                self._file.attrs[key] = json.dumps(_jsonable(value))
            else:
                self._file.attrs[key] = value

    def add_episode(self, episode: dict) -> str:
        name = f"demo_{self.count:04d}"
        group = self._data.create_group(name)
        obs = group.create_group("obs")
        for key in OBS_KEYS:
            array = np.asarray(episode["obs"][key])
            if key in ("agentview", "wrist_cam_left"):
                obs.create_dataset(key, data=array, compression="gzip", compression_opts=4)
            else:
                obs.create_dataset(key, data=np.asarray(array, dtype=np.float32))
        priv = group.create_group("privileged")
        priv.create_dataset("can_pose", data=np.asarray(episode["can_pose"], dtype=np.float32))
        group.create_dataset("actions", data=np.asarray(episode["actions"], dtype=np.float32))
        if "next_actions" in episode:
            group.create_dataset("next_actions", data=np.asarray(episode["next_actions"], dtype=np.float32))
        if "phase" in episode:
            group.create_dataset("phase", data=np.asarray(episode["phase"], dtype=np.int8))
        group.attrs["source"] = str(episode["source"])
        group.attrs["source_id"] = int(episode["source_id"])
        group.attrs["slowdown"] = float(episode["slowdown"])
        group.attrs["success"] = True
        group.attrs["success_step"] = int(episode["success_step"])
        if "phase_scheme" in episode:
            group.attrs["phase_scheme"] = str(episode["phase_scheme"])
        if "human_fraction" in episode:
            group.attrs["human_fraction"] = float(episode["human_fraction"])
        group.create_dataset("qpos0", data=np.asarray(episode["qpos0"], dtype=np.float64))
        group.create_dataset("qvel0", data=np.asarray(episode["qvel0"], dtype=np.float64))
        if "ctrl0" in episode:
            group.create_dataset("ctrl0", data=np.asarray(episode["ctrl0"], dtype=np.float64))
        if "time0" in episode:
            group.attrs["time0"] = float(episode["time0"])
        if "qacc_warmstart0" in episode:
            group.create_dataset(
                "qacc_warmstart0", data=np.asarray(episode["qacc_warmstart0"], dtype=np.float64)
            )
        if "qacc0" in episode:
            group.create_dataset("qacc0", data=np.asarray(episode["qacc0"], dtype=np.float64))
        if episode.get("act0") is not None:
            group.create_dataset("act0", data=np.asarray(episode["act0"], dtype=np.float64))
        self.count += 1
        self._file.flush()
        return name

    def close(self) -> None:
        if self._file is not None:
            self._file.attrs["n_demos"] = self.count
            self._file.close()
            self._file = None

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_demos(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle["data"].keys())


def load_episode(path: Path, name: str) -> dict:
    with h5py.File(path, "r") as handle:
        group = handle["data"][name]
        obs = {key: group["obs"][key][:] for key in OBS_KEYS}
        episode = {
            "obs": obs,
            "can_pose": group["privileged"]["can_pose"][:],
            "actions": group["actions"][:],
            "source": str(group.attrs["source"]),
            "source_id": int(group.attrs["source_id"]),
            "slowdown": float(group.attrs["slowdown"]),
            "success": bool(group.attrs["success"]),
            "success_step": int(group.attrs["success_step"]),
            "qpos0": group["qpos0"][:],
            "qvel0": group["qvel0"][:],
        }
        if "ctrl0" in group:
            episode["ctrl0"] = group["ctrl0"][:]
        if "next_actions" in group:
            episode["next_actions"] = group["next_actions"][:]
        if "phase" in group:
            episode["phase"] = group["phase"][:]
        if "phase_scheme" in group.attrs:
            episode["phase_scheme"] = str(group.attrs["phase_scheme"])
        if "human_fraction" in group.attrs:
            episode["human_fraction"] = float(group.attrs["human_fraction"])
        if "time0" in group.attrs:
            episode["time0"] = float(group.attrs["time0"])
        if "qacc_warmstart0" in group:
            episode["qacc_warmstart0"] = group["qacc_warmstart0"][:]
        if "qacc0" in group:
            episode["qacc0"] = group["qacc0"][:]
        if "act0" in group:
            episode["act0"] = group["act0"][:]
        return episode


def validate_dataset(path: Path) -> list[str]:
    """Return a list of problems. An empty list means the file checks out."""
    problems: list[str] = []
    path = Path(path)
    if not path.is_file():
        return [f"missing file {path}"]
    with h5py.File(path, "r") as handle:
        for key in (
            "control_hz",
            "action_convention",
            "reference_mode",
            "git_commit",
            "git_dirty",
            "file_sha256",
        ):
            if key not in handle.attrs:
                problems.append(f"missing root attr {key}")
        if "data" not in handle:
            return problems + ["missing data group"]
        names = sorted(handle["data"].keys())
        if not names:
            problems.append("no demonstrations")
        lengths = []
        for name in names:
            group = handle["data"][name]
            if "actions" not in group or "obs" not in group or "privileged" not in group:
                problems.append(f"{name}: missing actions, obs, or privileged")
                continue
            actions = group["actions"]
            if actions.shape[1:] != (8,) or actions.dtype != np.float32:
                problems.append(f"{name}: actions must be float32 (T, 8), got {actions.shape} {actions.dtype}")
            length = int(actions.shape[0])
            lengths.append(length)
            if "next_actions" not in group:
                problems.append(f"{name}: missing next_actions")
            else:
                nxt = group["next_actions"]
                if nxt.shape != actions.shape or nxt.dtype != np.float32:
                    problems.append(f"{name}: next_actions must be float32 {actions.shape}")
            if "phase" not in group:
                problems.append(f"{name}: missing phase")
            else:
                phase = group["phase"]
                if int(phase.shape[0]) != length:
                    problems.append(f"{name}: phase length {phase.shape[0]}")
            if "qacc_warmstart0" not in group:
                problems.append(f"{name}: missing qacc_warmstart0")
            if "time0" not in group.attrs:
                problems.append(f"{name}: missing time0")
            norms = np.linalg.norm(actions[:, 3:7], axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-3):
                problems.append(f"{name}: action quaternion is not unit length")
            for key, shape_tail in (
                ("agentview", IMAGE_SHAPE),
                ("wrist_cam_left", IMAGE_SHAPE),
                ("tcp_pose", (7,)),
                ("gripper_opening", (1,)),
                ("q", (6,)),
                ("qvel", (6,)),
            ):
                if key not in group["obs"]:
                    problems.append(f"{name}: missing obs/{key}")
                    continue
                array = group["obs"][key]
                if int(array.shape[0]) != length or tuple(array.shape[1:]) != shape_tail:
                    problems.append(f"{name}: obs/{key} shape {array.shape}")
                if key in ("agentview", "wrist_cam_left"):
                    if array.dtype != np.uint8:
                        problems.append(f"{name}: obs/{key} dtype {array.dtype}")
                    if int(array.shape[0]) > 0 and int(np.max(array[0])) == int(np.min(array[0])):
                        problems.append(f"{name}: obs/{key} first frame is a constant image")
            if "can_pose" not in group["privileged"]:
                problems.append(f"{name}: missing privileged/can_pose")
            elif group["privileged"]["can_pose"].shape != (length, 7):
                problems.append(f"{name}: can_pose shape {group['privileged']['can_pose'].shape}")
            for key in ("source", "source_id", "slowdown", "success", "success_step"):
                if key not in group.attrs:
                    problems.append(f"{name}: missing attr {key}")
            if "qpos0" not in group or "qvel0" not in group:
                problems.append(f"{name}: missing initial state")
    return problems


def length_stats(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        lengths = [int(handle["data"][name]["actions"].shape[0]) for name in handle["data"].keys()]
    array = np.asarray(lengths, dtype=float)
    if len(array) == 0:
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
