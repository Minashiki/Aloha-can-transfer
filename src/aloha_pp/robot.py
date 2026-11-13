"""Shared names and paths for the ALOHA left arm."""

from __future__ import annotations

from pathlib import Path

ARM_JOINTS: tuple[str, ...] = (
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
)
GRIPPER_ACTUATOR = "left/gripper"
GRIPPER_JOINT = "left/left_finger"
RIGHT_FINGER_JOINT = "left/right_finger"
BASE_BODY = "left/base_link"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def robot_xml() -> Path:
    return repo_root() / "aloha" / "aloha.xml"


def scene_xml(version: str) -> Path:
    if version not in ("motor", "position"):
        raise ValueError(f"unknown actuator version {version!r}")
    return repo_root() / "aloha" / f"aloha_{version}.xml"
