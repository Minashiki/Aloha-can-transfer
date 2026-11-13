"""Smoke tests for the M0 GPU and MuJoCo checks.

GPU tests fail when CUDA is missing; they are not skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import torch

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "aloha" / "aloha.xml"


def test_cuda_available() -> None:
    assert torch.cuda.is_available()


def test_arch_list_contains_sm_120() -> None:
    assert "sm_120" in torch.cuda.get_arch_list()


def test_mujoco_load_and_step() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    assert model.neq == 1
    data = mujoco.MjData(model)
    for _ in range(20):
        mujoco.mj_step(model, data)


def test_wrist_cam_offscreen() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=96, width=96)
    renderer.update_scene(data, camera="wrist_cam_left")
    image = renderer.render()
    renderer.close()
    assert image.shape == (96, 96, 3)
    assert image.size > 0
    assert int(image.min()) != int(image.max())
