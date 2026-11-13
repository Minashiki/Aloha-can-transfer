"""M0 environment self-check.

Prints each check and exits 0 only when every item passes.
Run with: conda run -n AIControl python scripts/check_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before mujoco is imported; the GL backend is chosen at import time.
os.environ["MUJOCO_GL"] = "egl"

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "aloha" / "aloha.xml"
IMAGE_PATH = ROOT / "outputs" / "m0_wrist_cam.png"
MATMUL_N = 2048
# fp32 GEMM accumulation order differs between CPU and GPU; this bound is
# loose enough for a 2048x2048 product of standard-normal matrices.
MATMUL_RTOL = 1e-2
MATMUL_ATOL = 1e-2
N_STEPS = 20

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def check_versions() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"mujoco: {mujoco.__version__}")
    print(f"numpy: {np.__version__}")


def check_cuda() -> bool:
    available = bool(torch.cuda.is_available())
    report("cuda_available", available, str(available))
    return available


def check_arch() -> None:
    archs = list(torch.cuda.get_arch_list())
    report("sm_120", "sm_120" in archs, str(archs))


def check_matmul() -> None:
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(MATMUL_N, MATMUL_N, generator=generator)
    right = torch.randn(MATMUL_N, MATMUL_N, generator=generator)
    cpu = left @ right
    gpu = (left.cuda() @ right.cuda()).cpu()
    max_err = (cpu - gpu).abs().max().item()
    ok = torch.allclose(cpu, gpu, rtol=MATMUL_RTOL, atol=MATMUL_ATOL)
    report(
        "gpu_matmul",
        ok,
        f"shape={MATMUL_N}x{MATMUL_N} max_abs_err={max_err:.3e} "
        f"rtol={MATMUL_RTOL} atol={MATMUL_ATOL}",
    )


def load_model():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    for _ in range(N_STEPS):
        mujoco.mj_step(model, data)
    return model, data


def check_mujoco() -> tuple[object | None, object | None]:
    try:
        model, data = load_model()
    except Exception as exc:
        report("mujoco_load_step", False, f"{XML_PATH}: {exc!r}")
        return None, None
    report("mujoco_load_step", model.neq == 1, f"neq={model.neq} steps={N_STEPS}")
    if model.neq != 1:
        return None, None
    return model, data


def check_render(model, data) -> None:
    if model is None or data is None:
        report("wrist_cam", False, "skipped because the model did not load")
        return
    try:
        renderer = mujoco.Renderer(model, height=96, width=96)
        renderer.update_scene(data, camera="wrist_cam_left")
        image = renderer.render()
        renderer.close()
    except Exception as exc:
        report("wrist_cam", False, repr(exc))
        return
    shape_ok = tuple(image.shape) == (96, 96, 3) and image.size > 0
    varies = shape_ok and int(image.min()) != int(image.max())
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(IMAGE_PATH, image)
    report(
        "wrist_cam",
        shape_ok and varies,
        f"shape={tuple(image.shape)} dtype={image.dtype} "
        f"min={int(image.min())} max={int(image.max())} saved={IMAGE_PATH}",
    )


def main() -> int:
    check_versions()
    cuda_ok = check_cuda()
    check_arch()
    if cuda_ok:
        try:
            check_matmul()
        except Exception as exc:
            report("gpu_matmul", False, repr(exc))
    else:
        report("gpu_matmul", False, "CUDA is not available")
    model, data = check_mujoco()
    check_render(model, data)
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
