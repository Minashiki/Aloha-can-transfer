"""Evaluate a diffusion-policy checkpoint on the M4 E2 initial conditions.

Run: conda run -n AIControl python -u scripts/m6_eval.py --checkpoint outputs/m6/train/ckpt_last.pt
Default seeds are 0-49, the same random-mode initial conditions as M4 E2.
The log ``eval.json`` uses format ``aloha_pp.eval.v1`` so M7 can compare BC and BC+DSRL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from aloha_pp.policy.checkpoint import load_checkpoint
from aloha_pp.policy.rollout import evaluate_policy
from aloha_pp.robot import repo_root


def _seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", default="0-49")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--videos", choices=("auto", "all"), default="auto")
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = repo_root()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, blob = load_checkpoint(ckpt, device)
    cfg = blob["extra"]["config"]
    timeout = float(cfg["episode_timeout_s"] if args.timeout_s is None else args.timeout_s)
    out = Path(args.out) if args.out else ckpt.parent / "eval"
    if not out.is_absolute():
        out = root / out
    policy.eval()
    evaluate_policy(
        policy,
        _seeds(args.seeds),
        out_dir=out,
        timeout_s=timeout,
        method="bc",
        checkpoint=str(ckpt),
        videos=args.videos,
        live_viewer=bool(args.live_viewer),
        video_fps=int(cfg.get("video_fps", 10)),
        video_max_mb=float(cfg.get("video_max_mb", 2.0)),
        exec_horizon=int(cfg["exec_horizon"]),
    )


if __name__ == "__main__":
    main()
