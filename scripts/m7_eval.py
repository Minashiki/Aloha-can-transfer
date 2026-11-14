"""Evaluate bc_dsrl, bc_zero, or bc_seeded. Always pass --out under outputs/m7.

bc_zero feeds w = 0. bc_seeded draws N(0, I) from a generator seeded by the
episode seed. bc_dsrl uses the deterministic DSRL actor. None of them call
DiffusionPolicy.sample with noise left as None.

Run:
  conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
    --method bc_zero --bc-checkpoint outputs/m6/train/ckpt_last.pt \
    --seeds 0-49 --videos none --out outputs/m7/eval_bc_zero
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from aloha_pp.policy.checkpoint import load_checkpoint
from aloha_pp.rl.checkpoint import load_actor
from aloha_pp.rl.evaluate import evaluate_latent
from aloha_pp.rl.trainer import assert_m7_output
from aloha_pp.robot import repo_root

METHODS = ("bc_dsrl", "bc_zero", "bc_seeded")


def _seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--bc-checkpoint", default="outputs/m6/train/ckpt_last.pt")
    parser.add_argument("--actor-checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="0-49")
    parser.add_argument("--videos", choices=("none", "auto", "all"), default="none")
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.method == "bc_dsrl" and not args.actor_checkpoint:
        raise SystemExit("bc_dsrl requires --actor-checkpoint")
    root = repo_root()
    bc = Path(args.bc_checkpoint)
    if not bc.is_absolute():
        bc = root / bc
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    assert_m7_output(out, root)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, blob = load_checkpoint(bc, device)
    policy.eval()
    actor = None
    checkpoint = str(bc)
    if args.method == "bc_dsrl":
        actor_path = Path(args.actor_checkpoint)
        if not actor_path.is_absolute():
            actor_path = root / actor_path
        actor, _actor_blob = load_actor(actor_path, device)
        checkpoint = f"{bc} actor {actor_path}"
    cfg = blob["extra"]["config"]
    evaluate_latent(
        policy,
        _seeds(args.seeds),
        out_dir=out,
        timeout_s=float(args.timeout_s),
        method=args.method,
        checkpoint=checkpoint,
        actor=actor,
        videos=args.videos,
        live_viewer=bool(args.live_viewer),
        video_fps=int(cfg.get("video_fps", 10)),
        video_max_mb=float(cfg.get("video_max_mb", 2.0)),
        exec_horizon=int(cfg["exec_horizon"]),
    )


if __name__ == "__main__":
    main()
