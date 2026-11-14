"""Checkpoint, git stamp, and RNG state for resume."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from aloha_pp.policy.normalizer import LimitNormalizer
from aloha_pp.policy.policy import DiffusionPolicy
from aloha_pp.robot import repo_root


def git_stamp() -> dict[str, str | bool]:
    root = repo_root()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip())
    return {"git_commit": commit, "git_dirty": dirty}


def rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def policy_from_config(cfg: dict, action: LimitNormalizer, state: LimitNormalizer) -> DiffusionPolicy:
    return DiffusionPolicy(
        n_obs_steps=int(cfg["n_obs_steps"]),
        pred_horizon=int(cfg["pred_horizon"]),
        action_dim=int(cfg["action_dim"]),
        state_dim=int(cfg["state_dim"]),
        image_size=int(cfg["image_size"]),
        denoising_steps=int(cfg["denoising_steps"]),
        ddim_steps=int(cfg["ddim_steps"]),
        denoised_clip=float(cfg["denoised_clip"]),
        ema_decay=float(cfg["ema_decay"]),
        augment=bool(cfg["augment"]),
        shift_pad=int(cfg["shift_pad"]),
        exec_horizon=int(cfg["exec_horizon"]),
        vit=cfg["vit"],
        unet=cfg["unet"],
        action_normalizer=action,
        state_normalizer=state,
    )


def save_checkpoint(path: Path, policy: DiffusionPolicy, optimizer, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[DiffusionPolicy, dict]:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    extra = blob["extra"]
    stats = extra["normalizer"]
    action = LimitNormalizer(float(stats["eps"]))
    state = LimitNormalizer(float(stats["eps"]))
    action.lo = np.asarray(stats["action_lo"], dtype=np.float32)
    action.hi = np.asarray(stats["action_hi"], dtype=np.float32)
    state.lo = np.asarray(stats["state_lo"], dtype=np.float32)
    state.hi = np.asarray(stats["state_hi"], dtype=np.float32)
    policy = policy_from_config(extra["config"], action, state)
    policy.load_state_dict(blob["policy"])
    policy.to(device)
    policy.ema.eval()
    policy.eval()
    return policy, blob
