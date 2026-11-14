"""Train the M6 image diffusion policy. Smoke runs override the epoch count.

Run: conda run -n AIControl python -u scripts/m6_train.py --config configs/m6_dp.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from aloha_pp.policy.checkpoint import git_stamp, load_checkpoint, policy_from_config, rng_state, save_checkpoint, set_rng
from aloha_pp.policy.rollout import evaluate_policy, render_environment
from aloha_pp.policy.sequences import collate_windows, load_splits
from aloha_pp.robot import repo_root


def epoch_lr(epoch: int, total: int, warmup: int, base: float, min_lr: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base * float(epoch + 1) / float(warmup)
    span = max(1, total - max(warmup, 0))
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base - min_lr) * cosine


def _apply(cfg: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "val_every": args.val_every,
        "n_val_episodes": args.n_val_episodes,
        "val_batches": args.val_batches,
        "val_loss_every": args.val_loss_every,
        "device": args.device,
        "ddim_steps": args.ddim_steps,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _normalizer_blob(policy, eps: float) -> dict:
    return {
        "eps": float(eps),
        "action_lo": policy.action_lo.detach().cpu().numpy(),
        "action_hi": policy.action_hi.detach().cpu().numpy(),
        "state_lo": policy.state_lo.detach().cpu().numpy(),
        "state_hi": policy.state_hi.detach().cpu().numpy(),
    }


def _plot(history: list[dict], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    if any(row.get("val_loss") is not None for row in history):
        ax.plot(epochs, [row.get("val_loss") or float("nan") for row in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("epsilon MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m6_dp.yaml")
    parser.add_argument("--out", default="outputs/m6/train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--n-val-episodes", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--val-loss-every", type=int, default=None)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = _apply(cfg, args)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    device_name = str(cfg["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"config device is {device_name} but CUDA is not available")
    device = torch.device(device_name)
    torch.manual_seed(int(cfg["seed"]))

    resume_blob = None
    start_epoch = 0
    history: list[dict] = []
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        _, resume_blob = load_checkpoint(resume_path, device)
        saved = resume_blob["extra"]
        start_epoch = int(saved["epoch"]) + 1
        history = list(saved.get("loss_history", []))
        global_step = int(saved.get("step", 0))
        if saved.get("rng") is not None:
            set_rng(saved["rng"])
        print(f"resume {resume_path} at epoch {start_epoch}", flush=True)

    dataset_path = Path(cfg["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    split = load_splits(
        dataset_path,
        n_train=int(cfg["n_train_demos"]),
        seed=int(cfg["subset_seed"]),
        n_obs=int(cfg["n_obs_steps"]),
        horizon=int(cfg["pred_horizon"]),
        range_eps=float(cfg["range_eps"]),
    )
    if resume_blob is not None:
        saved_ids = [int(v) for v in resume_blob["extra"]["train_source_ids"]]
        if saved_ids != list(split["train_source_ids"]):
            raise RuntimeError("resume training subset does not match this config")
    print(
        f"train demos {len(split['train_names'])} source_ids {split['train_source_ids']}",
        flush=True,
    )
    print(f"quaternion flips {split['quat_stats']}", flush=True)
    print(f"windows train {len(split['train'])} val {len(split['val'])}", flush=True)

    policy = policy_from_config(cfg, split["action_normalizer"], split["state_normalizer"]).to(device)
    if resume_blob is not None:
        policy.load_state_dict(resume_blob["policy"])
        policy.ema.eval()
    optimizer = torch.optim.AdamW(
        policy.diffusion.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    if resume_blob is not None and resume_blob["optimizer"] is not None:
        optimizer.load_state_dict(resume_blob["optimizer"])
    print(f"parameters {policy.n_parameters()} device {device} noise_shape {policy.noise_shape}", flush=True)
    hardware = render_environment()
    print(f"render {hardware}", flush=True)
    (out / "render.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    def save(epoch: int, name: str) -> None:
        stamp = git_stamp()
        extra = {
            "config": cfg,
            "epoch": int(epoch),
            "step": int(global_step),
            "train_names": split["train_names"],
            "train_source_ids": split["train_source_ids"],
            "val_names": split["val_names"],
            "quat_stats": split["quat_stats"],
            "normalizer": _normalizer_blob(policy, float(cfg["range_eps"])),
            "loss_history": history,
            "rng": rng_state(),
            **stamp,
        }
        path = out / name
        save_checkpoint(path, policy, optimizer, extra)
        meta = {
            "epoch": epoch,
            "step": global_step,
            "train_source_ids": split["train_source_ids"],
            "train_names": split["train_names"],
            "quat_stats": split["quat_stats"],
            "git_commit": stamp["git_commit"],
            "git_dirty": stamp["git_dirty"],
            "checkpoint": str(path),
            "n_parameters": policy.n_parameters(),
            "noise_shape": list(policy.noise_shape),
            "eta": policy.eta,
            "render": hardware,
        }
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_path = out / "train_log.jsonl"
    step_times: list[float] = []
    for epoch in range(start_epoch, int(cfg["epochs"])):
        lr = epoch_lr(epoch, int(cfg["epochs"]), int(cfg["warmup_epochs"]), float(cfg["lr"]), float(cfg["min_lr"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        policy.train()
        generator = torch.Generator()
        generator.manual_seed(int(cfg["seed"]) + epoch)
        loader = DataLoader(
            split["train"],
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_windows,
            generator=generator,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        losses = []
        epoch_t = time.perf_counter()
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = policy.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.diffusion.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            policy.update_ema()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
            global_step += 1
        train_loss = float(sum(losses) / max(1, len(losses)))
        do_loss = (epoch % int(cfg["val_loss_every"]) == 0) or epoch + 1 == int(cfg["epochs"])
        val_loss = None
        val_loss_s = None
        if do_loss:
            val_t = time.perf_counter()
            val_loss = _val_loss(policy, split["val"], device, int(cfg["val_batches"]), int(cfg["batch_size"]))
            val_loss_s = time.perf_counter() - val_t
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_s": val_loss_s,
            "lr": lr,
            "epoch_s": time.perf_counter() - epoch_t,
            "step_s": float(sum(step_times) / len(step_times)),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        _plot(history, out / "loss.png")
        print(
            f"epoch {epoch} train {train_loss:.6f} val {val_loss} lr {lr:.6e} "
            f"epoch_s {row['epoch_s']:.2f} step_s {row['step_s']:.4f}",
            flush=True,
        )
        do_val = (epoch + 1) % int(cfg["val_every"]) == 0 or epoch + 1 == int(cfg["epochs"])
        save(epoch, "ckpt_last.pt")
        if (epoch + 1) % int(cfg["save_every"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            save(epoch, f"ckpt_epoch{epoch:04d}.pt")
        if do_val and int(cfg["n_val_episodes"]) > 0:
            val_dir = out / f"val_epoch{epoch:04d}"
            summary = evaluate_policy(
                policy,
                list(range(int(cfg["n_val_episodes"]))),
                out_dir=val_dir,
                timeout_s=float(cfg["episode_timeout_s"]),
                method="bc",
                checkpoint=str(out / "ckpt_last.pt"),
                videos="all",
                live_viewer=bool(args.live_viewer),
                video_fps=int(cfg["video_fps"]),
                video_max_mb=float(cfg["video_max_mb"]),
                exec_horizon=int(cfg["exec_horizon"]),
            )
            print(f"val video dir {val_dir / 'videos'} success {summary['n_success']}/{summary['n']}", flush=True)

    steady = step_times[1:] if len(step_times) > 1 else step_times
    step_s = float(sum(steady) / len(steady))
    windows = len(split["train"])
    full_batch = 64
    full_epochs = 2000
    steps_per = math.ceil(windows / full_batch)
    hours = step_s * steps_per * full_epochs / 3600.0
    print(
        f"measured step_s {step_s:.4f} windows {windows} "
        f"projected {full_epochs} epochs batch {full_batch}: {hours:.2f} h",
        flush=True,
    )


@torch.no_grad()
def _val_loss(policy, dataset, device, max_batches: int, batch_size: int) -> float | None:
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)
    losses = []
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        cond = policy._cond(batch, state_is_normalized=True)
        action = batch["action"].to(device)
        losses.append(float(policy.ema.loss(action, cond)))
    if not losses:
        return None
    return float(sum(losses) / len(losses))


if __name__ == "__main__":
    main()
