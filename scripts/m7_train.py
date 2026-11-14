"""Train DSRL on a frozen M6 diffusion policy. Formal runs are started by the user.

Run:
  conda run -n AIControl --no-capture-output python -u scripts/m7_train.py --config configs/m7_dsrl.yaml
Smoke (a few minutes, not the formal step count):
  conda run -n AIControl --no-capture-output python -u scripts/m7_train.py --config configs/m7_dsrl.yaml --smoke --out outputs/m7/smoke/train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

from aloha_pp.policy.checkpoint import load_checkpoint
from aloha_pp.rl.checkpoint import assert_resume_config, build_models, sha256_file
from aloha_pp.rl.replay import Replay
from aloha_pp.rl.spec import privileged_dim
from aloha_pp.rl.trainer import (
    assert_m7_output,
    benchmark_collection,
    benchmark_updates,
    estimate_seconds,
    load_resumed,
    train,
)
from aloha_pp.robot import repo_root


SMOKE = {
    "n_envs": 4,
    "total_env_steps": 280,
    "init_rollout_steps": 55,
    "val_every": 140,
    "n_val_episodes": 2,
    "log_every": 20,
    "save_every": 140,
    "save_replay": True,
    "sac": {"batch_size": 64, "utd": 2},
}


def _apply(cfg: dict, args: argparse.Namespace) -> dict:
    if args.smoke:
        for key, value in SMOKE.items():
            if key == "sac":
                cfg["sac"].update(value)
            else:
                cfg[key] = value
    mapping = {
        "n_envs": args.n_envs,
        "total_env_steps": args.total_env_steps,
        "init_rollout_steps": args.init_rollout_steps,
        "val_every": args.val_every,
        "n_val_episodes": args.n_val_episodes,
        "bc_checkpoint": args.bc_checkpoint,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    if args.batch_size is not None:
        cfg["sac"]["batch_size"] = int(args.batch_size)
    if args.utd is not None:
        cfg["sac"]["utd"] = int(args.utd)
    if args.no_val_at_end:
        cfg["val_at_end"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m7_dsrl.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--init-rollout-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--n-val-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--utd", type=int, default=None)
    parser.add_argument("--no-val-at-end", action="store_true")
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--benchmark-envs", default="1,4,8")
    parser.add_argument("--benchmark-updates", type=int, default=10)
    args = parser.parse_args()
    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = _apply(cfg, args)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    bc = Path(cfg["bc_checkpoint"])
    if not bc.is_absolute():
        bc = root / bc
    cfg["bc_checkpoint"] = str(bc)
    cfg["bc_checkpoint_resolved"] = str(bc)
    out = Path(args.out) if args.out else root / "outputs" / "m7" / "train"
    if not out.is_absolute():
        out = root / out
    assert_m7_output(out, root)
    out.mkdir(parents=True, exist_ok=True)
    bc_hash = sha256_file(bc)
    print(f"bc {bc} sha256 {bc_hash} device {device}", flush=True)
    policy, _blob = load_checkpoint(bc, device)
    policy.eval()
    policy.requires_grad_(False)
    cfg["n_obs_steps"] = int(policy.n_obs_steps)
    cfg["pred_horizon"] = int(policy.pred_horizon)
    cfg["exec_horizon"] = int(policy.exec_horizon)
    cfg["action_dim"] = int(policy.action_dim)
    feat_dim = int(policy.actor_feature_dim())
    action_dim = int(np.prod(policy.noise_shape))
    print(
        f"noise_shape {tuple(policy.noise_shape)} actor_feature {feat_dim} "
        f"privileged {privileged_dim()} latent {action_dim}",
        flush=True,
    )
    if args.benchmark:
        counts = [int(part) for part in args.benchmark_envs.split(",") if part]
        rows = benchmark_collection(policy, cfg, counts, int(args.benchmark_steps), device)
        torch.manual_seed(int(cfg["seed"]))
        actor, agent = build_models(feat_dim, action_dim, cfg, device)
        replay = Replay(max(int(cfg["sac"]["batch_size"]), 64), feat_dim, privileged_dim(), action_dim)
        updates = benchmark_updates(agent, replay, int(cfg["sac"]["batch_size"]), int(args.benchmark_updates), device)
        recommend = max(rows, key=lambda row: row["transitions_per_s"])
        estimate = estimate_seconds(
            total_env_steps=int(cfg["total_env_steps"]),
            n_envs=int(recommend["n_envs"]),
            vec_step_s=float(recommend["vec_step_s"]),
            update_s=float(updates["update_s"]),
            utd=int(cfg["sac"]["utd"]),
            init_rollout_steps=int(cfg["init_rollout_steps"]),
            val_every=int(cfg["val_every"]),
            n_val=int(cfg["n_val_episodes"]),
            val_rl_steps=40,
        )
        payload = {"collection": rows, "updates": updates, "recommended_n_envs": recommend["n_envs"], "estimate": estimate}
        bench_path = out / "benchmark.json" if args.smoke or args.out else root / "outputs" / "m7" / "benchmark.json"
        assert_m7_output(bench_path, root)
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        print(f"wrote {bench_path}", flush=True)
        return
    resumed = None
    if args.resume:
        resume = Path(args.resume)
        if not resume.is_absolute():
            resume = root / resume
        actor, agent, resumed = load_resumed(resume, device, bc_hash)
        if int(resumed["feat_dim"]) != feat_dim or int(resumed["action_dim"]) != action_dim:
            raise SystemExit("resume actor shape does not match the BC policy")
        assert_resume_config(resumed["config"], cfg)
    else:
        torch.manual_seed(int(cfg["seed"]))
        np.random.seed(int(cfg["seed"]))
        actor, agent = build_models(feat_dim, action_dim, cfg, device)
    if int(actor.feat_dim) != feat_dim:
        raise SystemExit("actor feature width does not match encode_actor_obs")
    summary = train(cfg, policy=policy, actor=actor, agent=agent, out_dir=out, device=device, live_viewer=bool(args.live_viewer), resumed=resumed)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
