"""DSRL checkpoint: actor, critics, temperature, optimizers, and the BC stamp."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from aloha_pp.policy.checkpoint import git_stamp, rng_state, set_rng
from aloha_pp.rl.nets import SquashedGaussianActor, TwinCritic
from aloha_pp.rl.sac import SAC
from aloha_pp.rl.spec import privileged_dim


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_models(feat_dim: int, action_dim: int, cfg: dict, device: torch.device) -> tuple[SquashedGaussianActor, SAC]:
    sac_cfg = cfg["sac"]
    actor = SquashedGaussianActor(
        feat_dim,
        action_dim,
        hidden=int(sac_cfg["layer_size"]),
        n_layers=int(sac_cfg["num_layers"]),
        noise_scale=float(cfg["noise_scale"]),
        log_std_init=float(sac_cfg["log_std_init"]),
        log_std_min=float(sac_cfg["log_std_min"]),
        log_std_max=float(sac_cfg["log_std_max"]),
        use_layer_norm=bool(sac_cfg["use_layer_norm"]),
    ).to(device)
    critic = TwinCritic(
        privileged_dim(),
        feat_dim,
        action_dim,
        hidden=int(sac_cfg["layer_size"]),
        n_layers=int(sac_cfg["num_layers"]),
        use_layer_norm=bool(sac_cfg["use_layer_norm"]),
    ).to(device)
    agent = SAC(
        actor,
        critic,
        lr=float(sac_cfg["actor_lr"]),
        gamma=float(sac_cfg["discount"]),
        tau=float(sac_cfg["tau"]),
        target_entropy=float(sac_cfg["target_entropy"]),
    )
    return actor, agent


def save_training(path: Path, *, actor, agent, cfg: dict, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = git_stamp()
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.target.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "actor_opt": agent.actor_opt.state_dict(),
            "critic_opt": agent.critic_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
            "config": cfg,
            "git_commit": stamp["git_commit"],
            "git_dirty": stamp["git_dirty"],
            "rng": rng_state(),
            **extra,
        },
        path,
    )


def _cpu_byte(value) -> torch.Tensor:
    """``torch.save`` plus ``map_location=cuda`` turns the RNG state into a CUDA tensor.

    ``torch.set_rng_state`` only accepts a CPU ``ByteTensor``.
    """
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = value
    return torch.tensor(np.asarray(array), dtype=torch.uint8)


def restore_rng(state: dict) -> None:
    fixed = dict(state)
    fixed["torch"] = _cpu_byte(fixed["torch"])
    if fixed.get("cuda") is not None:
        fixed["cuda"] = [_cpu_byte(item) for item in fixed["cuda"]]
    numpy_state = fixed.get("numpy")
    if numpy_state is not None and torch.is_tensor(numpy_state[1]):
        array = numpy_state[1].detach().cpu().numpy()
        fixed["numpy"] = (numpy_state[0], array, *numpy_state[2:])
    set_rng(fixed)


def load_actor(path: Path, device: torch.device | str) -> tuple[SquashedGaussianActor, dict]:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = blob["config"]
    actor, _agent = build_models(int(blob["feat_dim"]), int(blob["action_dim"]), cfg, torch.device(device))
    actor.load_state_dict(blob["actor"])
    actor.eval()
    return actor, blob


# Compared on resume. n_envs is allowed to change. total_env_steps, val_every,
# and save_replay are not in this set: the run length and the replay file are
# operational, not the algorithm.
_RESUME_KEYS = (
    "noise_scale",
    "seed",
    "train_seed_start",
    "val_seed_start",
    "n_val_episodes",
    "eval_seed_start",
    "eval_seed_count",
)
_RESUME_SAC_KEYS = (
    "actor_lr",
    "batch_size",
    "tau",
    "discount",
    "utd",
    "layer_size",
    "num_layers",
    "use_layer_norm",
    "n_critics",
    "target_entropy",
    "log_std_init",
    "log_std_min",
    "log_std_max",
    "critic_include_actor_feature",
)


def _same_config_value(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0.0)
    return left == right


def assert_resume_config(saved: dict, current: dict) -> None:
    """Refuse to resume when the frozen algorithm settings differ.

    ``noise_scale``, the SAC hyperparameters, the network width and depth, and
    the seed ranges must match the checkpoint. ``n_envs`` is not compared.
    """
    mismatches = []
    for key in _RESUME_KEYS:
        if not _same_config_value(saved.get(key), current.get(key)):
            mismatches.append(f"{key}: checkpoint {saved.get(key)!r} current {current.get(key)!r}")
    saved_sac = saved.get("sac") or {}
    current_sac = current.get("sac") or {}
    for key in _RESUME_SAC_KEYS:
        if not _same_config_value(saved_sac.get(key), current_sac.get(key)):
            mismatches.append(f"sac.{key}: checkpoint {saved_sac.get(key)!r} current {current_sac.get(key)!r}")
    if mismatches:
        raise SystemExit("resume refused, immutable config differs:\n" + "\n".join(mismatches))


def load_training(path: Path, device: torch.device | str) -> tuple[SquashedGaussianActor, SAC, dict]:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = blob["config"]
    actor, agent = build_models(int(blob["feat_dim"]), int(blob["action_dim"]), cfg, torch.device(device))
    actor.load_state_dict(blob["actor"])
    agent.critic.load_state_dict(blob["critic"])
    agent.target.load_state_dict(blob["critic_target"])
    with torch.no_grad():
        agent.log_alpha.copy_(blob["log_alpha"].to(agent.log_alpha.device))
    agent.actor_opt.load_state_dict(blob["actor_opt"])
    agent.critic_opt.load_state_dict(blob["critic_opt"])
    agent.alpha_opt.load_state_dict(blob["alpha_opt"])
    if blob.get("rng") is not None:
        restore_rng(blob["rng"])
    return actor, agent, blob
