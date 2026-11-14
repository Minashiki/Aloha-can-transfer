"""Fixed-seed demo split and padded observation / action windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aloha_pp.data.dataset import list_demos, load_episode
from aloha_pp.policy.normalizer import LimitNormalizer
from aloha_pp.policy.quaternion import canonicalize_quaternions, pose_state


def select_indices(n_total: int, n_train: int, seed: int) -> np.ndarray:
    """Sorted indices of the training split. ``seed`` alone determines the draw."""
    if n_train < 1 or n_train > n_total:
        raise ValueError(f"n_train must be in 1..{n_total}, got {n_train}")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_total, size=int(n_train), replace=False))


def observation_window(series: np.ndarray, t: int, n_obs: int) -> np.ndarray:
    """``n_obs`` frames ending at ``t``. Missing early frames repeat index 0."""
    idx = [max(0, t - n_obs + 1 + k) for k in range(n_obs)]
    return np.stack([series[i] for i in idx], axis=0)


def action_window(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """``horizon`` actions starting at ``t``. Missing tail frames repeat the last action."""
    last = actions.shape[0] - 1
    idx = [min(t + k, last) for k in range(horizon)]
    return np.stack([actions[i] for i in idx], axis=0)


def _pack_episode(episode: dict, name: str) -> tuple[dict, dict]:
    actions = np.array(episode["actions"], dtype=np.float32, copy=True)
    tcp = np.array(episode["obs"]["tcp_pose"], dtype=np.float32, copy=True)
    grip = np.asarray(episode["obs"]["gripper_opening"], dtype=np.float32).reshape(-1, 1)
    actions[:, 3:7], action_quat = canonicalize_quaternions(actions[:, 3:7])
    state = pose_state(tcp, grip)
    _quat, obs_quat = canonicalize_quaternions(tcp[:, 3:7])
    packed = {
        "name": name,
        "source_id": int(episode["source_id"]),
        "agentview": np.asarray(episode["obs"]["agentview"]),
        "wrist": np.asarray(episode["obs"]["wrist_cam_left"]),
        "state": state,
        "action": actions,
    }
    return packed, {"action_quat": action_quat, "obs_quat": obs_quat}


class WindowDataset(Dataset):
    def __init__(self, episodes: list[dict], n_obs: int, horizon: int) -> None:
        self.episodes = episodes
        self.n_obs = int(n_obs)
        self.horizon = int(horizon)
        self.index: list[tuple[int, int]] = []
        for i, episode in enumerate(episodes):
            for t in range(episode["action"].shape[0]):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        which, t = self.index[item]
        episode = self.episodes[which]
        return {
            "agentview": observation_window(episode["agentview"], t, self.n_obs),
            "wrist_cam_left": observation_window(episode["wrist"], t, self.n_obs),
            "state": observation_window(episode["state"], t, self.n_obs),
            "action": action_window(episode["action"], t, self.horizon),
        }


def collate_windows(items: list[dict]) -> dict[str, torch.Tensor]:
    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([torch.as_tensor(item[key]) for item in items]).to(dtype)

    return {
        "agentview": stack("agentview", torch.uint8),
        "wrist_cam_left": stack("wrist_cam_left", torch.uint8),
        "state": stack("state", torch.float32),
        "action": stack("action", torch.float32),
    }


def load_splits(
    path: Path,
    *,
    n_train: int,
    seed: int,
    n_obs: int,
    horizon: int,
    range_eps: float = 1e-6,
) -> dict:
    """Load demos, apply the ``w >= 0`` quaternion rule, fit limits, and window them.

    The quaternion rule does not depend on the previous frame. Normalization
    statistics use the training demos only, after that rule and before end
    padding. Returned episode ``state`` and ``action`` arrays are already
    normalized.
    """
    path = Path(path)
    names = list_demos(path)
    chosen = select_indices(len(names), n_train, seed)
    chosen_set = set(int(i) for i in chosen)
    train_names = [names[i] for i in chosen]
    val_names = [names[i] for i in range(len(names)) if i not in chosen_set]
    train_eps = []
    val_eps = []
    quat_stats = {"action_hemisphere": 0, "action_continuity": 0, "obs_hemisphere": 0, "obs_continuity": 0}
    print(f"loading {len(train_names)} train and {len(val_names)} val demos from {path}", flush=True)

    def take(name: str) -> dict:
        packed, info = _pack_episode(load_episode(path, name), name)
        quat_stats["action_hemisphere"] += info["action_quat"]["hemisphere"]
        quat_stats["action_continuity"] += info["action_quat"]["continuity"]
        quat_stats["obs_hemisphere"] += info["obs_quat"]["hemisphere"]
        quat_stats["obs_continuity"] += info["obs_quat"]["continuity"]
        done = len(train_eps) + len(val_eps) + 1
        if done % 20 == 0 or done == len(train_names) + len(val_names):
            print(f"loaded {done}/{len(train_names) + len(val_names)}", flush=True)
        return packed

    for name in train_names:
        train_eps.append(take(name))
    for name in val_names:
        val_eps.append(take(name))

    action_fit = LimitNormalizer(range_eps).fit(np.concatenate([ep["action"] for ep in train_eps], axis=0))
    state_fit = LimitNormalizer(range_eps).fit(np.concatenate([ep["state"] for ep in train_eps], axis=0))
    for episode in train_eps + val_eps:
        episode["action"] = action_fit.normalize(episode["action"])
        episode["state"] = state_fit.normalize(episode["state"])
    return {
        "train": WindowDataset(train_eps, n_obs, horizon),
        "val": WindowDataset(val_eps, n_obs, horizon),
        "action_normalizer": action_fit,
        "state_normalizer": state_fit,
        "train_names": train_names,
        "val_names": val_names,
        "train_source_ids": [int(ep["source_id"]) for ep in train_eps],
        "quat_stats": quat_stats,
    }
