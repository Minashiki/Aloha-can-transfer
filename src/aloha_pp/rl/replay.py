"""Fixed-capacity replay of DSRL transitions.

Stored per step: actor feature, privileged vector, latent action, reward,
next feature, next privileged vector, terminated, truncated. Features are
stored so the update does not re-encode images. Sampling is with replacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Replay:
    def __init__(
        self,
        capacity: int,
        feat_dim: int,
        priv_dim: int,
        action_dim: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if int(capacity) < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.feat_dim = int(feat_dim)
        self.priv_dim = int(priv_dim)
        self.action_dim = int(action_dim)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.feat = np.zeros((self.capacity, self.feat_dim), dtype=np.float32)
        self.priv = np.zeros((self.capacity, self.priv_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_feat = np.zeros((self.capacity, self.feat_dim), dtype=np.float32)
        self.next_priv = np.zeros((self.capacity, self.priv_dim), dtype=np.float32)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        feat: np.ndarray,
        priv: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_feat: np.ndarray,
        next_priv: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        feat = np.asarray(feat, dtype=np.float32)
        n = int(feat.shape[0])
        priv = np.asarray(priv, dtype=np.float32).reshape(n, self.priv_dim)
        action = np.asarray(action, dtype=np.float32).reshape(n, self.action_dim)
        reward = np.asarray(reward, dtype=np.float32).reshape(n, 1)
        next_feat = np.asarray(next_feat, dtype=np.float32).reshape(n, self.feat_dim)
        next_priv = np.asarray(next_priv, dtype=np.float32).reshape(n, self.priv_dim)
        terminated = np.asarray(terminated, dtype=np.float32).reshape(n, 1)
        truncated = np.asarray(truncated, dtype=np.float32).reshape(n, 1)
        for i in range(n):
            j = self.pos
            self.feat[j] = feat[i]
            self.priv[j] = priv[i]
            self.action[j] = action[i]
            self.reward[j] = reward[i]
            self.next_feat[j] = next_feat[i]
            self.next_priv[j] = next_priv[i]
            self.terminated[j] = terminated[i]
            self.truncated[j] = truncated[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if self.size < 1:
            raise RuntimeError("replay is empty")
        index = self.rng.integers(0, self.size, size=int(batch))
        def take(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[index], device=device)
        return {
            "feat": take(self.feat),
            "priv": take(self.priv),
            "action": take(self.action),
            "reward": take(self.reward),
            "next_feat": take(self.next_feat),
            "next_priv": take(self.next_priv),
            "terminated": take(self.terminated),
            "truncated": take(self.truncated),
        }

    def state_dict(self) -> dict:
        return {
            "feat": self.feat[: self.size].copy(),
            "priv": self.priv[: self.size].copy(),
            "action": self.action[: self.size].copy(),
            "reward": self.reward[: self.size].copy(),
            "next_feat": self.next_feat[: self.size].copy(),
            "next_priv": self.next_priv[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "capacity": self.capacity,
            "feat_dim": self.feat_dim,
            "priv_dim": self.priv_dim,
            "action_dim": self.action_dim,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity does not match the checkpoint")
        n = int(state["feat"].shape[0])
        if n > self.capacity:
            raise ValueError("replay checkpoint is larger than the capacity")
        self.feat[:n] = state["feat"]
        self.priv[:n] = state["priv"]
        self.action[:n] = state["action"]
        self.reward[:n] = state["reward"]
        self.next_feat[:n] = state["next_feat"]
        self.next_priv[:n] = state["next_priv"]
        self.terminated[:n] = state["terminated"]
        self.truncated[:n] = state["truncated"]
        self.size = n
        self.pos = n % self.capacity
        self.rng.bit_generator.state = state["rng"]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, rng: np.random.Generator | None = None) -> "Replay":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        replay = cls(
            int(state["capacity"]),
            int(state["feat_dim"]),
            int(state["priv_dim"]),
            int(state["action_dim"]),
            rng=rng,
        )
        replay.load_state_dict(state)
        return replay
