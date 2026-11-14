"""Per-dimension min/max map onto ``[-1, 1]``."""

from __future__ import annotations

import numpy as np


class LimitNormalizer:
    """Affine map fitted on a training array of shape ``(N, D)``."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> LimitNormalizer:
        data = np.asarray(values, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"expected (N, D), got {data.shape}")
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        hi = np.maximum(hi, lo + np.float32(self.eps))
        self.lo = lo.astype(np.float32)
        self.hi = hi.astype(np.float32)
        return self

    def normalize(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float32)
        return (2.0 * (data - self.lo) / (self.hi - self.lo) - 1.0).astype(np.float32)

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float32)
        return ((data + 1.0) * 0.5 * (self.hi - self.lo) + self.lo).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"lo": self.lo.copy(), "hi": self.hi.copy(), "eps": np.float32(self.eps)}
