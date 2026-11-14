"""Execute ``exec_horizon`` steps from a predicted chunk and preview the next one.

Step ``k`` (0-based) commands chunk ``k`` and previews chunk ``k + 1``.
The 8th executed step (``k == 7``) therefore previews chunk index 8, the 9th step.
"""

from __future__ import annotations

import numpy as np


class ChunkExecutor:
    def __init__(self, pred_horizon: int = 16, exec_horizon: int = 8, action_dim: int = 8) -> None:
        if pred_horizon < exec_horizon + 1:
            raise ValueError("the chunk must contain the preview step after the last executed step")
        self.pred_horizon = int(pred_horizon)
        self.exec_horizon = int(exec_horizon)
        self.action_dim = int(action_dim)
        self.chunk: np.ndarray | None = None
        self.k = 0

    def reset(self) -> None:
        self.chunk = None
        self.k = 0

    def needs_chunk(self) -> bool:
        return self.chunk is None or self.k >= self.exec_horizon

    def push(self, chunk: np.ndarray) -> None:
        array = np.asarray(chunk, dtype=np.float32)
        if array.shape != (self.pred_horizon, self.action_dim):
            raise ValueError(f"chunk shape {array.shape} != {(self.pred_horizon, self.action_dim)}")
        self.chunk = array
        self.k = 0

    def pop(self) -> tuple[np.ndarray, np.ndarray]:
        if self.chunk is None or self.k >= self.exec_horizon:
            raise RuntimeError("push a chunk before pop")
        action = self.chunk[self.k]
        nxt = self.chunk[self.k + 1]
        self.k += 1
        return action, nxt
