"""Subprocess chunk environments, one EGL context each.

The parent holds the frozen policy and batches the N observations. Workers
only step physics and rendering. ``spawn`` is used so a CUDA context in the
parent is not inherited. ``MUJOCO_GL`` is set before MuJoCo is imported in
the child.
"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import multiprocessing as mp

import numpy as np


def _worker_main(conn, setup: dict) -> None:
    os.environ["MUJOCO_GL"] = "egl"
    try:
        from aloha_pp.rl.chunk_env import make_chunk_env

        env = make_chunk_env(
            timeout_s=float(setup["timeout_s"]),
            staged=setup["staged"],
            n_obs=int(setup["n_obs"]),
            pred_horizon=int(setup["pred_horizon"]),
            exec_horizon=int(setup["exec_horizon"]),
            action_dim=int(setup["action_dim"]),
            trace=bool(setup["trace"]),
            record=False,
            video_fps=int(setup["video_fps"]),
            live_viewer=False,
        )
        conn.send({"ok": True, "ready": True})
        while True:
            cmd = conn.recv()
            kind = cmd["cmd"]
            if kind == "close":
                env.close()
                conn.send({"ok": True, "closed": True})
                break
            if kind == "noop":
                conn.send({"ok": True, "noop": True})
                continue
            if kind == "reset":
                env.record = bool(cmd.get("record", False))
                reply = env.reset(seed=int(cmd["seed"]), mode=str(cmd.get("mode", "random")))
            elif kind == "step":
                reply = env.step(np.asarray(cmd["chunk"], dtype=np.float32))
            else:
                raise ValueError(f"unknown worker command {kind}")
            conn.send({"ok": True, "reply": reply})
    except Exception as exc:
        try:
            conn.send({"ok": False, "error": str(exc), "trace": traceback.format_exc()})
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _collate(replies: list[dict]) -> dict:
    return {
        "obs": {
            "agentview": np.stack([item["obs"]["agentview"] for item in replies]),
            "wrist_cam_left": np.stack([item["obs"]["wrist_cam_left"] for item in replies]),
            "tcp_pose": np.stack([item["obs"]["tcp_pose"] for item in replies]),
            "gripper_opening": np.stack([item["obs"]["gripper_opening"] for item in replies]),
        },
        "priv": np.stack([item["priv"] for item in replies]),
        "reward": np.asarray([item["reward"] for item in replies], dtype=np.float32),
        "terminated": np.asarray([item["terminated"] for item in replies], dtype=bool),
        "truncated": np.asarray([item["truncated"] for item in replies], dtype=bool),
        "info": [item["info"] for item in replies],
    }


class SerialChunkEnv:
    """In-process chunk environments. ``n=1`` is what ``--live-viewer`` uses."""

    def __init__(self, n: int, setup: dict, *, live_viewer: bool = False) -> None:
        from aloha_pp.rl.chunk_env import make_chunk_env

        self.n = int(n)
        self.envs = []
        for index in range(self.n):
            self.envs.append(make_chunk_env(
                timeout_s=float(setup["timeout_s"]),
                staged=setup["staged"],
                n_obs=int(setup["n_obs"]),
                pred_horizon=int(setup["pred_horizon"]),
                exec_horizon=int(setup["exec_horizon"]),
                action_dim=int(setup["action_dim"]),
                trace=bool(setup.get("trace", False)),
                record=False,
                video_fps=int(setup["video_fps"]),
                live_viewer=bool(live_viewer) and index == 0,
            ))

    def reset(self, seeds, mode: str = "random", record_index: int | None = None, indices: list[int] | None = None) -> dict:
        if indices is None:
            indices = list(range(len(seeds)))
        replies = []
        for index, seed in zip(indices, seeds):
            self.envs[int(index)].record = record_index is not None and int(index) == int(record_index)
            replies.append(self.envs[int(index)].reset(seed=int(seed), mode=mode))
        return _collate(replies)

    def step(self, chunks: np.ndarray, active: np.ndarray | None = None) -> dict:
        chunks = np.asarray(chunks, dtype=np.float32)
        n = int(chunks.shape[0])
        if active is None:
            active = np.ones(n, dtype=bool)
        replies = []
        for index in range(n):
            if bool(active[index]):
                replies.append(self.envs[index].step(np.asarray(chunks[index], dtype=np.float32)))
        if not replies:
            raise RuntimeError("step called with no active environments")
        return _collate(replies)

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.envs = []


class SubprocChunkEnv:
    """N chunk environments. ``reset`` and ``step`` talk to the first ``len(seeds)`` workers when fewer are active."""

    def __init__(self, n: int, setup: dict, *, timeout_s: float = 180.0) -> None:
        if int(n) < 1:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.timeout_s = float(timeout_s)
        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.conns = []
        self.procs = []
        for _index in range(self.n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child, setup), daemon=False)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        for conn in self.conns:
            self._recv(conn)

    def _recv(self, conn):
        if not conn.poll(self.timeout_s):
            raise TimeoutError("chunk-env worker did not reply")
        msg = conn.recv()
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("trace") or msg.get("error") or "worker failed")
        return msg

    def _exchange(self, commands: list[tuple[int, dict]]) -> list[dict]:
        for index, command in commands:
            self.conns[index].send(command)
        replies = []
        for index, _command in commands:
            msg = self._recv(self.conns[index])
            if "reply" not in msg:
                raise RuntimeError(f"worker {index} returned no step result: {msg}")
            replies.append(msg["reply"])
        return replies

    def reset(
        self,
        seeds: list[int],
        mode: str = "random",
        record_index: int | None = None,
        indices: list[int] | None = None,
    ) -> dict:
        if indices is None:
            indices = list(range(len(seeds)))
        if len(indices) != len(seeds):
            raise ValueError("seeds and indices must have the same length")
        commands = []
        for index, seed in zip(indices, seeds):
            commands.append((int(index), {
                "cmd": "reset",
                "seed": int(seed),
                "mode": mode,
                "record": record_index is not None and int(index) == int(record_index),
            }))
        return _collate(self._exchange(commands))

    def step(self, chunks: np.ndarray, active: np.ndarray | None = None) -> dict:
        chunks = np.asarray(chunks, dtype=np.float32)
        n = int(chunks.shape[0])
        if active is None:
            active = np.ones(n, dtype=bool)
        commands = []
        for index in range(n):
            if bool(active[index]):
                commands.append((index, {"cmd": "step", "chunk": np.ascontiguousarray(chunks[index])}))
        if not commands:
            raise RuntimeError("step called with no active environments")
        replies = self._exchange(commands)
        # Inactive envs are omitted. Callers pass a mask of the same length and only read active rows
        # when some are idle; the collated batch is in active-index order.
        return _collate(replies)

    def close(self) -> None:
        for conn, proc in zip(self.conns, self.procs):
            if proc.is_alive():
                try:
                    conn.send({"cmd": "close"})
                    if conn.poll(10.0):
                        conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        self.conns = []
        self.procs = []
