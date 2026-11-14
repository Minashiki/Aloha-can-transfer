"""One RL step is one diffusion chunk: 8 control steps, previewing the 9th.

Success ends the episode (terminal). The 25 s cap ends it as a truncation,
which the SAC backup still bootstraps. The environment is the shared C3
assembly, and ``reset`` calls ``reset_controller_after_settle`` the same way
M6 eval does.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any

import numpy as np

from aloha_pp.policy.chunk import ChunkExecutor
from aloha_pp.policy.quaternion import to_env_action
from aloha_pp.rl.reward import StageTracker, step_reward
from aloha_pp.rl.spec import pack_privileged
from aloha_pp.sim.c3 import make_c3_env, reset_controller_after_settle


def stack_history(hist: list[dict]) -> dict[str, np.ndarray]:
    grip = [np.asarray(item["gripper_opening"], dtype=np.float32).reshape(1) for item in hist]
    return {
        "agentview": np.stack([item["agentview"] for item in hist]),
        "wrist_cam_left": np.stack([item["wrist_cam_left"] for item in hist]),
        "tcp_pose": np.stack([np.asarray(item["tcp_pose"], dtype=np.float32) for item in hist]),
        "gripper_opening": np.stack(grip),
    }


def _stage_from_config(staged: dict, m1: dict, m5: dict, mode: str) -> StageTracker:
    cell = m1["target_cells"][mode]
    rewards = {name: float(staged[name]) for name in ("both_contact", "lifted", "above_cell", "released_in_cell")}
    return StageTracker(
        enabled=bool(staged.get("enabled", False)),
        rewards=rewards,
        lift_thresh=float(m5["lift_thresh_m"]),
        cell_center=np.asarray(cell["center"], dtype=np.float64),
        cell_half=np.asarray(cell["half_extents"], dtype=np.float64),
    )


class ChunkEnv:
    """Executes a latent-noise chunk on one can-task environment."""

    def __init__(
        self,
        env,
        *,
        n_obs: int,
        pred_horizon: int,
        exec_horizon: int,
        action_dim: int,
        timeout_s: float,
        stages: StageTracker,
        gripper_close: float,
        grip_tol: float,
        trace: bool = False,
        record: bool = False,
        video_fps: int = 10,
        live_viewer: bool = False,
    ) -> None:
        self.env = env
        self.n_obs = int(n_obs)
        self.exec_horizon = int(exec_horizon)
        self.player = ChunkExecutor(int(pred_horizon), int(exec_horizon), int(action_dim))
        self.timeout_s = float(timeout_s)
        hz = float(env.cfg["control_hz"])
        self.hz = hz
        self.limit = max(1, int(round(self.timeout_s * hz)))
        self.stages = stages
        self.gripper_close = float(gripper_close)
        self.grip_tol = float(grip_tol)
        self.trace = bool(trace)
        self.record = bool(record)
        self.video_every = max(1, int(round(hz / float(video_fps))))
        self.live_viewer = bool(live_viewer)
        self.viewer = None
        self.hist: list[dict] = []
        self.control_count = 0
        self.frames: list[np.ndarray] = []
        self._signals = self._blank_signals()
        self._open = False

    def _blank_signals(self) -> dict[str, Any]:
        return {
            "grasped": False,
            "contact_z": None,
            "max_after": None,
            "end_pos": np.zeros(3, dtype=np.float64),
            "end_released": True,
        }

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _observe_substep(self, live, closed_cmd: bool) -> None:
        contacts = np.asarray(live.finger_can_contacts(), dtype=bool).reshape(2)
        pos = np.asarray(live.can_center_base(), dtype=np.float64).reshape(3)
        both = bool(contacts[0] and contacts[1]) and closed_cmd
        already = self.stages.paid["both_contact"]
        if both and self._signals["contact_z"] is None and not already:
            self._signals["contact_z"] = float(pos[2])
            self._signals["max_after"] = float(pos[2])
        if both:
            self._signals["grasped"] = True
        if self._signals["contact_z"] is not None or already:
            height = float(pos[2])
            previous = self._signals["max_after"]
            self._signals["max_after"] = height if previous is None else max(float(previous), height)
        self._signals["end_pos"] = pos
        self._signals["end_released"] = not bool(contacts[0] or contacts[1])

    def _result(self, reward: float, terminated: bool, truncated: bool, n_control: int, trace: dict) -> dict:
        latest = self.hist[-1]
        frames = None
        if (terminated or truncated) and self.frames:
            frames = np.stack(self.frames)
            self.frames = []
        info = {"n_control": int(n_control), "control_count": int(self.control_count), "frames": frames}
        info.update(trace)
        return {
            "obs": stack_history(self.hist),
            "priv": pack_privileged(latest, self.env.get_privileged()),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }

    def reset(self, seed: int, mode: str = "random") -> dict:
        self.player.reset()
        self.stages.reset()
        self.control_count = 0
        self.frames = []
        self._signals = self._blank_signals()
        obs = self.env.reset(seed=int(seed), mode=mode, render=True)
        reset_controller_after_settle(self.env)
        if self.live_viewer and self.viewer is None:
            import mujoco

            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.hist = [obs] * self.n_obs
        self._open = True
        return self._result(0.0, False, False, 0, {})

    def step(self, chunk: np.ndarray) -> dict:
        if not self._open:
            raise RuntimeError("reset the chunk env before stepping a finished episode")
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        self.player.push(array)
        self._signals = self._blank_signals()
        actions = []
        next_actions = []
        tcp_rows = []
        q_rows = []
        can_rows = []
        n_control = 0
        terminated = False
        while n_control < self.exec_horizon and self.control_count < self.limit and not terminated:
            if self.record and self.control_count % self.video_every == 0:
                self.frames.append(self.env.render("agentview", 256, 256))
            action, nxt = self.player.pop()
            action = to_env_action(action)
            nxt = to_env_action(nxt)
            closed_cmd = abs(float(action[7]) - self.gripper_close) <= self.grip_tol

            def on_substep(live, _closed=closed_cmd) -> None:
                self._observe_substep(live, _closed)

            obs, success, _info = self.env.step(action, on_substep=on_substep, render=True, next_action=nxt)
            self.hist = (self.hist + [obs])[-self.n_obs :]
            self.control_count += 1
            n_control += 1
            if self.trace:
                actions.append(np.asarray(action, dtype=np.float64).copy())
                next_actions.append(np.asarray(nxt, dtype=np.float64).copy())
                tcp_rows.append(np.asarray(obs["tcp_pose"], dtype=np.float64).reshape(7).copy())
                q_rows.append(np.asarray(obs["q"], dtype=np.float64).reshape(-1).copy())
                can_pos, can_quat = self.env.can_pose_base()
                can_rows.append(np.concatenate([np.asarray(can_pos, dtype=np.float64), np.asarray(can_quat, dtype=np.float64)]))
            if self.viewer is not None:
                self.viewer.sync()
            if success:
                terminated = True
        truncated = (not terminated) and self.control_count >= self.limit
        if n_control == 0:
            bonus = 0.0
        else:
            bonus = self.stages.update(
                grasped_contact=bool(self._signals["grasped"]),
                contact_z=self._signals["contact_z"],
                max_z_after_contact=self._signals["max_after"],
                end_pos=self._signals["end_pos"],
                end_released=bool(self._signals["end_released"]),
            )
        reward = step_reward(terminated, bonus, staged=self.stages.enabled)
        if terminated or truncated:
            self._open = False
        trace = {}
        if self.trace and n_control:
            trace = {
                "actions": np.stack(actions),
                "next_actions": np.stack(next_actions),
                "tcp": np.stack(tcp_rows),
                "q": np.stack(q_rows),
                "can_pose": np.stack(can_rows),
            }
        return self._result(reward, terminated, truncated, n_control, trace)


def make_chunk_env(
    *,
    timeout_s: float,
    staged: dict,
    n_obs: int = 2,
    pred_horizon: int = 16,
    exec_horizon: int = 8,
    action_dim: int = 8,
    trace: bool = False,
    record: bool = False,
    video_fps: int = 10,
    live_viewer: bool = False,
    mode_for_cell: str = "random",
) -> ChunkEnv:
    """C3 can task wrapped as a chunk environment. ``mode_for_cell`` picks the stage cell."""
    from aloha_pp.policy.rollout import task_configs

    m1, m3, m4, m5 = task_configs()
    env = make_c3_env(m1, m3, m4)
    stages = _stage_from_config(staged, m1, m5, mode_for_cell)
    return ChunkEnv(
        env,
        n_obs=n_obs,
        pred_horizon=pred_horizon,
        exec_horizon=exec_horizon,
        action_dim=action_dim,
        timeout_s=timeout_s,
        stages=stages,
        gripper_close=float(m1["gripper_close"]),
        grip_tol=float(m5["grip_closed_tol"]),
        trace=trace,
        record=record,
        video_fps=video_fps,
        live_viewer=live_viewer,
    )
