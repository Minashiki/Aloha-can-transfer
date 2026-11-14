"""Closed-loop rollout and the eval log M7 reuses (``aloha_pp.eval.v1``)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml

from aloha_pp.planning.metrics import placement_errors, wilson_interval
from aloha_pp.policy.chunk import ChunkExecutor
from aloha_pp.policy.failure import classify_physics
from aloha_pp.policy.quaternion import to_env_action
from aloha_pp.robot import repo_root
from aloha_pp.sim.c3 import make_c3_env, reset_controller_after_settle
from aloha_pp.sim.env import load_config

EVAL_FORMAT = "aloha_pp.eval.v1"


def _yaml(name: str) -> dict:
    with (repo_root() / "configs" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def task_configs() -> tuple[dict, dict, dict, dict]:
    return load_config(), _yaml("m3_track.yaml"), _yaml("m4_task.yaml"), _yaml("m5_retarget.yaml")


def write_video(frames: list[np.ndarray], path: Path, fps: int, max_mb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(max_mb * 1e6)
    chosen = list(frames)
    stride = 1
    while chosen:
        for crf in (28, 32, 36, 40, 44):
            writer = imageio.get_writer(
                path, fps=max(1, fps // stride), codec="libx264", ffmpeg_params=["-crf", str(crf)]
            )
            try:
                for frame in chosen:
                    writer.append_data(frame)
            finally:
                writer.close()
            if path.stat().st_size <= limit:
                return
        stride *= 2
        chosen = frames[::stride]
        if len(chosen) < 2:
            break
    raise RuntimeError(f"video {path} is {path.stat().st_size} bytes, above {limit}")


def _stack_hist(hist: list[dict]) -> dict:
    grip = [np.asarray(item["gripper_opening"], dtype=np.float32).reshape(-1) for item in hist]
    return {
        "agentview": np.stack([item["agentview"] for item in hist]),
        "wrist_cam_left": np.stack([item["wrist_cam_left"] for item in hist]),
        "tcp_pose": np.stack([np.asarray(item["tcp_pose"], dtype=np.float32) for item in hist]),
        "gripper_opening": np.stack(grip),
    }


def render_environment(env=None) -> dict:
    """MuJoCo build, GL backend, and the GPU that produced the pixels."""
    from OpenGL import GL

    created = env is None
    if created:
        m1, m3, m4, _m5 = task_configs()
        env = make_c3_env(m1, m3, m4)
        env.reset(seed=0, render=False)
    try:
        renderer = env._get_renderer(int(env.image_height), int(env.image_width))
        renderer._gl_context.make_current()

        def text(value) -> str:
            if value is None:
                return ""
            if isinstance(value, (bytes, bytearray)):
                return value.decode()
            return str(value)

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        driver = ""
        try:
            line = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip().splitlines()[0].strip()
            driver = line
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver = ""
        return {
            "mujoco": mujoco.__version__,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", ""),
            "gl_renderer": text(GL.glGetString(GL.GL_RENDERER)),
            "gl_vendor": text(GL.glGetString(GL.GL_VENDOR)),
            "gpu": gpu,
            "driver": driver,
        }
    finally:
        if created:
            env.close()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rollout_episode(
    env,
    policy,
    *,
    seed: int,
    mode: str,
    timeout_s: float,
    live_viewer: bool,
    record: bool,
    video_path: Path | None,
    video_fps: int,
    video_max_mb: float,
    m1: dict,
    m4: dict,
    m5: dict,
) -> dict:
    """One episode. A new chunk is sampled every ``exec_horizon`` control steps."""
    device = policy.action_lo.device
    n_obs = int(policy.n_obs_steps)
    hz = float(env.cfg["control_hz"])
    limit = max(1, int(round(float(timeout_s) * hz)))
    exec_horizon = int(getattr(policy, "exec_horizon", 8))
    player = ChunkExecutor(policy.pred_horizon, exec_horizon, policy.action_dim)
    obs = env.reset(seed=int(seed), mode=mode, render=True)
    reset_controller_after_settle(env)
    hist = [obs] * n_obs
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if live_viewer else None
    contact: list[np.ndarray] = []
    cans: list[np.ndarray] = []
    vels: list[np.ndarray] = []
    closed_flags: list[bool] = []
    holder = {"closed": False}
    infer: list[float] = []
    frames: list[np.ndarray] = []
    ik_current_fail = 0
    ik_preview_fail = 0
    ik_either = 0
    success = False
    t0 = float(env.data.time)
    steps = 0
    close = np.float32(m1["gripper_close"])
    tol = np.float32(m5["grip_closed_tol"])
    video_every = max(1, int(round(hz / float(video_fps))))

    def on_substep(live) -> None:
        contact.append(live.finger_can_contacts().copy())
        cans.append(live.can_center_base().copy())
        vels.append(live.can_linvel().copy())
        closed_flags.append(bool(holder["closed"]))

    try:
        for step_index in range(limit):
            if record and step_index % video_every == 0:
                frames.append(env.render("agentview", 256, 256))
            if player.needs_chunk():
                _sync(device)
                started = time.perf_counter()
                chunk = policy.sample(_stack_hist(hist))
                _sync(device)
                infer.append(time.perf_counter() - started)
                player.push(chunk[0].detach().cpu().numpy())
            action, nxt = player.pop()
            action = to_env_action(action)
            nxt = to_env_action(nxt)
            holder["closed"] = abs(np.float32(action[7]) - close) <= tol
            obs, success, info = env.step(action, on_substep=on_substep, render=True, next_action=nxt)
            hist = (hist + [obs])[-n_obs:]
            current_bad = not bool(info.get("ik_current_success", True))
            preview_bad = bool(info.get("ik_preview_present", False)) and not bool(info.get("ik_preview_success", True))
            ik_current_fail += int(current_bad)
            ik_preview_fail += int(preview_bad)
            ik_either += int(current_bad or preview_bad)
            steps += 1
            if viewer is not None:
                viewer.sync()
            if success:
                break
    finally:
        if viewer is not None:
            viewer.close()

    can = np.stack(cans)
    cell = m1["target_cells"][mode]
    center = np.asarray(cell["center"], dtype=float)
    half = np.asarray(cell["half_extents"], dtype=float)
    place = np.asarray(m1.get("modes", {}).get(mode, {}).get("place_pos", center), dtype=float)
    reason = classify_physics(
        success=bool(success),
        contact=np.stack(contact),
        closed=np.asarray(closed_flags, dtype=bool),
        can=can,
        vel=np.stack(vels),
        center=center,
        half_extents=half,
        vel_thresh=float(m1["success"]["vel_thresh"]),
        drop_z=float(m4["drop_z"]),
        lift_thresh=float(m5["lift_thresh_m"]),
        lost_steps=int(m5["lost_physics_steps"]),
    )
    placed = placement_errors(can[-1], center, place)
    video = None
    if record and frames and video_path is not None:
        write_video(frames, video_path, int(video_fps), float(video_max_mb))
        video = str(video_path)
    infer_arr = np.asarray(infer, dtype=float)
    return {
        "seed": int(seed),
        "mode": mode,
        "success": bool(success),
        "failure": reason,
        "ik_current_fail_ticks": int(ik_current_fail),
        "ik_preview_fail_ticks": int(ik_preview_fail),
        "ik_fail_ticks": int(ik_either),
        "ik_fail_fraction": float(ik_either) / float(steps) if steps else 0.0,
        "ik_any": bool(ik_current_fail or ik_preview_fail),
        "place_xy": placed["xy"],
        "place_xyz": placed["xyz"],
        "place_vs_place_pos": placed["place"],
        "duration_s": float(steps / hz),
        "sim_time_s": float(env.data.time - t0),
        "n_control_steps": int(steps),
        "timed_out": bool(steps >= limit and not success),
        "infer_s": [float(v) for v in infer_arr],
        "infer_s_median": float(np.median(infer_arr)) if infer_arr.size else None,
        "infer_s_max": float(np.max(infer_arr)) if infer_arr.size else None,
        "n_chunks": int(infer_arr.size),
        "video": video,
    }


def summarize_episodes(episodes: list[dict], *, method: str, checkpoint: str, timeout_s: float) -> dict:
    n = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    low, high = wilson_interval(successes, n)
    counts = Counter(ep["failure"] for ep in episodes if ep["failure"])
    xy = np.asarray([ep["place_xy"] for ep in episodes], dtype=float)
    dur = np.asarray([ep["duration_s"] for ep in episodes], dtype=float)
    chunks = [v for ep in episodes for v in ep["infer_s"]]
    chunk_arr = np.asarray(chunks, dtype=float)
    current_ticks = [int(ep.get("ik_current_fail_ticks", 0)) for ep in episodes]
    preview_ticks = [int(ep.get("ik_preview_fail_ticks", 0)) for ep in episodes]
    either_ticks = [int(ep.get("ik_fail_ticks", 0)) for ep in episodes]
    return {
        "format": EVAL_FORMAT,
        "method": method,
        "checkpoint": checkpoint,
        "episode_timeout_s": float(timeout_s),
        "n": n,
        "n_success": int(successes),
        "success_rate": float(successes) / float(n),
        "wilson_95": [low, high],
        "failure_counts": dict(counts),
        "ik_episodes": int(sum(1 for ep in episodes if ep.get("ik_any"))),
        "ik_current_fail_ticks_sum": int(sum(current_ticks)),
        "ik_preview_fail_ticks_sum": int(sum(preview_ticks)),
        "ik_fail_ticks_sum": int(sum(either_ticks)),
        "place_xy_mean": float(np.mean(xy)),
        "place_xy_median": float(np.median(xy)),
        "duration_s_median": float(np.median(dur)),
        "infer_s_median": float(np.median(chunk_arr)) if chunk_arr.size else None,
        "infer_s_max": float(np.max(chunk_arr)) if chunk_arr.size else None,
        "episodes": episodes,
    }


def write_eval_log(summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = out_dir / "episodes.csv"
    fields = [
        "seed", "success", "failure",
        "ik_current_fail_ticks", "ik_preview_fail_ticks", "ik_fail_ticks", "ik_fail_fraction", "ik_any",
        "place_xy", "place_xyz", "place_vs_place_pos",
        "duration_s", "n_control_steps", "infer_s_median", "infer_s_max", "n_chunks", "video",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for episode in summary["episodes"]:
            row = dict(episode)
            row["failure"] = episode["failure"] or ""
            writer.writerow(row)
    return path


def evaluate_policy(
    policy,
    seeds: list[int],
    *,
    out_dir: Path,
    timeout_s: float,
    method: str,
    checkpoint: str,
    videos: str = "auto",
    live_viewer: bool = False,
    video_fps: int = 10,
    video_max_mb: float = 2.0,
    mode: str = "random",
    exec_horizon: int = 8,
) -> dict:
    """Roll out ``seeds`` and write ``eval.json``. ``videos`` is ``auto`` or ``all``.

    ``auto`` keeps recording until one success and one failure have been saved,
    and always records the first episode.
    """
    m1, m3, m4, m5 = task_configs()
    policy.exec_horizon = int(exec_horizon)
    env = make_c3_env(m1, m3, m4)
    hardware = render_environment(env)
    episodes = []
    have_success = False
    have_failure = False
    try:
        for index, seed in enumerate(seeds):
            want = videos == "all" or index == 0 or not (have_success and have_failure)
            video_path = Path(out_dir) / "videos" / f"seed{int(seed):02d}.mp4" if want else None
            episode = rollout_episode(
                env,
                policy,
                seed=int(seed),
                mode=mode,
                timeout_s=timeout_s,
                live_viewer=live_viewer,
                record=want,
                video_path=video_path,
                video_fps=video_fps,
                video_max_mb=video_max_mb,
                m1=m1,
                m4=m4,
                m5=m5,
            )
            episodes.append(episode)
            have_success = have_success or episode["success"]
            have_failure = have_failure or not episode["success"]
            print(
                f"seed {seed} success {episode['success']} failure {episode['failure']} "
                f"ik_ticks {episode['ik_current_fail_ticks']}/{episode['ik_preview_fail_ticks']} "
                f"place_xy {episode['place_xy']:.4f} duration {episode['duration_s']:.2f}s "
                f"infer_median {episode['infer_s_median']} video {episode['video']}",
                flush=True,
            )
    finally:
        env.close()
    summary = summarize_episodes(episodes, method=method, checkpoint=checkpoint, timeout_s=timeout_s)
    summary["render"] = hardware
    write_eval_log(summary, Path(out_dir))
    print(
        f"success {summary['n_success']}/{summary['n']} "
        f"rate {summary['success_rate']:.3f} wilson {summary['wilson_95']} "
        f"failures {summary['failure_counts']} "
        f"ik_episodes {summary['ik_episodes']} "
        f"ik_ticks current {summary['ik_current_fail_ticks_sum']} "
        f"preview {summary['ik_preview_fail_ticks_sum']}",
        flush=True,
    )
    return summary
