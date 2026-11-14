"""Evaluate bc_zero, bc_seeded, and bc_dsrl with the M6 rollout log."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from aloha_pp.policy.rollout import (
    render_environment,
    rollout_episode,
    summarize_episodes,
    task_configs,
    write_eval_log,
)
from aloha_pp.rl.noise import seeded_generator, zero_noise
from aloha_pp.sim.c3 import make_c3_env


def chunk_noise_fn(method: str, policy, actor, seed: int):
    """Noise callback for one episode. ``actor`` is required for ``bc_dsrl``."""
    shape = tuple(policy.noise_shape)
    if method == "bc_zero":
        def zeros(_obs):
            return zero_noise(shape)
        return zeros
    if method == "bc_seeded":
        rng = seeded_generator(int(seed))
        def draw(_obs):
            return rng.standard_normal(shape).astype(np.float32)
        return draw
    if method == "bc_dsrl":
        if actor is None:
            raise ValueError("bc_dsrl needs the DSRL actor")
        def policy_noise(obs):
            feat = policy.encode_actor_obs(obs)
            action = actor.deterministic(feat)
            return action.reshape(shape).detach().cpu().numpy()
        return policy_noise
    raise ValueError(f"unknown method {method}")


def record_episode(videos: str, index: int, have_success: bool, have_failure: bool) -> bool:
    """Whether this episode is written to a video file.

    ``none`` never records. ``all`` records every seed. ``auto`` records the
    first episode and keeps going until one success and one failure are saved.
    """
    if videos == "none":
        return False
    if videos == "all":
        return True
    if videos == "auto":
        return index == 0 or not (have_success and have_failure)
    raise ValueError(f"videos must be none, auto, or all, got {videos!r}")


def evaluate_latent(
    policy,
    seeds: list[int],
    *,
    out_dir: Path,
    timeout_s: float,
    method: str,
    checkpoint: str,
    actor=None,
    videos: str = "none",
    live_viewer: bool = False,
    video_fps: int = 10,
    video_max_mb: float = 2.0,
    mode: str = "random",
    exec_horizon: int = 8,
) -> dict:
    """Same log as M6 eval. The only change is the initial noise of each chunk.

    ``videos="none"`` does not record, so the rollout is not asked to render
    the agentview video frames.
    """
    if videos not in {"none", "auto", "all"}:
        raise ValueError(f"videos must be none, auto, or all, got {videos!r}")
    m1, m3, m4, m5 = task_configs()
    policy.exec_horizon = int(exec_horizon)
    env = make_c3_env(m1, m3, m4)
    hardware = render_environment(env)
    episodes = []
    have_success = False
    have_failure = False
    try:
        for index, seed in enumerate(seeds):
            want = record_episode(videos, index, have_success, have_failure)
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
                chunk_noise=chunk_noise_fn(method, policy, actor, int(seed)),
            )
            episodes.append(episode)
            have_success = have_success or episode["success"]
            have_failure = have_failure or not episode["success"]
            print(
                f"{method} seed {seed} success {episode['success']} failure {episode['failure']} "
                f"ik_ticks {episode['ik_current_fail_ticks']}/{episode['ik_preview_fail_ticks']} "
                f"place_xy {episode['place_xy']:.4f} duration {episode['duration_s']:.2f}s "
                f"video {episode['video']}",
                flush=True,
            )
    finally:
        env.close()
    summary = summarize_episodes(episodes, method=method, checkpoint=checkpoint, timeout_s=timeout_s)
    summary["videos"] = videos
    summary["video_seeds"] = [int(episode["seed"]) for episode in episodes if episode.get("video")]
    summary["render"] = hardware
    summary["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "cpu_count": os.cpu_count(),
        "driver": hardware.get("driver", ""),
        "gpu": hardware.get("gpu", ""),
        "mujoco": hardware.get("mujoco", ""),
        "MUJOCO_GL": hardware.get("MUJOCO_GL", ""),
    }
    write_eval_log(summary, Path(out_dir))
    print(
        f"{method} success {summary['n_success']}/{summary['n']} "
        f"rate {summary['success_rate']:.3f} wilson {summary['wilson_95']} "
        f"failures {summary['failure_counts']}",
        flush=True,
    )
    return summary
