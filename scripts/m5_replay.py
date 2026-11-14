"""Replay stored demonstrations, write videos, and check that they succeed again.

Run: conda run -n AIControl python -u scripts/m5_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m4_run import _controller, _gains, _motor_cfg, _write_video  # noqa: E402

from aloha_pp.data.dataset import list_demos, load_episode
from aloha_pp.robot import repo_root
from aloha_pp.sim.env import AlohaCanEnv, load_config

import yaml


def _load(name: str) -> dict:
    path = repo_root() / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _pick(dataset: Path, backup: Path) -> list[tuple[Path, str]]:
    names = list_demos(dataset)
    chosen = []
    noise = []
    retarget = []
    for name in names:
        episode = load_episode(dataset, name)
        if episode["source"] == "planner_noise":
            noise.append((dataset, name))
        else:
            retarget.append((dataset, name))
    if noise:
        chosen.append(noise[0])
    elif backup.is_file() and list_demos(backup):
        chosen.append((backup, list_demos(backup)[0]))
    for item in retarget:
        if len(chosen) >= 5:
            break
        chosen.append(item)
    for item in noise[1:]:
        if len(chosen) >= 5:
            break
        chosen.append(item)
    return chosen[:5]


def main() -> None:
    root = repo_root()
    out = root / "outputs" / "m5"
    dataset = out / "dataset" / "demos.hdf5"
    backup = out / "dataset" / "planner_noise.hdf5"
    video_dir = out / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    m1 = load_config()
    m3 = _load("m3_track.yaml")
    m4 = _load("m4_task.yaml")
    env = AlohaCanEnv(_motor_cfg(m1))
    env.controller = _controller(env, m3, *_gains(m3, m4))
    picks = _pick(dataset, backup)
    if len(picks) < 5:
        raise RuntimeError(f"need 5 demonstrations to replay, found {len(picks)}")
    deviations = []
    try:
        for index, (path, name) in enumerate(picks):
            episode = load_episode(path, name)
            state = {"qpos": episode["qpos0"], "qvel": episode["qvel0"]}
            if "ctrl0" in episode:
                state["ctrl"] = episode["ctrl0"]
            if "time0" in episode:
                state["time"] = episode["time0"]
            if "qacc_warmstart0" in episode:
                state["qacc_warmstart"] = episode["qacc_warmstart0"]
            if "qacc0" in episode:
                state["qacc"] = episode["qacc0"]
            if "act0" in episode:
                state["act"] = episode["act0"]
            env.reset(mode="random", state=state, render=False)
            actions = episode["actions"]
            stored_next = episode.get("next_actions")
            max_q = 0.0
            max_q32 = 0.0
            frames = []
            success = False
            height = int(m4["video_height"])
            width = int(m4["video_width"])
            every = max(1, int(round(m1["control_hz"] / float(m4["video_fps"]))))
            for t in range(len(actions)):
                q = env.data.qpos[env.arm_qpos_adr]
                stored = episode["obs"]["q"][t]
                max_q = max(max_q, float(np.max(np.abs(q - stored))))
                max_q32 = max(max_q32, float(np.max(np.abs(np.asarray(q, dtype=np.float32) - stored))))
                if t % every == 0:
                    frames.append(env.render("agentview", height, width))
                if stored_next is not None:
                    nxt = stored_next[t]
                else:
                    nxt = actions[t + 1] if t + 1 < len(actions) else actions[t]
                _obs, success, _info = env.step(actions[t], render=False, next_action=nxt)
                if success:
                    break
            deviations.append((path.name, name, episode["source"], episode["source_id"], success, max_q, max_q32))
            print(
                f"replay {name} source {episode['source']} id {episode['source_id']} "
                f"success {success} max_q {max_q:.3e} max_q32 {max_q32:.3e}",
                flush=True,
            )
            if not success:
                raise RuntimeError(f"{name} did not succeed on replay")
            _write_video(frames, video_dir / f"replay_{index:02d}_{episode['source']}_{episode['source_id']}.mp4",
                         float(m4["video_max_mb"]), int(m4["video_fps"]))
            # 96x96 pair from the stored observations.
            agent = episode["obs"]["agentview"][len(episode["obs"]["agentview"]) // 2]
            wrist = episode["obs"]["wrist_cam_left"][len(episode["obs"]["wrist_cam_left"]) // 2]
            pair = np.concatenate((agent, wrist), axis=1)
            imageio.imwrite(video_dir / f"obs_{index:02d}_{episode['source']}_{episode['source_id']}.png", pair)
    finally:
        env.close()
    print("max joint deviation vs float32 obs", max(item[-2] for item in deviations))
    print("max float32(replay q) vs stored obs", max(item[-1] for item in deviations))


if __name__ == "__main__":
    main()
