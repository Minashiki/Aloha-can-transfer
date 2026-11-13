"""Scripted top-down grasp on fixed_1: approach, descend, close, lift, hold.

Run: conda run -n AIControl python scripts/m1_grasp_smoke.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import imageio.v2 as imageio
import numpy as np

from aloha_pp.kin.ik import quat_to_mat, top_down_quat
from aloha_pp.robot import repo_root
from aloha_pp.sim.env import AlohaCanEnv
from aloha_pp.sim.success import tcp_local_offset


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return b.reshape(1, -1)
    alphas = np.linspace(0.0, 1.0, n)
    return (1.0 - alphas)[:, None] * a + alphas[:, None] * b


def main() -> None:
    out_dir = repo_root() / "outputs" / "m1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with AlohaCanEnv() as env:
        obs = env.reset(seed=0, mode="fixed_1")
        imageio.imwrite(out_dir / "agentview_reset_96.png", env.render("agentview", 96, 96))
        imageio.imwrite(out_dir / "agentview_reset_256.png", env.render("agentview", 256, 256))
        can0 = env.can_center_base().copy()
        yaw = float(np.arctan2(can0[1], can0[0]))
        quat = top_down_quat(yaw, 0)
        open_g = float(env.cfg["gripper_open"])
        close_g = float(env.cfg["gripper_close"])
        grasp = can0.copy()
        approach = can0.copy()
        approach[2] += 0.08
        lifted = can0.copy()
        lifted[2] += 0.10
        start = obs["tcp_pose"][:3].copy()

        segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        segments.append((_lerp(start, approach, 20), np.full(20, open_g), np.array(["move"] * 20)))
        segments.append((_lerp(approach, grasp, 20), np.full(20, open_g), np.array(["down"] * 20)))
        segments.append((_lerp(grasp, grasp, 12), np.linspace(open_g, close_g, 12), np.array(["close"] * 12)))
        segments.append((_lerp(grasp, grasp, 8), np.full(8, close_g), np.array(["settle"] * 8)))
        segments.append((_lerp(grasp, lifted, 20), np.full(20, close_g), np.array(["lift"] * 20)))
        segments.append((_lerp(lifted, lifted, 60), np.full(60, close_g), np.array(["hold"] * 60)))

        video_path = out_dir / "grasp.mp4"
        writer = imageio.get_writer(
            video_path,
            fps=10,
            codec="libx264",
            ffmpeg_params=["-crf", "28", "-pix_fmt", "yuv420p"],
        )
        hold_local: list[np.ndarray] = []
        can_z = [float(can0[2])]
        frames = 0
        ik_failures = 0
        saved_grasp = False
        hold_actions = 0
        phase_box = {"phase": "move"}

        def on_substep(current: AlohaCanEnv) -> None:
            can = current.can_center_base()
            can_z.append(float(can[2]))
            if phase_box["phase"] != "hold":
                return
            tcp_pos, tcp_quat = current.tcp_pose()
            hold_local.append(tcp_local_offset(can, tcp_pos, quat_to_mat(tcp_quat)))

        try:
            for positions, grips, phases in segments:
                for pos, grip, phase in zip(positions, grips, phases):
                    phase_box["phase"] = str(phase)
                    action = np.concatenate((pos, quat, [grip]))
                    obs, _success, info = env.step(action, on_substep=on_substep)
                    if not info["ik_success"]:
                        ik_failures += 1
                    if phase == "hold":
                        hold_actions += 1
                    if phase == "settle" and not saved_grasp:
                        imageio.imwrite(out_dir / "agentview_grasp.png", env.render("agentview", 256, 256))
                        saved_grasp = True
                    if frames % 2 == 0:
                        # Nearest-neighbor scale of the 96x96 agentview keeps the file small.
                        frame = np.repeat(np.repeat(obs["agentview"], 4, axis=0), 4, axis=1)
                        writer.append_data(frame)
                    frames += 1
        finally:
            writer.close()

        agent = obs["agentview"]
        wrist = obs["wrist_cam_left"]
        imageio.imwrite(out_dir / "agentview.png", agent)
        imageio.imwrite(out_dir / "wrist_cam_left.png", wrist)
        rel = np.stack(hold_local)
        slip = np.linalg.norm(rel - rel[0], axis=1)
        lift = max(can_z) - can_z[0]
        # Lift is measured from the reset height to the lowest physics-step
        # sample of the hold, so a brief bounce does not count.
        hold_z = can_z[-len(hold_local) :]
        hold_lift = float(np.min(hold_z) - can_z[0])
        hold_s = hold_actions / env.cfg["control_hz"]
        size_mb = video_path.stat().st_size / 1e6
        print(f"video {video_path} bytes {video_path.stat().st_size} mb {size_mb:.3f}")
        print(f"screenshots {out_dir / 'agentview.png'} {out_dir / 'wrist_cam_left.png'}")
        print(
            f"reset {out_dir / 'agentview_reset_96.png'} {out_dir / 'agentview_reset_256.png'} "
            f"grasp {out_dir / 'agentview_grasp.png'}"
        )
        print(
            f"agentview min {int(agent.min())} max {int(agent.max())} "
            f"wrist min {int(wrist.min())} max {int(wrist.max())}"
        )
        print(f"can_z0 {can_z[0]:.4f} hold_min_z {min(hold_z):.4f} hold_lift {hold_lift:.4f} peak_lift {lift:.4f}")
        print(f"hold_samples {len(hold_local)} hold_actions {hold_actions} hold_s {hold_s:.2f}")
        print(f"max_slip_m {float(slip.max()):.6f}")
        print(f"ik_failures {ik_failures}")
        ok = hold_lift >= 0.05 and hold_s >= 3.0 and float(slip.max()) < 0.005 and size_mb <= 2.0
        print("GRASP_OK" if ok else "GRASP_FAIL")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
