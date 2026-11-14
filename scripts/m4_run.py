"""M4 task and reference-mode runs. No camera images except the video pass.

Run: conda run -n AIControl python -u scripts/m4_run.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

from aloha_pp.control.logs import save_run
from aloha_pp.control.reference import build_speed_family, pick_place_waypoints
from aloha_pp.control.tracking import velocity_limits
from aloha_pp.kin.ik import DlsIk
from aloha_pp.planning.metrics import classify_episode, signals_from_log
from aloha_pp.planning.planner import PickPlacePlanner
from aloha_pp.robot import repo_root, scene_xml
from aloha_pp.sim.c3 import c3_controller, c3_gains, motor_cfg
from aloha_pp.sim.env import AlohaCanEnv, load_config

JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()


def _load(name: str) -> dict:
    path = repo_root() / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(part) for part in text.split(",") if part]


def _gains(m3: dict, m4: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return c3_gains(m3, m4)


def _controller(env: AlohaCanEnv, m3: dict, kp, kd, ki):
    return c3_controller(env, m3, kp, kd, ki)


def _motor_cfg(m1: dict) -> dict:
    return motor_cfg(m1)


def _empty_rows() -> dict[str, list]:
    keys = (
        "t", "q", "qd", "q_d", "qd_d", "qdd_d", "tau_cmd", "tau_applied", "saturated",
        "vel_exceeded", "tcp_pos", "tcp_quat", "tcp_ref_pos", "tcp_ref_quat",
        "tcp_qd_pos", "tcp_qd_quat", "ik_pos_err", "ik_rot_err", "phase", "integral",
        "can_pos", "can_quat", "finger_can_contact", "gripper_cmd", "gripper_q",
        "can_linvel", "finger_contact", "planner_state", "success_count", "success",
        "ik_solve_s", "ik_next_solve_s", "ik_success", "dropped_knot",
    )
    return {key: [] for key in keys}


def _stack(rows: dict[str, list]) -> dict[str, np.ndarray]:
    stacked = {}
    for key, value in rows.items():
        stacked[key] = np.stack(value) if np.asarray(value[0]).ndim else np.asarray(value)
    for key in ("phase", "planner_state"):
        stacked[key] = stacked[key].astype(np.int16)
    stacked["success_count"] = stacked["success_count"].astype(np.int32)
    return stacked


def _base_meta(name, mode, speed, m3, gains, commit, motion_time, extra) -> dict:
    meta = {
        "name": name,
        "controller": "C3",
        "mode": mode,
        "speed": speed,
        "actuator": "motor",
        "gravity_compensation": False,
        "kp": [float(v) for v in gains[0]],
        "kd": [float(v) for v in gains[1]],
        "ki": [0.0] * 6,
        "torque_limit": [10.0, 20.0, 20.0, 10.0, 10.0, 4.0],
        "velocity_limit": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        "motion_time": float(motion_time),
        "hold_s": float(m3["hold_s"]),
        "git_commit": commit,
        "ik_pos_tol": float(m3["ik_pos_tol"]),
        "ik_rot_tol": float(m3["ik_rot_tol"]),
        "omega": float(m3["omega"]),
        "zeta": float(m3["zeta"]),
    }
    meta.update(extra)
    return meta


def _write_video(frames: list[np.ndarray], path: Path, max_mb: float, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(max_mb * 1e6)
    for crf in (28, 32, 36, 40, 44):
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"]
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        if path.stat().st_size <= limit:
            print(f"video {path.name} bytes {path.stat().st_size} crf {crf}", flush=True)
            return
    raise RuntimeError(f"video {path} is {path.stat().st_size} bytes, above {limit}")


def run_task(
    m1: dict,
    m3: dict,
    m4: dict,
    gains,
    *,
    mode: str,
    speed: str,
    seed: int,
    name: str,
    commit: str,
    record: bool,
    viewer: bool,
    zero_arm_friction: bool = False,
) -> dict:
    env = AlohaCanEnv(_motor_cfg(m1))
    if zero_arm_friction:
        env.model.dof_frictionloss[env.arm_dof_adr] = 0.0
    frames: list[np.ndarray] = []
    passive = None
    try:
        controller = _controller(env, m3, *gains)
        env.controller = controller
        env.reset(seed=seed, mode=mode, render=False)
        planner = PickPlacePlanner(m1, m4, m3, speed)
        print(f"plan {name}", flush=True)
        planner.reset(env, mode=mode)
        if viewer:
            passive = mujoco.viewer.launch_passive(env.model, env.data)
        rows = _empty_rows()
        physics_dt = 1.0 / float(env.cfg["physics_hz"])
        n_sub = env.n_substeps
        box = {"n": 0, "i": 0, "pos": None, "quat": None, "info_ok": True}
        height = int(m4["video_height"])
        width = int(m4["video_width"])
        every = max(1, int(round(env.cfg["control_hz"] / float(m4["video_fps"]))))

        def on_substep(current: AlohaCanEnv) -> None:
            sample = controller.sample
            if sample is None or not np.all(np.isfinite(sample.tau_applied)):
                raise RuntimeError("physics step finished without an applied-torque sample")
            i = box["i"]
            n = box["n"]
            pos, quat = current.tcp_pose()
            planner.note_tcp(pos)
            contacts = current.finger_can_contacts()
            can_pos, can_quat = current.can_pose_base()
            rows["t"].append((n + 1) * physics_dt)
            rows["q"].append(current.data.qpos[current.arm_qpos_adr].copy())
            rows["qd"].append(current.data.qvel[current.arm_dof_adr].copy())
            rows["q_d"].append(sample.q_d)
            rows["qd_d"].append(sample.qd_d)
            rows["qdd_d"].append(sample.qdd_d)
            rows["tau_cmd"].append(sample.tau_cmd)
            rows["tau_applied"].append(sample.tau_applied)
            rows["saturated"].append(sample.saturated)
            rows["vel_exceeded"].append(sample.vel_exceeded)
            rows["tcp_pos"].append(pos)
            rows["tcp_quat"].append(quat)
            rows["tcp_ref_pos"].append(box["pos"][i])
            rows["tcp_ref_quat"].append(box["quat"][i])
            rows["tcp_qd_pos"].append(sample.tcp_qd_pos)
            rows["tcp_qd_quat"].append(sample.tcp_qd_quat)
            rows["ik_pos_err"].append(sample.ik_pos_err)
            rows["ik_rot_err"].append(sample.ik_rot_err)
            rows["phase"].append(planner.schedule[min(planner.k, len(planner.schedule) - 1)].state)
            rows["integral"].append(sample.integral)
            rows["can_pos"].append(can_pos)
            rows["can_quat"].append(can_quat)
            rows["finger_can_contact"].append(bool(contacts[0] or contacts[1]))
            rows["gripper_cmd"].append(controller._grip_cmd)
            rows["gripper_q"].append(float(current.data.qpos[current.grip_qpos_adr]))
            rows["can_linvel"].append(current.can_linvel())
            rows["finger_contact"].append(contacts.copy())
            rows["planner_state"].append(rows["phase"][-1])
            rows["success_count"].append(int(current.tracker.count))
            rows["success"].append(bool(current.success))
            rows["ik_solve_s"].append(controller.ik_solve_s)
            rows["ik_next_solve_s"].append(controller.ik_next_solve_s)
            rows["ik_success"].append(bool(controller._ik_ok))
            rows["dropped_knot"].append(bool(sample.dropped_knot))
            if record and i == n_sub - 1 and (planner.k % every) == 0:
                frames.append(current.render("agentview", height, width))
            if passive is not None:
                passive.sync()
            box["i"] = i + 1
            box["n"] = n + 1

        while not planner.done:
            action, nxt, _tick = planner.action_pair()
            box["pos"], box["quat"] = planner.dense_reference()
            box["i"] = 0
            _obs, success, _info = env.step(action, on_substep=on_substep, render=False, next_action=nxt)
            planner.after_step(env)
            if success:
                break
        arrays = _stack(rows)
        succeeded = bool(env.success)
        meta = _base_meta(
            name, mode, speed, m3, gains, commit, planner.planned_motion_time,
            {
                "experiment": "E1" if mode != "random" else "E2",
                "seed": int(seed),
                "reference_mode": "preview",
                "can_init_pos": [float(v) for v in planner.can_init[:3]],
                "can_init_quat": [float(v) for v in planner.can_init[3:]],
                "target_center": [float(v) for v in planner.cell_center],
                "target_half_extents": [float(v) for v in planner.cell_half],
                "place_pos": [float(v) for v in planner.place_pos],
                "place_tcp": [float(v) for v in planner.place_tcp],
                "arm_frictionloss": [float(v) for v in env.model.dof_frictionloss[env.arm_dof_adr]],
                "yaw": float(planner.yaw),
                "grasp_offset": [float(v) for v in planner.offset],
                "tcp_err_close": float(planner.tcp_err_close),
                "tcp_err_open": float(planner.tcp_err_open),
                "planned_total_time": float(planner.planned_total_time),
                "motion_ticks": [int(v) for v in planner.motion_ticks],
                "drop_z": float(m4["drop_z"]),
                "vel_thresh": float(m1["success"]["vel_thresh"]),
                "success": succeeded,
                "timed_out": bool(planner.done and not succeeded),
                "place_gap": float(m4["place_gap"]),
                "settle_s": float(m4["settle_s"]),
            },
        )
        record_flags = signals_from_log(arrays, meta)
        meta["failure"] = classify_episode(record_flags)
        meta["success_time"] = float(arrays["t"][arrays["success"]][0]) if np.any(arrays["success"]) else None
        return {"arrays": arrays, "meta": meta, "frames": frames}
    finally:
        if passive is not None:
            passive.close()
        env.close()


def _home_pose(model, site: str, home_q: np.ndarray, grip: float):
    ik = DlsIk(model, site, max_iters=1)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home_q
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/left_finger")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/right_finger")
    data.qpos[int(model.jnt_qposadr[left])] = grip
    data.qpos[int(model.jnt_qposadr[right])] = grip
    mujoco.mj_kinematics(model, data)
    return ik.site_pose_in_base(data)


def run_e4_case(m1, m3, gains, traj, *, mode, speed, scheme_name, commit, park=True) -> dict:
    env = AlohaCanEnv(_motor_cfg(m1))
    try:
        controller = _controller(env, m3, *gains)
        env.controller = controller
        env.reset(seed=0, mode=mode, render=False)
        if park:
            adr = env.can_qpos_adr
            env.data.qpos[adr : adr + 3] = np.array([1.5, 1.5, 0.2])
            env.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            env.data.qvel[env.can_dof_adr : env.can_dof_adr + 6] = 0.0
            mujoco.mj_forward(env.model, env.data)
        rows = _empty_rows()
        physics_dt = 1.0 / float(env.cfg["physics_hz"])
        n_sub = env.n_substeps
        box = {"n": 0, "i": 0}
        actions = traj.actions
        preview = scheme_name == "preview"

        def on_substep(current: AlohaCanEnv) -> None:
            sample = controller.sample
            if sample is None or not np.all(np.isfinite(sample.tau_applied)):
                raise RuntimeError("physics step finished without an applied-torque sample")
            n = box["n"]
            i = box["i"]
            pos, quat = current.tcp_pose()
            contacts = current.finger_can_contacts()
            can_pos, can_quat = current.can_pose_base()
            rows["t"].append(float(traj.t[n]))
            rows["q"].append(current.data.qpos[current.arm_qpos_adr].copy())
            rows["qd"].append(current.data.qvel[current.arm_dof_adr].copy())
            rows["q_d"].append(sample.q_d)
            rows["qd_d"].append(sample.qd_d)
            rows["qdd_d"].append(sample.qdd_d)
            rows["tau_cmd"].append(sample.tau_cmd)
            rows["tau_applied"].append(sample.tau_applied)
            rows["saturated"].append(sample.saturated)
            rows["vel_exceeded"].append(sample.vel_exceeded)
            rows["tcp_pos"].append(pos)
            rows["tcp_quat"].append(quat)
            rows["tcp_ref_pos"].append(traj.tcp_pos[n])
            rows["tcp_ref_quat"].append(traj.tcp_quat[n])
            rows["tcp_qd_pos"].append(sample.tcp_qd_pos)
            rows["tcp_qd_quat"].append(sample.tcp_qd_quat)
            rows["ik_pos_err"].append(sample.ik_pos_err)
            rows["ik_rot_err"].append(sample.ik_rot_err)
            rows["phase"].append(int(traj.phase[n]))
            rows["integral"].append(sample.integral)
            rows["can_pos"].append(can_pos)
            rows["can_quat"].append(can_quat)
            rows["finger_can_contact"].append(bool(contacts[0] or contacts[1]))
            rows["gripper_cmd"].append(controller._grip_cmd)
            rows["gripper_q"].append(float(current.data.qpos[current.grip_qpos_adr]))
            rows["can_linvel"].append(current.can_linvel())
            rows["finger_contact"].append(contacts.copy())
            rows["planner_state"].append(int(traj.phase[n]))
            rows["success_count"].append(0)
            rows["success"].append(False)
            rows["ik_solve_s"].append(controller.ik_solve_s)
            rows["ik_next_solve_s"].append(controller.ik_next_solve_s)
            rows["ik_success"].append(bool(controller._ik_ok))
            rows["dropped_knot"].append(bool(sample.dropped_knot))
            box["i"] = i + 1
            box["n"] = n + 1

        for index, action in enumerate(actions):
            box["i"] = 0
            nxt = actions[index + 1] if index + 1 < len(actions) else action
            if preview:
                env.step(action, on_substep=on_substep, render=False, next_action=nxt)
            else:
                env.step(action, on_substep=on_substep, render=False)
        arrays = _stack(rows)
        name = f"E4_C3_{mode}_{speed}_{scheme_name}"
        meta = _base_meta(
            name, mode, speed, m3, gains, commit, traj.motion_time,
            {
                "experiment": "E4",
                "seed": 0,
                "reference_mode": scheme_name,
                "can_init_pos": [0.0, 0.0, 0.0],
                "can_init_quat": [1.0, 0.0, 0.0, 0.0],
                "target_center": [0.0, 0.0, 0.0],
                "target_half_extents": [1.0, 1.0, 1.0],
                "place_pos": [0.0, 0.0, 0.0],
                "success": False,
                "timed_out": False,
                "failure": None,
                "vel_thresh": float(m1["success"]["vel_thresh"]),
                "drop_z": 0.03,
                "can_parked": True,
            },
        )
        return {"arrays": arrays, "meta": meta, "frames": []}
    finally:
        env.close()


def _save(run_dir: Path, result: dict) -> None:
    meta = dict(result["meta"])
    save_run(run_dir / f"{meta['name']}.npz", result["arrays"], meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", action="store_true")
    parser.add_argument("--e2", action="store_true")
    parser.add_argument("--e4", action="store_true")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--speeds", nargs="*", default=None)
    parser.add_argument("--e2-seeds", default="0-49")
    parser.add_argument("--zero-arm-friction", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    selected = args.e1 or args.e2 or args.e4 or args.videos or args.viewer
    do_e1 = args.e1 or not selected
    do_e2 = args.e2 or not selected
    do_e4 = args.e4 or not selected
    do_videos = args.videos or not selected
    if args.viewer and not (args.e1 or args.e2 or args.e4):
        do_e1, do_e2, do_e4, do_videos = True, False, False, False
    if args.zero_arm_friction and not selected:
        do_e1, do_e2, do_e4, do_videos = True, False, False, False

    started = time.perf_counter()
    root = repo_root()
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
    elif args.zero_arm_friction:
        out = root / "outputs" / "m4" / "ablation_nofriction"
    else:
        out = root / "outputs" / "m4"
    run_dir = out if args.zero_arm_friction else out / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    m1 = load_config()
    m3 = _load("m3_track.yaml")
    m4 = _load("m4_task.yaml")
    gains = _gains(m3, m4)
    commit = _git_commit()
    modes = args.modes or ["fixed_1", "fixed_2"]
    speeds = args.speeds or ["fast", "medium", "slow"]
    if args.zero_arm_friction and not selected:
        if args.modes is None:
            modes = ["fixed_1", "fixed_2"]
        if args.speeds is None:
            speeds = ["medium"]
    section_t = 0.0

    if do_e4:
        print("E4 reference comparison", flush=True)
        plan_model = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
        home_q = np.asarray(m1["home_q"], dtype=float)
        grip = float(m1["gripper_open"])
        home_pos, home_quat = _home_pose(plan_model, m1["tcp_site"], home_q, grip)
        probe = AlohaCanEnv(m1)
        can_pos = {}
        try:
            for mode in modes:
                probe.reset(seed=0, mode=mode, render=False)
                can_pos[mode] = probe.can_center_base().copy()
        finally:
            probe.close()
        ik = DlsIk(
            plan_model,
            m1["tcp_site"],
            damping=float(m3["ik_damping"]),
            max_iters=int(m3["ik_max_iters"]),
            pos_tol=float(m3["ik_pos_tol"]),
            rot_tol=float(m3["ik_rot_tol"]),
            rot_weight=float(m3["ik_rot_weight"]),
        )
        families = {}
        for scheme in ("lookahead", "preview"):
            families[scheme] = {}
            for mode in modes:
                poses = pick_place_waypoints(
                    can_pos[mode],
                    np.asarray(m1["modes"][mode]["place_pos"], dtype=float),
                    home_pos,
                    home_quat,
                    clearance=float(m3["pregrasp_clearance"]),
                    lift=float(m3["lift_height"]),
                )
                families[scheme][mode] = build_speed_family(
                    ik,
                    poses,
                    home_q=home_q,
                    home_pos=home_pos,
                    home_quat=home_quat,
                    grip=grip,
                    control_dt=1.0 / float(m1["control_hz"]),
                    n_substeps=int(m1["physics_hz"]) // int(m1["control_hz"]),
                    v_max=velocity_limits(m1),
                    hold_s=float(m3["hold_s"]),
                    label=f"{mode}_{scheme}",
                    scheme=scheme,
                    a_max=np.asarray(m3["accel_limit"], dtype=float),
                )
        for scheme in ("lookahead", "preview"):
            label = "wait" if scheme == "lookahead" else "preview"
            for mode in modes:
                for speed in speeds:
                    name = f"E4_C3_{mode}_{speed}_{label}"
                    t0 = time.perf_counter()
                    result = run_e4_case(
                        m1, m3, gains, families[scheme][mode][speed],
                        mode=mode, speed=speed, scheme_name=label, commit=commit,
                    )
                    _save(run_dir, result)
                    dt = time.perf_counter() - t0
                    section_t += dt
                    print(f"run {name} {dt:.2f}s steps {result['arrays']['t'].shape[0]}", flush=True)

    task_results = []
    if do_e1:
        print("E1 fixed pick-and-place", flush=True)
        e1_modes = modes
        e1_speeds = speeds
        first = True
        for mode in e1_modes:
            for speed in e1_speeds:
                name = f"E1_C3_{mode}_{speed}"
                t0 = time.perf_counter()
                use_viewer = bool(args.viewer and first)
                result = run_task(
                    m1, m3, m4, gains, mode=mode, speed=speed, seed=0, name=name,
                    commit=commit, record=False, viewer=use_viewer,
                    zero_arm_friction=args.zero_arm_friction,
                )
                first = False
                result["meta"]["experiment"] = "E1"
                _save(run_dir, result)
                dt = time.perf_counter() - t0
                section_t += dt
                task_results.append(result["meta"])
                print(
                    f"run {name} {dt:.2f}s success {result['meta']['success']} "
                    f"failure {result['meta']['failure']}",
                    flush=True,
                )
                if args.viewer and not (args.e1 or args.e2 or args.e4):
                    break
            else:
                continue
            break

    if do_e2:
        print("E2 random", flush=True)
        for seed in _parse_seeds(args.e2_seeds):
            name = f"E2_C3_random_s{seed:02d}"
            t0 = time.perf_counter()
            result = run_task(
                m1, m3, m4, gains, mode="random", speed="medium", seed=seed, name=name,
                commit=commit, record=False, viewer=False,
                zero_arm_friction=args.zero_arm_friction,
            )
            _save(run_dir, result)
            dt = time.perf_counter() - t0
            section_t += dt
            task_results.append(result["meta"])
            print(
                f"run {name} {dt:.2f}s success {result['meta']['success']} "
                f"failure {result['meta']['failure']}",
                flush=True,
            )

    if do_videos and task_results:
        print("videos", flush=True)
        jobs = []
        for meta in task_results:
            if meta["experiment"] == "E1":
                jobs.append(meta)
        seen = set()
        success_done = False
        for meta in task_results:
            if meta["experiment"] != "E2":
                continue
            if meta["success"] and not success_done:
                jobs.append(meta)
                success_done = True
            reason = meta["failure"]
            if reason and reason not in seen:
                seen.add(reason)
                jobs.append(meta)
        video_dir = out / "videos"
        for meta in jobs:
            t0 = time.perf_counter()
            result = run_task(
                m1, m3, m4, gains,
                mode=meta["mode"], speed=meta["speed"], seed=int(meta["seed"]),
                name=meta["name"], commit=commit, record=True, viewer=False,
                zero_arm_friction=args.zero_arm_friction,
            )
            if not result["frames"]:
                raise RuntimeError(f"no frames for {meta['name']}")
            _write_video(result["frames"], video_dir / f"{meta['name']}.mp4", float(m4["video_max_mb"]), int(m4["video_fps"]))
            section_t += time.perf_counter() - t0

    print(f"section_s {section_t:.2f} total_s {time.perf_counter() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
