"""Batch M3 tracking runs. No camera images except two short videos.

Run: conda run -n AIControl python scripts/m3_run.py
"""

from __future__ import annotations

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

from aloha_pp.control.arm import arm_index
from aloha_pp.control.logs import load_run, save_run
from aloha_pp.control.quintic import scheme_lag_s
from aloha_pp.control.reference import (
    benchmark_ik,
    build_speed_family,
    choose_tolerance,
    contact_trajectory,
    path_samples,
    pick_place_waypoints,
    Trajectory,
)
from aloha_pp.control.tracking import TrackingController, velocity_limits
from aloha_pp.kin.ik import DlsIk
from aloha_pp.robot import repo_root, scene_xml
from aloha_pp.sim.env import AlohaCanEnv, load_config
from aloha_pp.sim.gravity import apply_gravity_compensation

CONTROLLERS = ("B0", "B1", "C1", "C2", "C3", "C4")
MODES = ("fixed_1", "fixed_2")
SPEEDS = ("fast", "medium", "slow")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()


def _load_m3() -> dict:
    path = repo_root() / "configs" / "m3_track.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _home_pose(model, site: str, home_q: np.ndarray, grip: float) -> tuple[np.ndarray, np.ndarray]:
    ik = DlsIk(model, site, max_iters=1)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home_q
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/left_finger")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left/right_finger")
    data.qpos[int(model.jnt_qposadr[left])] = grip
    data.qpos[int(model.jnt_qposadr[right])] = grip
    mujoco.mj_kinematics(model, data)
    return ik.site_pose_in_base(data)


def _make_controller(
    env: AlohaCanEnv,
    kind: str,
    m3: dict,
    tol: dict,
    gains: dict,
    *,
    vel_feedback: str = "error",
) -> TrackingController:
    kp = np.asarray(gains["kp"], dtype=float)
    kd = np.asarray(gains["kd"], dtype=float)
    ki = np.asarray(gains["ki"], dtype=float)
    return TrackingController(
        env.model,
        kind=kind,
        site_name=env.cfg["tcp_site"],
        v_max=velocity_limits(env.cfg),
        control_dt=1.0 / float(env.cfg["control_hz"]),
        n_substeps=env.n_substeps,
        gripper_open=float(env.cfg["gripper_open"]),
        ik_damping=float(m3["ik_damping"]),
        ik_max_iters=int(m3["ik_max_iters"]),
        ik_pos_tol=float(tol["pos_tol"]),
        ik_rot_tol=float(tol["rot_tol"]),
        ik_rot_weight=float(m3["ik_rot_weight"]),
        kp=kp,
        kd=kd,
        ki=ki,
        tt=1.0 / float(m3["omega"]),
        scheme=str(m3["reference_scheme"]),
        a_max=np.asarray(m3["accel_limit"], dtype=float),
        vel_feedback=vel_feedback,
    )


def _execute(
    env: AlohaCanEnv,
    controller: TrackingController,
    traj: Trajectory,
    *,
    record: bool,
    video_size: tuple[int, int],
    park_can: bool = False,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    env.controller = controller
    mode = "fixed_2" if "fixed_2" in traj.name else "fixed_1"
    env.reset(seed=0, mode=mode, render=False)
    if park_can:
        # The Cartesian waypoints were already built from the original can pose.
        # Parking the body removes finger contact without moving that reference.
        adr = env.can_qpos_adr
        env.data.qpos[adr : adr + 3] = np.array([1.5, 1.5, 0.2])
        env.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[env.can_dof_adr : env.can_dof_adr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
    n_sub = env.n_substeps
    if traj.actions.shape[0] * n_sub != traj.tcp_pos.shape[0]:
        raise RuntimeError("trajectory action count does not match the physics reference")
    rows: dict[str, list] = {key: [] for key in (
        "t", "q", "qd", "q_d", "qd_d", "qdd_d", "tau_cmd", "tau_applied", "saturated",
        "vel_exceeded", "tcp_pos", "tcp_quat", "tcp_ref_pos", "tcp_ref_quat",
        "tcp_qd_pos", "tcp_qd_quat", "ik_pos_err", "ik_rot_err", "phase", "integral",
        "can_pos", "can_quat", "finger_can_contact",
    )}
    frames: list[np.ndarray] = []
    index = {"n": 0}

    def on_substep(current: AlohaCanEnv) -> None:
        sample = controller.sample
        if sample is None or not np.all(np.isfinite(sample.tau_applied)):
            raise RuntimeError("physics step finished without an applied-torque sample")
        n = index["n"]
        pos, quat = current.tcp_pose()
        rows["t"].append(traj.t[n])
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
        rows["phase"].append(traj.phase[n])
        rows["integral"].append(sample.integral)
        can_pos, can_quat = current.can_pose_base()
        rows["can_pos"].append(can_pos)
        rows["can_quat"].append(can_quat)
        rows["finger_can_contact"].append(not current.fingers_released())
        if record and n % 50 == 0:
            frames.append(current.render("agentview", video_size[0], video_size[1]))
        index["n"] = n + 1

    for action in traj.actions:
        env.step(action, on_substep=on_substep, render=False)
    if index["n"] != traj.tcp_pos.shape[0]:
        raise RuntimeError(f"logged {index['n']} steps, trajectory has {traj.tcp_pos.shape[0]}")
    stacked = {key: np.stack(value) if np.asarray(value[0]).ndim else np.asarray(value) for key, value in rows.items()}
    stacked["phase"] = stacked["phase"].astype(np.int8)
    return stacked, frames


def _write_video(frames: list[np.ndarray], path: Path, max_mb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(max_mb * 1e6)
    for crf in (28, 32, 36, 40, 44):
        writer = imageio.get_writer(
            path,
            fps=10,
            codec="libx264",
            ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        if path.stat().st_size <= limit:
            print(f"video {path} bytes {path.stat().st_size} crf {crf}")
            return
    raise RuntimeError(f"video {path} is {path.stat().st_size} bytes, above {limit}")


def _meta(kind, mode, speed, m3, tol, gains, traj: Trajectory, commit: str, actuator: str, grav: bool, extra: dict) -> dict:
    meta = {
        "name": extra.get("name", f"{kind}_{mode}_{speed}"),
        "controller": kind,
        "mode": mode,
        "speed": speed,
        "actuator": actuator,
        "gravity_compensation": grav,
        "kp": list(map(float, gains["kp"])),
        "kd": list(map(float, gains["kd"])),
        "ki": list(map(float, gains["ki"])) if kind == "C4" else [0.0] * 6,
        "omega": float(m3["omega"]),
        "zeta": float(m3["zeta"]),
        "torque_limit": [10.0, 20.0, 20.0, 10.0, 10.0, 4.0],
        "velocity_limit": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        "motion_time": float(traj.motion_time),
        "hold_s": float(m3["hold_s"]),
        "git_commit": commit,
        "ik_pos_tol": float(tol["pos_tol"]),
        "ik_rot_tol": float(tol["rot_tol"]),
        "peak_qd_plan": [float(v) for v in traj.peak_qd],
    }
    meta.update(extra)
    return meta


def _plant(model, q, qd, grip: float):
    data = mujoco.MjData(model)
    arm = arm_index(model)
    data.qpos[arm.qpos_adr] = q
    data.qvel[arm.dof_adr] = qd
    for name in ("left/left_finger", "left/right_finger"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = grip
    data.ctrl[arm.grip_act] = grip
    return data, arm


def _write_noff_report(run_dir: Path, m1: dict, m3: dict) -> None:
    """Max |q| gap between B1 and C2 with position-style damping, plus same-state torques."""
    b1, _meta_b1 = load_run(run_dir / "B1_fixed_1_medium.npz")
    noff, _meta_noff = load_run(run_dir / "C2_menagerie_noff_fixed_1_medium.npz")
    gap = np.abs(b1["q"] - noff["q"])
    flat = int(np.argmax(gap))
    step, joint = np.unravel_index(flat, gap.shape)
    report: dict = {
        "max_abs_q": float(gap[step, joint]),
        "step": int(step),
        "joint": int(joint),
        "t": float(b1["t"][step]),
    }
    kp = np.asarray(m3["menagerie_kp"], dtype=float)
    kv = np.asarray(m3["menagerie_kv"], dtype=float)
    grip = float(m1["gripper_open"])
    motor = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
    position = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    apply_gravity_compensation(position, True)
    formula_gap = 0.0
    for index in range(0, b1["q"].shape[0], 5):
        g_tau, grav_tau = _both_torques(motor, position, b1, index, kp, kv, grip)
        formula_gap = max(formula_gap, float(np.max(np.abs(g_tau - grav_tau))))
    q = b1["q"][step]
    qd = b1["qd"][step]
    q_d = b1["q_d"][step]
    g_tau, grav_tau = _both_torques(motor, position, b1, int(step), kp, kv, grip)
    motor_data, motor_arm = _plant(motor, q, qd, grip)
    motor_data.ctrl[motor_arm.act_ids] = np.clip(g_tau, motor_arm.force_lo, motor_arm.force_hi)
    mujoco.mj_step(motor, motor_data)
    pos_data, pos_arm = _plant(position, q, qd, grip)
    pos_data.ctrl[pos_arm.act_ids] = q_d
    mujoco.mj_step(position, pos_data)
    report["same_state"] = {
        "source": "B1 measured state and that step's q_d",
        "tau_motor_g": [float(v) for v in g_tau],
        "tau_position_gravcomp": [float(v) for v in grav_tau],
        "tau_formula_max_abs": float(np.max(np.abs(g_tau - grav_tau))),
        "tau_formula_max_abs_over_trajectory": formula_gap,
        "one_step_q_motor": [float(v) for v in motor_data.qpos[motor_arm.qpos_adr]],
        "one_step_q_position": [float(v) for v in pos_data.qpos[pos_arm.qpos_adr]],
        "one_step_q_abs": float(
            np.max(np.abs(motor_data.qpos[motor_arm.qpos_adr] - pos_data.qpos[pos_arm.qpos_adr]))
        ),
    }
    path = run_dir.parent / "noff_vs_b1.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("noff vs B1 " + json.dumps(report))


def _both_torques(motor, position, log, step: int, kp, kv, grip: float):
    q = log["q"][step]
    qd = log["qd"][step]
    q_d = log["q_d"][step]
    motor_data, motor_arm = _plant(motor, q, np.zeros(6), grip)
    mujoco.mj_forward(motor, motor_data)
    gravity = motor_data.qfrc_bias[motor_arm.dof_adr].copy()
    pos_data, pos_arm = _plant(position, q, np.zeros(6), grip)
    mujoco.mj_forward(position, pos_data)
    gravcomp = pos_data.qfrc_gravcomp[pos_arm.dof_adr].copy()
    shared = kp * (q_d - q) - kv * qd
    return shared + gravity, shared + gravcomp


def main() -> None:
    started = time.perf_counter()
    root = repo_root()
    out = root / "outputs" / "m3"
    run_dir = out / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    m1 = load_config()
    m3 = _load_m3()
    commit = _git_commit()
    v_max = velocity_limits(m1)
    control_dt = 1.0 / float(m1["control_hz"])
    n_sub = int(m1["physics_hz"]) // int(m1["control_hz"])
    grip = float(m1["gripper_open"])
    home_q = np.asarray(m1["home_q"], dtype=float)
    plan_model = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
    home_pos, home_quat = _home_pose(plan_model, m1["tcp_site"], home_q, grip)

    # Settled can positions, one reset per mode, position model, no rendering.
    can_pos = {}
    probe = AlohaCanEnv(m1)
    try:
        for mode in MODES:
            probe.reset(seed=0, mode=mode, render=False)
            can_pos[mode] = probe.can_center_base().copy()
            print(f"can {mode} {np.array2string(can_pos[mode], precision=4)}")
    finally:
        probe.close()

    samples = []
    for mode in MODES:
        poses = pick_place_waypoints(
            can_pos[mode],
            np.asarray(m1["modes"][mode]["place_pos"], dtype=float),
            home_pos,
            home_quat,
            clearance=float(m3["pregrasp_clearance"]),
            lift=float(m3["lift_height"]),
        )
        samples.extend(
            path_samples(poses, total_time=float(m3["ik_benchmark_time"]), control_dt=control_dt)
        )
    ik_rows = benchmark_ik(
        plan_model,
        samples,
        home_q,
        m3["ik_candidates"],
        site_name=m1["tcp_site"],
        damping=float(m3["ik_damping"]),
        max_iters=int(m3["ik_max_iters"]),
        rot_weight=float(m3["ik_rot_weight"]),
    )
    suggestion = choose_tolerance(ik_rows)
    chosen = {"pos_tol": float(m3["ik_pos_tol"]), "rot_tol": float(m3["ik_rot_tol"])}
    for row in ik_rows:
        same = (
            abs(row["pos_tol"] - chosen["pos_tol"]) < 1e-15
            and abs(row["rot_tol"] - chosen["rot_tol"]) < 1e-15
        )
        suggested = (
            abs(row["pos_tol"] - suggestion["pos_tol"]) < 1e-15
            and abs(row["rot_tol"] - suggestion["rot_tol"]) < 1e-15
        )
        row["chosen"] = same
        row["benchmark_suggestion"] = suggested
    (out / "ik_tolerance.json").write_text(json.dumps(ik_rows, indent=2), encoding="utf-8")
    print("ik tolerance " + json.dumps(ik_rows))
    print(
        f"configured pos_tol {chosen['pos_tol']} rot_tol {chosen['rot_tol']} "
        f"benchmark_suggestion {suggestion['pos_tol']} {suggestion['rot_tol']}"
    )
    if abs(suggestion["pos_tol"] - chosen["pos_tol"]) > 1e-15:
        print("benchmark suggestion differs from the configured tolerance; keeping the configuration")

    scheme = str(m3["reference_scheme"])
    a_max = np.asarray(m3["accel_limit"], dtype=float)
    print(f"reference scheme {scheme} lag_s {scheme_lag_s(scheme, control_dt):.3f}")

    ik = DlsIk(
        plan_model,
        m1["tcp_site"],
        damping=float(m3["ik_damping"]),
        max_iters=int(m3["ik_max_iters"]),
        pos_tol=float(chosen["pos_tol"]),
        rot_tol=float(chosen["rot_tol"]),
        rot_weight=float(m3["ik_rot_weight"]),
    )
    families = {}
    for mode in MODES:
        poses = pick_place_waypoints(
            can_pos[mode],
            np.asarray(m1["modes"][mode]["place_pos"], dtype=float),
            home_pos,
            home_quat,
            clearance=float(m3["pregrasp_clearance"]),
            lift=float(m3["lift_height"]),
        )
        families[mode] = build_speed_family(
            ik,
            poses,
            home_q=home_q,
            home_pos=home_pos,
            home_quat=home_quat,
            grip=grip,
            control_dt=control_dt,
            n_substeps=n_sub,
            v_max=v_max,
            hold_s=float(m3["hold_s"]),
            label=mode,
            scheme=scheme,
            a_max=a_max,
        )
    speed_dump = {
        mode: {
            speed: {
                "motion_time": traj.motion_time,
                "peak_qd": [float(v) for v in traj.peak_qd],
                "peak_over_limit": [float(v) for v in traj.peak_qd / v_max],
            }
            for speed, traj in speeds.items()
        }
        for mode, speeds in families.items()
    }
    (out / "speeds.json").write_text(json.dumps(speed_dump, indent=2), encoding="utf-8")

    gains = {"kp": m3["kp"], "kd": m3["kd"], "ki": m3["ki"]}
    menagerie = {"kp": m3["menagerie_kp"], "kd": m3["menagerie_kv"], "ki": [0.0] * 6}
    jobs = []
    for kind in CONTROLLERS:
        for mode in MODES:
            for speed in SPEEDS:
                jobs.append((kind, mode, speed, gains, None, "error"))
    jobs.append(("C2", "fixed_1", "medium", menagerie, "C2_menagerie_fixed_1_medium", "error"))
    jobs.append(("C2", "fixed_1", "medium", menagerie, "C2_menagerie_noff_fixed_1_medium", "absolute"))

    video_height = int(m3["video_height"])
    video_width = int(m3["video_width"])

    def _run_tracking(park_can: bool) -> list[str]:
        hits = []
        for kind, mode, speed, gain, name_override, vel_feedback in jobs:
            actuator = "position" if kind in ("B0", "B1") else "motor"
            grav = kind == "B1"
            cfg = dict(m1)
            cfg["actuator"] = actuator
            cfg["gravity_compensation"] = grav
            env = AlohaCanEnv(cfg)
            name = name_override or f"{kind}_{mode}_{speed}"
            record = name in ("C3_fixed_1_fast", "B0_fixed_1_fast")
            try:
                controller = _make_controller(env, kind, m3, chosen, gain, vel_feedback=vel_feedback)
                if kind in ("B0", "B1"):
                    used_gains = {
                        "kp": controller.kp_pos.tolist(),
                        "kd": controller.kv_pos.tolist(),
                        "ki": [0.0] * 6,
                    }
                else:
                    used_gains = gain
                traj = families[mode][speed]
                t0 = time.perf_counter()
                arrays, frames = _execute(
                    env,
                    controller,
                    traj,
                    record=record,
                    video_size=(video_height, video_width),
                    park_can=park_can,
                )
                meta = _meta(
                    kind, mode, speed, m3, chosen, used_gains, traj, commit, actuator, grav,
                    {
                        "name": name,
                        "menagerie_gains": name_override is not None and vel_feedback == "error",
                        "vel_feedback": vel_feedback,
                        "reference_scheme": scheme,
                        "reference_lag_s": scheme_lag_s(scheme, control_dt),
                        "can_parked": park_can,
                    },
                )
                save_run(run_dir / f"{name}.npz", arrays, meta)
                if record:
                    _write_video(frames, out / "videos" / f"{name}.mp4", float(m3["video_max_mb"]))
                elapsed = time.perf_counter() - t0
                contact_steps = int(np.count_nonzero(arrays["finger_can_contact"]))
                if contact_steps and not park_can:
                    hits.append(name)
                print(
                    f"run {name} {elapsed:.2f}s steps {arrays['t'].shape[0]} "
                    f"can_contact {contact_steps}",
                    flush=True,
                )
            finally:
                env.close()
        return hits

    contact_hits = _run_tracking(False)
    if contact_hits:
        print(f"finger-can contact in {contact_hits}; re-running tracking with the can parked")
        _run_tracking(True)
    else:
        print("main tracking runs: no finger-can contact")
    _write_noff_report(run_dir, m1, m3)

    contact_cfg = m3["contact"]
    contact = contact_trajectory(
        ik,
        home_q=home_q,
        home_pos=home_pos,
        home_quat=home_quat,
        grip=grip,
        xy=np.asarray(contact_cfg["xy"], dtype=float),
        above_z=float(contact_cfg["above_z"]),
        press_z=float(contact_cfg["press_z"]),
        control_dt=control_dt,
        n_substeps=n_sub,
        v_max=v_max,
        descend_s=float(contact_cfg["descend_s"]),
        press_s=float(contact_cfg["press_s"]),
        lift_s=float(contact_cfg["lift_s"]),
        settle_s=float(contact_cfg["settle_s"]),
        scheme=scheme,
        a_max=a_max,
    )
    for kind in ("C2", "C4"):
        cfg = dict(m1)
        cfg["actuator"] = "motor"
        cfg["gravity_compensation"] = False
        env = AlohaCanEnv(cfg)
        name = f"{kind}_contact"
        try:
            controller = _make_controller(env, kind, m3, chosen, gains)
            t0 = time.perf_counter()
            arrays, _frames = _execute(
                env, controller, contact, record=False, video_size=(video_height, video_width)
            )
            meta = _meta(
                kind, "contact", "contact", m3, chosen, gains, contact, commit, "motor", False,
                {
                    "name": name,
                    "contact_above_z": float(contact_cfg["above_z"]),
                    "contact_press_z": float(contact_cfg["press_z"]),
                    "hold_s": float(contact_cfg["settle_s"]),
                },
            )
            # Contact's steady window is the post-lift settle, whose length is
            # settle_s. The required metadata key stays present.
            meta["hold_s"] = float(contact_cfg["settle_s"])
            save_run(run_dir / f"{name}.npz", arrays, meta)
            print(f"run {name} {time.perf_counter() - t0:.2f}s steps {arrays['t'].shape[0]}", flush=True)
        finally:
            env.close()

    print(f"total_s {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
