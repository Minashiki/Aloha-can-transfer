"""Retarget Can-PH and, if needed, fill with noisy planner demonstrations.

One closed-loop execution both decides success and records the images that
are written. Actions are cast to float32 before the first ``env.step``.
``next_actions`` stores the preview that step actually received.

Run: conda run -n AIControl python -u scripts/m5_generate.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m4_run import _controller, _gains, _git_commit, _motor_cfg  # noqa: E402

from aloha_pp.control.quintic import simulate_stream
from aloha_pp.data.dataset import DatasetWriter
from aloha_pp.data.retarget import HUMAN_INFO_PHASES, PHASE_NAMES, retarget_demo
from aloha_pp.kin.ik import DlsIk, quat_to_mat
from aloha_pp.planning.noise import NoisyPickPlacePlanner
from aloha_pp.planning.schedule import STATE_NAMES
from aloha_pp.robot import repo_root, scene_xml
from aloha_pp.sim.env import AlohaCanEnv, load_config
from aloha_pp.sim.success import in_target, is_settled

import mujoco


JOINTS = (
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
)


def _load(name: str) -> dict:
    path = repo_root() / "configs" / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _demo_names(handle: h5py.File) -> list[str]:
    return sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))


def _read_demo(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = group["obs"]
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    if "object" not in obs:
        raise KeyError("obs/object is missing; the can pose is the first 7 values of that dataset")
    obj = np.asarray(obs["object"], dtype=float)
    if obj.shape[-1] < 7:
        raise ValueError(f"object obs width {obj.shape[-1]} is below 7")
    return eef, obj[:, :3], grip, quat


def _quantize(action: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(action, dtype=np.float32), dtype=np.float64)


def _snapshot(env: AlohaCanEnv, obs: dict) -> dict:
    privileged = env.get_privileged()
    return {
        "agentview": obs["agentview"],
        "wrist_cam_left": obs["wrist_cam_left"],
        "tcp_pose": obs["tcp_pose"].astype(np.float32),
        "gripper_opening": obs["gripper_opening"].astype(np.float32),
        "q": obs["q"].astype(np.float32),
        "qvel": obs["qvel"].astype(np.float32),
        "can_pose": privileged["can_pose"].astype(np.float32),
    }


def _consecutive(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _rollout(
    env: AlohaCanEnv,
    actions: np.ndarray,
    *,
    render: bool,
    extra: int,
    phase: np.ndarray | None,
    closed_value: float,
    closed_tol: float,
    v_max: np.ndarray,
) -> dict:
    """Execute stored-precision actions. Record images in this same pass."""
    actions = np.asarray(actions, dtype=np.float64)
    obs = env.get_obs(render=render)
    frames: list[dict] = []
    used: list[np.ndarray] = []
    nxts: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    q_cmds: list[np.ndarray] = []
    phases: list[int] = []
    phys_contact: list[np.ndarray] = []
    phys_can: list[np.ndarray] = []
    phys_vel: list[np.ndarray] = []
    phys_qvel: list[np.ndarray] = []
    closed_flags: list[bool] = []
    holder = {"closed": False}
    close = np.float32(closed_value)
    tol = np.float32(closed_tol)

    def on_substep(live: AlohaCanEnv) -> None:
        phys_contact.append(live.finger_can_contacts().copy())
        phys_can.append(live.can_center_base().copy())
        phys_vel.append(live.can_linvel().copy())
        phys_qvel.append(live.data.qvel[live.arm_dof_adr].copy())
        closed_flags.append(bool(holder["closed"]))

    ik_ok = True
    success = False
    total = len(actions) + int(extra)
    last_phase = 0 if phase is None or len(phase) == 0 else int(phase[-1])
    for t in range(total):
        frames.append(_snapshot(env, obs))
        qpos_rows.append(env.data.qpos.copy())
        index = min(t, len(actions) - 1)
        nxt_index = min(t + 1, len(actions) - 1)
        action = _quantize(actions[index])
        next_action = _quantize(actions[nxt_index])
        holder["closed"] = abs(np.float32(action[7]) - close) <= tol
        obs, success, info = env.step(
            action, on_substep=on_substep, render=render, next_action=next_action
        )
        used.append(action.copy())
        nxts.append(next_action.copy())
        q_cmds.append(np.asarray(env.controller._seed, dtype=float).copy())
        if phase is None:
            phases.append(0)
        elif t < len(phase):
            phases.append(int(phase[t]))
        else:
            phases.append(last_phase)
        ik_ok = ik_ok and bool(info.get("ik_success", True))
        if success:
            break
    qvel = np.stack(phys_qvel)
    exec_ratio, exec_step, exec_joint = _exec_peak(qvel, v_max)
    return {
        "frames": frames,
        "actions": np.stack(used),
        "next_actions": np.stack(nxts),
        "phase": np.asarray(phases, dtype=np.int8),
        "qpos": np.stack(qpos_rows),
        "q_cmds": np.stack(q_cmds),
        "phys_contact": np.stack(phys_contact),
        "phys_can": np.stack(phys_can),
        "phys_vel": np.stack(phys_vel),
        "phys_qvel": qvel,
        "phys_closed": np.asarray(closed_flags, dtype=bool),
        "ik_ok": ik_ok,
        "success": bool(success),
        "exec_peak_ratio": exec_ratio,
        "exec_step": exec_step,
        "exec_joint": exec_joint,
    }


def _exec_peak(qvel: np.ndarray, v_max: np.ndarray) -> tuple[float, int, int]:
    ratio = np.abs(qvel) / v_max.reshape(1, -1)
    flat = int(np.argmax(ratio))
    step, joint = np.unravel_index(flat, ratio.shape)
    return float(ratio[step, joint]), int(step), int(joint)


def _classify(env: AlohaCanEnv, rollout: dict, m5: dict, drop_z: float) -> tuple[str | None, float]:
    """Process labels. ``not_lifted`` and ``dropped_after_lift`` precede ``outside``."""
    if rollout["success"]:
        contact = rollout["phys_contact"]
        closed = rollout["phys_closed"]
        cans = rollout["phys_can"]
        lift = 0.0
        if np.any(closed):
            start = int(np.flatnonzero(closed)[0])
            lift = float(np.max(cans[closed, 2]) - cans[start, 2])
        return None, lift
    if not rollout["ik_ok"]:
        return "ik_fail", 0.0
    contact = rollout["phys_contact"]
    closed = rollout["phys_closed"]
    cans = rollout["phys_can"]
    both = contact[:, 0] & contact[:, 1]
    if not np.any(closed):
        grasped = False
        lift = 0.0
        not_lifted = False
        dropped = False
        ref_z = float(cans[0, 2])
    else:
        start = int(np.flatnonzero(closed)[0])
        end = start
        while end < len(closed) and closed[end]:
            end += 1
        window = np.zeros(len(closed), dtype=bool)
        window[start:end] = True
        ref_z = float(cans[start, 2])
        rise = cans[:, 2] - ref_z
        lift = float(np.max(rise[window]))
        thresh = float(m5["lift_thresh_m"])
        not_lifted = lift < thresh
        grasped = bool(np.any(both & window))
        running = np.maximum.accumulate(np.where(window, rise, -np.inf))
        after = window & (running >= thresh)
        neither = ~(contact[:, 0] | contact[:, 1])
        lost = _consecutive(after & neither) >= int(m5["lost_physics_steps"])
        fell = bool(np.any(after & (cans[:, 2] < ref_z - float(drop_z))))
        dropped = bool(grasped and (not not_lifted) and (lost or fell))
    released = not bool(contact[-1, 0] or contact[-1, 1])
    cell = in_target(cans[-1], env.tracker.center, env.tracker.half_extents)
    still = is_settled(rollout["phys_vel"][-1], float(env.cfg["success"]["vel_thresh"]))
    if not grasped:
        return "not_grasped", lift
    if not_lifted:
        return "not_lifted", lift
    if dropped:
        return "dropped_after_lift", lift
    if not released:
        return "not_released", lift
    if not cell:
        return "outside", lift
    if not still:
        return "not_still", lift
    return "timeout", lift


def _tcp_offset(tcp: np.ndarray, quat: np.ndarray, can: np.ndarray) -> tuple[float, float, float]:
    """Finger-close, perpendicular, and world-vertical components of can - tcp.

    ``top_down_quat`` puts the approach on site +X (down) and the finger
    opening on site +Y. The third site axis is the other horizontal.
    """
    local = quat_to_mat(quat).T @ (np.asarray(can, dtype=float) - np.asarray(tcp, dtype=float))
    return float(local[1]), float(local[2]), float(can[2] - tcp[2])


def _source_offset(eef: np.ndarray, quat: np.ndarray, can: np.ndarray) -> tuple[float, float, float, float]:
    """Panda grip-site offset (approach is +Z) and that axis's world-z component."""
    rot = quat_to_mat(quat)
    local = rot.T @ (np.asarray(can, dtype=float) - np.asarray(eef, dtype=float))
    return float(local[0]), float(local[1]), float(local[2]), float(rot[2, 2])


def _reference_peak(
    q_cmds: np.ndarray,
    q0: np.ndarray,
    v_max: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
    a_max: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, bool]:
    speeds, _acc, flags = simulate_stream(
        q_cmds,
        q0,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme="preview",
        a_max=a_max,
    )
    peak = np.max(np.abs(speeds), axis=0)
    ok = bool(np.all(peak <= float(ratio) * v_max) and not np.any(flags))
    return peak, ok


def _replay_qpos(env: AlohaCanEnv, state: dict, actions: np.ndarray, next_actions: np.ndarray) -> np.ndarray:
    env.reset(mode="random", state=state, render=False)
    rows = []
    for t in range(len(actions)):
        rows.append(env.data.qpos.copy())
        env.step(actions[t], render=False, next_action=next_actions[t])
    return np.stack(rows)


def _manifest() -> dict[str, str]:
    root = repo_root()
    paths = sorted((root / "src" / "aloha_pp" / "data").rglob("*"))
    rels = [path.relative_to(root) for path in paths if path.is_file()]
    for rel in (
        "src/aloha_pp/planning/noise.py",
        "src/aloha_pp/sim/env.py",
        "scripts/m5_generate.py",
        "scripts/m4_run.py",
        "configs/m1_pick_place.yaml",
        "configs/m3_track.yaml",
        "configs/m4_task.yaml",
        "configs/m5_retarget.yaml",
        "configs/m5_noise.yaml",
    ):
        rels.append(Path(rel))
    out = {}
    for rel in rels:
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        out[str(rel)] = digest
    return out


def _git_dirty() -> str:
    text = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root(), text=True)
    return "true" if text.strip() else "false"


class _Writer:
    def __init__(self, path: Path, attrs: dict) -> None:
        self.path = path
        self.attrs = attrs
        self.inner: DatasetWriter | None = None
        self.count = 0

    def add(self, episode: dict) -> None:
        if self.inner is None:
            self.inner = DatasetWriter(self.path)
            self.inner.set_attrs(self.attrs)
        self.inner.add_episode(episode)
        self.count += 1

    def close(self) -> None:
        if self.inner is not None:
            self.inner.close()


def _episode(rollout: dict, *, source: str, source_id: int, slowdown: float, state: dict, scheme: str) -> dict:
    frames = rollout["frames"]
    obs = {key: np.stack([frame[key] for frame in frames]) for key in (
        "agentview", "wrist_cam_left", "tcp_pose", "gripper_opening", "q", "qvel"
    )}
    phase = np.asarray(rollout["phase"], dtype=np.int8)
    human = float(np.mean(np.isin(phase, HUMAN_INFO_PHASES))) if scheme == "retarget" else float("nan")
    episode = {
        "obs": obs,
        "can_pose": np.stack([frame["can_pose"] for frame in frames]),
        "actions": np.asarray(rollout["actions"], dtype=np.float32),
        "next_actions": np.asarray(rollout["next_actions"], dtype=np.float32),
        "phase": phase,
        "source": source,
        "source_id": int(source_id),
        "slowdown": float(slowdown),
        "success_step": len(frames) - 1,
        "phase_scheme": scheme,
        "human_fraction": human,
        "qpos0": state["qpos"],
        "qvel0": state["qvel"],
        "ctrl0": state["ctrl"],
        "time0": state["time"],
        "qacc_warmstart0": state["qacc_warmstart"],
        "qacc0": state["qacc"],
    }
    if "act" in state:
        episode["act0"] = state["act"]
    return episode


def _row(source: str, source_id: int, success: bool, reason: str | None, **fields) -> dict:
    row = {
        "source": source,
        "source_id": source_id,
        "success": success,
        "reason": "" if reason is None else reason,
        "kept": False,
    }
    row.update(fields)
    return row


def _scale_cols(result: dict) -> dict:
    scales = result.get("segment_scales") or {}
    columns = {
        "synth_approach": "scale_synth_approach",
        "human_pregrasp": "scale_human_pregrasp",
        "carry": "scale_carry",
        "place_slide": "scale_place_slide",
        "place_down": "scale_place_down",
        "retreat": "scale_retreat",
    }
    return {
        column: f"{float(scales[name]):.4f}" if name in scales else ""
        for name, column in columns.items()
    }


def _blank(**fields) -> dict:
    base = {
        "slowdown": "",
        "length": 0,
        "ref_peak_ratio": "",
        "ref_unlimited_ratio": "",
        "exec_peak_ratio": "",
        "exec_peak_joint": "",
        "lift_m": "",
        "can_x": "",
        "can_y": "",
        "place_xy": "",
        "clamped": "",
        "human_fraction": "",
        "scale_synth_approach": "",
        "scale_human_pregrasp": "",
        "scale_carry": "",
        "scale_place_slide": "",
        "scale_place_down": "",
        "scale_retreat": "",
    }
    base.update(fields)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="retarget only the first N demos (0 = all)")
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    root = repo_root()
    out = root / "outputs" / "m5"
    out.mkdir(parents=True, exist_ok=True)
    source = args.source if args.source is not None else root / "datasets" / "robomimic" / "can" / "ph" / "low_dim_v141.hdf5"
    m1 = load_config()
    m3 = _load("m3_track.yaml")
    m4 = _load("m4_task.yaml")
    m5 = _load("m5_retarget.yaml")
    noise_cfg = _load("m5_noise.yaml")
    v_max = np.array([m1["velocity_limit"][name] for name in JOINTS], dtype=float)
    a_max = np.asarray(m3["accel_limit"], dtype=float)
    plan_model = mujoco.MjModel.from_xml_path(str(scene_xml("motor")))
    ik = DlsIk(
        plan_model,
        m1["tcp_site"],
        damping=float(m3["ik_damping"]),
        max_iters=int(m3["ik_max_iters"]),
        pos_tol=float(m3["ik_pos_tol"]),
        rot_tol=float(m3["ik_rot_tol"]),
        rot_weight=float(m3["ik_rot_weight"]),
    )
    env = AlohaCanEnv(_motor_cfg(m1))
    env.controller = _controller(env, m3, *_gains(m3, m4))
    env.reset(seed=0, mode="fixed_1", render=False)
    env.controller.reset(env.model, env.data)
    home_pos, home_quat = env.tcp_pose()
    q_init = np.asarray(env.controller.home_q, dtype=float).copy()
    dt = 1.0 / float(m1["control_hz"])
    extra = int(round(float(m5["success_wait_s"]) / dt))
    drop_z = float(m4["drop_z"])
    ratio = float(m5["speed_ratio"])
    rows: list[dict] = []
    geometry: list[dict] = []
    ik_rows: list[dict] = []
    over: list[dict] = []
    checked = {"retarget": False, "planner_noise": False}
    fieldnames = [
        "source", "source_id", "success", "reason", "kept", "slowdown", "length",
        "ref_peak_ratio", "ref_unlimited_ratio", "exec_peak_ratio", "exec_peak_joint",
        "lift_m", "can_x", "can_y", "place_xy", "clamped", "human_fraction",
        "scale_synth_approach", "scale_human_pregrasp", "scale_carry",
        "scale_place_slide", "scale_place_down", "scale_retreat",
    ]

    def flush() -> None:
        with (out / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def place_xy(rollout: dict) -> float:
        delta = rollout["phys_can"][-1, :2] - env.tracker.center[:2]
        return float(np.hypot(delta[0], delta[1]))

    def consider(kind: str, state: dict, rollout: dict) -> None:
        if checked[kind]:
            return
        replayed = _replay_qpos(env, state, rollout["actions"], rollout["next_actions"])
        diff = float(np.max(np.abs(replayed - rollout["qpos"])))
        print(f"bit-exact {kind} max_qpos {diff:.3e}", flush=True)
        if diff != 0.0:
            raise RuntimeError(f"{kind} generate/replay qpos max diff {diff}")
        checked[kind] = True

    commit = _git_commit()
    attrs = {
        "control_hz": int(m1["control_hz"]),
        "camera_names": ["agentview", "wrist_cam_left"],
        "image_height": int(m1["camera"]["height"]),
        "image_width": int(m1["camera"]["width"]),
        "action_convention": (
            "base-frame absolute TCP pose, xyz metres, quaternion wxyz, gripper opening metres; "
            "action t is the pose at t+1 (last repeats); next_actions[t] is the preview passed to step"
        ),
        "reference_mode": "preview",
        "method": "部分重定向 + 合成放置",
        "git_commit": commit,
        "git_dirty": _git_dirty(),
        "file_sha256": _manifest(),
        "config_json": json.dumps({"m5_retarget": m5, "m5_noise": noise_cfg, "m4_drop_z": drop_z}),
    }
    main_writer = _Writer(out / "dataset" / "demos.hdf5", attrs)
    backup_writer = _Writer(out / "dataset" / "planner_noise.hdf5", attrs)

    def run_actions(actions, phase, state) -> dict:
        del state
        q0 = np.asarray(env.controller.home_q, dtype=float).copy()
        rollout = _rollout(
            env, actions, render=True, extra=extra, phase=phase,
            closed_value=float(m1["gripper_close"]),
            closed_tol=float(m5["grip_closed_tol"]),
            v_max=v_max,
        )
        rollout["q0"] = q0
        return rollout

    t0 = time.perf_counter()
    with h5py.File(source, "r") as handle:
        names = _demo_names(handle)
        if args.limit:
            names = names[: args.limit]
        print(f"source {source} demos {len(names)}", flush=True)
        openings = []
        for name in names:
            _eef, _can, grip, _quat = _read_demo(handle["data"][name])
            openings.append(np.abs(grip).sum(axis=-1))
        opening = np.concatenate(openings)
        print(
            "gripper opening min/p10/median/p90/max "
            f"{opening.min():.4f} {np.percentile(opening, 10):.4f} "
            f"{np.median(opening):.4f} {np.percentile(opening, 90):.4f} {opening.max():.4f}",
            flush=True,
        )
        for index, name in enumerate(names):
            eef, can, grip, quat = _read_demo(handle["data"][name])
            demo_id = int(name.split("_")[-1])
            try:
                result = retarget_demo(
                    eef, can, grip, m1=m1, m5=m5,
                    home_pos=home_pos, home_quat=home_quat,
                    ik=ik, q_init=q_init, v_max=v_max,
                    n_substeps=env.n_substeps, a_max=a_max,
                )
            except ValueError as exc:
                rows.append(_row("retarget", demo_id, False, "no_grasp_segment", **_blank()))
                print(f"retarget {demo_id} no_grasp_segment {exc}", flush=True)
                continue
            ref_ratio = ""
            unlimited = ""
            if result["ref_peak"] is not None:
                ref_ratio = float(np.max(result["ref_peak"] / v_max))
                unlimited = float(np.max(result["ref_peak_unlimited"] / v_max))
            sx, sy, sz, approach_z = _source_offset(
                eef[result["source_grasp"]], quat[result["source_grasp"]], can[result["source_grasp"]]
            )
            g = int(result["grasp_index"])
            close_off, perp_off, height_off = _tcp_offset(
                result["actions"][g, :3], result["actions"][g, 3:7], result["can_pos"]
            )
            geometry.append({
                "source_id": demo_id,
                "success": "",
                "reason": "",
                "close_offset_m": f"{close_off:.6f}",
                "perp_offset_m": f"{perp_off:.6f}",
                "height_offset_m": f"{height_off:.6f}",
                "source_x_m": f"{sx:.6f}",
                "source_y_m": f"{sy:.6f}",
                "source_z_m": f"{sz:.6f}",
                "source_approach_z": f"{approach_z:.6f}",
            })
            ik_err = result["ik_pos_err"]
            unreachable = bool(ik_err is not None and float(np.max(ik_err)) > float(m5["ik_skip_pos_err"]))
            if unreachable and ik_err is not None:
                worst = int(np.argmax(ik_err))
                ik_rows.append({
                    "source_id": demo_id,
                    "phase": PHASE_NAMES[int(result["phase"][worst])],
                    "pos_x": f"{result['actions'][worst, 0]:.5f}",
                    "pos_y": f"{result['actions'][worst, 1]:.5f}",
                    "pos_z": f"{result['actions'][worst, 2]:.5f}",
                    "pos_err_m": f"{float(ik_err[worst]):.6f}",
                })
            if unreachable or result["can_out_of_bin"] or not result["speed_ok"]:
                if result["can_out_of_bin"]:
                    reason = "can_out_of_bin"
                elif unreachable:
                    reason = "ik_unreachable"
                else:
                    reason = "ref_speed"
                rows.append(_row(
                    "retarget", demo_id, False, reason,
                    **_blank(
                        slowdown=f"{result['slowdown']:.4f}",
                        ref_peak_ratio=f"{ref_ratio:.4f}" if ref_ratio != "" else "",
                        ref_unlimited_ratio=f"{unlimited:.4f}" if unlimited != "" else "",
                        can_x=f"{result['can_pos'][0]:.5f}",
                        can_y=f"{result['can_pos'][1]:.5f}",
                        clamped=result["can_clamped"],
                        **_scale_cols(result),
                    ),
                ))
                geometry[-1]["reason"] = reason
                print(f"retarget {demo_id} {reason} slowdown {result['slowdown']:.2f} ref {ref_ratio}", flush=True)
                continue
            can_pose = np.concatenate((result["can_pos"], result["can_quat"]))
            env.reset(mode="random", can_pose=can_pose, render=False)
            state = env.capture_state()
            env.controller.reset(env.model, env.data)
            rollout = run_actions(result["actions"], result["phase"], state)
            exec_peak, exec_ok = _reference_peak(
                rollout["q_cmds"], rollout["q0"], v_max,
                control_dt=dt, n_substeps=env.n_substeps, a_max=a_max, ratio=ratio,
            )
            exec_ref = float(np.max(exec_peak / v_max))
            reason, lift = _classify(env, rollout, m5, drop_z)
            success = reason is None
            kept = bool(success and exec_ok)
            human = float(np.mean(np.isin(rollout["phase"], HUMAN_INFO_PHASES)))
            rows.append(_row(
                "retarget", demo_id, success, reason, kept=kept,
                **_blank(
                    slowdown=f"{result['slowdown']:.4f}",
                    length=len(rollout["frames"]),
                    ref_peak_ratio=f"{exec_ref:.4f}",
                    ref_unlimited_ratio=f"{unlimited:.4f}" if unlimited != "" else "",
                    exec_peak_ratio=f"{rollout['exec_peak_ratio']:.4f}",
                    exec_peak_joint=JOINTS[rollout["exec_joint"]],
                    lift_m=f"{lift:.5f}",
                    can_x=f"{result['can_pos'][0]:.5f}",
                    can_y=f"{result['can_pos'][1]:.5f}",
                    place_xy=f"{place_xy(rollout):.6f}",
                    clamped=result["can_clamped"],
                    human_fraction=f"{human:.4f}",
                    **_scale_cols(result),
                ),
            ))
            geometry[-1]["success"] = success
            geometry[-1]["reason"] = "" if reason is None else reason
            if rollout["exec_peak_ratio"] > 0.6:
                over.append({
                    "source": "retarget",
                    "source_id": demo_id,
                    "exec_peak_ratio": f"{rollout['exec_peak_ratio']:.4f}",
                    "joint": JOINTS[rollout["exec_joint"]],
                    "physics_step": rollout["exec_step"],
                    "ref_peak_ratio": f"{exec_ref:.4f}",
                    "kept": kept,
                })
            if kept:
                consider("retarget", state, rollout)
                main_writer.add(_episode(
                    rollout, source="retarget", source_id=demo_id,
                    slowdown=result["slowdown"], state=state, scheme="retarget",
                ))
                print(f"wrote retarget {demo_id}", flush=True)
            print(
                f"retarget {index + 1}/{len(names)} id {demo_id} "
                f"{'ok' if success else reason} kept {kept} slowdown {result['slowdown']:.2f} "
                f"ref {exec_ref:.3f} exec {rollout['exec_peak_ratio']:.3f} "
                f"ticks {len(rollout['frames'])} iters {result['iterations']}",
                flush=True,
            )
            if (index + 1) % 10 == 0:
                flush()
    retarget_s = time.perf_counter() - t0
    n_ok = main_writer.count
    print(f"retarget successes {n_ok} in {retarget_s:.1f}s", flush=True)
    flush()

    t1 = time.perf_counter()
    need = 0 if n_ok >= int(m5["min_success"]) else int(m5["min_success"]) - n_ok
    backup_target = int(noise_cfg["backup_episodes"]) if need == 0 else need
    noise_ok = 0
    filled = 0
    seed = int(noise_cfg["seed_start"])
    attempts = 0
    while noise_ok < backup_target and attempts < int(noise_cfg["max_attempts"]):
        attempts += 1
        env.reset(seed=seed, mode="random", render=False)
        state = env.capture_state()
        env.controller.reset(env.model, env.data)
        planner = NoisyPickPlacePlanner(m1, m4, m3, "medium", noise_cfg, np.random.default_rng(seed))
        planner.reset(env, mode="random", log=lambda *_a, **_k: None)
        slowdown = float(np.max(planner._draw["time"])) if planner._draw is not None else 1.0
        q0 = np.asarray(env.controller.home_q, dtype=float).copy()
        obs = env.get_obs(render=True)
        frames = []
        used = []
        nxts = []
        qpos_rows = []
        q_cmds = []
        phases = []
        phys_contact = []
        phys_can = []
        phys_vel = []
        phys_qvel = []
        closed_flags = []
        holder = {"closed": False}
        close = np.float32(m1["gripper_close"])
        tol = np.float32(m5["grip_closed_tol"])

        def on_substep(live: AlohaCanEnv) -> None:
            phys_contact.append(live.finger_can_contacts().copy())
            phys_can.append(live.can_center_base().copy())
            phys_vel.append(live.can_linvel().copy())
            phys_qvel.append(live.data.qvel[live.arm_dof_adr].copy())
            closed_flags.append(bool(holder["closed"]))

        ik_ok = True
        success = False
        guard = 0
        limit = len(planner.schedule) + extra
        while not success and guard < limit:
            frames.append(_snapshot(env, obs))
            qpos_rows.append(env.data.qpos.copy())
            action, nxt, tick = planner.action_pair()
            action = _quantize(action)
            nxt = _quantize(nxt)
            holder["closed"] = abs(np.float32(action[7]) - close) <= tol
            obs, success, info = env.step(action, on_substep=on_substep, render=True, next_action=nxt)
            used.append(action.copy())
            nxts.append(nxt.copy())
            q_cmds.append(np.asarray(env.controller._seed, dtype=float).copy())
            phases.append(int(tick.state))
            ik_ok = ik_ok and bool(info.get("ik_success", True))
            planner.after_step(env)
            guard += 1
        qvel = np.stack(phys_qvel)
        exec_ratio, _step, joint = _exec_peak(qvel, v_max)
        rollout = {
            "frames": frames,
            "actions": np.stack(used),
            "next_actions": np.stack(nxts),
            "phase": np.asarray(phases, dtype=np.int8),
            "qpos": np.stack(qpos_rows),
            "q_cmds": np.stack(q_cmds),
            "q0": q0,
            "phys_contact": np.stack(phys_contact),
            "phys_can": np.stack(phys_can),
            "phys_vel": np.stack(phys_vel),
            "phys_qvel": qvel,
            "phys_closed": np.asarray(closed_flags, dtype=bool),
            "ik_ok": ik_ok,
            "success": bool(success),
            "exec_peak_ratio": exec_ratio,
            "exec_joint": joint,
            "exec_step": _step,
        }
        reason, lift = _classify(env, rollout, m5, drop_z)
        success = reason is None
        exec_peak, exec_ok = _reference_peak(
            rollout["q_cmds"], q0, v_max,
            control_dt=dt, n_substeps=env.n_substeps, a_max=a_max, ratio=ratio,
        )
        exec_ref = float(np.max(exec_peak / v_max))
        if success and not exec_ok:
            reason = "ref_speed"
            success = False
        kept = success
        can0 = frames[0]["can_pose"][:3]
        rows.append(_row(
            "planner_noise", seed, success, reason, kept=kept,
            **_blank(
                slowdown=f"{slowdown:.4f}",
                length=len(frames),
                ref_peak_ratio=f"{exec_ref:.4f}",
                exec_peak_ratio=f"{exec_ratio:.4f}",
                exec_peak_joint=JOINTS[joint],
                lift_m=f"{lift:.5f}",
                can_x=f"{can0[0]:.5f}",
                can_y=f"{can0[1]:.5f}",
                place_xy=f"{place_xy(rollout):.6f}",
                clamped=False,
            ),
        ))
        if exec_ratio > 0.6:
            over.append({
                "source": "planner_noise",
                "source_id": seed,
                "exec_peak_ratio": f"{exec_ratio:.4f}",
                "joint": JOINTS[joint],
                "physics_step": _step,
                "ref_peak_ratio": f"{exec_ref:.4f}",
                "kept": kept,
            })
        if kept:
            consider("planner_noise", state, rollout)
            episode = _episode(
                rollout, source="planner_noise", source_id=seed,
                slowdown=slowdown, state=state, scheme="planner",
            )
            backup_writer.add(episode)
            if filled < need:
                main_writer.add(episode)
                filled += 1
            noise_ok += 1
        print(
            f"planner seed {seed} {'ok' if success else reason} kept {kept} "
            f"ref {exec_ref:.3f} ({noise_ok}/{backup_target})",
            flush=True,
        )
        seed += 1
    planner_s = time.perf_counter() - t1
    flush()
    main_writer.close()
    backup_writer.close()
    env.close()

    def _write_csv(path: Path, table: list[dict]) -> None:
        if not table:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    _write_csv(out / "grasp_geometry.csv", geometry)
    _write_csv(out / "ik_unreachable.csv", ik_rows)
    _write_csv(out / "exec_peak_over_0.6.csv", over)
    total = time.perf_counter() - started
    timing = {
        "retarget_s": retarget_s,
        "planner_s": planner_s,
        "render_s": 0.0,
        "total_s": total,
        "retarget_success": n_ok,
        "planner_success": noise_ok,
        "main_episodes": main_writer.count,
        "backup_episodes": backup_writer.count,
        "source": str(source),
        "note": "images are recorded in the deciding execution; render_s is included in retarget_s and planner_s",
        "r1_two_pass_flips": "9/122 retarget successes flipped on the second render pass because qacc_warmstart was dropped and actions were quantized only at save time",
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2), flush=True)
    # STATE_NAMES is the planner phase vocabulary stored under phase_scheme=planner.
    print("planner phase names", ",".join(STATE_NAMES), flush=True)


if __name__ == "__main__":
    main()
