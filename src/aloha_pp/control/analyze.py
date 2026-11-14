"""Read-only summaries of M3 logs. No simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aloha_pp.control.logs import load_run
from aloha_pp.control.reference import PHASE_HOLD, PHASE_LIFT, PHASE_PRESS

MAIN_CONTROLLERS = ("B0", "B1", "C1", "C2", "C3", "C4")
SPEEDS = ("slow", "medium", "fast")


def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _position_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def summarize_run(arrays: dict[str, np.ndarray], meta: dict) -> dict:
    q_err = arrays["q"] - arrays["q_d"]
    hold = arrays["phase"] == PHASE_HOLD
    if not np.any(hold):
        hold = arrays["t"] >= float(arrays["t"][-1]) - float(meta["hold_s"])
    tcp_ref = _position_error(arrays["tcp_pos"], arrays["tcp_ref_pos"])
    tcp_fk = _position_error(arrays["tcp_pos"], arrays["tcp_qd_pos"])
    limits = np.asarray(meta["torque_limit"], dtype=float)
    applied = np.abs(arrays["tau_applied"])
    peak_ratio = float(np.max(applied / limits))
    row = {
        "name": meta["name"],
        "controller": meta["controller"],
        "mode": meta["mode"],
        "speed": meta["speed"],
        "motion_time": float(meta["motion_time"]),
        "q_rmse": float(np.sqrt(np.mean(q_err**2))),
        "q_max": float(np.max(np.abs(q_err))),
        "tcp_ref_rmse": float(np.sqrt(np.mean(tcp_ref**2))),
        "tcp_ref_max": float(np.max(tcp_ref)),
        "tcp_fk_rmse": float(np.sqrt(np.mean(tcp_fk**2))),
        "tcp_fk_max": float(np.max(tcp_fk)),
        "tcp_ref_rot_rmse": float(np.sqrt(np.mean(_quat_angle(arrays["tcp_quat"], arrays["tcp_ref_quat"]) ** 2))),
        "steady_q_mae": float(np.mean(np.abs(q_err[hold]))),
        "steady_tcp_ref_mae": float(np.mean(tcp_ref[hold])),
        "steady_tcp_fk_mae": float(np.mean(tcp_fk[hold])),
        "sat_ratio": float(np.mean(arrays["saturated"])),
        "vel_ratio": float(np.mean(arrays["vel_exceeded"])),
        "tau_peak_ratio": peak_ratio,
        "ik_pos_max": float(np.max(arrays["ik_pos_err"])),
        "ik_rot_max": float(np.max(arrays["ik_rot_err"])),
        "can_disp_max": float(np.max(np.linalg.norm(arrays["can_pos"] - arrays["can_pos"][0], axis=1))),
        "can_contact_steps": int(np.count_nonzero(arrays["finger_can_contact"])),
    }
    return row


def load_directory(runs_dir: Path) -> list[tuple[dict[str, np.ndarray], dict, dict]]:
    rows = []
    for path in sorted(Path(runs_dir).glob("*.npz")):
        arrays, meta = load_run(path)
        meta.setdefault("name", path.stem)
        rows.append((arrays, meta, summarize_run(arrays, meta)))
    if not rows:
        raise FileNotFoundError(f"no logs in {runs_dir}")
    return rows


def write_tables(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "name",
        "controller",
        "mode",
        "speed",
        "motion_time",
        "q_rmse",
        "q_max",
        "tcp_ref_rmse",
        "tcp_ref_max",
        "tcp_fk_rmse",
        "tcp_fk_max",
        "steady_q_mae",
        "steady_tcp_ref_mae",
        "sat_ratio",
        "vel_ratio",
        "tau_peak_ratio",
        "ik_pos_max",
        "can_disp_max",
        "can_contact_steps",
    ]
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})
    md_path = out_dir / "summary.md"
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _by_name(loaded) -> dict[str, tuple]:
    return {meta["name"]: (arrays, meta, row) for arrays, meta, row in loaded}


def plot_all(loaded, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    found = _by_name(loaded)
    paths = []
    paths.append(_plot_fast_family(found, fig_dir, plt))
    paths.append(_plot_rmse_speed(loaded, fig_dir, plt))
    paths.append(_plot_b0_sag(found, fig_dir, plt))
    paths.append(_plot_c2_b1(found, fig_dir, plt))
    paths.append(_plot_contact(found, fig_dir, plt))
    return paths


def _plot_fast_family(found, fig_dir: Path, plt) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in MAIN_CONTROLLERS:
        key = f"{name}_fixed_1_fast"
        arrays, _meta, _row = found[key]
        err = np.linalg.norm(arrays["q"] - arrays["q_d"], axis=1)
        tau = np.linalg.norm(arrays["tau_applied"], axis=1)
        axes[0].plot(arrays["t"], err, label=name, linewidth=1.0)
        axes[1].plot(arrays["t"], tau, label=name, linewidth=1.0)
    axes[0].set_ylabel("||q - q_d|| (rad)")
    axes[1].set_ylabel("||tau applied|| (Nm)")
    axes[1].set_xlabel("t (s)")
    axes[0].set_title("fixed_1 fast: joint error and torque")
    axes[0].legend(ncol=6, fontsize=8)
    fig.tight_layout()
    path = fig_dir / "fixed1_fast_error_torque.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_rmse_speed(loaded, fig_dir: Path, plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, mode in zip(axes, ("fixed_1", "fixed_2")):
        for controller in MAIN_CONTROLLERS:
            ys = []
            for speed in SPEEDS:
                match = [
                    row["q_rmse"]
                    for _a, _m, row in loaded
                    if row["controller"] == controller and row["mode"] == mode and row["speed"] == speed
                ]
                ys.append(match[0] if match else np.nan)
            axis.plot(SPEEDS, ys, marker="o", label=controller)
        axis.set_title(mode)
        axis.set_xlabel("speed")
        axis.set_ylabel("joint RMSE (rad)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Joint RMSE across speed")
    fig.tight_layout()
    path = fig_dir / "rmse_vs_speed.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_b0_sag(found, fig_dir: Path, plt) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, style in (("B0_fixed_1_slow", "-"), ("B1_fixed_1_slow", "--")):
        arrays, _meta, _row = found[name]
        err = arrays["q"] - arrays["q_d"]
        axes[0].plot(arrays["t"], err[:, 1], style, label=name + " shoulder")
        axes[1].plot(arrays["t"], err[:, 2], style, label=name + " elbow")
    axes[0].set_ylabel("shoulder q - q_d (rad)")
    axes[1].set_ylabel("elbow q - q_d (rad)")
    axes[1].set_xlabel("t (s)")
    axes[0].set_title("B0 gravity sag vs B1, fixed_1 slow")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "b0_gravity_sag.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_c2_b1(found, fig_dir: Path, plt) -> Path:
    fig, axis = plt.subplots(figsize=(10, 4))
    for name in (
        "B1_fixed_1_medium",
        "C2_menagerie_fixed_1_medium",
        "C2_menagerie_noff_fixed_1_medium",
        "C2_fixed_1_medium",
    ):
        arrays, _meta, _row = found[name]
        err = np.linalg.norm(arrays["q"] - arrays["q_d"], axis=1)
        axis.plot(arrays["t"], err, label=name, linewidth=1.0)
    axis.set_xlabel("t (s)")
    axis.set_ylabel("||q - q_d|| (rad)")
    axis.set_title("C2 with Menagerie kp/kv vs B1, fixed_1 medium")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "c2_vs_b1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_contact(found, fig_dir: Path, plt) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for name in ("C2_contact", "C4_contact"):
        arrays, meta, _row = found[name]
        press = arrays["phase"] == PHASE_PRESS
        axes[0].plot(arrays["t"], arrays["tcp_pos"][:, 2], label=name)
        axes[1].plot(arrays["t"], np.linalg.norm(arrays["integral"], axis=1), label=name)
        axes[2].plot(arrays["t"], np.mean(arrays["saturated"], axis=1), label=name)
        axes[3].plot(arrays["t"], np.linalg.norm(arrays["q"] - arrays["q_d"], axis=1), label=name)
        if np.any(press):
            t0 = float(arrays["t"][press][0])
            t1 = float(arrays["t"][press][-1])
            for axis in axes:
                axis.axvspan(t0, t1, color="0.85", zorder=0)
        above = float(meta.get("contact_above_z", np.nan))
        if np.isfinite(above):
            axes[0].axhline(above, color="k", linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("TCP z (m)")
    axes[1].set_ylabel("||integral||")
    axes[2].set_ylabel("sat. fraction")
    axes[3].set_ylabel("||q - q_d||")
    axes[3].set_xlabel("t (s)")
    axes[0].set_title("Contact: press 1 cm under the bin floor, then lift")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "c4_contact.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def contact_notes(loaded) -> dict:
    """Numbers the delivery uses for the integral-windup comparison."""
    found = _by_name(loaded)
    notes = {}
    for name in ("C2_contact", "C4_contact"):
        arrays, meta, row = found[name]
        press = arrays["phase"] == PHASE_PRESS
        lift = (arrays["phase"] == PHASE_LIFT) | (arrays["phase"] == PHASE_HOLD)
        above = float(meta["contact_above_z"])
        z = arrays["tcp_pos"][:, 2]
        notes[name] = {
            "press_sat_ratio": float(np.mean(arrays["saturated"][press])) if np.any(press) else float("nan"),
            "press_integral_max": float(np.max(np.abs(arrays["integral"][press]))) if np.any(press) else 0.0,
            "lift_z_overshoot": float(np.max(z[lift]) - above) if np.any(lift) else float("nan"),
            "lift_q_max": float(np.max(np.abs(arrays["q"][lift] - arrays["q_d"][lift]))) if np.any(lift) else float("nan"),
            "q_rmse": row["q_rmse"],
            "sat_ratio": row["sat_ratio"],
        }
    return notes
