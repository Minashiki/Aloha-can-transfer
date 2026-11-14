"""Read outputs/m4/runs and write the summary tables and figures. No simulation.

Run: conda run -n AIControl python -u scripts/m4_analyze.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from aloha_pp.control.logs import load_run
from aloha_pp.planning.metrics import (
    hold_mask,
    placement_errors,
    qdd_ripple,
    release_split,
    slip_peak,
    tilt_angle,
    wilson_interval,
    wrist_peaks,
)
from aloha_pp.robot import repo_root

JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
LIMITS = np.array([10.0, 20.0, 20.0, 10.0, 10.0, 4.0])


def _rmse(err: np.ndarray) -> float:
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.asarray(err, dtype=float) ** 2)))


def _tcp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def summarize(arrays: dict, meta: dict) -> dict:
    q_err = arrays["q"] - arrays["q_d"]
    tcp_ref = _tcp(arrays["tcp_pos"], arrays["tcp_ref_pos"])
    tcp_fk = _tcp(arrays["tcp_pos"], arrays["tcp_qd_pos"])
    limits = np.asarray(meta.get("torque_limit", LIMITS), dtype=float)
    tau = np.asarray(arrays["tau_cmd"], dtype=float)
    sat = np.asarray(arrays["saturated"], dtype=bool)
    row = {
        "name": meta["name"],
        "experiment": meta.get("experiment", ""),
        "controller": meta["controller"],
        "mode": meta["mode"],
        "speed": meta["speed"],
        "seed": meta.get("seed", ""),
        "reference_mode": meta.get("reference_mode", ""),
        "success": bool(meta.get("success", False)),
        "failure": meta.get("failure") or "",
        "success_time": meta.get("success_time") if meta.get("success_time") is not None else "",
        "motion_time": float(meta["motion_time"]),
        "episode_time": float(arrays["t"][-1]),
        "q_rmse": _rmse(q_err),
        "q_max": float(np.max(np.abs(q_err))),
        "tcp_ref_rmse": _rmse(tcp_ref),
        "tcp_fk_rmse": _rmse(tcp_fk),
        "sat_ratio": float(np.mean(sat)),
        "vel_ratio": float(np.mean(arrays["vel_exceeded"])),
        "tau_peak": float(np.max(np.abs(tau))),
        "tau_peak_ratio": float(np.max(np.abs(tau) / limits)),
    }
    for index, joint in enumerate(JOINTS):
        row[f"sat_{joint}"] = float(np.mean(sat[:, index]))
        row[f"tau_peak_{joint}"] = float(np.max(np.abs(tau[:, index])))
    ripple = qdd_ripple(arrays["qdd_d"], joint=1, window=25)
    row.update(ripple)
    if meta.get("experiment") == "E4":
        return row
    mask = hold_mask(arrays["planner_state"])
    row["q_rmse_hold"] = _rmse(q_err[mask])
    row["q_max_hold"] = float(np.max(np.abs(q_err[mask]))) if np.any(mask) else float("nan")
    row["tcp_ref_rmse_hold"] = _rmse(tcp_ref[mask]) if np.any(mask) else float("nan")
    row["tcp_fk_rmse_hold"] = _rmse(tcp_fk[mask]) if np.any(mask) else float("nan")
    final = arrays["can_pos"][-1]
    center = np.asarray(meta["target_center"], dtype=float)
    place = np.asarray(meta["place_pos"], dtype=float)
    errors = placement_errors(final, center, place)
    row["place_xy"] = errors["xy"]
    row["place_xyz"] = errors["xyz"]
    row["place_vs_place_pos"] = errors["place"]
    row["tilt"] = tilt_angle(arrays["can_quat"][-1])
    row["slip"] = slip_peak(arrays["can_pos"], arrays["tcp_pos"], arrays["tcp_quat"], mask)
    row["tcp_err_close"] = float(meta.get("tcp_err_close", float("nan")))
    row["tcp_err_open"] = float(meta.get("tcp_err_open", float("nan")))
    full = wrist_peaks(q_err, tau, limits, None)
    held = wrist_peaks(q_err, tau, limits, mask)
    for key, value in full.items():
        row[key] = value
    for key, value in held.items():
        row[f"{key}_hold"] = value
    row["peak_qd"] = float(np.max(np.abs(arrays["qd"])))
    row.update(release_split(arrays, meta))
    return row


def _fmt(value) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _e2_summary_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)
    k = sum(1 for row in rows if row["success"])
    low, high = wilson_interval(k, n)
    counts = Counter(row["failure"] or "success" for row in rows)
    xy = np.array([row["place_xy"] for row in rows], dtype=float)
    xyz = np.array([row["place_xyz"] for row in rows], dtype=float)
    summary = {key: "" for key in rows[0]}
    summary.update({
        "name": "E2_summary",
        "experiment": "E2",
        "controller": "C3",
        "mode": "random",
        "speed": "medium",
        "reference_mode": "preview",
        "success": k == n,
        "failure": ", ".join(f"{name}:{count}" for name, count in sorted(counts.items())),
        "n": n,
        "n_success": k,
        "success_rate": k / n,
        "wilson_low": low,
        "wilson_high": high,
        "place_xy": float(np.median(xy)),
        "place_xyz": float(np.median(xyz)),
        "place_xy_max": float(np.max(xy)),
        "place_xyz_max": float(np.max(xyz)),
        "q_rmse": float(np.median([row["q_rmse"] for row in rows])),
        "tau_peak": float(np.max([row["tau_peak"] for row in rows])),
        "slip": float(np.max([row["slip"] for row in rows])),
    })
    return summary


def _e2_block(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    n = len(rows)
    k = sum(1 for row in rows if row["success"])
    low, high = wilson_interval(k, n)
    counts = Counter(row["failure"] or "success" for row in rows)
    xy = np.array([row["place_xy"] for row in rows], dtype=float)
    xyz = np.array([row["place_xyz"] for row in rows], dtype=float)
    lines = [
        "",
        "## E2",
        "",
        f"Successes {k} / {n} = {k / n:.3f}. Wilson 95% interval [{low:.3f}, {high:.3f}].",
        "",
        "Failure counts: " + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
        "",
        f"Place xy error median {np.median(xy):.4f} m, max {np.max(xy):.4f} m. "
        f"3D median {np.median(xyz):.4f} m, max {np.max(xyz):.4f} m.",
        "",
    ]
    dist_cols = (
        "open_tcp_xy", "open_tcp_z", "open_can_xy", "open_can_z",
        "open_shoulder_err", "open_elbow_err",
        "detach_tcp_xy", "detach_tcp_z", "detach_can_xy", "detach_can_z",
        "detach_shoulder_err", "detach_elbow_err", "detach_can_xy_disp",
    )
    lines.append("Release split, min / median / max:")
    lines.append("")
    for col in dist_cols:
        values = np.array([row[col] for row in rows if col in row], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        lines.append(
            f"- {col}: {np.min(values):.6g} / {np.median(values):.6g} / {np.max(values):.6g}"
        )
    lines.append("")
    return lines


def _ik_block(loaded) -> list[str]:
    current = []
    nxt = []
    for arrays, meta, _row in loaded:
        if meta.get("experiment") not in ("E1", "E2"):
            continue
        if "ik_solve_s" not in arrays:
            continue
        current.append(np.asarray(arrays["ik_solve_s"], dtype=float)[::25])
        nxt.append(np.asarray(arrays["ik_next_solve_s"], dtype=float)[::25])
    if not current:
        return []
    cur = np.concatenate(current)
    following = np.concatenate(nxt)
    return [
        "",
        "## IK time",
        "",
        "One sample per control tick: index 0, 25, 50, ... of the physics-step log, "
        "so the 25 repeated copies of one solve are not 25 solves. "
        "Current target and next target are separate. E1 and E2 together.",
        "",
        f"Current target: median {np.median(cur) * 1e6:.3f} µs, max {np.max(cur) * 1e6:.3f} µs.",
        f"Next target: median {np.median(following) * 1e6:.3f} µs, max {np.max(following) * 1e6:.3f} µs.",
        "",
    ]


_TCP_REF_NOTE = (
    "E1 and E2 `tcp_ref_rmse` compare the measured TCP with the planner's time law. "
    "The action is sampled at the end of the tick, one tick ahead of that law, which "
    "cancels the 50 ms preview lag. Lag comparisons use E4 only. An M5 demonstration "
    "built from this planner carries the same one-tick lead."
)


def _plot(loaded, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    by_name = {meta["name"]: (arrays, meta, row) for arrays, meta, row in loaded}
    e1 = [item for item in loaded if item[1].get("experiment") == "E1"]
    e2 = [item for item in loaded if item[1].get("experiment") == "E2"]
    e4 = [item for item in loaded if item[1].get("experiment") == "E4"]

    focus = by_name.get("E1_C3_fixed_1_medium")
    if focus is not None:
        arrays, _meta, _row = focus
        t = arrays["t"]
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        for joint in range(6):
            axes[0].plot(t, arrays["q"][:, joint], color="C0", alpha=0.8)
            axes[0].plot(t, arrays["q_d"][:, joint], color="C1", alpha=0.8, linestyle="--")
        axes[0].set_ylabel("q / q_d (rad)")
        limits = LIMITS
        for joint in range(6):
            axes[1].plot(t, arrays["tau_cmd"][:, joint])
            axes[1].axhline(limits[joint], color="k", linewidth=0.4)
            axes[1].axhline(-limits[joint], color="k", linewidth=0.4)
        axes[1].set_ylabel("tau_cmd (Nm)")
        axes[2].plot(t, arrays["gripper_q"], label="opening")
        axes[2].plot(t, arrays["gripper_cmd"], linestyle="--", label="command")
        axes[2].legend()
        axes[2].set_ylabel("gripper (m)")
        axes[3].plot(t, arrays["planner_state"])
        axes[3].set_ylabel("state")
        axes[3].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_fixed1_medium.png", dpi=120)
        plt.close(fig)

    if e1:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        for arrays, meta, _row in e1:
            axes[0].plot(arrays["tcp_pos"][:, 0], arrays["tcp_pos"][:, 1], label=f"tcp {meta['mode']} {meta['speed']}")
            axes[0].plot(arrays["can_pos"][:, 0], arrays["can_pos"][:, 1], linestyle="--")
            axes[1].plot(arrays["tcp_pos"][:, 0], arrays["tcp_pos"][:, 2])
            axes[1].plot(arrays["can_pos"][:, 0], arrays["can_pos"][:, 2], linestyle="--")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title("top: solid TCP, dashed can")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("z (m)")
        axes[1].set_title("side")
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_paths.png", dpi=120)
        plt.close(fig)

        speeds = ["fast", "medium", "slow"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for mode in ("fixed_1", "fixed_2"):
            xs, err, tau, place = [], [], [], []
            for index, speed in enumerate(speeds):
                found = [row for _a, meta, row in e1 if meta["mode"] == mode and meta["speed"] == speed]
                if not found:
                    continue
                xs.append(index)
                err.append(found[0]["q_rmse"])
                tau.append(found[0]["tau_peak"])
                place.append(found[0]["place_xy"])
            axes[0].plot(xs, err, marker="o", label=mode)
            axes[1].plot(xs, tau, marker="o", label=mode)
            axes[2].plot(xs, place, marker="o", label=mode)
        for axis, title in zip(axes, ("joint RMSE (rad)", "peak |tau_cmd| (Nm)", "place xy (m)")):
            axis.set_xticks([0, 1, 2], speeds)
            axis.set_title(title)
            axis.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_speed.png", dpi=120)
        plt.close(fig)

    if e2:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = {}
        palette = plt.cm.tab10.colors
        for arrays, meta, _row in e2:
            label = "success" if meta.get("success") else (meta.get("failure") or "other")
            if label not in colors:
                colors[label] = palette[len(colors) % len(palette)]
            xy = np.asarray(meta["can_init_pos"], dtype=float)
            ax.scatter(xy[0], xy[1], color=colors[label], label=label)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        ax.set_xlabel("can x (m)")
        ax.set_ylabel("can y (m)")
        ax.set_title("E2 initial can position")
        fig.tight_layout()
        fig.savefig(fig_dir / "e2_scatter.png", dpi=120)
        plt.close(fig)

    if e4:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        labels = []
        rmse, ripple, peak = [], [], []
        for arrays, meta, row in sorted(e4, key=lambda item: item[1]["name"]):
            labels.append(f"{meta['mode'][:2]} {meta['speed'][:3]} {meta['reference_mode'][:4]}")
            rmse.append(row["q_rmse"])
            ripple.append(row["qdd_rms"])
            peak.append(row["tau_peak"])
        x = np.arange(len(labels))
        axes[0].bar(x, rmse)
        axes[1].bar(x, ripple)
        axes[2].bar(x, peak)
        for axis, title in zip(axes, ("joint RMSE", "shoulder qdd RMS", "peak |tau_cmd|")):
            axis.set_title(title)
            axis.set_xticks(x, labels, rotation=90, fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "e4_compare.png", dpi=120)
        plt.close(fig)


def _load_dir(run_dir: Path):
    loaded = []
    for path in sorted(run_dir.glob("*.npz")):
        arrays, meta = load_run(path)
        meta.setdefault("name", path.stem)
        loaded.append((arrays, meta, summarize(arrays, meta)))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = repo_root()
    run_dir = Path(args.runs) if args.runs else root / "outputs" / "m4" / "runs"
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    out = Path(args.out) if args.out else run_dir.parent
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    loaded = _load_dir(run_dir)
    if not loaded:
        raise SystemExit(f"no logs in {run_dir}")
    columns = list(loaded[0][2].keys())
    for _arrays, _meta, row in loaded[1:]:
        for key in row:
            if key not in columns:
                columns.append(key)
    rows = [row for _a, _m, row in loaded]
    e2_rows = [row for row in rows if row["experiment"] == "E2"]
    summary_row = _e2_summary_row(e2_rows)
    if summary_row is not None:
        rows.append(summary_row)
        for key in summary_row:
            if key not in columns:
                columns.append(key)
    csv_path = out / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    main_cols = [
        "name", "experiment", "mode", "speed", "reference_mode", "success", "failure",
        "success_time", "motion_time", "episode_time", "q_rmse", "q_max",
        "tcp_ref_rmse", "tcp_fk_rmse", "sat_ratio", "tau_peak", "vel_ratio",
        "place_xy", "place_xyz", "place_vs_place_pos", "tilt", "slip",
        "tcp_err_close", "tcp_err_open", "qdd_rms", "qdd_rms_ma25",
    ]
    wrist_cols = [
        "name", "wrist_angle_err", "wrist_rotate_err", "wrist_angle_tau", "wrist_rotate_tau",
        "wrist_angle_tau_ratio", "wrist_rotate_tau_ratio",
        "wrist_angle_err_hold", "wrist_rotate_err_hold", "wrist_angle_tau_hold", "wrist_rotate_tau_hold",
    ]
    release_cols = [
        "name", "mode", "speed",
        "open_tcp_x", "open_tcp_y", "open_tcp_z", "open_tcp_xy",
        "open_can_xy", "open_can_z", "open_shoulder_err", "open_elbow_err",
        "detach_tcp_x", "detach_tcp_y", "detach_tcp_z", "detach_tcp_xy",
        "detach_can_xy", "detach_can_z", "detach_shoulder_err", "detach_elbow_err",
        "detach_can_xy_disp", "place_xy",
    ]
    task_rows = [row for row in rows if row["experiment"] in ("E1", "E2") and row["name"] != "E2_summary"]
    text = ["# M4 summary", "", _TCP_REF_NOTE, ""]
    _write_table(out / "summary_main.md", rows, [col for col in main_cols if any(col in row for row in rows)])
    text.append((out / "summary_main.md").read_text(encoding="utf-8"))
    text.append(_TCP_REF_NOTE)
    text.append("")
    text.append("## Release split")
    text.append("")
    text.append(
        "`open_*` is the first physics step of the open ramp. "
        "`detach_*` is the first physics step after a two-finger grasp on which neither finger touches the can. "
        "`detach_can_xy_disp` is the length of the can xy displacement from that step to the end. "
        "TCP components are measured minus the place waypoint."
    )
    text.append("")
    _write_table(out / "summary_release.md", task_rows, [col for col in release_cols if any(col in row for row in task_rows)])
    text.append((out / "summary_release.md").read_text(encoding="utf-8"))
    text.extend(_e2_block(e2_rows))
    text.extend(_ik_block(loaded))
    ablation_dir = root / "outputs" / "m4" / "ablation_nofriction"
    if args.runs is None and any(ablation_dir.glob("*.npz")):
        ablation = _load_dir(ablation_dir)
        text.extend(_ablation_block(task_rows, [row for _a, _m, row in ablation]))
    text.append("## Wrists")
    text.append("")
    _write_table(
        out / "summary_wrist.md",
        [row for row in rows if row["experiment"] != "E4" and row["name"] != "E2_summary"],
        wrist_cols,
    )
    text.append((out / "summary_wrist.md").read_text(encoding="utf-8"))
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    if args.runs is None:
        _plot(loaded, out / "figures")
    print(f"runs {len(loaded)}", flush=True)


def _ablation_block(task_rows: list[dict], ablation_rows: list[dict]) -> list[str]:
    lines = ["", "## Friction ablation", "", "Arm `dof_frictionloss` set to 0 in memory. `aloha/` is unchanged.", ""]
    cols = [
        "name", "place_xy", "open_tcp_xy", "open_can_xy", "open_shoulder_err", "open_elbow_err",
        "detach_can_xy_disp", "detach_can_xy", "q_rmse", "tau_peak",
    ]
    by_name = {row["name"]: row for row in task_rows}
    paired = []
    for row in ablation_rows:
        base = by_name.get(row["name"])
        labeled = dict(row)
        labeled["name"] = row["name"] + " friction0"
        paired.append(labeled)
        if base is None:
            continue
        delta = {"name": row["name"] + " friction0-nominal"}
        for col in cols:
            if col == "name":
                continue
            if col in row and col in base and isinstance(row[col], float) and isinstance(base[col], float):
                delta[col] = row[col] - base[col]
        paired.append(delta)
    lines.append("`friction0` is the rerun. `friction0-nominal` is that rerun minus the matching E1 run.")
    lines.append("")
    _ = paired
    # The table is written by the caller's file only when this returns text.
    # Build the markdown here so the section is self-contained.
    header = [col for col in cols if any(col in row for row in paired)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in paired:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in header) + " |")
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
