"""Joint coupling from the hand-written mass matrix and inverse dynamics.

Run: conda run -n AIControl python scripts/m2_coupling.py

The normalized coefficient is M_ij / sqrt(M_ii M_jj). Armature is configuration
independent, so it changes the coefficient only by inflating the diagonal.
One quintic moves a joint by 0.5 rad in 1 s. A second quintic uses the same
distance and reaches a peak speed of 2 rad/s. Other joints are held by
M_ij qdd_j + (C qd)_i + g_i. The comparison figure drops gravity.
The driven joint's own torque adds viscous damping and frictionloss, which the
inverse dynamics do not include.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from aloha_pp.dyn.rnea import coriolis_force, gravity_torque, mass_matrix
from aloha_pp.kin.fk import ArmChain
from aloha_pp.robot import robot_xml

TYPICAL_Q = np.array([0.0, -0.24632554, 0.35635693, 0.0, 1.46044343, 0.0])
SHORT = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
PAIRS = ((1, 2, "shoulder-elbow"), (1, 4, "shoulder-wrist_angle"), (2, 4, "elbow-wrist_angle"))
N_LINE = 61
N_GRID = 21
DURATION = 1.0
DISTANCE = 0.5
N_TRAJ = 101
# max(30 s^2 - 60 s^3 + 30 s^4) on [0, 1] is 15/8 at s = 1/2.
QUINTIC_PEAK_FACTOR = 15.0 / 8.0
VELOCITY_LIMIT = 2.0
FAST_DURATION = DISTANCE * QUINTIC_PEAK_FACTOR / VELOCITY_LIMIT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m2"


def kappa(matrix: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(matrix), 1e-16, None))
    return matrix / np.outer(scale, scale)


def quintic(time: np.ndarray, duration: float, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = time / duration
    position = distance * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
    velocity = distance * (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    acceleration = distance * (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    return position, velocity, acceleration


def _line_scan(chain: ArmChain, joint: int, grid: np.ndarray, with_armature: bool) -> np.ndarray:
    values = np.zeros((grid.size, len(PAIRS)))
    for i, sample in enumerate(grid):
        q = TYPICAL_Q.copy()
        q[joint] = sample
        coeff = kappa(mass_matrix(chain, q, armature=with_armature))
        for col, (a, b, _name) in enumerate(PAIRS):
            values[i, col] = coeff[a, b]
    return values


def _heatmap(chain: ArmChain, with_armature: bool) -> np.ndarray:
    return kappa(mass_matrix(chain, TYPICAL_Q, armature=with_armature))


def _save_lines(
    path: Path,
    grid: np.ndarray,
    with_arm: np.ndarray,
    without_arm: np.ndarray,
    q_name: str = "q",
) -> None:
    header = [q_name]
    columns = [grid]
    for _flag, block in (("with_armature", with_arm), ("no_armature", without_arm)):
        for name in (pair[2] for pair in PAIRS):
            header.append(f"{name}_{_flag}")
        columns.append(block)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def _plot_lines(
    elbow: np.ndarray,
    shoulder: np.ndarray,
    elbow_with: np.ndarray,
    elbow_without: np.ndarray,
    shoulder_with: np.ndarray,
    shoulder_without: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=True)
    scans = (
        (elbow, elbow_with, elbow_without, "elbow (rad)"),
        (shoulder, shoulder_with, shoulder_without, "shoulder (rad)"),
    )
    for row, (grid, with_arm, without_arm, xlabel) in enumerate(scans):
        for col, (_a, _b, name) in enumerate(PAIRS):
            ax = axes[row, col]
            ax.plot(grid, with_arm[:, col], label="with armature")
            ax.plot(grid, without_arm[:, col], ls="--", label="no armature")
            ax.set_title(name)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("M_ij / sqrt(M_ii M_jj)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Normalized mass coupling along shoulder and elbow")
    fig.tight_layout()
    fig.savefig(OUT / "coupling_curves.png", dpi=140)
    plt.close(fig)


def _plot_heatmaps(with_arm: np.ndarray, without_arm: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], with_arm, "with armature"),
        (axes[1], without_arm, "no armature"),
    ):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(6), SHORT, rotation=45, ha="right")
        ax.set_yticks(range(6), SHORT)
        ax.set_title(title)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.86, pad=0.03, label="M_ij / sqrt(M_ii M_jj)")
    fig.suptitle("Normalized mass matrix at the typical pose")
    fig.savefig(OUT / "coupling_heatmap.png", dpi=140)
    plt.close(fig)


def _plot_grid(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    with_arm: np.ndarray,
    without_arm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], with_arm, "with armature"),
        (axes[1], without_arm, "no armature"),
    ):
        image = ax.pcolormesh(elbow, shoulder, matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, shading="auto")
        ax.set_xlabel("elbow (rad)")
        ax.set_ylabel("shoulder (rad)")
        ax.set_title(title)
    fig.colorbar(image, ax=axes, shrink=0.86, pad=0.03, label="shoulder-elbow coupling")
    fig.suptitle("Normalized shoulder-elbow coupling")
    fig.savefig(OUT / "coupling_shoulder_elbow_map.png", dpi=140)
    plt.close(fig)


def _plot_wrist_lines(grid: np.ndarray, with_arm: np.ndarray, without_arm: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for col, (_a, _b, name) in enumerate(PAIRS):
        ax = axes[col]
        ax.plot(grid, with_arm[:, col], label="with armature")
        ax.plot(grid, without_arm[:, col], ls="--", label="no armature")
        ax.set_title(name)
        ax.set_xlabel("wrist_angle (rad)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("M_ij / sqrt(M_ii M_jj)")
            ax.legend(fontsize=8)
    fig.suptitle("Normalized mass coupling along wrist_angle (other joints at home)")
    fig.tight_layout()
    fig.savefig(OUT / "coupling_vs_wrist_angle.png", dpi=140)
    plt.close(fig)


def _plot_elbow_wrist_map(elbow: np.ndarray, wrist: np.ndarray, matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    image = ax.pcolormesh(elbow, wrist, matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, shading="auto")
    ax.set_xlabel("elbow (rad)")
    ax.set_ylabel("wrist_angle (rad)")
    ax.set_title("shoulder-wrist_angle coupling, with armature")
    fig.colorbar(image, ax=ax, shrink=0.86, pad=0.03, label="M_ij / sqrt(M_ii M_jj)")
    fig.savefig(OUT / "coupling_shoulder_wrist_map.png", dpi=140)
    plt.close(fig)


def _trajectory(chain: ArmChain, joint: int, duration: float) -> dict[str, np.ndarray]:
    start = float(TYPICAL_Q[joint])
    if start + DISTANCE <= chain.arm_range[joint, 1]:
        distance = DISTANCE
    elif start - DISTANCE >= chain.arm_range[joint, 0]:
        distance = -DISTANCE
    else:
        raise RuntimeError(f"{SHORT[joint]} has no room for a 0.5 rad quintic")
    time = np.linspace(0.0, duration, N_TRAJ)
    position, velocity, acceleration = quintic(time, duration, distance)
    inertial = np.zeros((N_TRAJ, 6))
    coriolis = np.zeros((N_TRAJ, 6))
    gravity = np.zeros((N_TRAJ, 6))
    for i, (p, v, a) in enumerate(zip(position, velocity, acceleration)):
        q = TYPICAL_Q.copy()
        q[joint] = start + p
        qd = np.zeros(6)
        qdd = np.zeros(6)
        qd[joint] = v
        qdd[joint] = a
        inertial[i] = mass_matrix(chain, q)[:, joint] * a
        coriolis[i] = coriolis_force(chain, q, qd)
        gravity[i] = gravity_torque(chain, q)
    # Other joints are held, so their acceleration is zero and the inertial
    # coupling onto the driven joint is zero. What remains is C(q, qd) qd.
    damping = chain.arm_damping[joint] * velocity
    friction = chain.arm_frictionloss[joint] * np.sign(velocity)
    total = inertial + coriolis + gravity
    return {
        "time": time,
        "duration": np.array([duration]),
        "distance": np.array([distance]),
        "q": start + position,
        "qd": velocity,
        "qdd": acceleration,
        "inertial": inertial,
        "coriolis": coriolis,
        "gravity": gravity,
        "total": total,
        "driven_mii_qdd": inertial[:, joint].copy(),
        "driven_coupling": coriolis[:, joint].copy(),
        "driven_gravity": gravity[:, joint].copy(),
        "driven_damping": damping,
        "driven_friction": friction,
        "driven_total": inertial[:, joint] + coriolis[:, joint] + gravity[:, joint] + damping + friction,
    }


def _save_trajectory(path: Path, data: dict[str, np.ndarray]) -> None:
    columns = [data["time"], data["q"], data["qd"], data["qdd"]]
    header = ["time", "q_driven", "qd_driven", "qdd_driven"]
    for label, block in (
        ("inertial", data["inertial"]),
        ("coriolis", data["coriolis"]),
        ("gravity", data["gravity"]),
        ("total", data["total"]),
    ):
        for name in SHORT:
            header.append(f"{label}_{name}")
        columns.append(block)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def _plot_trajectory(
    path: Path,
    joint: int,
    data: dict[str, np.ndarray],
    limits: np.ndarray,
    title: str,
) -> None:
    others = [i for i in range(6) if i != joint]
    fig, axes = plt.subplots(len(others), 1, figsize=(7.4, 9.0), sharex=True)
    time = data["time"]
    for ax, i in zip(axes, others):
        ax.plot(time, data["inertial"][:, i], label="M_ij qdd_j")
        ax.plot(time, data["coriolis"][:, i], label="Coriolis / centrifugal")
        ax.plot(time, data["gravity"][:, i], label="gravity")
        ax.plot(time, data["total"][:, i], color="k", lw=1.2, label="total")
        peak = max(float(np.max(np.abs(data["total"][:, i]))), 1e-3)
        ax.set_ylim(-1.35 * peak, 1.35 * peak)
        if limits[i] <= 1.35 * peak:
            ax.axhline(limits[i], color="0.4", ls="--", lw=0.8)
            ax.axhline(-limits[i], color="0.4", ls="--", lw=0.8)
        else:
            ax.text(
                0.98,
                0.92,
                f"limit ±{limits[i]:.0f} N·m",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
        ax.set_ylabel(f"{SHORT[i]} (N·m)")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _summarize(joint: int, data: dict[str, np.ndarray], limits: np.ndarray, profile: str) -> None:
    print(f"drive {SHORT[joint]} {profile} distance {float(data['distance'][0]):+.1f} rad")
    print(f"  peak |qd| {np.max(np.abs(data['qd'])):.4f} rad/s  peak |qdd| {np.max(np.abs(data['qdd'])):.4f} rad/s^2")
    for i in range(6):
        if i == joint:
            continue
        total = float(np.max(np.abs(data["total"][:, i])))
        pieces = {
            "inertial": float(np.max(np.abs(data["inertial"][:, i]))),
            "coriolis": float(np.max(np.abs(data["coriolis"][:, i]))),
            "gravity": float(np.max(np.abs(data["gravity"][:, i]))),
        }
        print(
            f"  {SHORT[i]:16s} max|total| {total:.4f} N·m  "
            f"ratio {total / limits[i]:.4f}  "
            f"max|Mij qdd| {pieces['inertial']:.4f}  "
            f"max|C| {pieces['coriolis']:.4f}  max|g| {pieces['gravity']:.4f}  "
            f"limit {limits[i]:.1f}"
        )


def _coupled_dynamic(data: dict[str, np.ndarray], joint: int) -> np.ndarray:
    return data["inertial"][:, joint] + data["coriolis"][:, joint]


def _endpoints_ok(data: dict[str, np.ndarray]) -> bool:
    ends = (data["qd"][0], data["qdd"][0], data["qd"][-1], data["qdd"][-1])
    return all(abs(float(value)) <= 1e-12 for value in ends)


def _summarize_dynamic(joint: int, data: dict[str, np.ndarray], limits: np.ndarray, profile: str) -> None:
    print(f"dynamic no gravity, drive {SHORT[joint]} {profile}")
    for i in range(6):
        if i == joint:
            continue
        peak = float(np.max(np.abs(_coupled_dynamic(data, i))))
        print(
            f"  {SHORT[i]:16s} max|Mij qdd + C| {peak:.4f} N·m  "
            f"ratio {peak / limits[i]:.4f}  limit {limits[i]:.1f}"
        )


def _summarize_driven(joint: int, data: dict[str, np.ndarray], limits: np.ndarray, profile: str) -> None:
    total = data["driven_total"]
    index = int(np.argmax(np.abs(total)))
    peak = float(abs(total[index]))
    limit = float(limits[joint])
    saturated = "yes" if peak >= limit else "no"
    print(
        f"driven {SHORT[joint]} {profile}: peak |tau| {peak:.4f} N·m  "
        f"ratio {peak / limit:.4f}  limit {limit:.1f}  saturated {saturated}"
    )
    print(
        f"  at peak t={data['time'][index]:.4f} s: "
        f"Mii qdd {data['driven_mii_qdd'][index]:+.4f}  "
        f"coupling(C) {data['driven_coupling'][index]:+.4f}  "
        f"g {data['driven_gravity'][index]:+.4f}  "
        f"B qd {data['driven_damping'][index]:+.4f}  "
        f"friction {data['driven_friction'][index]:+.4f}"
    )
    print(
        f"  separate peaks |Mii qdd| {np.max(np.abs(data['driven_mii_qdd'])):.4f}  "
        f"|C| {np.max(np.abs(data['driven_coupling'])):.4f}  "
        f"|g| {np.max(np.abs(data['driven_gravity'])):.4f}  "
        f"|B qd| {np.max(np.abs(data['driven_damping'])):.4f}  "
        f"|friction| {np.max(np.abs(data['driven_friction'])):.4f}"
    )


def _plot_dynamic(cases: dict[tuple[int, str], dict[str, np.ndarray]], limits: np.ndarray) -> None:
    slow_peak = float(np.max(np.abs(cases[(1, "slow")]["qd"])))
    fast_peak = float(np.max(np.abs(cases[(1, "fast")]["qd"])))
    fast_duration = float(cases[(1, "fast")]["duration"][0])
    slow_label = f"1 s, peak {slow_peak:.3f} rad/s"
    fast_label = f"{fast_duration:.3f} s, peak {fast_peak:.3f} rad/s"
    fig, axes = plt.subplots(5, 2, figsize=(10.4, 12.0), sharex=True)
    for col, joint in enumerate((1, 2)):
        others = [i for i in range(6) if i != joint]
        slow = cases[(joint, "slow")]
        fast = cases[(joint, "fast")]
        phase = slow["time"] / float(slow["duration"][0])
        for row, i in enumerate(others):
            ax = axes[row, col]
            dyn_slow = _coupled_dynamic(slow, i)
            dyn_fast = _coupled_dynamic(fast, i)
            ax.plot(phase, dyn_slow, label=slow_label)
            ax.plot(phase, dyn_fast, label=fast_label)
            peak_slow = float(np.max(np.abs(dyn_slow)))
            peak_fast = float(np.max(np.abs(dyn_fast)))
            ax.text(
                0.98,
                0.96,
                f"peak {peak_slow:.3f} / {peak_fast:.3f} N·m\n"
                f"vs limit {peak_slow / limits[i]:.4f} / {peak_fast / limits[i]:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.25",
            )
            ax.set_ylabel(f"{SHORT[i]} (N·m)")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"{SHORT[joint]} moving, no gravity")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right")
        axes[-1, col].set_xlabel("normalized time")
    fig.suptitle("Dynamic coupling without gravity: M_ij qdd_j + Coriolis/centrifugal")
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_coupling_no_gravity.png", dpi=140)
    plt.close(fig)


def _save_dynamic(cases: dict[tuple[int, str], dict[str, np.ndarray]], limits: np.ndarray) -> None:
    phase = cases[(1, "slow")]["time"] / float(cases[(1, "slow")]["duration"][0])
    columns = [phase]
    header = ["phase"]
    for joint in (1, 2):
        for i in range(6):
            if i == joint:
                continue
            for profile in ("slow", "fast"):
                header.append(f"{SHORT[joint]}_drive_{SHORT[i]}_{profile}")
                columns.append(_coupled_dynamic(cases[(joint, profile)], i))
    np.savetxt(
        OUT / "dynamic_coupling_no_gravity.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    with (OUT / "dynamic_coupling_peaks.csv").open("w", encoding="utf-8") as handle:
        handle.write("profile,drive,joint,peak_abs_nm,ratio_to_limit,torque_limit_nm\n")
        for profile in ("slow", "fast"):
            for joint in (1, 2):
                data = cases[(joint, profile)]
                for i in range(6):
                    if i == joint:
                        continue
                    peak = float(np.max(np.abs(_coupled_dynamic(data, i))))
                    handle.write(
                        f"{profile},{SHORT[joint]},{SHORT[i]},{peak:.8e},"
                        f"{peak / limits[i]:.8e},{limits[i]:.8e}\n"
                    )


def _save_driven(cases: dict[tuple[int, str], dict[str, np.ndarray]], limits: np.ndarray) -> None:
    with (OUT / "driven_joint_torque.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "profile,joint,time,qd,qdd,mii_qdd,coupling,gravity,damping,frictionloss,total,torque_limit\n"
        )
        for profile in ("slow", "fast"):
            for joint in (1, 2):
                data = cases[(joint, profile)]
                for i in range(int(data["time"].size)):
                    handle.write(
                        f"{profile},{SHORT[joint]},{data['time'][i]:.8e},{data['qd'][i]:.8e},"
                        f"{data['qdd'][i]:.8e},{data['driven_mii_qdd'][i]:.8e},"
                        f"{data['driven_coupling'][i]:.8e},{data['driven_gravity'][i]:.8e},"
                        f"{data['driven_damping'][i]:.8e},{data['driven_friction'][i]:.8e},"
                        f"{data['driven_total'][i]:.8e},{limits[joint]:.8e}\n"
                    )
    with (OUT / "driven_joint_torque_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "profile,joint,peak_abs_nm,ratio_to_limit,torque_limit_nm,saturated,"
            "t_at_peak,mii_qdd_at_peak,coupling_at_peak,gravity_at_peak,damping_at_peak,friction_at_peak,"
            "peak_abs_mii,peak_abs_coupling,peak_abs_gravity,peak_abs_damping,peak_abs_friction\n"
        )
        for profile in ("slow", "fast"):
            for joint in (1, 2):
                data = cases[(joint, profile)]
                total = data["driven_total"]
                index = int(np.argmax(np.abs(total)))
                peak = float(abs(total[index]))
                limit = float(limits[joint])
                saturated = "yes" if peak >= limit else "no"
                handle.write(
                    f"{profile},{SHORT[joint]},{peak:.8e},{peak / limit:.8e},{limit:.8e},{saturated},"
                    f"{data['time'][index]:.8e},{data['driven_mii_qdd'][index]:.8e},"
                    f"{data['driven_coupling'][index]:.8e},{data['driven_gravity'][index]:.8e},"
                    f"{data['driven_damping'][index]:.8e},{data['driven_friction'][index]:.8e},"
                    f"{np.max(np.abs(data['driven_mii_qdd'])):.8e},"
                    f"{np.max(np.abs(data['driven_coupling'])):.8e},"
                    f"{np.max(np.abs(data['driven_gravity'])):.8e},"
                    f"{np.max(np.abs(data['driven_damping'])):.8e},"
                    f"{np.max(np.abs(data['driven_friction'])):.8e}\n"
                )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(robot_xml()))
    chain = ArmChain(model, finger_q=0.037)
    for value, limit in zip(TYPICAL_Q, chain.arm_range):
        if not (limit[0] <= value <= limit[1]):
            print("typical pose outside joint range", file=sys.stderr)
            return 1

    elbow = np.linspace(chain.arm_range[2, 0], chain.arm_range[2, 1], N_LINE)
    shoulder = np.linspace(chain.arm_range[1, 0], chain.arm_range[1, 1], N_LINE)
    elbow_with = _line_scan(chain, 2, elbow, True)
    elbow_without = _line_scan(chain, 2, elbow, False)
    shoulder_with = _line_scan(chain, 1, shoulder, True)
    shoulder_without = _line_scan(chain, 1, shoulder, False)
    wrist = np.linspace(chain.arm_range[4, 0], chain.arm_range[4, 1], N_LINE)
    wrist_with = _line_scan(chain, 4, wrist, True)
    wrist_without = _line_scan(chain, 4, wrist, False)
    _save_lines(OUT / "coupling_vs_elbow.csv", elbow, elbow_with, elbow_without)
    _save_lines(OUT / "coupling_vs_shoulder.csv", shoulder, shoulder_with, shoulder_without)
    _save_lines(
        OUT / "coupling_vs_wrist_angle.csv",
        wrist,
        wrist_with,
        wrist_without,
        q_name="wrist_angle",
    )
    _plot_lines(elbow, shoulder, elbow_with, elbow_without, shoulder_with, shoulder_without)
    _plot_wrist_lines(wrist, wrist_with, wrist_without)

    heat_with = _heatmap(chain, True)
    heat_without = _heatmap(chain, False)
    raw_with = mass_matrix(chain, TYPICAL_Q, armature=True)
    raw_without = mass_matrix(chain, TYPICAL_Q, armature=False)
    _plot_heatmaps(heat_with, heat_without)
    np.savetxt(
        OUT / "coupling_heatmap.csv",
        np.vstack((heat_with, heat_without)),
        delimiter=",",
        header="first 6 rows with armature, next 6 rows without; order " + " ".join(SHORT),
        comments="",
    )

    shoulder_grid = np.linspace(chain.arm_range[1, 0], chain.arm_range[1, 1], N_GRID)
    elbow_grid = np.linspace(chain.arm_range[2, 0], chain.arm_range[2, 1], N_GRID)
    grid_with = np.zeros((N_GRID, N_GRID))
    grid_without = np.zeros((N_GRID, N_GRID))
    for i, shoulder_q in enumerate(shoulder_grid):
        for j, elbow_q in enumerate(elbow_grid):
            q = TYPICAL_Q.copy()
            q[1] = shoulder_q
            q[2] = elbow_q
            grid_with[i, j] = kappa(mass_matrix(chain, q, armature=True))[1, 2]
            grid_without[i, j] = kappa(mass_matrix(chain, q, armature=False))[1, 2]
    _plot_grid(shoulder_grid, elbow_grid, grid_with, grid_without)

    wrist_grid = np.linspace(chain.arm_range[4, 0], chain.arm_range[4, 1], N_GRID)
    shoulder_wrist_map = np.zeros((N_GRID, N_GRID))
    for i, wrist_q in enumerate(wrist_grid):
        for j, elbow_q in enumerate(elbow_grid):
            q = TYPICAL_Q.copy()
            q[2] = elbow_q
            q[4] = wrist_q
            shoulder_wrist_map[i, j] = kappa(mass_matrix(chain, q, armature=True))[1, 4]
    _plot_elbow_wrist_map(elbow_grid, wrist_grid, shoulder_wrist_map)
    with (OUT / "coupling_shoulder_wrist_map.csv").open("w", encoding="utf-8") as handle:
        handle.write("wrist_angle,elbow,shoulder_wrist_kappa_with_armature\n")
        for i, wrist_q in enumerate(wrist_grid):
            for j, elbow_q in enumerate(elbow_grid):
                handle.write(f"{wrist_q:.8e},{elbow_q:.8e},{shoulder_wrist_map[i, j]:.8e}\n")

    np.savez(
        OUT / "coupling_data.npz",
        typical_q=TYPICAL_Q,
        elbow=elbow,
        shoulder=shoulder,
        elbow_with=elbow_with,
        elbow_without=elbow_without,
        shoulder_with=shoulder_with,
        shoulder_without=shoulder_without,
        kappa_with=heat_with,
        kappa_without=heat_without,
        M_with=raw_with,
        M_without=raw_without,
        shoulder_grid=shoulder_grid,
        elbow_grid=elbow_grid,
        map_with=grid_with,
        map_without=grid_without,
        short_names=np.array(SHORT),
        wrist=wrist,
        wrist_with=wrist_with,
        wrist_without=wrist_without,
        wrist_grid=wrist_grid,
        shoulder_wrist_map_with=shoulder_wrist_map,
    )

    print("typical pose", np.array2string(TYPICAL_Q, precision=6))
    print("M diag with armature   ", np.array2string(np.diag(raw_with), precision=5))
    print("M diag without armature", np.array2string(np.diag(raw_without), precision=5))
    print("armature               ", np.array2string(chain.arm_armature, precision=4))
    for a, b, name in PAIRS:
        print(
            f"kappa {name:24s} typical with {heat_with[a, b]:+.4f}  "
            f"without {heat_without[a, b]:+.4f}  "
            f"scan elbow [{elbow_with[:, PAIRS.index((a, b, name))].min():+.4f}, "
            f"{elbow_with[:, PAIRS.index((a, b, name))].max():+.4f}] with  "
            f"[{elbow_without[:, PAIRS.index((a, b, name))].min():+.4f}, "
            f"{elbow_without[:, PAIRS.index((a, b, name))].max():+.4f}] without"
        )
    print(
        f"shoulder-elbow map with armature [{grid_with.min():+.4f}, {grid_with.max():+.4f}]  "
        f"without [{grid_without.min():+.4f}, {grid_without.max():+.4f}]"
    )
    print(
        "vs shoulder the three coefficients stay constant: "
        f"with {np.array2string(shoulder_with[0], precision=4)} "
        f"without {np.array2string(shoulder_without[0], precision=4)}"
    )
    print(
        "shoulder scan max abs deviation "
        f"with {np.max(np.abs(shoulder_with - shoulder_with[0])):.3e} "
        f"without {np.max(np.abs(shoulder_without - shoulder_without[0])):.3e}"
    )
    print(
        "elbow-wrist_angle span along elbow "
        f"with {float(np.max(elbow_with[:, 2]) - np.min(elbow_with[:, 2])):.3e} "
        f"without {float(np.max(elbow_without[:, 2]) - np.min(elbow_without[:, 2])):.3e}"
    )
    for col, (_a, _b, name) in enumerate(PAIRS):
        print(
            f"kappa {name:24s} vs wrist_angle "
            f"with [{wrist_with[:, col].min():+.4f}, {wrist_with[:, col].max():+.4f}]  "
            f"without [{wrist_without[:, col].min():+.4f}, {wrist_without[:, col].max():+.4f}]"
        )
    print(
        "shoulder-wrist_angle on elbow x wrist_angle, with armature "
        f"[{shoulder_wrist_map.min():+.4f}, {shoulder_wrist_map.max():+.4f}]"
    )

    def _largest(matrix: np.ndarray) -> None:
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                pairs.append((abs(float(matrix[i, j])), float(matrix[i, j]), SHORT[i], SHORT[j]))
        for _mag, value, left, right in sorted(pairs, reverse=True)[:4]:
            print(f"  {left}-{right} {value:+.4f}")

    print("largest |kappa| with armature")
    _largest(heat_with)
    print("largest |kappa| without armature")
    _largest(heat_without)

    print(
        f"fast quintic duration {FAST_DURATION:.5f} s, distance {DISTANCE:.1f} rad, "
        f"target peak speed {VELOCITY_LIMIT:.1f} rad/s"
    )
    cases: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for joint, stem in ((1, "single_joint_shoulder"), (2, "single_joint_elbow")):
        for profile, duration in (("slow", DURATION), ("fast", FAST_DURATION)):
            data = _trajectory(chain, joint, duration)
            cases[(joint, profile)] = data
            filename = stem if profile == "slow" else f"{stem}_fast"
            dist = float(data["distance"][0])
            peak = float(np.max(np.abs(data["qd"])))
            if profile == "slow":
                title = f"Holding torque while {SHORT[joint]} moves {dist:+.1f} rad in 1 s"
            else:
                title = (
                    f"Holding torque while {SHORT[joint]} moves {dist:+.1f} rad "
                    f"in {duration:.5f} s (peak |qd| {peak:.3f} rad/s)"
                )
            _save_trajectory(OUT / f"{filename}.csv", data)
            _plot_trajectory(OUT / f"{filename}.png", joint, data, chain.arm_torque_limit, title)
            _summarize(joint, data, chain.arm_torque_limit, profile)
            _summarize_dynamic(joint, data, chain.arm_torque_limit, profile)
            _summarize_driven(joint, data, chain.arm_torque_limit, profile)
            if not _endpoints_ok(data):
                print("quintic endpoint velocity or acceleration is not zero", file=sys.stderr)
                return 1
            if profile == "fast" and abs(peak - VELOCITY_LIMIT) > 1e-9:
                print(f"fast peak speed {peak} is not {VELOCITY_LIMIT}", file=sys.stderr)
                return 1
    _plot_dynamic(cases, chain.arm_torque_limit)
    _save_dynamic(cases, chain.arm_torque_limit)
    _save_driven(cases, chain.arm_torque_limit)
    print("COUPLING_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
