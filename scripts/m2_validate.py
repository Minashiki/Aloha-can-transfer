"""Cross-check the hand-written model against MuJoCo.

Run: conda run -n AIControl python scripts/m2_validate.py

Tolerances
----------
ABS_TOL and REL_TOL are 1e-8. Double precision is about 2e-16. The arm is
shorter than 1 m and the joint angles in these samples are O(1), so a correct
product of rotations lands near 1e-15 m and 1e-15 rad. Mass-matrix entries are
O(1e-3) to O(1) kg m^2 and joint torques in the sample box are O(1) to O(10)
N·m, so the same residual is far below 1e-12 N·m. 1e-8 sits four to seven
orders above that roundoff and well below a millimetre, a milliradian, or a
millinewton-metre. Relative error is taken only on entries whose reference
magnitude exceeds 1e-8, so a 1e-16 residual on a true zero does not inflate it.

Non-finite samples are rejected before the max reduction. ``max`` of a finite
value and NaN keeps the finite value, so a NaN or Inf result would otherwise
be recorded as a zero error and still PASS.

MuJoCo 3.17's mj_rne includes Coriolis, centrifugal and gravity once the
spatial velocity has been updated (mj_fwdVelocity / mj_comVel). It does not
include armature. The armature term is what separates mj_rne from mj_fullM.
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

from aloha_pp.dyn.rnea import (
    CHRISTOFFEL_STEP,
    coriolis_force,
    coriolis_matrix,
    gravity_torque,
    mass_matrix,
    mass_matrix_energy,
    rnea,
    skew_residual,
    write_inertia_table,
)
from aloha_pp.kin.fk import TCP_SITE, ArmChain, write_kinematic_table
from aloha_pp.kin.spatial import rotation_angle
from aloha_pp.robot import robot_xml

SEED = 20261113
N_SAMPLES = 200
FINGER_OPENINGS = (0.037, 0.015)
QD_LIMIT = 2.0
QDD_LIMIT = 10.0
ABS_TOL = 1e-8
REL_TOL = 1e-8
REF_FLOOR = 1e-8
# The estimate is sigma_max / max(sigma_min, COND_DENOM_FLOOR). At the coaxial
# pose sigma_min is theoretically zero, so the condition number is infinite.
COND_DENOM_FLOOR = 1e-16
COND_ESTIMATE_NAME = "分母下限 1e-16 的条件数估计"
SINGULAR_LABEL = "理论上为无穷大"
# M1 pick-and-place home. Wrist angle is away from the coaxial singularity.
TYPICAL_Q = np.array([0.0, -0.24632554, 0.35635693, 0.0, 1.46044343, 0.0])

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m2"


class Tracker:
    def __init__(self) -> None:
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.nonfinite = False

    def add(self, got: np.ndarray, ref: np.ndarray) -> None:
        got_arr = np.asarray(got, dtype=float)
        ref_arr = np.asarray(ref, dtype=float)
        if not (np.all(np.isfinite(got_arr)) and np.all(np.isfinite(ref_arr))):
            self.nonfinite = True
            return
        diff = np.abs(got_arr - ref_arr)
        if diff.size == 0:
            return
        self.max_abs = max(self.max_abs, float(np.max(diff)))
        denom = np.abs(ref_arr)
        mask = denom > REF_FLOOR
        if np.any(mask):
            self.max_rel = max(self.max_rel, float(np.max(diff[mask] / denom[mask])))

    def add_scalar(self, value: float) -> None:
        number = float(value)
        if not np.isfinite(number):
            self.nonfinite = True
            return
        self.max_abs = max(self.max_abs, number)
        self.max_rel = max(self.max_rel, number)

    @property
    def ok(self) -> bool:
        return (not self.nonfinite) and self.max_abs <= ABS_TOL and self.max_rel <= REL_TOL


def _prepare(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chain: ArmChain,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> None:
    idx = chain.arm_dof
    data.qpos[:] = chain.pack_q(q)
    data.qvel[:] = 0.0
    data.qvel[idx] = qd
    data.qacc[:] = 0.0
    mujoco.mj_fwdPosition(model, data)
    data.qvel[:] = 0.0
    data.qvel[idx] = qd
    data.qacc[:] = 0.0
    data.qacc[idx] = qdd
    mujoco.mj_fwdVelocity(model, data)
    data.qacc[:] = 0.0
    data.qacc[idx] = qdd


def _wrist_curve(chain: ArmChain, model: mujoco.MjModel, roll: float) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    # Neighbourhood of the coaxial pose. 0 is on this grid (step = 0.01).
    angles = np.linspace(-1.2, 1.2, 241)
    sigma = np.zeros_like(angles)
    cond = np.zeros_like(angles)
    sigma_mj = np.zeros_like(angles)
    cross = np.zeros_like(angles)
    distance = np.zeros_like(angles)
    q = TYPICAL_Q.copy()
    q[3] = roll
    for i, angle in enumerate(angles):
        q[4] = angle
        jac = chain.tcp_jacobian(q)
        singular = np.linalg.svd(jac, compute_uv=False)
        sigma[i] = singular[-1]
        cond[i] = singular[0] / max(singular[-1], COND_DENOM_FLOOR)
        align = chain.wrist_alignment(q)
        cross[i] = align["cross_norm"]
        distance[i] = align["line_distance"]
        data.qpos[:] = chain.pack_q(q)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        ref = np.vstack((jacp[:, chain.arm_dof], jacr[:, chain.arm_dof]))
        singular_mj = np.linalg.svd(ref, compute_uv=False)
        sigma_mj[i] = singular_mj[-1]
    return {
        "wrist_angle": angles,
        "sigma_min": sigma,
        "sigma_min_mj": sigma_mj,
        "cond": cond,
        "cross_norm": cross,
        "line_distance": distance,
    }


def _status(tracker: Tracker) -> str:
    if tracker.nonfinite:
        return "FAIL nonfinite"
    if tracker.ok:
        return "PASS"
    return "FAIL"


def _write_wrist(curves: dict[float, dict[str, np.ndarray]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True)
    for roll, curve in curves.items():
        label = f"forearm_roll = {roll:.2f}"
        axes[0].plot(curve["wrist_angle"], curve["sigma_min"], label=label)
        axes[1].plot(curve["wrist_angle"], curve["cond"], label=label)
    axes[0].axvline(0.0, color="0.4", ls="--", lw=1, label="wrist_angle = 0")
    axes[1].axvline(0.0, color="0.4", ls="--", lw=1, label=f"wrist_angle = 0，{SINGULAR_LABEL}")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("smallest singular value")
    axes[1].set_ylabel(COND_ESTIMATE_NAME)
    axes[1].set_xlabel("wrist_angle (rad)")
    axes[1].set_yscale("log")
    axes[1].annotate(
        SINGULAR_LABEL,
        xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=9,
        color="0.15",
        ha="left",
        va="top",
    )
    axes[0].set_title("TCP Jacobian near the forearm_roll / wrist_rotate alignment")
    fig.tight_layout()
    fig.savefig(OUT / "wrist_singularity.png", dpi=140)
    plt.close(fig)
    roll0 = next(iter(curves.values()))
    columns = [
        roll0["wrist_angle"],
        roll0["sigma_min"],
        roll0["sigma_min_mj"],
        roll0["cond"],
        roll0["cross_norm"],
        roll0["line_distance"],
    ]
    header = f"wrist_angle,sigma_min,sigma_min_mj,{COND_ESTIMATE_NAME},axis_cross,line_distance"
    extra = curves.get(1.0)
    if extra is not None:
        columns.append(extra["sigma_min"])
        header += ",sigma_min_roll_1"
    stacked = np.column_stack(columns)
    with (OUT / "wrist_singularity.csv").open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        np.savetxt(handle, stacked, delimiter=",")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(robot_xml()))
    data = mujoco.MjData(model)
    chain = ArmChain(model, finger_q=FINGER_OPENINGS[0])
    gravity = model.opt.gravity.copy()
    write_kinematic_table(chain, OUT)
    write_inertia_table(chain, OUT, FINGER_OPENINGS)

    rng = np.random.default_rng(SEED)
    q_samples = rng.uniform(chain.arm_range[:, 0], chain.arm_range[:, 1], size=(N_SAMPLES, 6))
    qd_samples = rng.uniform(-QD_LIMIT, QD_LIMIT, size=(N_SAMPLES, 6))
    qdd_samples = rng.uniform(-QDD_LIMIT, QDD_LIMIT, size=(N_SAMPLES, 6))

    names = [
        "fk_pos_world",
        "fk_rot_world",
        "fk_pos_base",
        "fk_rot_base",
        "jacobian",
        "M",
        "M_energy",
        "g",
        "Cqd",
        "rnea_body_vs_mj_rne",
        "rnea_full_vs_mj_rne_plus_armature",
        "rnea_full_vs_Mqdd_plus_bias",
        "passive_vs_minus_B_qd",
    ]
    trackers = {name: Tracker() for name in names}
    christoffel = Tracker()
    skew = Tracker()
    mass_by_opening: dict[float, list[np.ndarray]] = {opening: [] for opening in FINGER_OPENINGS}
    finger_coupling = 0.0
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    idx = chain.arm_dof
    finger_dof = chain.jnt_dof[chain.finger_joint_ids]

    print(
        f"samples {N_SAMPLES} seed {SEED} openings {FINGER_OPENINGS} "
        f"qd U[-{QD_LIMIT},{QD_LIMIT}] qdd U[-{QDD_LIMIT},{QDD_LIMIT}]"
    )
    try:
        for opening in FINGER_OPENINGS:
            chain.set_finger_q(opening)
            for q, qd, qdd in zip(q_samples, qd_samples, qdd_samples):
                _prepare(model, data, chain, q, qd, qdd)
                world, in_base = chain.poses(q)
                state = chain.forward(q)
                for body in range(model.nbody):
                    trackers["fk_pos_world"].add(state.xpos[body], data.xpos[body])
                    trackers["fk_rot_world"].add_scalar(
                        rotation_angle(state.xmat[body], data.xmat[body].reshape(3, 3))
                    )
                base_rot = data.xmat[chain.base_id].reshape(3, 3)
                base_pos = data.xpos[chain.base_id]
                for name, body in chain.link_ids:
                    trackers["fk_pos_base"].add(in_base[name].pos, base_rot.T @ (data.xpos[body] - base_pos))
                    trackers["fk_rot_base"].add_scalar(
                        rotation_angle(in_base[name].rot, base_rot.T @ data.xmat[body].reshape(3, 3))
                    )
                tcp_pos = base_rot.T @ (data.site_xpos[site] - base_pos)
                tcp_rot = base_rot.T @ data.site_xmat[site].reshape(3, 3)
                trackers["fk_pos_base"].add(in_base[TCP_SITE].pos, tcp_pos)
                trackers["fk_rot_base"].add_scalar(rotation_angle(in_base[TCP_SITE].rot, tcp_rot))
                trackers["fk_pos_world"].add(world[TCP_SITE].pos, data.site_xpos[site])
                trackers["fk_rot_world"].add_scalar(
                    rotation_angle(world[TCP_SITE].rot, data.site_xmat[site].reshape(3, 3))
                )

                mujoco.mj_jacSite(model, data, jacp, jacr, site)
                jac_ref = np.vstack((jacp[:, idx], jacr[:, idx]))
                trackers["jacobian"].add(chain.tcp_jacobian(q, state), jac_ref)

                dense = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, data, dense)
                block = dense[np.ix_(idx, idx)]
                got_m = mass_matrix(chain, q, state=state)
                trackers["M"].add(got_m, block)
                trackers["M_energy"].add(mass_matrix_energy(chain, q), got_m)
                mass_by_opening[opening].append(got_m.copy())
                finger_coupling = max(finger_coupling, float(np.max(np.abs(dense[np.ix_(idx, finger_dof)]))))

                rne_tau = np.zeros(model.nv)
                mujoco.mj_rne(model, data, 1, rne_tau)
                body_tau = rnea(chain, q, qd, qdd, armature=False, state=state)
                full_tau = rnea(chain, q, qd, qdd, armature=True, state=state)
                trackers["rnea_body_vs_mj_rne"].add(body_tau, rne_tau[idx])
                trackers["rnea_full_vs_mj_rne_plus_armature"].add(
                    full_tau, rne_tau[idx] + chain.arm_armature * qdd
                )
                trackers["rnea_full_vs_Mqdd_plus_bias"].add(full_tau, block @ qdd + data.qfrc_bias[idx])
                trackers["passive_vs_minus_B_qd"].add(data.qfrc_passive[idx], -chain.arm_damping * qd)

                data.qvel[:] = 0.0
                mujoco.mj_fwdVelocity(model, data)
                trackers["g"].add(gravity_torque(chain, q, state=state), data.qfrc_bias[idx])

                model.opt.gravity[:] = 0.0
                data.qvel[:] = 0.0
                data.qvel[idx] = qd
                mujoco.mj_fwdVelocity(model, data)
                model.opt.gravity[:] = gravity
                trackers["Cqd"].add(coriolis_force(chain, q, qd, state=state), data.qfrc_bias[idx])
    finally:
        model.opt.gravity[:] = gravity

    chain.set_finger_q(FINGER_OPENINGS[0])
    christoffel_rng = np.random.default_rng(SEED + 7)
    for _ in range(8):
        q = christoffel_rng.uniform(chain.arm_range[:, 0], chain.arm_range[:, 1])
        qd = christoffel_rng.uniform(-QD_LIMIT, QD_LIMIT, size=6)
        matrix, mdot = coriolis_matrix(chain, q, qd, step=CHRISTOFFEL_STEP)
        christoffel.add(matrix @ qd, coriolis_force(chain, q, qd))
        skew.add(skew_residual(mdot, matrix), np.zeros_like(mdot))

    print(
        f"{'item':42s} {'max_abs':12s} {'max_rel':12s} {'abs_tol':10s} {'rel_tol':10s} status"
    )
    failed = []
    rows = []
    for name, tracker in trackers.items():
        status = _status(tracker)
        if not tracker.ok:
            failed.append(name)
        line = (
            f"{name:42s} {tracker.max_abs:12.3e} {tracker.max_rel:12.3e} "
            f"{ABS_TOL:10.1e} {REL_TOL:10.1e} {status}"
        )
        print(line)
        rows.append((name, tracker.max_abs, tracker.max_rel, status, tracker.nonfinite))
    for name, tracker in (("christoffel_Cqd", christoffel), ("Mdot_minus_2C_skew", skew)):
        status = _status(tracker)
        if not tracker.ok:
            failed.append(name)
        line = (
            f"{name:42s} {tracker.max_abs:12.3e} {tracker.max_rel:12.3e} "
            f"{ABS_TOL:10.1e} {REL_TOL:10.1e} {status}"
        )
        print(line)
        rows.append((name, tracker.max_abs, tracker.max_rel, status, tracker.nonfinite))
    print(
        f"christoffel 4th-order central difference step {CHRISTOFFEL_STEP:g} rad, "
        f"8 samples, finger_q {FINGER_OPENINGS[0]}"
    )

    paired = np.stack(mass_by_opening[0.037]) - np.stack(mass_by_opening[0.015])
    print(
        f"finger M_aa |q=0.037 - q=0.015| max {np.max(np.abs(paired)):.3e} "
        f"mean {np.mean(np.abs(paired)):.3e}"
    )
    print(f"finger coupling block |M_af| max {finger_coupling:.3e}")

    # Nonzero finger velocity changes the arm bias; the rigid-attachment model assumes it is zero.
    chain.set_finger_q(0.037)
    q = q_samples[0]
    qd = qd_samples[0]
    _prepare(model, data, chain, q, qd, np.zeros(6))
    bias_locked = data.qfrc_bias[idx].copy()
    data.qvel[finger_dof] = 0.05
    mujoco.mj_fwdVelocity(model, data)
    bias_moving = data.qfrc_bias[idx].copy()
    print(
        f"arm bias change when both finger joints run at 0.05 m/s: "
        f"max abs {np.max(np.abs(bias_moving - bias_locked)):.3e}"
    )

    if not _friction_report(model, chain):
        failed.append("friction_demo")
    curves = {0.0: _wrist_curve(chain, model, 0.0), 1.0: _wrist_curve(chain, model, 1.0)}
    _write_wrist(curves)
    sigma_gap = max(np.max(np.abs(curve["sigma_min"] - curve["sigma_min_mj"])) for curve in curves.values())
    print(f"wrist sigma_min vs mj_jacSite max abs {sigma_gap:.3e}")
    if sigma_gap > ABS_TOL:
        failed.append("wrist_sigma")
    for roll, curve in curves.items():
        i_cross = int(np.argmin(curve["cross_norm"]))
        i_zero = int(np.argmin(np.abs(curve["wrist_angle"])))
        i_off = int(np.argmin(np.abs(curve["wrist_angle"] - 0.4)))
        print(
            f"roll {roll:.2f} axis cross minimum {curve['cross_norm'][i_cross]:.3e} "
            f"at wrist_angle {curve['wrist_angle'][i_cross]:.4f} "
            f"line_distance {curve['line_distance'][i_zero]:.3e}"
        )
        print(
            f"roll {roll:.2f} sigma_min(0) {curve['sigma_min'][i_zero]:.6e} "
            f"{COND_ESTIMATE_NAME}(0) {curve['cond'][i_zero]:.6e} ({SINGULAR_LABEL}) "
            f"sigma_min(0.4) {curve['sigma_min'][i_off]:.6e} "
            f"{COND_ESTIMATE_NAME}(0.4) {curve['cond'][i_off]:.6e}"
        )
        if abs(float(curve["wrist_angle"][i_cross])) > 0.02 or curve["cross_norm"][i_zero] > 1e-8:
            failed.append(f"wrist_axis_roll_{roll}")
        if curve["sigma_min"][i_zero] >= curve["sigma_min"][i_off]:
            failed.append(f"wrist_sigma_dip_roll_{roll}")

    table = OUT / "validation_errors.csv"
    with table.open("w", encoding="utf-8") as handle:
        handle.write("item,max_abs,max_rel,status,nonfinite\n")
        for name, max_abs, max_rel, status, nonfinite in rows:
            handle.write(f"{name},{max_abs:.8e},{max_rel:.8e},{status},{int(nonfinite)}\n")
    if failed:
        print("FAILED", ", ".join(failed))
        return 1
    print("ALL WITHIN TOLERANCE")
    return 0


def _friction_report(model: mujoco.MjModel, chain: ArmChain) -> bool:
    data = mujoco.MjData(model)
    q = TYPICAL_Q.copy()
    qd = np.zeros(6)
    qd[1] = 0.4
    qd[2] = -0.25
    data.qpos[:] = chain.pack_q(q)
    data.qvel[:] = 0.0
    data.qvel[chain.arm_dof] = qd
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    passive = data.qfrc_passive[chain.arm_dof].copy()
    constraint = data.qfrc_constraint[chain.arm_dof].copy()
    expected_passive = -chain.arm_damping * qd
    print(f"friction demo ncon {data.ncon}")
    print(f"qfrc_passive arm {np.array2string(passive, precision=6)}")
    print(f"-B qd          {np.array2string(expected_passive, precision=6)}")
    passive_error = float(np.max(np.abs(passive - expected_passive)))
    print(f"passive error  {passive_error:.3e}")
    print(f"qfrc_constraint arm {np.array2string(constraint, precision=6)}")
    print(f"frictionloss arm    {np.array2string(chain.arm_frictionloss, precision=4)}")
    friction_error = 0.0
    for joint, rate in ((1, qd[1]), (2, qd[2])):
        mobilized = -chain.arm_frictionloss[joint] * np.sign(rate)
        friction_error = max(friction_error, abs(float(constraint[joint] - mobilized)))
        print(
            f"  {chain.joint_name[int(chain.arm_joint_ids[joint])]} "
            f"constraint {constraint[joint]:.6f} expected mobilized {mobilized:.6f}"
        )
    finger_c = data.qfrc_constraint[chain.jnt_dof[chain.finger_joint_ids]]
    print(f"finger equality qfrc_constraint {np.array2string(finger_c, precision=6)}")
    # Kinetic friction on a moving joint is frictionloss opposing qvel, and it
    # is a constraint force. Damping stays in qfrc_passive and does not contain it.
    return passive_error <= ABS_TOL and friction_error <= ABS_TOL and data.ncon == 0


if __name__ == "__main__":
    sys.exit(main())
