"""Hand-written inverse dynamics for the six arm joints.

The recursion is Newton-Euler in the world frame. It reads only the parameters
stored on ``ArmChain`` (mass, centre of mass, inertia, joint axis). Finger
joints stay at the chain's opening with zero velocity and acceleration, so the
fingers and the fixed gripper bodies contribute as rigid attachments.

The returned torque matches

    M(q) qdd + C(q, qd) qd + g(q)

with armature on the diagonal of M when ``armature`` is true. Damping and
frictionloss are not included. ``g(q)`` is the same recursion at zero velocity
and acceleration. ``C(q, qd) qd`` is the same recursion at zero acceleration
and zero gravity. The Coriolis matrix itself is built from Christoffel symbols
of M, using a central difference of M (see ``CHRISTOFFEL_STEP``).

MuJoCo's recursive inverse dynamics includes the velocity product and gravity
and omits armature. A comparison has to add the armature term separately, and
the reference spatial velocity has to be current. This module does not call it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aloha_pp.kin.fk import HINGE, ArmChain, ChainState

# Fourth-order central-difference step for ∂M/∂q, in radians.
# A second-order step of 1e-6 left relative residuals near 1e-7 on the smaller
# Coriolis components. This stencil keeps both the absolute and the relative
# residual under 1e-10 against the velocity-product recursion.
CHRISTOFFEL_STEP = 1e-3


def _as_arm(vec: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(6)
    if arr.shape != (6,):
        raise ValueError(f"{name} must have shape (6,)")
    return arr


def rnea(
    chain: ArmChain,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    gravity: np.ndarray | None = None,
    *,
    armature: bool = True,
    state: ChainState | None = None,
) -> np.ndarray:
    """Joint torque (6,) for the arm. Fingers stay locked."""
    q = _as_arm(q, "q")
    qd = _as_arm(qd, "qd")
    qdd = _as_arm(qdd, "qdd")
    gravity_vec = chain.gravity if gravity is None else np.asarray(gravity, dtype=float)
    state = chain.forward(q) if state is None else state

    qd_full = np.zeros(chain.nq)
    qdd_full = np.zeros(chain.nq)
    qd_full[chain.arm_qadr] = qd
    qdd_full[chain.arm_qadr] = qdd

    omega = np.zeros((chain.nbody, 3))
    alpha = np.zeros((chain.nbody, 3))
    origin_acc = np.zeros((chain.nbody, 3))
    for i in range(1, chain.nbody):
        pid = int(chain.parent[i])
        offset = state.xpos[i] - state.xpos[pid]
        joint = int(chain.body_joint[i])
        if joint < 0:
            omega[i] = omega[pid]
            alpha[i] = alpha[pid]
            origin_acc[i] = _rigid_acc(origin_acc[pid], alpha[pid], omega[pid], offset)
            continue
        axis = state.axis_w[joint]
        rate = float(qd_full[int(chain.jnt_qadr[joint])])
        acc = float(qdd_full[int(chain.jnt_qadr[joint])])
        if int(chain.jnt_type[joint]) == HINGE:
            omega[i] = omega[pid] + axis * rate
            alpha[i] = alpha[pid] + axis * acc + np.cross(omega[pid], axis * rate)
            origin_acc[i] = _rigid_acc(origin_acc[pid], alpha[pid], omega[pid], offset)
        else:
            omega[i] = omega[pid]
            alpha[i] = alpha[pid]
            origin_acc[i] = _rigid_acc(origin_acc[pid], alpha[pid], omega[pid], offset)
            origin_acc[i] = origin_acc[i] + 2.0 * np.cross(omega[pid], axis * rate) + axis * acc

    force = np.zeros((chain.nbody, 3))
    moment = np.zeros((chain.nbody, 3))
    for i in range(chain.nbody - 1, 0, -1):
        rot = state.xmat[i]
        com_local = rot @ chain.body_ipos[i]
        com_acc = (
            origin_acc[i]
            + np.cross(alpha[i], com_local)
            + np.cross(omega[i], np.cross(omega[i], com_local))
        )
        inertia_w = rot @ chain.I_com_body[i] @ rot.T
        own_force = chain.body_mass[i] * (com_acc - gravity_vec)
        own_moment = (
            np.cross(com_local, own_force)
            + inertia_w @ alpha[i]
            + np.cross(omega[i], inertia_w @ omega[i])
        )
        force[i] += own_force
        moment[i] += own_moment
        pid = int(chain.parent[i])
        if pid != 0:
            offset = state.xpos[i] - state.xpos[pid]
            force[pid] += force[i]
            moment[pid] += moment[i] + np.cross(offset, force[i])

    tau = np.zeros(6)
    for col, joint in enumerate(chain.arm_joint_ids):
        body = int(chain.jnt_body[int(joint)])
        tau[col] = float(np.dot(state.axis_w[int(joint)], moment[body]))
        if armature:
            tau[col] += float(chain.arm_armature[col]) * float(qdd[col])
    return tau


def _rigid_acc(
    parent_acc: np.ndarray,
    parent_alpha: np.ndarray,
    parent_omega: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    return (
        parent_acc
        + np.cross(parent_alpha, offset)
        + np.cross(parent_omega, np.cross(parent_omega, offset))
    )


def gravity_torque(chain: ArmChain, q: np.ndarray, *, state: ChainState | None = None) -> np.ndarray:
    zero = np.zeros(6)
    return rnea(chain, q, zero, zero, armature=False, state=state)


def coriolis_force(chain: ArmChain, q: np.ndarray, qd: np.ndarray, *, state: ChainState | None = None) -> np.ndarray:
    """``C(q, qd) qd`` from the velocity product, with gravity removed."""
    return rnea(chain, q, qd, np.zeros(6), gravity=np.zeros(3), armature=False, state=state)


def mass_matrix(
    chain: ArmChain,
    q: np.ndarray,
    *,
    armature: bool = True,
    state: ChainState | None = None,
) -> np.ndarray:
    """Mass matrix from six unit-acceleration RNEA solves. Includes armature by default."""
    state = chain.forward(q) if state is None else state
    zero = np.zeros(6)
    matrix = np.zeros((6, 6))
    for col in range(6):
        qdd = np.zeros(6)
        qdd[col] = 1.0
        matrix[:, col] = rnea(chain, q, zero, qdd, gravity=np.zeros(3), armature=armature, state=state)
    return matrix


def mass_matrix_energy(chain: ArmChain, q: np.ndarray, *, armature: bool = True) -> np.ndarray:
    """Mass matrix from body Jacobians and the kinetic energy, a second implementation."""
    state = chain.forward(q)
    matrix = np.zeros((6, 6))
    for body in range(1, chain.nbody):
        rot = state.xmat[body]
        com = state.xpos[body] + rot @ chain.body_ipos[body]
        inertia_w = rot @ chain.I_com_body[body] @ rot.T
        jac_w = np.zeros((3, 6))
        jac_v = np.zeros((3, 6))
        for col, joint in enumerate(chain.arm_joint_ids):
            if not chain.arm_affects[col, body]:
                continue
            axis = state.axis_w[int(joint)]
            anchor = state.anchor_w[int(joint)]
            jac_w[:, col] = axis
            jac_v[:, col] = np.cross(axis, com - anchor)
        matrix += jac_w.T @ inertia_w @ jac_w
        matrix += float(chain.body_mass[body]) * (jac_v.T @ jac_v)
    if armature:
        matrix += np.diag(chain.arm_armature)
    return matrix


def coriolis_matrix(
    chain: ArmChain,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    step: float = CHRISTOFFEL_STEP,
    armature: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Christoffel matrix ``C(q, qd)`` and ``Mdot``.

    Partial derivatives of M use a fourth-order central difference with
    spacing ``step`` (radians):

        (M(q-2h) - 8 M(q-h) + 8 M(q+h) - M(q+2h)) / (12 h)

    Armature does not depend on q, so it does not enter the derivatives.
    Returns ``(C, Mdot)``. ``Mdot - 2 C`` is skew-symmetric when M is symmetric.
    """
    q = _as_arm(q, "q")
    qd = _as_arm(qd, "qd")
    if step <= 0.0:
        raise ValueError("step must be positive")
    deriv = np.zeros((6, 6, 6))
    for k in range(6):
        direction = np.zeros(6)
        direction[k] = 1.0
        plus_two = mass_matrix(chain, q + 2.0 * step * direction, armature=armature)
        plus_one = mass_matrix(chain, q + step * direction, armature=armature)
        minus_one = mass_matrix(chain, q - step * direction, armature=armature)
        minus_two = mass_matrix(chain, q - 2.0 * step * direction, armature=armature)
        deriv[k] = (minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two) / (12.0 * step)
    deriv = 0.5 * (deriv + np.transpose(deriv, (0, 2, 1)))
    coriolis = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            total = 0.0
            for k in range(6):
                gamma = 0.5 * (deriv[k, i, j] + deriv[j, i, k] - deriv[i, j, k])
                total += gamma * float(qd[k])
            coriolis[i, j] = total
    mdot = np.zeros((6, 6))
    for k in range(6):
        mdot += deriv[k] * float(qd[k])
    return coriolis, mdot


def skew_residual(mdot: np.ndarray, coriolis: np.ndarray) -> np.ndarray:
    """``(Mdot - 2 C) + (Mdot - 2 C).T``, which vanishes for a valid C."""
    skew = mdot - 2.0 * coriolis
    return skew + skew.T


def composite_inertia(chain: ArmChain, q: np.ndarray | None = None) -> list[dict[str, object]]:
    """Merge each arm link with the bodies rigidly attached to it.

    Fixed descendants are included. Finger bodies are included at ``finger_q``
    because those joints are locked. The result is expressed in the link frame,
    about the composite centre of mass, and does not depend on the arm pose.
    """
    if q is None:
        q = np.zeros(6)
    state = chain.forward(q)
    arm_bodies = set(int(b) for b in chain.arm_body)
    rows: list[dict[str, object]] = []
    for name, body in chain.link_ids:
        cluster = _rigid_cluster(chain, int(body), arm_bodies)
        mass = 0.0
        com_world = np.zeros(3)
        for member in cluster:
            member_mass = float(chain.body_mass[member])
            member_com = state.xpos[member] + state.xmat[member] @ chain.body_ipos[member]
            mass += member_mass
            com_world += member_mass * member_com
        if mass <= 0.0:
            raise ValueError(f"{name} has no mass")
        com_world /= mass
        inertia_w = np.zeros((3, 3))
        for member in cluster:
            rot = state.xmat[member]
            member_com = state.xpos[member] + rot @ chain.body_ipos[member]
            offset = member_com - com_world
            member_inertia = rot @ chain.I_com_body[member] @ rot.T
            member_mass = float(chain.body_mass[member])
            shift = member_mass * ((float(np.dot(offset, offset)) * np.eye(3)) - np.outer(offset, offset))
            inertia_w += member_inertia + shift
        link_rot = state.xmat[int(body)]
        com_link = link_rot.T @ (com_world - state.xpos[int(body)])
        inertia_link = link_rot.T @ inertia_w @ link_rot
        inertia_link = 0.5 * (inertia_link + inertia_link.T)
        rows.append(
            {
                "link": name,
                "bodies": [chain.body_name[i] for i in cluster],
                "mass": mass,
                "com": com_link,
                "inertia": inertia_link,
                "principal": np.linalg.eigvalsh(inertia_link),
                "finger_q": chain.finger_q,
            }
        )
    return rows


def _rigid_cluster(chain: ArmChain, root: int, arm_bodies: set[int]) -> list[int]:
    cluster = [root]
    stack = [root]
    while stack:
        current = stack.pop()
        for child in chain.children[current]:
            if child in arm_bodies:
                continue
            cluster.append(child)
            stack.append(child)
    return cluster


def _report_number(value: float) -> str:
    number = float(value)
    if abs(number) < 1e-12:
        number = 0.0
    return f"{number:.8g}"


def write_inertia_table(chain: ArmChain, directory: Path, finger_openings: tuple[float, ...]) -> tuple[Path, Path]:
    """Composite mass, centre of mass and inertia for each arm link."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "inertia_parameters.csv"
    md_path = directory / "inertia_parameters.md"
    header = (
        "finger_q,link,mass,com_x,com_y,com_z,"
        "Ixx,Ixy,Ixz,Iyy,Iyz,Izz,I1,I2,I3,bodies"
    )
    lines = [header]
    md = [
        "# Composite inertia",
        "",
        "Each row merges the link with every descendant that is not another arm link. "
        "Finger joints are locked at `finger_q`, so the fingers are part of `left/gripper_link`. "
        "The centre of mass and the inertia are expressed in the link frame, about the composite "
        "centre of mass. Arm pose does not change these link-frame values.",
        "",
        "| finger_q | link | mass (kg) | com in link (m) | principal moments |",
        "|---|---|---|---|---|",
    ]
    saved = chain.finger_q
    try:
        for opening in finger_openings:
            chain.set_finger_q(opening)
            for row in composite_inertia(chain):
                com = np.asarray(row["com"], dtype=float)
                inertia = np.asarray(row["inertia"], dtype=float)
                principal = np.asarray(row["principal"], dtype=float)
                bodies = " ".join(row["bodies"])  # type: ignore[arg-type]
                lines.append(
                    ",".join(
                        [
                            f"{opening:.8g}",
                            str(row["link"]),
                            _report_number(float(row["mass"])),
                            *(_report_number(float(v)) for v in com),
                            _report_number(inertia[0, 0]),
                            _report_number(inertia[0, 1]),
                            _report_number(inertia[0, 2]),
                            _report_number(inertia[1, 1]),
                            _report_number(inertia[1, 2]),
                            _report_number(inertia[2, 2]),
                            *(_report_number(float(v)) for v in principal),
                            bodies.replace(",", " "),
                        ]
                    )
                )
                md.append(
                    f"| {opening:.3f} | {row['link']} | {_report_number(float(row['mass']))} | "
                    f"{_report_number(com[0])} {_report_number(com[1])} {_report_number(com[2])} | "
                    f"{_report_number(principal[0])} {_report_number(principal[1])} {_report_number(principal[2])} |"
                )
    finally:
        chain.set_finger_q(saved)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path, md_path
