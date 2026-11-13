"""Hand-written forward kinematics for the ALOHA left arm.

Parameters are copied out of ``MjModel`` (body and joint placements, site
offsets). Pose and Jacobian are then computed with numpy only.

Frames
------
World is the ``aloha.xml`` world frame. The base frame is the body frame of
``left/base_link``, which is fixed to the world. Joint axes and ``body_pos`` /
``body_quat`` follow MuJoCo: the body offset is expressed in the parent frame,
and the joint axis and anchor are expressed in the child body frame before the
joint motion is applied.

The geometric Jacobian of ``left/tcp`` is 6×6 in the world frame, linear part
on top and angular part below, matching MuJoCo's site Jacobian layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from aloha_pp.kin.spatial import mat_to_quat, quat_to_mat, rot_about_axis
from aloha_pp.robot import ARM_JOINTS, BASE_BODY, GRIPPER_JOINT, RIGHT_FINGER_JOINT

TCP_SITE = "left/tcp"
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    rot: np.ndarray

    def expressed_in(self, frame: Pose) -> Pose:
        aligned = frame.rot.T @ (self.pos - frame.pos)
        return Pose(aligned, frame.rot.T @ self.rot)

    @property
    def quat(self) -> np.ndarray:
        return mat_to_quat(self.rot)


@dataclass
class ChainState:
    """Body frames and joint axes after a forward-kinematics evaluation."""

    xpos: np.ndarray
    xmat: np.ndarray
    axis_w: np.ndarray
    anchor_w: np.ndarray


class ArmChain:
    """Serial arm plus gripper bodies, with the fingers locked at one opening.

    ``q`` passed to the kinematics and dynamics methods is the 6-vector of arm
    joints in ``ARM_JOINTS`` order. Both finger joints are held at ``finger_q``.
    """

    def __init__(self, model: mujoco.MjModel, finger_q: float = 0.037) -> None:
        self.finger_q = float(finger_q)
        self.gravity = np.asarray(model.opt.gravity, dtype=float).copy()
        self.nbody = int(model.nbody)
        self.njnt = int(model.njnt)
        self.nq = int(model.nq)

        self.parent = np.asarray(model.body_parentid, dtype=int).copy()
        self.body_pos = np.asarray(model.body_pos, dtype=float).copy()
        self.body_quat = np.asarray(model.body_quat, dtype=float).copy()
        self.body_rot = np.stack([quat_to_mat(self.body_quat[i]) for i in range(self.nbody)])
        self.body_mass = np.asarray(model.body_mass, dtype=float).copy()
        self.body_ipos = np.asarray(model.body_ipos, dtype=float).copy()
        self.body_iquat = np.asarray(model.body_iquat, dtype=float).copy()
        self.body_inertia = np.asarray(model.body_inertia, dtype=float).copy()
        self.I_com_body = np.zeros((self.nbody, 3, 3))
        for i in range(self.nbody):
            principal = quat_to_mat(self.body_iquat[i])
            inertia = principal @ np.diag(self.body_inertia[i]) @ principal.T
            self.I_com_body[i] = 0.5 * (inertia + inertia.T)

        self.jnt_type = np.asarray(model.jnt_type, dtype=int).copy()
        self.jnt_axis = np.asarray(model.jnt_axis, dtype=float).copy()
        self.jnt_pos = np.asarray(model.jnt_pos, dtype=float).copy()
        self.jnt_body = np.asarray(model.jnt_bodyid, dtype=int).copy()
        self.jnt_qadr = np.asarray(model.jnt_qposadr, dtype=int).copy()
        self.jnt_dof = np.asarray(model.jnt_dofadr, dtype=int).copy()
        self.jnt_range = np.asarray(model.jnt_range, dtype=float).copy()
        self.dof_armature = np.asarray(model.dof_armature, dtype=float).copy()
        self.dof_damping = np.asarray(model.dof_damping, dtype=float).copy()
        self.dof_frictionloss = np.asarray(model.dof_frictionloss, dtype=float).copy()
        if np.max(np.abs(self.jnt_pos)) > 1e-12:
            raise ValueError("RNEA assumes every joint anchor is the body origin")
        if np.any((self.jnt_type != HINGE) & (self.jnt_type != SLIDE)):
            raise ValueError("only hinge and slide joints are supported")

        self.body_name = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.nbody)
        ]
        self.joint_name = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(self.njnt)
        ]
        self.body_joint = np.full(self.nbody, -1, dtype=int)
        for j in range(self.njnt):
            body = int(self.jnt_body[j])
            if self.body_joint[body] >= 0:
                raise ValueError(f"body {self.body_name[body]} has more than one joint")
            self.body_joint[body] = j

        self.children: list[list[int]] = [[] for _ in range(self.nbody)]
        for i in range(1, self.nbody):
            self.children[int(self.parent[i])].append(i)

        self.arm_joint_ids = np.array([self._joint_id(name) for name in ARM_JOINTS], dtype=int)
        self.arm_qadr = self.jnt_qadr[self.arm_joint_ids].copy()
        self.arm_dof = self.jnt_dof[self.arm_joint_ids].copy()
        self.arm_body = self.jnt_body[self.arm_joint_ids].copy()
        self.arm_range = self.jnt_range[self.arm_joint_ids].copy()
        self.arm_armature = self.dof_armature[self.arm_dof].copy()
        self.arm_damping = self.dof_damping[self.arm_dof].copy()
        self.arm_frictionloss = self.dof_frictionloss[self.arm_dof].copy()
        self.arm_torque_limit = np.zeros(6)
        for col, jid in enumerate(self.arm_joint_ids):
            limited = int(model.jnt_actfrclimited[jid])
            if not limited:
                raise ValueError(f"{self.joint_name[jid]} has no actuator force range")
            hi = float(model.jnt_actfrcrange[jid, 1])
            lo = float(model.jnt_actfrcrange[jid, 0])
            if hi <= 0.0 or abs(hi + lo) > 1e-9:
                raise ValueError(f"expected a symmetric force range for {self.joint_name[jid]}")
            self.arm_torque_limit[col] = hi

        finger_ids = [self._joint_id(GRIPPER_JOINT), self._joint_id(RIGHT_FINGER_JOINT)]
        self.finger_joint_ids = np.array(finger_ids, dtype=int)
        self.finger_qadr = self.jnt_qadr[self.finger_joint_ids].copy()

        self.base_id = self._body_id(BASE_BODY)
        self.link_ids = [(self.body_name[int(b)], int(b)) for b in self.arm_body]
        tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if tcp < 0:
            raise ValueError(f"site not found: {TCP_SITE}")
        self.tcp_body = int(model.site_bodyid[tcp])
        self.tcp_pos = np.asarray(model.site_pos[tcp], dtype=float).copy()
        self.tcp_rot = quat_to_mat(np.asarray(model.site_quat[tcp], dtype=float))

        self.arm_affects = np.zeros((6, self.nbody), dtype=bool)
        for col, body in enumerate(self.arm_body):
            for candidate in range(self.nbody):
                walker = candidate
                while walker != 0:
                    if walker == int(body):
                        self.arm_affects[col, candidate] = True
                        break
                    walker = int(self.parent[walker])

    def set_finger_q(self, finger_q: float) -> None:
        self.finger_q = float(finger_q)

    def _body_id(self, name: str) -> int:
        for i, body_name in enumerate(self.body_name):
            if body_name == name:
                return i
        raise ValueError(f"body not found: {name}")

    def _joint_id(self, name: str) -> int:
        for i, joint_name in enumerate(self.joint_name):
            if joint_name == name:
                return i
        raise ValueError(f"joint not found: {name}")

    def pack_q(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=float).reshape(6)
        qpos = np.zeros(self.nq)
        qpos[self.arm_qadr] = q_arm
        qpos[self.finger_qadr] = self.finger_q
        return qpos

    def forward(self, q_arm: np.ndarray) -> ChainState:
        qpos = self.pack_q(q_arm)
        xpos = np.zeros((self.nbody, 3))
        xmat = np.zeros((self.nbody, 3, 3))
        xmat[0] = np.eye(3)
        axis_w = np.zeros((self.njnt, 3))
        anchor_w = np.zeros((self.njnt, 3))
        for i in range(1, self.nbody):
            pid = int(self.parent[i])
            rot = xmat[pid] @ self.body_rot[i]
            pos = xpos[pid] + xmat[pid] @ self.body_pos[i]
            joint = int(self.body_joint[i])
            if joint >= 0:
                axis = rot @ self.jnt_axis[joint]
                anchor = pos + rot @ self.jnt_pos[joint]
                axis_w[joint] = axis
                anchor_w[joint] = anchor
                angle = float(qpos[int(self.jnt_qadr[joint])])
                if int(self.jnt_type[joint]) == HINGE:
                    delta = rot_about_axis(axis, angle)
                    pos = anchor + delta @ (pos - anchor)
                    rot = delta @ rot
                else:
                    pos = pos + axis * angle
            xpos[i] = pos
            xmat[i] = rot
        return ChainState(xpos, xmat, axis_w, anchor_w)

    def tcp_pose(self, state: ChainState) -> Pose:
        rot = state.xmat[self.tcp_body] @ self.tcp_rot
        pos = state.xpos[self.tcp_body] + state.xmat[self.tcp_body] @ self.tcp_pos
        return Pose(pos, rot)

    def poses(self, q_arm: np.ndarray) -> tuple[dict[str, Pose], dict[str, Pose]]:
        """Link and TCP poses in the world frame and in the base frame."""
        state = self.forward(q_arm)
        world: dict[str, Pose] = {}
        for name, body in self.link_ids:
            world[name] = Pose(state.xpos[body].copy(), state.xmat[body].copy())
        world[TCP_SITE] = self.tcp_pose(state)
        base = Pose(state.xpos[self.base_id].copy(), state.xmat[self.base_id].copy())
        in_base = {name: pose.expressed_in(base) for name, pose in world.items()}
        return world, in_base

    def tcp_jacobian(self, q_arm: np.ndarray, state: ChainState | None = None) -> np.ndarray:
        """6×6 geometric Jacobian of ``left/tcp`` in the world frame."""
        state = self.forward(q_arm) if state is None else state
        point = self.tcp_pose(state).pos
        jac = np.zeros((6, 6))
        for col, joint in enumerate(self.arm_joint_ids):
            axis = state.axis_w[int(joint)]
            anchor = state.anchor_w[int(joint)]
            jac[:3, col] = np.cross(axis, point - anchor)
            jac[3:, col] = axis
        return jac

    def wrist_alignment(self, q_arm: np.ndarray) -> dict[str, float]:
        """Alignment of the forearm-roll and wrist-rotate axes.

        The two hinges are collinear when ``left/wrist_angle`` is an integer
        multiple of π. Inside the joint range the only such value is 0.
        """
        state = self.forward(q_arm)
        roll = int(self.arm_joint_ids[3])
        rotate = int(self.arm_joint_ids[5])
        axis_roll = state.axis_w[roll]
        axis_rot = state.axis_w[rotate]
        anchor_roll = state.anchor_w[roll]
        anchor_rot = state.anchor_w[rotate]
        cross = np.cross(axis_roll, axis_rot)
        separation = anchor_rot - anchor_roll
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-12:
            distance = float(np.linalg.norm(np.cross(separation, axis_roll)))
        else:
            distance = float(abs(np.dot(separation, cross)) / cross_norm)
        return {
            "cross_norm": cross_norm,
            "line_distance": distance,
            "axis_dot": float(np.dot(axis_roll, axis_rot)),
        }


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in np.asarray(values).reshape(-1))


def write_kinematic_table(chain: ArmChain, directory: Path) -> tuple[Path, Path]:
    """Joint axes and parent-relative placements, for the report."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "kinematic_parameters.csv"
    md_path = directory / "kinematic_parameters.md"
    header = (
        "joint,child_body,parent_body,axis_x,axis_y,axis_z,"
        "anchor_x,anchor_y,anchor_z,pos_x,pos_y,pos_z,"
        "quat_w,quat_x,quat_y,quat_z,q_min,q_max,armature,damping,frictionloss"
    )
    rows = [header]
    md_rows = [
        "| joint | child | parent | axis in child | pos in parent | quat (wxyz) |",
        "|---|---|---|---|---|---|",
    ]
    for joint in chain.arm_joint_ids:
        body = int(chain.jnt_body[int(joint)])
        parent = int(chain.parent[body])
        axis = chain.jnt_axis[int(joint)]
        anchor = chain.jnt_pos[int(joint)]
        pos = chain.body_pos[body]
        quat = chain.body_quat[body]
        span = chain.jnt_range[int(joint)]
        dof = int(chain.jnt_dof[int(joint)])
        rows.append(
            ",".join(
                [
                    chain.joint_name[int(joint)],
                    chain.body_name[body],
                    chain.body_name[parent],
                    *(f"{float(v):.8g}" for v in axis),
                    *(f"{float(v):.8g}" for v in anchor),
                    *(f"{float(v):.8g}" for v in pos),
                    *(f"{float(v):.8g}" for v in quat),
                    f"{float(span[0]):.8g}",
                    f"{float(span[1]):.8g}",
                    f"{float(chain.dof_armature[dof]):.8g}",
                    f"{float(chain.dof_damping[dof]):.8g}",
                    f"{float(chain.dof_frictionloss[dof]):.8g}",
                ]
            )
        )
        md_rows.append(
            f"| {chain.joint_name[int(joint)]} | {chain.body_name[body]} | "
            f"{chain.body_name[parent]} | {_fmt(axis)} | {_fmt(pos)} | {_fmt(quat)} |"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    note = (
        "# Kinematic parameters\n\n"
        "World frame is the `aloha.xml` world. `left/base_link` is welded to the world; "
        "its body frame is the base frame. `axis` and `anchor` are in the child body "
        "frame before the joint moves. `pos` and `quat` (wxyz) place that child frame "
        "in the parent frame at zero joint angle, before the joint motion.\n\n"
        "Wrist singularity: `left/forearm_roll` and `left/wrist_rotate` are both local "
        "+X hinges, and the only rotation between them is `left/wrist_angle` about local Y. "
        "They are collinear exactly when `wrist_angle` is an integer multiple of π. "
        "Inside the joint range the only solution is `wrist_angle = 0`.\n\n"
    )
    md_path.write_text(note + "\n".join(md_rows) + "\n", encoding="utf-8")
    return csv_path, md_path
