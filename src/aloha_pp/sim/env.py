"""ALOHA can pick-and-place environment.

Actions are absolute TCP poses in the base frame plus a gripper opening.
M1's low-level controller is position actuators driven by DLS IK; pass another
``LowLevelController`` to replace it.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import yaml

from aloha_pp.robot import ARM_JOINTS, GRIPPER_JOINT, RIGHT_FINGER_JOINT, scene_xml
from aloha_pp.sim.controller import LowLevelController, PositionIkController
from aloha_pp.sim.gravity import apply_gravity_compensation
from aloha_pp.sim.success import SuccessTracker

MODES = ("fixed_1", "fixed_2", "random")


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs" / "m1_pick_place.yaml"
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class AlohaCanEnv:
    def __init__(
        self,
        config: dict | None = None,
        config_path: Path | None = None,
        controller: LowLevelController | None = None,
    ) -> None:
        self.cfg = config if config is not None else load_config(config_path)
        version = self.cfg["actuator"]
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml(version)))
        self.data = mujoco.MjData(self.model)
        apply_gravity_compensation(self.model, bool(self.cfg.get("gravity_compensation", False)))

        timestep = float(self.model.opt.timestep)
        self.n_substeps = int(round((1.0 / self.cfg["control_hz"]) / timestep))
        if self.n_substeps != int(self.cfg["physics_hz"]) // int(self.cfg["control_hz"]):
            raise RuntimeError(
                f"substeps {self.n_substeps} do not match "
                f"{self.cfg['physics_hz']} Hz / {self.cfg['control_hz']} Hz"
            )

        site_name = self.cfg["tcp_site"]
        self.controller = controller if controller is not None else PositionIkController(
            self.model,
            site_name,
            damping=float(self.cfg["ik"]["damping"]),
            max_iters=int(self.cfg["ik"]["max_iters"]),
            pos_tol=float(self.cfg["ik"]["pos_tol"]),
            rot_tol=float(self.cfg["ik"]["rot_tol"]),
            rot_weight=float(self.cfg["ik"]["rot_weight"]),
        )
        self._ik = self.controller.ik if isinstance(self.controller, PositionIkController) else None
        if self._ik is None:
            # Pose readout still needs a site Jacobian helper's frame math.
            from aloha_pp.kin.ik import DlsIk

            self._ik = DlsIk(self.model, site_name, max_iters=1)

        self.arm_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in ARM_JOINTS
            ],
            dtype=int,
        )
        self.arm_dof_adr = np.array(
            [
                self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in ARM_JOINTS
            ],
            dtype=int,
        )
        self.grip_qpos_adr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
        )
        self.right_qpos_adr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_FINGER_JOINT)
            ]
        )
        self.can_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can")
        self.can_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_collision")
        left_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left/left_finger_link")
        right_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left/right_finger_link")
        self.left_finger_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == left_body
        }
        self.right_finger_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == right_body
        }
        self.finger_geoms = self.left_finger_geoms | self.right_finger_geoms
        can_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_joint")
        self.can_qpos_adr = int(self.model.jnt_qposadr[can_joint])
        self.can_dof_adr = int(self.model.jnt_dofadr[can_joint])
        self.base_id = self._ik.base_id
        self.home_q = np.asarray(self.cfg["home_q"], dtype=float)
        cam = self.cfg["camera"]
        self.image_height = int(cam["height"])
        self.image_width = int(cam["width"])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        cell = self.cfg["target_cells"]["fixed_1"]
        self.tracker = SuccessTracker(
            center=np.asarray(cell["center"], dtype=float),
            half_extents=np.asarray(cell["half_extents"], dtype=float),
            vel_thresh=float(self.cfg["success"]["vel_thresh"]),
            hold_steps=int(self.cfg["success"]["hold_steps"]),
            require_release=bool(self.cfg["success"].get("require_release", True)),
        )
        self.success = False
        self._rng = np.random.default_rng()

    def __enter__(self) -> AlohaCanEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    @property
    def _renderer(self) -> mujoco.Renderer | None:
        """Default-resolution renderer, if one has been created.

        Existing checks use this attribute. Extra resolutions live in
        ``_renderers`` and are released by ``close`` as well.
        """
        if not self._renderers:
            return None
        key = (int(self.image_height), int(self.image_width))
        return self._renderers.get(key, next(iter(self._renderers.values())))

    def reset(
        self,
        seed: int | None = None,
        mode: str = "fixed_1",
        *,
        render: bool = True,
        can_pose: np.ndarray | None = None,
        state: dict | None = None,
    ) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if state is not None:
            return self._reset_from_state(state, mode, render=render)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_adr] = self.home_q
        self.data.qvel[:] = 0.0
        opening = float(self.cfg["gripper_open"])
        self.data.qpos[self.grip_qpos_adr] = opening
        self.data.qpos[self.right_qpos_adr] = opening
        if isinstance(self.controller, PositionIkController):
            self.data.ctrl[self.controller.arm_act] = self.home_q
            self.data.ctrl[self.controller.grip_act] = opening
        # Base pose does not depend on q, but reset leaves xpos stale.
        mujoco.mj_kinematics(self.model, self.data)
        if can_pose is None:
            pos, quat = self._sample_can(mode)
        else:
            pose = np.asarray(can_pose, dtype=float).reshape(7)
            pos = pose[:3].copy()
            quat = pose[3:].copy()
            quat = quat / np.linalg.norm(quat)
        self.data.qpos[self.can_qpos_adr : self.can_qpos_adr + 3] = self._base_to_world(pos)
        self.data.qpos[self.can_qpos_adr + 3 : self.can_qpos_adr + 7] = _base_quat_to_world(
            self.data, self.base_id, quat
        )
        self._apply_target_cell(mode)
        self.controller.reset(self.model, self.data)
        self.tracker.reset()
        self.success = False
        mujoco.mj_forward(self.model, self.data)
        # Let the can settle on the bin floor while the arm holds home.
        # PositionIkController already wrote a constant ctrl. A per-step
        # controller instead refreshes and commands inside the loop; the
        # PositionIk path is the original mj_step-only loop.
        for _ in range(50):
            if self._uses_substep_controller():
                self.controller.compute(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        # Settle integrates qpos but leaves xpos / contacts one step behind.
        # Sync before the observation reset returns. Success is not counted
        # during settle; the tracker was just cleared.
        self._sync_derived()
        self.success = False
        self.tracker.reset()
        return self.get_obs(render=render)

    def capture_state(self) -> dict:
        """Simulator state after ``reset``, for a bit-exact replay.

        ``qacc_warmstart`` is the constraint solver's warm start. Dropping it
        makes a later ``reset(state=...)`` integrate a different first step
        even when ``qpos``, ``qvel``, and ``ctrl`` match. ``time`` is the
        settled clock (the 50-step settle leaves it at ``50 * timestep``).
        """
        state = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "qacc_warmstart": self.data.qacc_warmstart.copy(),
            "qacc": self.data.qacc.copy(),
        }
        if int(self.model.na) > 0:
            state["act"] = self.data.act.copy()
        return state

    def _reset_from_state(self, state: dict, mode: str, *, render: bool) -> dict:
        """Restore a captured state and skip the settle loop.

        The default ``reset`` path never calls this. Settle is skipped because
        those 50 steps would leave the saved state.
        """
        qpos = np.asarray(state["qpos"], dtype=float).reshape(-1)
        qvel = np.asarray(state["qvel"], dtype=float).reshape(-1)
        if qpos.shape != self.data.qpos.shape or qvel.shape != self.data.qvel.shape:
            raise ValueError("state qpos/qvel shape does not match the model")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._apply_target_cell(mode)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset(self.model, self.data)
        if "ctrl" in state:
            ctrl = np.asarray(state["ctrl"], dtype=float).reshape(-1)
            if ctrl.shape != self.data.ctrl.shape:
                raise ValueError("state ctrl shape does not match the model")
            self.data.ctrl[:] = ctrl
        # After forward / controller.reset, so the first mj_step sees the
        # captured solver state rather than the zeros from mj_resetData.
        if "time" in state:
            self.data.time = float(state["time"])
        if "qacc_warmstart" in state:
            warm = np.asarray(state["qacc_warmstart"], dtype=float).reshape(-1)
            if warm.shape != self.data.qacc_warmstart.shape:
                raise ValueError("state qacc_warmstart shape does not match the model")
            self.data.qacc_warmstart[:] = warm
        if "qacc" in state:
            qacc = np.asarray(state["qacc"], dtype=float).reshape(-1)
            if qacc.shape != self.data.qacc.shape:
                raise ValueError("state qacc shape does not match the model")
            self.data.qacc[:] = qacc
        if "act" in state and int(self.model.na) > 0:
            act = np.asarray(state["act"], dtype=float).reshape(-1)
            if act.shape != self.data.act.shape:
                raise ValueError("state act shape does not match the model")
            self.data.act[:] = act
        self.tracker.reset()
        self.success = False
        return self.get_obs(render=render)

    def step(
        self,
        action: np.ndarray,
        on_substep: Callable[[AlohaCanEnv], None] | None = None,
        *,
        render: bool = True,
        next_action: np.ndarray | None = None,
    ) -> tuple[dict, bool, dict]:
        if self._uses_substep_controller():
            # Keyword is omitted when there is no preview so a controller whose
            # set_target still takes (model, data, action) keeps that signature.
            if next_action is None:
                self.controller.set_target(self.model, self.data, action)
            else:
                self.controller.set_target(self.model, self.data, action, next_action)
        else:
            self.controller.apply(self.model, self.data, action)
        info = self._ik_info()
        for _ in range(self.n_substeps):
            if self._uses_substep_controller():
                # compute() refreshes qfrc_bias and M onto the current state,
                # then writes ctrl. mj_step applies that ctrl. read_applied
                # copies qfrc_actuator before _sync_derived replaces it.
                self.controller.compute(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if self._uses_substep_controller() and hasattr(self.controller, "read_applied"):
                self.controller.read_applied(self.model, self.data)
            # Success is evaluated once per physics step, on the post-step
            # state. mj_step has already integrated qpos and qvel, but xpos,
            # xmat, site_xpos and contacts still describe the pre-step
            # configuration. Refresh those before the hold check.
            self._sync_derived()
            released = self.fingers_released()
            self.success = self.tracker.update(self.can_center_base(), self.can_linvel(), released)
            if on_substep is not None:
                on_substep(self)
        return self.get_obs(render=render), self.success, info

    def _uses_substep_controller(self) -> bool:
        ctrl = self.controller
        return callable(getattr(ctrl, "set_target", None)) and callable(getattr(ctrl, "compute", None))

    def get_obs(self, render: bool = True) -> dict:
        self._sync_derived()
        tcp_pos, tcp_quat = self._ik.site_pose_in_base(self.data)
        if render:
            agent = self.render("agentview")
            wrist = self.render("wrist_cam_left")
        else:
            agent = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            wrist = np.zeros_like(agent)
        return {
            "agentview": agent,
            "wrist_cam_left": wrist,
            "tcp_pose": np.concatenate((tcp_pos, tcp_quat)),
            "gripper_opening": np.array([self.data.qpos[self.grip_qpos_adr]], dtype=float),
            "q": self.data.qpos[self.arm_qpos_adr].copy(),
            "qvel": self.data.qvel[self.arm_dof_adr].copy(),
        }

    def get_privileged(self) -> dict:
        pos, quat = self.can_pose_base()
        return {
            "can_pose": np.concatenate((pos, quat)),
            "can_linvel": self.can_linvel(),
        }

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self._sync_derived()
        return self._ik.site_pose_in_base(self.data)

    def render(self, camera: str, height: int | None = None, width: int | None = None) -> np.ndarray:
        height = self.image_height if height is None else height
        width = self.image_width if width is None else width
        renderer = self._get_renderer(height, width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def can_center_base(self) -> np.ndarray:
        self._sync_derived()
        return self._world_to_base(self.data.geom_xpos[self.can_geom])

    def can_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Collision-cylinder position and quaternion in the base frame."""
        self._sync_derived()
        pos = self._world_to_base(self.data.geom_xpos[self.can_geom])
        quat = self._xmat_quat_base(self.data.geom_xmat[self.can_geom])
        return pos, quat

    def can_linvel(self) -> np.ndarray:
        return self.data.qvel[self.can_dof_adr : self.can_dof_adr + 3].copy()

    def fingers_released(self) -> bool:
        """True when no finger geom is in contact with the can collision geom."""
        return not bool(np.any(self.finger_can_contacts()))

    def finger_can_contacts(self) -> np.ndarray:
        """``(left, right)`` contact with ``can_collision``. Contacts must be current."""
        left = False
        right = False
        can_geom = int(self.can_geom)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if can_geom not in pair:
                continue
            other = pair - {can_geom}
            if not other:
                continue
            geom = other.pop()
            if geom in self.left_finger_geoms:
                left = True
            if geom in self.right_finger_geoms:
                right = True
        return np.array([left, right], dtype=bool)

    def _apply_target_cell(self, mode: str) -> None:
        cell = self.cfg["target_cells"][mode]
        self.tracker.center = np.asarray(cell["center"], dtype=float)
        self.tracker.half_extents = np.asarray(cell["half_extents"], dtype=float)

    def _ik_info(self) -> dict:
        result = getattr(self.controller, "last_result", None)
        preview = getattr(self.controller, "last_next_result", None)
        if result is None:
            current = True
            pos = 0.0
            rot = 0.0
        else:
            current = bool(result.success)
            pos = float(result.pos_err)
            rot = float(result.rot_err)
        preview_present = preview is not None
        preview_ok = True if preview is None else bool(preview.success)
        return {
            "ik_success": current and preview_ok,
            "ik_pos_err": pos,
            "ik_rot_err": rot,
            "ik_current_success": current,
            "ik_preview_success": preview_ok,
            "ik_preview_present": preview_present,
        }

    def _sync_derived(self) -> None:
        """Recompute kinematics and contacts from the current qpos and qvel.

        ``mj_step`` integrates the state and leaves xpos, xmat, site_xpos and
        the contact list at the pre-step configuration. Call this before
        reading any of those fields. It does not integrate, so the timestep
        and the number of physics steps per action stay unchanged.
        """
        mujoco.mj_forward(self.model, self.data)

    def _sample_can(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode == "random":
            box = self.cfg["random_can"]
            xy = self._rng.uniform(np.asarray(box["xy_low"], dtype=float), np.asarray(box["xy_high"], dtype=float))
            yaw = float(self._rng.uniform(0.0, 2.0 * np.pi))
            pos = np.array([xy[0], xy[1], float(self.cfg["can_rest_z"])])
            quat = _yaw_quat(yaw)
            return pos, quat
        spec = self.cfg["modes"][mode]
        return np.asarray(spec["can_pos"], dtype=float), np.asarray(spec["can_quat"], dtype=float)

    def _world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        base_pos = self.data.xpos[self.base_id]
        base_mat = self.data.xmat[self.base_id].reshape(3, 3)
        return base_mat.T @ (np.asarray(pos_world) - base_pos)

    def _base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        base_pos = self.data.xpos[self.base_id]
        base_mat = self.data.xmat[self.base_id].reshape(3, 3)
        return base_pos + base_mat @ np.asarray(pos_base)

    def _xmat_quat_base(self, xmat: np.ndarray) -> np.ndarray:
        base_quat = np.zeros(4)
        mujoco.mju_mat2Quat(base_quat, self.data.xmat[self.base_id])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=float).reshape(-1))
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, base_quat)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, inv, quat)
        return out

    def _get_renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (int(height), int(width))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=key[0], width=key[1])
            self._renderers[key] = renderer
        return renderer


def _yaw_quat(yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
    return quat


def _base_quat_to_world(data: mujoco.MjData, base_id: int, quat_base: np.ndarray) -> np.ndarray:
    base_quat = np.zeros(4)
    mujoco.mju_mat2Quat(base_quat, data.xmat[base_id])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, base_quat, np.asarray(quat_base, dtype=float))
    return out
