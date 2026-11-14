"""Tick-by-tick pick-and-place planner.

The planner is the privileged baseline: it reads the can pose from
``get_privileged()`` after reset and aims at the mode's target cell. It does
not add demonstration noise.

Each ``action_pair`` returns the current 20 Hz action and the following one.
The runner passes that pair to ``env.step(..., next_action=...)``. The last
action repeats itself as the preview so the final knot is actually played.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.control.quintic import simulate_stream
from aloha_pp.control.reference import allocate_ticks, segment_distances, solve_cartesian
from aloha_pp.kin.ik import DlsIk, top_down_quat
from aloha_pp.planning.schedule import (
    CARRY,
    CLOSE,
    N_MOTION,
    OPEN,
    Tick,
    build_schedule,
    grip_wait_ticks,
    seconds_to_ticks,
)

# Left bin, base frame. The base sits at world (-0.041018, -0.00014, 0) with
# no rotation, and the left-bin body is at world (0.258982, 0.12486). Inner
# faces are one wall thickness (3.6 mm) inside the 0.072 x 0.09 floor.
LEFT_BIN_CENTER = np.array([0.300, 0.125])
LEFT_BIN_INNER_HALF = np.array([0.0684, 0.0864])

_FAST_TICKS: dict[tuple, np.ndarray] = {}


def choose_yaw(can_xy: np.ndarray) -> float:
    """Point the finger opening along the roomier horizontal axis of the left bin.

    The can is a cylinder, so the grasp yaw does not change the contact on the
    can. It does change which way the open fingers stick out. Site +Y is the
    opening. ``yaw = 0`` puts that axis on base +Y, ``yaw = π/2`` puts it on
    base -X. The wider axis faces the wall that still has more clearance, and
    the same yaw is kept through the place so the low-gain wrist does not have
    to twist while it is holding the can.
    """
    margin = LEFT_BIN_INNER_HALF - np.abs(np.asarray(can_xy, dtype=float).reshape(2) - LEFT_BIN_CENTER)
    if float(margin[1]) >= float(margin[0]):
        return 0.0
    return 0.5 * np.pi


def place_tcp(cell_center: np.ndarray, offset: np.ndarray, gap: float, min_above: float) -> np.ndarray:
    """TCP pose translation that puts the can ``gap`` above the cell center.

    ``offset`` is ``can_center - tcp`` measured after the fingers close. The
    TCP is not driven closer than ``min_above`` over the cell's vertical center.
    """
    cell = np.asarray(cell_center, dtype=float).reshape(3)
    delta = np.asarray(offset, dtype=float).reshape(3)
    tcp = cell - delta
    tcp[2] = cell[2] + float(gap) - float(delta[2])
    tcp[2] = max(float(tcp[2]), cell[2] + float(min_above))
    return tcp


def make_waypoints(
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    can_pos: np.ndarray,
    place_tcp_pos: np.ndarray,
    yaw: float,
    clearance: float,
    lift: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    can = np.asarray(can_pos, dtype=float).reshape(3)
    place = np.asarray(place_tcp_pos, dtype=float).reshape(3)
    quat = top_down_quat(float(yaw), 0)
    above = np.array([can[0], can[1], can[2] + clearance])
    grasp = can.copy()
    lifted = np.array([can[0], can[1], can[2] + lift])
    above_place = np.array([place[0], place[1], place[2] + lift])
    retreat = above_place.copy()
    return [
        (np.asarray(home_pos, dtype=float).copy(), np.asarray(home_quat, dtype=float).copy()),
        (above, quat.copy()),
        (grasp, quat.copy()),
        (lifted, quat.copy()),
        (above_place, quat.copy()),
        (place.copy(), quat.copy()),
        (retreat, quat.copy()),
        (np.asarray(home_pos, dtype=float).copy(), np.asarray(home_quat, dtype=float).copy()),
    ]


def _nominal_place(cell_center: np.ndarray, gap: float) -> np.ndarray:
    place = np.asarray(cell_center, dtype=float).copy()
    place[2] = place[2] + float(gap)
    return place


class PickPlacePlanner:
    def __init__(self, m1: dict, m4: dict, m3: dict, speed: str) -> None:
        if speed not in ("fast", "medium", "slow"):
            raise ValueError(f"unknown speed {speed}")
        self.m1 = m1
        self.m4 = m4
        self.m3 = m3
        self.speed = speed
        self.scale = {"fast": 1, "medium": 2, "slow": 4}[speed]
        self.control_dt = 1.0 / float(m1["control_hz"])
        self.n_substeps = int(m1["physics_hz"]) // int(m1["control_hz"])
        self.gripper_open = float(m1["gripper_open"])
        self.gripper_close = float(m1["gripper_close"])
        self.schedule: list[Tick] = []
        self.k = 0
        self.measured = False
        self.offset = np.zeros(3)
        self.yaw = 0.0
        self.can_init = np.zeros(7)
        self.cell_center = np.zeros(3)
        self.cell_half = np.zeros(3)
        self.mode = "fixed_1"
        self.place_pos = np.zeros(3)
        self.place_tcp = np.zeros(3)
        self.grasp_pos = np.zeros(3)
        self.motion_ticks = np.ones(N_MOTION, dtype=int)
        self.tcp_err_close = float("nan")
        self.tcp_err_open = float("nan")
        self._saw_close = False
        self._saw_open = False
        self._waypoints: list[tuple[np.ndarray, np.ndarray]] = []
        self._wait: dict[str, int] = {}
        self._home_q = np.zeros(6)
        self._home_pos = np.zeros(3)
        self._home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._ik: DlsIk | None = None
        self._v_max = np.ones(6)

    def reset(self, env, *, mode: str, log=print) -> None:
        controller = env.controller
        self._home_q = np.asarray(controller.home_q, dtype=float).copy()
        self._home_pos = np.asarray(controller.home_pos, dtype=float).copy()
        self._home_quat = np.asarray(controller.home_quat, dtype=float).copy()
        self._v_max = np.asarray(controller.stream.v_max, dtype=float).copy()
        privileged = env.get_privileged()["can_pose"]
        self.can_init = np.asarray(privileged, dtype=float).copy()
        can_pos = self.can_init[:3].copy()
        self.yaw = choose_yaw(can_pos[:2])
        self.mode = mode
        self.cell_center = np.asarray(env.tracker.center, dtype=float).copy()
        self.cell_half = np.asarray(env.tracker.half_extents, dtype=float).copy()
        if mode in self.m1["modes"] and "place_pos" in self.m1["modes"][mode]:
            self.place_pos = np.asarray(self.m1["modes"][mode]["place_pos"], dtype=float).copy()
        else:
            self.place_pos = self.cell_center.copy()
        self.offset = np.zeros(3)
        self.measured = False
        self._saw_close = False
        self._saw_open = False
        self.tcp_err_close = float("nan")
        self.tcp_err_open = float("nan")
        self._ik = DlsIk(
            env.model,
            env.cfg["tcp_site"],
            damping=float(self.m3["ik_damping"]),
            max_iters=int(self.m3["ik_max_iters"]),
            pos_tol=float(self.m3["ik_pos_tol"]),
            rot_tol=float(self.m3["ik_rot_tol"]),
            rot_weight=float(self.m3["ik_rot_weight"]),
        )
        self._wait = self._wait_counts()
        nominal = _nominal_place(self.cell_center, float(self.m4["place_gap"]))
        self._waypoints = make_waypoints(
            self._home_pos,
            self._home_quat,
            can_pos,
            nominal,
            self.yaw,
            float(self.m4["pregrasp_clearance"]),
            float(self.m4["lift_height"]),
        )
        self.grasp_pos = self._waypoints[2][0].copy()
        self.place_tcp = self._waypoints[5][0].copy()
        fast = self._fast_motion_ticks(log)
        self.motion_ticks = fast * self.scale
        self.schedule = self._assemble(self._waypoints, self.motion_ticks)
        self.k = 0

    def action_pair(self) -> tuple[np.ndarray, np.ndarray, Tick]:
        index = min(self.k, len(self.schedule) - 1)
        current = self.schedule[index]
        nxt = self.schedule[min(index + 1, len(self.schedule) - 1)]
        return current.action(), nxt.action(), current

    def dense_reference(self) -> tuple[np.ndarray, np.ndarray]:
        index = min(self.k, len(self.schedule) - 1)
        return self.schedule[index].dense(self.n_substeps)

    def after_step(self, env) -> None:
        index = min(self.k, len(self.schedule) - 1)
        tick = self.schedule[index]
        if not self.measured and tick.state == CLOSE:
            following = index + 1
            last_close = following >= len(self.schedule) or self.schedule[following].state != CLOSE
            if last_close:
                self._measure(env)
                self._rebuild_carry()
        if self.k + 1 < len(self.schedule):
            self.k += 1
        else:
            self.k = len(self.schedule)

    def note_tcp(self, tcp_pos: np.ndarray) -> None:
        """Record the TCP error at the first physics step of the close and open ramps."""
        if self.k >= len(self.schedule):
            return
        state = self.schedule[self.k].state
        pos = np.asarray(tcp_pos, dtype=float)
        if state == CLOSE and not self._saw_close:
            self.tcp_err_close = float(np.linalg.norm(pos - self.grasp_pos))
            self._saw_close = True
        if state == OPEN and not self._saw_open:
            self.tcp_err_open = float(np.linalg.norm(pos - self._waypoints[5][0]))
            self._saw_open = True

    @property
    def done(self) -> bool:
        return self.k >= len(self.schedule)

    @property
    def planned_motion_time(self) -> float:
        return float(np.sum(self.motion_ticks) * self.control_dt)

    @property
    def planned_total_time(self) -> float:
        return float(len(self.schedule) * self.control_dt)

    def _wait_counts(self) -> dict[str, int]:
        lag = 1  # preview mode reaches the knot on the next control tick
        dt = self.control_dt
        return {
            "pregrasp_hold_ticks": seconds_to_ticks(float(self.m4["pregrasp_hold_s"]), dt),
            "wait_close_ticks": grip_wait_ticks(lag, float(self.m4["settle_s"]), dt),
            "grip_ramp_ticks": int(self.m4["grip_ramp_ticks"]),
            "dwell_closed_ticks": seconds_to_ticks(float(self.m4["post_close_dwell_s"]), dt),
            "wait_open_ticks": grip_wait_ticks(lag, float(self.m4["settle_s"]), dt),
            "dwell_open_ticks": seconds_to_ticks(float(self.m4["post_open_dwell_s"]), dt),
            "success_wait_ticks": seconds_to_ticks(float(self.m4["success_wait_s"]), dt),
        }

    def _assemble(self, waypoints, motion_ticks: np.ndarray) -> list[Tick]:
        return build_schedule(
            waypoints,
            motion_ticks,
            gripper_open=self.gripper_open,
            gripper_close=self.gripper_close,
            **self._wait,
        )

    def _measure(self, env) -> None:
        can = env.can_center_base()
        tcp, _quat = env.tcp_pose()
        offset = np.asarray(can, dtype=float) - np.asarray(tcp, dtype=float)
        if float(np.linalg.norm(offset)) > 0.05:
            offset = np.zeros(3)
        self.offset = offset
        self.measured = True

    def _rebuild_carry(self) -> None:
        tcp_place = place_tcp(
            self.cell_center,
            self.offset,
            float(self.m4["place_gap"]),
            float(self.m4["place_min_above"]),
        )
        can_pos = self.can_init[:3]
        self._waypoints = make_waypoints(
            self._home_pos,
            self._home_quat,
            can_pos,
            tcp_place,
            self.yaw,
            float(self.m4["pregrasp_clearance"]),
            float(self.m4["lift_height"]),
        )
        rebuilt = self._assemble(self._waypoints, self.motion_ticks)
        start = next(i for i, tick in enumerate(self.schedule) if tick.state == CARRY)
        # The prefix through the lift does not depend on the measured offset.
        if start != next(i for i, tick in enumerate(rebuilt) if tick.state == CARRY):
            raise RuntimeError("rebuilt schedule moved the carry boundary")
        self.schedule = self.schedule[:start] + rebuilt[start:]
        self.place_tcp = self._waypoints[5][0].copy()

    def _fast_motion_ticks(self, log) -> np.ndarray:
        key = (
            round(float(self.can_init[0]), 5),
            round(float(self.can_init[1]), 5),
            round(float(self.can_init[2]), 5),
            round(float(self.cell_center[0]), 5),
            round(float(self.cell_center[1]), 5),
            round(float(self.yaw), 5),
        )
        cached = _FAST_TICKS.get(key)
        if cached is not None:
            return cached.copy()
        distances = segment_distances(self._waypoints)
        v_max = self._v_max
        # The success hold is stationary and sits after the motion, so it does
        # not change the speed search. Keep one tick of it to limit IK work.
        saved_tail = self._wait["success_wait_ticks"]
        self._wait["success_wait_ticks"] = 1

        def feasible(total: int) -> tuple[bool, np.ndarray]:
            counts = allocate_ticks(distances, total)
            ticks = self._assemble(self._waypoints, counts)
            q_rows = []
            seed = self._home_q.copy()
            prev_pos = None
            prev_quat = None
            ok = True
            for tick in ticks:
                same = (
                    prev_pos is not None
                    and np.allclose(tick.pos, prev_pos)
                    and np.allclose(tick.quat, prev_quat)
                )
                if same:
                    q_rows.append(seed.copy())
                    continue
                result = solve_cartesian(
                    self._ik,
                    tick.pos,
                    tick.quat,
                    seed,
                    self._home_q,
                    self._home_pos,
                    self._home_quat,
                )
                if result.success:
                    seed = np.asarray(result.q, dtype=float).copy()
                else:
                    ok = False
                q_rows.append(seed.copy())
                prev_pos = tick.pos
                prev_quat = tick.quat
            if not ok:
                return False, np.full(6, np.inf)
            speeds, _acc, flags = simulate_stream(
                np.stack(q_rows),
                self._home_q,
                control_dt=self.control_dt,
                n_substeps=self.n_substeps,
                v_max=v_max,
                scheme="preview",
                a_max=np.asarray(self.m3["accel_limit"], dtype=float),
            )
            peak = np.max(np.abs(speeds), axis=0)
            good = (not np.any(flags)) and bool(np.all(peak <= v_max * (1.0 + 1e-5)))
            return good, peak

        try:
            total = int(distances.shape[0])
            ok, peak = feasible(total)
            while not ok:
                total = int(total * 1.5) + 1
                if total > 4000:
                    raise RuntimeError("no velocity-feasible duration under 4000 control ticks")
                ok, peak = feasible(total)
                log(f"  duration search {total} ticks feasible={ok} peak/limit={np.max(peak / v_max):.3f}")
            lo = int(distances.shape[0])
            hi = total
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                ok, peak = feasible(mid)
                log(f"  duration search {mid} ticks feasible={ok} peak/limit={np.max(peak / v_max):.3f}")
                if ok:
                    hi = mid
                else:
                    lo = mid
            _ok, peak = feasible(hi)
        finally:
            self._wait["success_wait_ticks"] = saved_tail
        counts = allocate_ticks(distances, hi)
        _FAST_TICKS[key] = counts.copy()
        log(
            f"  fast ticks {counts.tolist()} motion {hi * self.control_dt:.2f}s "
            f"peak/limit {np.max(peak / v_max):.3f}"
        )
        return counts
