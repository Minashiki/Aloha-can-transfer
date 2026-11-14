"""Cartesian pick-place references and the 20 Hz command stream.

Moving segments use a quintic time law, so Cartesian velocity is zero at every
waypoint and the samples in between are smooth. Those samples are the 20 Hz
commands. Joint-space timing is whatever :class:`JointStream` does with them.
``lookahead`` reaches the command from time ``k T`` at ``(k+2) T`` (100 ms).
``causal`` reaches it at ``(k+1) T`` (50 ms) unless its acceleration bound
is active, in which case the rest of the move spills into later ticks.

Fast is the shortest total duration, on the 20 Hz grid, whose joint stream
stays inside the velocity limits without the online clamp. Medium and slow
repeat every fast segment for 2× and 4× as many ticks, so their durations are
exact multiples and the path is the same polyline in time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from aloha_pp.control.quintic import SCHEME_LAG_TICKS, simulate_stream
from aloha_pp.kin.ik import DlsIk, IkResult, top_down_quat

PHASE_MOTION = 0
PHASE_CATCHUP = 1
PHASE_HOLD = 2
PHASE_PRESS = 3
PHASE_LIFT = 4


@dataclass
class Trajectory:
    name: str
    actions: np.ndarray
    tcp_pos: np.ndarray
    tcp_quat: np.ndarray
    phase: np.ndarray
    t: np.ndarray
    motion_time: float
    peak_qd: np.ndarray
    q_ik: np.ndarray


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    denom = np.sin(theta)
    w0 = np.sin((1.0 - s) * theta) / denom
    w1 = np.sin(s * theta) / denom
    return w0 * q0 + w1 * q1


def quintic_blend(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return (10.0 * u**3) - (15.0 * u**4) + (6.0 * u**5)


def _top(xy: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([float(xy[0]), float(xy[1]), float(z)])
    yaw = float(np.arctan2(pos[1], pos[0]))
    return pos, top_down_quat(yaw)


def pick_place_waypoints(
    can_pos: np.ndarray,
    place_pos: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    *,
    clearance: float,
    lift: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """home → above can → grasp height → lift → above cell → place height → lift → home."""
    can = np.asarray(can_pos, dtype=float)
    place = np.asarray(place_pos, dtype=float)
    above_can, above_can_q = _top(can[:2], can[2] + clearance)
    grasp, grasp_q = _top(can[:2], can[2])
    lifted, lifted_q = _top(can[:2], can[2] + lift)
    above_place, above_place_q = _top(place[:2], place[2] + lift)
    drop, drop_q = _top(place[:2], place[2])
    depart, depart_q = _top(place[:2], place[2] + lift)
    return [
        (np.asarray(home_pos, dtype=float).copy(), np.asarray(home_quat, dtype=float).copy()),
        (above_can, above_can_q),
        (grasp, grasp_q),
        (lifted, lifted_q),
        (above_place, above_place_q),
        (drop, drop_q),
        (depart, depart_q),
        (np.asarray(home_pos, dtype=float).copy(), np.asarray(home_quat, dtype=float).copy()),
    ]


def segment_distances(poses: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Cartesian length plus 0.15 m per radian of orientation change."""
    lengths = []
    for (p0, q0), (p1, q1) in zip(poses[:-1], poses[1:]):
        dot = min(1.0, abs(float(np.dot(q0, q1))))
        angle = 2.0 * float(np.arccos(dot))
        lengths.append(float(np.linalg.norm(p1 - p0)) + 0.15 * angle)
    return np.asarray(lengths, dtype=float)


def allocate_ticks(distances: np.ndarray, total_ticks: int) -> np.ndarray:
    """Give every segment at least one control tick, the rest proportional to length."""
    distances = np.asarray(distances, dtype=float)
    n = int(distances.shape[0])
    if total_ticks < n:
        raise ValueError(f"need at least {n} ticks, got {total_ticks}")
    ticks = np.ones(n, dtype=int)
    extra = int(total_ticks) - n
    if extra == 0:
        return ticks
    raw = distances / float(distances.sum()) * extra
    add = np.floor(raw).astype(int)
    ticks = ticks + add
    left = extra - int(add.sum())
    order = np.argsort(-(raw - add))
    for index in order[:left]:
        ticks[index] += 1
    if int(ticks.sum()) != int(total_ticks):
        raise RuntimeError("tick allocation does not sum to the requested total")
    return ticks


def pose_at(poses: list[tuple[np.ndarray, np.ndarray]], ticks: np.ndarray, control_dt: float, t: float):
    bounds = np.cumsum(ticks) * control_dt
    starts = bounds - ticks * control_dt
    if t <= 0.0:
        return poses[0]
    if t >= float(bounds[-1]):
        return poses[-1]
    index = int(np.searchsorted(bounds, t, side="right"))
    index = min(index, len(ticks) - 1)
    duration = float(bounds[index] - starts[index])
    u = 0.0 if duration <= 0.0 else (t - float(starts[index])) / duration
    blend = quintic_blend(u)
    p0, q0 = poses[index]
    p1, q1 = poses[index + 1]
    return p0 + blend * (p1 - p0), slerp(q0, q1, blend)


def solve_cartesian(
    ik: DlsIk,
    pos: np.ndarray,
    quat: np.ndarray,
    seed: np.ndarray,
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
) -> IkResult:
    """One 20 Hz solve. The exact home pose returns ``home_q`` so the ends do not drift."""
    quat = np.asarray(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    pos = np.asarray(pos, dtype=float)
    if float(np.linalg.norm(pos - home_pos)) <= 1e-10 and abs(abs(float(np.dot(quat, home_quat))) - 1.0) <= 1e-10:
        return IkResult(True, np.asarray(home_q, dtype=float).copy(), 0.0, 0.0, 0)
    return ik.solve(pos, quat, seed)


def _command_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
    ticks: np.ndarray,
    control_dt: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    total = float(np.sum(ticks) * control_dt)
    n_motion = int(np.sum(ticks))
    commands = [pose_at(poses, ticks, control_dt, k * control_dt) for k in range(n_motion)]
    commands.append(pose_at(poses, ticks, control_dt, total))
    return commands, total


def _ik_commands(
    ik: DlsIk,
    commands: list[tuple[np.ndarray, np.ndarray]],
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = np.asarray(home_q, dtype=float).copy()
    q_rows = []
    pos_err = []
    rot_err = []
    ok = []
    for pos, quat in commands:
        result = solve_cartesian(ik, pos, quat, seed, home_q, home_pos, home_quat)
        if result.success:
            seed = np.asarray(result.q, dtype=float).copy()
        q_rows.append(seed.copy())
        pos_err.append(result.pos_err)
        rot_err.append(result.rot_err)
        ok.append(bool(result.success))
    return (
        np.stack(q_rows),
        np.asarray(pos_err),
        np.asarray(rot_err),
        np.asarray(ok, dtype=bool),
    )


def stream_peak(
    q_targets: np.ndarray,
    q0: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    speeds, _acc, flags = simulate_stream(
        q_targets,
        q0,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme=scheme,
        a_max=a_max,
    )
    peak = np.max(np.abs(speeds), axis=0)
    return peak, bool(np.any(flags))


def search_fast_ticks(
    ik: DlsIk,
    poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
    log=print,
) -> np.ndarray:
    """Shortest tick budget whose 20 Hz IK stream does not trip the speed clamp."""
    distances = segment_distances(poses)
    v_max = np.asarray(v_max, dtype=float)

    def feasible(total: int) -> tuple[bool, np.ndarray]:
        ticks = allocate_ticks(distances, total)
        commands, _total_time = _command_poses(poses, ticks, control_dt)
        q_ik, _pos, _rot, ok = _ik_commands(ik, commands, home_q, home_pos, home_quat)
        if not np.all(ok):
            return False, np.full(6, np.inf)
        # The stream starts at home and consumes every command, including the
        # catch-up sample at the final waypoint. Drop that target from the
        # "targets after the start" list: simulate_stream pushes each row, and
        # the first command is already the home the stream was reset to.
        peak, flagged = stream_peak(
            q_ik,
            home_q,
            control_dt=control_dt,
            n_substeps=n_substeps,
            v_max=v_max,
            scheme=scheme,
            a_max=a_max,
        )
        return (not flagged) and bool(np.all(peak <= v_max * (1.0 + 1e-5))), peak

    total = int(distances.shape[0])
    ok, peak = feasible(total)
    while not ok:
        total = int(total * 1.5) + 1
        if total > 4000:
            raise RuntimeError("no velocity-feasible duration under 4000 control ticks")
        ok, peak = feasible(total)
        log(f"  duration search {total} ticks feasible={ok} peak/limit={np.max(peak / v_max):.3f}")
    hi = total
    lo = int(distances.shape[0])
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, peak = feasible(mid)
        log(f"  duration search {mid} ticks feasible={ok} peak/limit={np.max(peak / v_max):.3f}")
        if ok:
            hi = mid
        else:
            lo = mid
    ticks = allocate_ticks(distances, hi)
    _ok, peak = feasible(hi)
    log(
        f"  fast ticks {ticks.tolist()} total {hi} "
        f"duration {hi * control_dt:.2f}s peak/limit {np.max(peak / v_max):.3f}"
    )
    return ticks


def build_trajectory(
    name: str,
    ik: DlsIk,
    poses: list[tuple[np.ndarray, np.ndarray]],
    ticks: np.ndarray,
    *,
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    grip: float,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    hold_s: float,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
) -> Trajectory:
    commands, motion_time = _command_poses(poses, ticks, control_dt)
    q_ik, _pos, _rot, ok = _ik_commands(ik, commands, home_q, home_pos, home_quat)
    if not np.all(ok):
        missed = int(np.size(ok) - np.count_nonzero(ok))
        raise RuntimeError(f"{name}: IK failed on {missed} command(s)")
    # _command_poses already repeats the final pose for one tick. Lookahead
    # needs a second one so the 0.5 s hold starts after the reference arrives.
    extra_catch = SCHEME_LAG_TICKS[scheme] - 1
    for _ in range(extra_catch):
        commands.append(commands[-1])
        q_ik = np.concatenate((q_ik, q_ik[-1:]), axis=0)
    hold_n = int(round(hold_s / control_dt))
    physics_dt = control_dt / n_substeps
    n_motion = int(np.sum(ticks))
    # commands is n_motion samples plus the catch-up at t = motion_time.
    actions = []
    ref_pos = []
    ref_quat = []
    phases = []
    times = []

    def append_command(pos, quat, phase: int, k: int) -> None:
        actions.append(np.concatenate((pos, quat, [grip])))
        for i in range(n_substeps):
            t_end = (k * n_substeps + i + 1) * physics_dt
            sample_t = k * control_dt + (i + 1) * physics_dt
            if phase == PHASE_MOTION:
                p, q = pose_at(poses, ticks, control_dt, sample_t)
            else:
                p, q = poses[-1]
            ref_pos.append(np.asarray(p, dtype=float))
            ref_quat.append(np.asarray(q, dtype=float))
            phases.append(phase)
            times.append(t_end)

    for k in range(n_motion):
        pos, quat = commands[k]
        append_command(pos, quat, PHASE_MOTION, k)
    for c in range(SCHEME_LAG_TICKS[scheme]):
        catch_pos, catch_quat = commands[n_motion + c]
        append_command(catch_pos, catch_quat, PHASE_CATCHUP, n_motion + c)
    final_pos, final_quat = poses[-1]
    for h in range(hold_n):
        append_command(final_pos, final_quat, PHASE_HOLD, n_motion + SCHEME_LAG_TICKS[scheme] + h)

    hold_q = np.repeat(np.asarray(home_q, dtype=float).reshape(1, -1), max(hold_n, 1), axis=0)
    peak, flagged = stream_peak(
        np.concatenate((q_ik, hold_q), axis=0),
        home_q,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme=scheme,
        a_max=a_max,
    )
    if flagged:
        raise RuntimeError(f"{name}: joint stream exceeds the velocity limit")
    return Trajectory(
        name=name,
        actions=np.stack(actions),
        tcp_pos=np.stack(ref_pos),
        tcp_quat=np.stack(ref_quat),
        phase=np.asarray(phases, dtype=np.int8),
        t=np.asarray(times, dtype=float),
        motion_time=motion_time,
        peak_qd=peak,
        q_ik=q_ik,
    )


def build_speed_family(
    ik: DlsIk,
    poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    grip: float,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    hold_s: float,
    label: str,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
    log=print,
) -> dict[str, Trajectory]:
    log(f"search {label}")
    fast_ticks = search_fast_ticks(
        ik,
        poses,
        home_q=home_q,
        home_pos=home_pos,
        home_quat=home_quat,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme=scheme,
        a_max=a_max,
        log=log,
    )
    families = {
        "fast": fast_ticks,
        "medium": fast_ticks * 2,
        "slow": fast_ticks * 4,
    }
    out = {}
    for speed, ticks in families.items():
        traj = build_trajectory(
            f"{label}_{speed}",
            ik,
            poses,
            ticks,
            home_q=home_q,
            home_pos=home_pos,
            home_quat=home_quat,
            grip=grip,
            control_dt=control_dt,
            n_substeps=n_substeps,
            v_max=v_max,
            hold_s=hold_s,
            scheme=scheme,
            a_max=a_max,
        )
        ratio = np.max(traj.peak_qd / np.asarray(v_max, dtype=float))
        log(
            f"  {label} {speed}: motion {traj.motion_time:.2f}s "
            f"peak qd {np.array2string(traj.peak_qd, precision=3)} "
            f"peak/limit {ratio:.3f}"
        )
        out[speed] = traj
    return out


def contact_trajectory(
    ik: DlsIk,
    *,
    home_q: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    grip: float,
    xy: np.ndarray,
    above_z: float,
    press_z: float,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    descend_s: float = 1.0,
    press_s: float = 2.0,
    lift_s: float = 1.0,
    settle_s: float = 0.5,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
) -> Trajectory:
    """Move above a bin-floor point, command 1 cm under the floor, hold, then lift.

    The press block repeats the same Cartesian target. Extra control ticks
    cover the reference lag so the arm is actually on that target for ``press_s``.
    """
    above_pos, above_quat = _top(xy, above_z)
    press_pos, press_quat = _top(xy, press_z)
    segments = [
        (home_pos, home_quat, above_pos, above_quat, _ticks(0.8, control_dt), PHASE_MOTION, True),
        (above_pos, above_quat, press_pos, press_quat, _ticks(descend_s, control_dt), PHASE_MOTION, True),
        (
            press_pos,
            press_quat,
            press_pos,
            press_quat,
            _ticks(press_s, control_dt) + SCHEME_LAG_TICKS[scheme],
            PHASE_PRESS,
            False,
        ),
        (press_pos, press_quat, above_pos, above_quat, _ticks(lift_s, control_dt), PHASE_LIFT, True),
        (above_pos, above_quat, above_pos, above_quat, _ticks(settle_s, control_dt), PHASE_HOLD, False),
    ]
    physics_dt = control_dt / n_substeps
    actions = []
    ref_pos = []
    ref_quat = []
    phases = []
    times = []
    q_rows = []
    seed = np.asarray(home_q, dtype=float).copy()
    step = 0
    for p0, q0, p1, q1, ticks, phase, moving in segments:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        for k in range(ticks):
            blend = quintic_blend(k / ticks) if moving else 0.0
            pos = p0 + blend * (p1 - p0)
            quat = slerp(q0, q1, blend) if moving else q0
            result = solve_cartesian(ik, pos, quat, seed, home_q, home_pos, home_quat)
            if not result.success:
                raise RuntimeError(
                    f"contact IK failed at phase {phase}: pos err {result.pos_err:.3e} "
                    f"rot err {result.rot_err:.3e}"
                )
            seed = np.asarray(result.q, dtype=float).copy()
            q_rows.append(seed.copy())
            actions.append(np.concatenate((pos, quat, [grip])))
            for i in range(n_substeps):
                u = (k + (i + 1) / n_substeps) / ticks
                blend_u = quintic_blend(u) if moving else 0.0
                sample = p0 + blend_u * (p1 - p0)
                sample_q = slerp(q0, q1, blend_u) if moving else q0
                ref_pos.append(sample)
                ref_quat.append(sample_q)
                phases.append(phase)
                times.append((step + 1) * physics_dt)
                step += 1
    q_ik = np.stack(q_rows)
    peak, flagged = stream_peak(
        q_ik,
        home_q,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme=scheme,
        a_max=a_max,
    )
    if flagged:
        raise RuntimeError("contact trajectory exceeds the joint velocity limit")
    motion_time = float(times[phases.index(PHASE_PRESS)] - physics_dt) if PHASE_PRESS in phases else float(times[-1])
    return Trajectory(
        name="contact",
        actions=np.stack(actions),
        tcp_pos=np.stack(ref_pos),
        tcp_quat=np.stack(ref_quat),
        phase=np.asarray(phases, dtype=np.int8),
        t=np.asarray(times, dtype=float),
        motion_time=motion_time,
        peak_qd=peak,
        q_ik=q_ik,
    )


def _ticks(seconds: float, control_dt: float) -> int:
    return max(1, int(round(seconds / control_dt)))


def benchmark_ik(
    model: mujoco.MjModel,
    samples: list[tuple[np.ndarray, np.ndarray]],
    home_q: np.ndarray,
    candidates: list[dict],
    *,
    site_name: str,
    damping: float,
    max_iters: int,
    rot_weight: float,
) -> list[dict]:
    """Time and convergence of each tolerance on the same warm-started samples."""
    rows = []
    for candidate in candidates:
        ik = DlsIk(
            model,
            site_name,
            damping=damping,
            max_iters=max_iters,
            pos_tol=float(candidate["pos_tol"]),
            rot_tol=float(candidate["rot_tol"]),
            rot_weight=rot_weight,
        )
        seed = np.asarray(home_q, dtype=float).copy()
        times = []
        pos_err = []
        rot_err = []
        ok = []
        iters = []
        for pos, quat in samples:
            start = time.perf_counter()
            result = ik.solve(pos, quat, seed)
            times.append(time.perf_counter() - start)
            if result.success:
                seed = np.asarray(result.q, dtype=float).copy()
            pos_err.append(result.pos_err)
            rot_err.append(result.rot_err)
            ok.append(bool(result.success))
            iters.append(int(result.iterations))
        times_a = np.asarray(times)
        rows.append(
            {
                "pos_tol": float(candidate["pos_tol"]),
                "rot_tol": float(candidate["rot_tol"]),
                "n": int(len(samples)),
                "success_rate": float(np.mean(ok)),
                "median_s": float(np.median(times_a)),
                "max_s": float(np.max(times_a)),
                "median_pos_err": float(np.median(pos_err)),
                "max_pos_err": float(np.max(pos_err)),
                "median_rot_err": float(np.median(rot_err)),
                "max_rot_err": float(np.max(rot_err)),
                "median_iters": float(np.median(iters)),
                "max_iters_used": int(np.max(iters)),
            }
        )
    return rows


def choose_tolerance(rows: list[dict]) -> dict:
    """Prefer 0.1 mm / 1 mrad when it converges and finishes well inside 50 ms.

    The looser 1 mm / 10 mrad budget is the same size as the tracking error we
    are trying to measure, so it would leak into the TCP-versus-reference
    number. It is the fallback when the tight solve is slow or unreliable.
    """
    if len(rows) < 2:
        raise ValueError("need both IK tolerances")
    loose, tight = rows[0], rows[1]
    fast_enough = tight["median_s"] < 0.020 and tight["max_s"] < 0.040
    reliable = tight["success_rate"] >= 0.99 and tight["success_rate"] + 1e-9 >= loose["success_rate"]
    tight["chosen"] = bool(fast_enough and reliable)
    loose["chosen"] = not tight["chosen"]
    return tight if tight["chosen"] else loose


def path_samples(
    poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    total_time: float,
    control_dt: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    distances = segment_distances(poses)
    total_ticks = max(int(distances.shape[0]), int(round(total_time / control_dt)))
    ticks = allocate_ticks(distances, total_ticks)
    n_motion = int(np.sum(ticks))
    return [pose_at(poses, ticks, control_dt, k * control_dt) for k in range(n_motion + 1)]
