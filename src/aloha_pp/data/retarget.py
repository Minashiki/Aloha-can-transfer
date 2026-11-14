"""Object-centric retarget of one Panda Can-PH demonstration.

Tool frames. Robosuite's ``robot0_eef_pos`` / ``robot0_eef_quat`` are the
world pose of the Panda ``grip_site`` (``panda_gripper.xml`` body ``eef``,
site ``grip_site``). That site sits between the fingertips; the site ``+Z``
axis continues out through the fingers, so it is the approach axis. This
project's ``left/tcp`` is the midpoint of the two fingertip pads and its
``+X`` axis is the approach axis. Both are grasp centers. Orientations are
not copied: every retargeted pose is top-down about ``left/tcp`` ``+X``,
with yaw from :func:`aloha_pp.planning.planner.choose_yaw`. The can is a
cylinder, so the human yaw does not change the contact, and M4 already
checked that those two yaws keep the open fingers off the nearer wall.

Segmentation uses the gripper opening (sum of absolute finger positions).
Before the grasp, TCP positions are the source offset from the current can
position, placed on this project's can. After the release, they are the
source offset from the robosuite can-target cell, placed on ``fixed_1``.
The carry removes the straight source translation from the grasp point to
the release point and keeps that residual on top of a smoothstep blend
between our grasp point and our release point, so the 0.53 m bin gap is
not copied. A sine envelope lifts the middle of the carry if it would pass
below ``carry_clearance_z``. Distances of those offsets are not scaled.

The source gripper opens while the can is still well above the bin (median
about 0.24 m over the robosuite cell) and the can then falls. This project's
cell is much smaller, so that drop misses it. After the source release
pose, the path slides across at that height, descends onto the cell center
plus the grasp offset, and only then opens. A short retreat follows so the
fingers leave the can. This is partial retargeting plus a synthetic place:
human relative positions before the grasp, a synthetic baseline plus the
human residual on the carry, and a synthetic approach, slide, descent, and
retreat. Orientation is rewritten top-down. Gripper timing is rewritten.

The approach from the reset pose is a quintic in position and orientation
whose first sample is the home TCP. Each motion segment (approach, human
pre-grasp, carry, slide, descent, retreat) is then slowed by its own
preview peak. The slowdown blends with a quintic across the segment
boundary, so the knot spacing — and therefore the preview ``q_d`` and
``q̇_d``, which the stream already matches at each knot — does not step.
Gripper waits, ramps, and dwells are inserted after that scaling and keep
the M4 tick counts; they are not multiplied by the segment scale.

The whole float32 action sequence is accepted only when its preview
reference stays at or under ``speed_ratio * v_max`` with every overspeed
flag clear. The check starts at the controller's home joint angles, solves
IK with the controller's seed and failure fallback, and replays
``simulate_stream(..., scheme="preview")``. The search repeats until that
holds or a segment is already at ``max_scale``.

Actions are the next retargeted pose, ``[xyz, quat wxyz, gripper]``, which
is the same one-tick lead the M4 planner uses against preview. The last
action repeats the final pose. Gripper open / close commands are
``gripper_open`` / ``gripper_close``. ``grasp_index`` and ``release_index``
are the first close-ramp and open-ramp samples on that final action
sequence. ``source_grasp`` / ``source_release`` are the indices in the
input demonstration, before the prefix, the waits, and the resampling.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.control.quintic import eval_quintic, simulate_stream
from aloha_pp.control.reference import quintic_blend, slerp, solve_cartesian
from aloha_pp.data.frames import can_target_center, map_can_xy
from aloha_pp.kin.ik import DlsIk, top_down_quat
from aloha_pp.planning.planner import choose_yaw
from aloha_pp.planning.schedule import grip_wait_ticks, seconds_to_ticks

# Labels on the final stored action, one per control tick.
PHASE_SYNTH_APPROACH = 0
PHASE_HUMAN_PREGRASP = 1
PHASE_GRIP_WAIT = 2
PHASE_GRIP_RAMP = 3
PHASE_GRIP_DWELL = 4
PHASE_CARRY = 5
PHASE_PLACE_SLIDE = 6
PHASE_PLACE_DOWN = 7
PHASE_RETREAT = 8

PHASE_NAMES = (
    "synth_approach",
    "human_pregrasp",
    "grip_wait",
    "grip_ramp",
    "grip_dwell",
    "carry",
    "place_slide",
    "place_down",
    "retreat",
)

# Ticks that still carry a human position or a human carry residual.
# Repeated gripper holds and the synthetic approach / place / retreat do not.
HUMAN_INFO_PHASES = (PHASE_HUMAN_PREGRASP, PHASE_CARRY)

# Motion segments that receive a slowdown. Gripper phases are inserted later.
MOTION_PHASES = (
    PHASE_SYNTH_APPROACH,
    PHASE_HUMAN_PREGRASP,
    PHASE_CARRY,
    PHASE_PLACE_SLIDE,
    PHASE_PLACE_DOWN,
    PHASE_RETREAT,
)


def gripper_opening(qpos: np.ndarray) -> np.ndarray:
    """Scalar opening from ``robot0_gripper_qpos``. Larger means more open."""
    fingers = np.asarray(qpos, dtype=float)
    if fingers.ndim == 1:
        return np.abs(fingers)
    return np.abs(fingers).sum(axis=-1)


def grasp_release(
    opening: np.ndarray,
    close_below: float,
    min_frames: int = 1,
) -> tuple[int, int]:
    """Start of the first sustained close, and the next open.

    Can-PH resets the gripper on a closed-looking width for a few frames
    before the approach, so a close shorter than ``min_frames`` is ignored.
    The release index is the first open sample after that close, or the last
    sample when the gripper never reopens. Raises if nothing lasts long enough.
    """
    closed = np.asarray(opening, dtype=float) < float(close_below)
    need = max(int(min_frames), 1)
    start = 0
    n = len(closed)
    while start < n:
        if not closed[start]:
            start += 1
            continue
        end = start + 1
        while end < n and closed[end]:
            end += 1
        if end - start >= need:
            release = end if end < n else n - 1
            return int(start), int(release)
        start = end
    raise ValueError("gripper never closes")


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def object_centric_positions(
    eef_pos: np.ndarray,
    can_pos: np.ndarray,
    *,
    grasp: int,
    release: int,
    can_new: np.ndarray,
    cell_src: np.ndarray,
    cell_new: np.ndarray,
    carry_clearance_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Retargeted TCP positions and a per-frame phase id.

    ``can_new`` and ``cell_new`` are in the base frame. Source positions stay
    in the robosuite world frame; with ``R = I`` the offsets transfer as-is.
    """
    eef = np.asarray(eef_pos, dtype=float).reshape(-1, 3)
    can = np.asarray(can_pos, dtype=float).reshape(-1, 3)
    if len(eef) != len(can):
        raise ValueError("eef and can trajectories differ in length")
    if not (0 <= grasp <= release < len(eef)):
        raise ValueError("grasp/release index out of range")
    cell_src = np.asarray(cell_src, dtype=float).reshape(3)
    cell_new = np.asarray(cell_new, dtype=float).reshape(3)
    can_new = np.asarray(can_new, dtype=float).reshape(3)
    out = np.empty_like(eef)
    phase = np.empty(len(eef), dtype=np.int8)
    grasp_off = eef[grasp] - can[grasp]
    our_grasp = can_new + grasp_off
    our_release = cell_new + (eef[release] - cell_src)
    span = max(int(release) - int(grasp), 1)
    for t in range(len(eef)):
        if t <= grasp:
            out[t] = can_new + (eef[t] - can[t])
            phase[t] = PHASE_HUMAN_PREGRASP
        elif t >= release:
            out[t] = cell_new + (eef[t] - cell_src)
            phase[t] = PHASE_PLACE_SLIDE
        else:
            u = (t - grasp) / span
            src_linear = (1.0 - u) * eef[grasp] + u * eef[release]
            residual = eef[t] - src_linear
            blend = _smoothstep(u)
            out[t] = (1.0 - blend) * our_grasp + blend * our_release + residual
            deficit = float(carry_clearance_z) - float(out[t, 2])
            if deficit > 0.0:
                out[t, 2] += np.sin(np.pi * u) * deficit
            phase[t] = PHASE_CARRY
    # The grasp sample is the endpoint of both the approach and the carry.
    out[grasp] = our_grasp
    phase[grasp] = PHASE_HUMAN_PREGRASP
    return out, phase


def _block(pos: np.ndarray, quat: np.ndarray, index: int, n: int, grip_value: float, phase_id: int):
    row_pos = np.repeat(pos[index : index + 1], n, axis=0)
    row_quat = np.repeat(quat[index : index + 1], n, axis=0)
    grip = np.full(n, float(grip_value))
    phases = np.full(n, int(phase_id), dtype=np.int8)
    return row_pos, row_quat, grip, phases


def _ramp_block(pos: np.ndarray, quat: np.ndarray, index: int, n: int, g0: float, g1: float):
    row_pos, row_quat, _, phases = _block(pos, quat, index, n, g0, PHASE_GRIP_RAMP)
    levels = np.linspace(float(g0), float(g1), n + 1)[1:]
    return row_pos, row_quat, levels, phases


def _cat(blocks: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([item[0] for item in blocks]),
        np.concatenate([item[1] for item in blocks]),
        np.concatenate([item[2] for item in blocks]),
        np.concatenate([item[3] for item in blocks]),
    )


def insert_gripper_waits(
    pos: np.ndarray,
    quat: np.ndarray,
    phase: np.ndarray,
    *,
    grasp: int,
    release: int,
    gripper_open: float,
    gripper_close: float,
    wait_ticks: int,
    ramp_ticks: int,
    dwell_close: int,
    dwell_open: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Hold, then ramp, then dwell at the close and at the open.

    Geometric samples are kept. The extra frames repeat the pose at the
    event. The returned indices are the first tick of each ramp.
    """
    if ramp_ticks < 1:
        raise ValueError("ramp_ticks must be positive")
    open_g = float(gripper_open)
    close_g = float(gripper_close)
    blocks = []
    if grasp > 0:
        blocks.append((pos[:grasp], quat[:grasp], np.full(grasp, open_g), phase[:grasp]))
    blocks.append(_block(pos, quat, grasp, wait_ticks, open_g, PHASE_GRIP_WAIT))
    blocks.append(_ramp_block(pos, quat, grasp, ramp_ticks, open_g, close_g))
    blocks.append(_block(pos, quat, grasp, dwell_close, close_g, PHASE_GRIP_DWELL))
    grasp_out = grasp + wait_ticks
    mid_end = release
    if mid_end > grasp:
        # The grasp sample itself is represented by the blocks above.
        sl = slice(grasp + 1, release)
        count = release - grasp - 1
        if count > 0:
            blocks.append((pos[sl], quat[sl], np.full(count, close_g), phase[sl]))
        blocks.append(_block(pos, quat, release, wait_ticks, close_g, PHASE_GRIP_WAIT))
        blocks.append(_ramp_block(pos, quat, release, ramp_ticks, close_g, open_g))
        blocks.append(_block(pos, quat, release, dwell_open, open_g, PHASE_GRIP_DWELL))
    else:
        blocks.append(_block(pos, quat, release, wait_ticks, close_g, PHASE_GRIP_WAIT))
        blocks.append(_ramp_block(pos, quat, release, ramp_ticks, close_g, open_g))
        blocks.append(_block(pos, quat, release, dwell_open, open_g, PHASE_GRIP_DWELL))
    # Frames strictly after the release sample stay open.
    if release + 1 < len(pos):
        tail = slice(release + 1, None)
        n_tail = len(pos) - release - 1
        blocks.append((pos[tail], quat[tail], np.full(n_tail, open_g), phase[tail]))
    # release ramp starts after: approach + close-wait + close-ramp + close-dwell
    # + carry frames + open-wait.
    carry = max(release - grasp - 1, 0)
    release_out = grasp + wait_ticks + ramp_ticks + dwell_close + carry + wait_ticks
    out = _cat([block for block in blocks if len(block[2])])
    return out[0], out[1], out[2], out[3], grasp_out, release_out


def lead_actions(pos: np.ndarray, quat: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Action at t is the pose at t+1. The last action repeats the final pose."""
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    grip = np.asarray(grip, dtype=float).reshape(-1)
    n = len(pos)
    actions = np.empty((n, 8), dtype=float)
    if n == 1:
        actions[0, :3] = pos[0]
        actions[0, 3:7] = quat[0]
        actions[0, 7] = grip[0]
        return actions
    actions[:-1, :3] = pos[1:]
    actions[:-1, 3:7] = quat[1:]
    actions[:-1, 7] = grip[1:]
    actions[-1] = actions[-2]
    norms = np.linalg.norm(actions[:, 3:7], axis=1, keepdims=True)
    actions[:, 3:7] /= norms
    return actions


def _smooth_segment(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    """Smoothstep samples from ``start`` to ``end``, excluding ``start``."""
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    dist = float(np.linalg.norm(end - start))
    n = max(4, int(np.ceil(dist / float(step))))
    out = np.empty((n, 3))
    for k in range(1, n + 1):
        u = k / n
        blend = u * u * (3.0 - 2.0 * u)
        out[k - 1] = (1.0 - blend) * start + blend * end
    return out


def append_place_down(
    pos: np.ndarray,
    phase: np.ndarray,
    release: int,
    place_tcp: np.ndarray,
    retreat_z: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slide over the cell, descend, then retreat. Gripper opens at the bottom.

    Frames after the source release stay at the drop height, so they are
    replaced. The returned index is the lowered pose, which is where the
    gripper-open wait is inserted.
    """
    pos = np.asarray(pos, dtype=float)
    phase = np.asarray(phase)
    start = pos[int(release)]
    place = np.asarray(place_tcp, dtype=float).reshape(3)
    hover = place.copy()
    hover[2] = max(float(start[2]), float(place[2]) + float(retreat_z))
    over = _smooth_segment(start, hover, step)
    down = _smooth_segment(hover, place, step)
    retreat = place.copy()
    retreat[2] = float(place[2]) + float(retreat_z)
    up = _smooth_segment(place, retreat, step)
    new_pos = np.vstack((pos[: int(release) + 1], over, down, up))
    new_phase = np.concatenate((
        phase[: int(release) + 1],
        np.full(len(over), PHASE_PLACE_SLIDE, dtype=np.int8),
        np.full(len(down), PHASE_PLACE_DOWN, dtype=np.int8),
        np.full(len(up), PHASE_RETREAT, dtype=np.int8),
    ))
    return new_pos, new_phase, int(release) + len(over) + len(down)


def constant_quat(yaw: float, n: int) -> np.ndarray:
    quat = top_down_quat(float(yaw), 0)
    return np.repeat(quat.reshape(1, 4), n, axis=0)


def prepend_home(
    pos: np.ndarray,
    quat: np.ndarray,
    grip: np.ndarray,
    phase: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
    n_ticks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quintic from the home TCP onto the first retargeted pose.

    Sample 0 is the home pose (``u = 0``). The existing first sample is the
    arrival (``u = 1``) and is not duplicated. Position and orientation both
    use :func:`aloha_pp.control.reference.quintic_blend`.
    """
    n = int(n_ticks)
    if n < 1:
        return pos, quat, grip, phase
    home_pos = np.asarray(home_pos, dtype=float).reshape(3)
    home_quat = np.asarray(home_quat, dtype=float).reshape(4)
    target_pos = np.asarray(pos[0], dtype=float).reshape(3)
    target_quat = np.asarray(quat[0], dtype=float).reshape(4)
    extra_pos = np.empty((n, 3))
    extra_quat = np.empty((n, 4))
    for k in range(n):
        blend = quintic_blend(k / n)
        extra_pos[k] = (1.0 - blend) * home_pos + blend * target_pos
        extra_quat[k] = slerp(home_quat, target_quat, blend)
    extra_grip = np.full(n, float(grip[0]))
    extra_phase = np.full(n, PHASE_SYNTH_APPROACH, dtype=np.int8)
    return (
        np.concatenate((extra_pos, pos)),
        np.concatenate((extra_quat, quat)),
        np.concatenate((extra_grip, grip)),
        np.concatenate((extra_phase, phase)),
    )


def map_index(index: int, n_src: int, n_dst: int) -> int:
    """Index of the same fraction along a resampled sequence."""
    if n_dst <= 1 or n_src <= 1:
        return 0
    mapped = int(np.rint(float(index) * (n_dst - 1) / (n_src - 1)))
    return int(np.clip(mapped, 0, n_dst - 1))


def lead_phase(phase: np.ndarray) -> np.ndarray:
    """Phase of the pose each action commands. The last action repeats."""
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    out = np.empty(len(phase), dtype=np.int8)
    if len(phase) <= 1:
        out[:] = phase
        return out
    out[:-1] = phase[1:]
    out[-1] = phase[-1]
    return out


def stored_actions(actions: np.ndarray) -> np.ndarray:
    """Float64 view of the float32 values that are written and replayed."""
    return np.asarray(np.asarray(actions, dtype=np.float32), dtype=np.float64)


def ik_actions(
    ik: DlsIk,
    actions: np.ndarray,
    q_init: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """IK each stored action the way ``TrackingController`` does.

    The seed starts at ``q_init``. The exact home pose returns ``q_init``.
    A failed solve keeps the previous seed instead of the rejected solution.
    """
    seed = np.asarray(q_init, dtype=float).copy()
    home_q = seed.copy()
    home_pos = np.asarray(home_pos, dtype=float).reshape(3)
    home_quat = np.asarray(home_quat, dtype=float).reshape(4)
    qs = np.empty((len(actions), seed.size))
    err = np.empty(len(actions))
    for t in range(len(actions)):
        quat = np.asarray(actions[t, 3:7], dtype=float)
        norm = float(np.linalg.norm(quat))
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-8 else quat / norm
        result = solve_cartesian(ik, actions[t, :3], quat, seed, home_q, home_pos, home_quat)
        if result.success:
            seed = np.asarray(result.q, dtype=float).copy()
        qs[t] = seed
        err[t] = float(result.pos_err)
    return qs, err


def unbraked_preview_series(
    q_targets: np.ndarray,
    q0: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
) -> np.ndarray:
    """Per-tick peak |qd| of the unbraked preview quintic, shape ``(T, dof)``.

    This is the polynomial ``JointStream`` plays when every segment fits in
    ``v_max``: central velocity, central acceleration, incoming acceleration
    kept. Samples are the physics-step fractions, matching ``simulate_stream``.
    """
    targets = np.asarray(q_targets, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    prev = q.copy()
    duration = float(control_dt)
    s = np.arange(1, int(n_substeps) + 1, dtype=float) / float(n_substeps)
    series = np.zeros_like(targets)
    for index in range(len(targets)):
        target = targets[index]
        future = targets[index + 1] if index + 1 < len(targets) else target
        v1 = (future - prev) / (2.0 * duration)
        a1 = (future - 2.0 * target + prev) / (duration * duration)
        _q, qd, _qdd = eval_quintic(q, v, a, target, v1, a1, duration, s)
        qd = np.asarray(qd, dtype=float).copy()
        qd[-1] = v1
        series[index] = np.max(np.abs(qd), axis=0)
        prev = np.asarray(target, dtype=float).copy()
        q = np.asarray(target, dtype=float).copy()
        v = v1.copy()
        a = a1.copy()
    return series


def unbraked_preview_peak(
    q_targets: np.ndarray,
    q0: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
) -> np.ndarray:
    """Per-joint peak |qd| of the unbraked preview quintic."""
    series = unbraked_preview_series(
        q_targets, q0, control_dt=control_dt, n_substeps=n_substeps
    )
    if len(series) == 0:
        return np.zeros_like(np.asarray(q0, dtype=float))
    return np.max(series, axis=0)


def _limited_peak(
    q_targets: np.ndarray,
    q0: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    a_max: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    speeds, _acc, flags = simulate_stream(
        q_targets,
        q0,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme="preview",
        a_max=a_max,
    )
    return np.max(np.abs(speeds), axis=0), np.any(flags, axis=0)


def _under_ratio(peak: np.ndarray, flagged: np.ndarray, v_max: np.ndarray, ratio: float) -> bool:
    limit = float(ratio) * np.asarray(v_max, dtype=float)
    return bool(np.all(peak <= limit) and not np.any(flagged))


def ik_joint_path(
    ik: DlsIk,
    pos: np.ndarray,
    quat: np.ndarray,
    q_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """IK each pose, always accepting the solution. Returns ``(q, pos_err)``."""
    seed = np.asarray(q_init, dtype=float).copy()
    qs = np.empty((len(pos), len(seed)))
    err = np.empty(len(pos))
    for t in range(len(pos)):
        result = ik.solve(pos[t], quat[t], seed)
        seed = np.asarray(result.q, dtype=float).copy()
        qs[t] = seed
        err[t] = float(result.pos_err)
    return qs, err


def peak_joint_speed(q: np.ndarray, control_dt: float) -> np.ndarray:
    if len(q) < 2:
        return np.zeros(q.shape[-1])
    return np.max(np.abs(np.diff(q, axis=0)) / float(control_dt), axis=0)


def resample_trajectory(
    pos: np.ndarray,
    quat: np.ndarray,
    grip: np.ndarray,
    phase: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stretch time by ``scale`` (>= 1). Quaternions are slerped."""
    scale = float(scale)
    if scale < 1.0:
        raise ValueError("scale must be >= 1")
    n = len(pos)
    if n < 2 or scale == 1.0:
        return pos.copy(), quat.copy(), grip.copy(), phase.copy()
    n_new = int(np.ceil((n - 1) * scale)) + 1
    src = np.linspace(0.0, n - 1, n)
    dst = np.linspace(0.0, n - 1, n_new)
    out_pos = np.empty((n_new, 3))
    out_quat = np.empty((n_new, 4))
    for axis in range(3):
        out_pos[:, axis] = np.interp(dst, src, pos[:, axis])
    out_grip = np.interp(dst, src, grip)
    out_phase = phase[np.clip(np.rint(dst).astype(int), 0, n - 1)]
    for i, u in enumerate(dst):
        lo = int(np.floor(u))
        hi = min(lo + 1, n - 1)
        frac = float(u - lo)
        out_quat[i] = slerp(quat[lo], quat[hi], frac)
    return out_pos, out_quat, out_grip, out_phase


def _solve_stored(
    ik: DlsIk,
    pos: np.ndarray,
    quat: np.ndarray,
    grip: np.ndarray,
    q_init: np.ndarray,
    home_pos: np.ndarray,
    home_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = stored_actions(lead_actions(pos, quat, grip))
    q, err = ik_actions(ik, actions, q_init, home_pos, home_quat)
    return actions, q, err


def _segment_runs(phase: np.ndarray) -> list[tuple[int, int, int]]:
    """Inclusive ``(start, end, phase_id)`` runs."""
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    if len(phase) == 0:
        return []
    runs = []
    start = 0
    for index in range(1, len(phase)):
        if phase[index] != phase[start]:
            runs.append((start, index - 1, int(phase[start])))
            start = index
    runs.append((start, len(phase) - 1, int(phase[start])))
    return runs


def smooth_segment_scales(phase: np.ndarray, scales: dict, blend_ticks: int) -> np.ndarray:
    """Piecewise segment slowdown with a quintic blend at each boundary.

    The blend uses at most half of each adjacent run, so a long segment keeps
    its own coefficient in the interior and the two windows do not overlap.
    ``quintic_blend`` has zero slope at both ends, so the slowdown and its
    first difference match the flat interior.
    """
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    runs = _segment_runs(phase)
    sigma = np.ones(len(phase), dtype=float)
    for start, end, pid in runs:
        sigma[start : end + 1] = float(scales.get(int(pid), 1.0))
    width = max(int(blend_ticks), 0)
    for (start0, end0, pid0), (start1, end1, pid1) in zip(runs, runs[1:]):
        left = float(scales.get(int(pid0), 1.0))
        right = float(scales.get(int(pid1), 1.0))
        if abs(left - right) < 1e-12:
            continue
        n_left = min(width, max((end0 - start0 + 1) // 2, 0))
        n_right = min(width, max((end1 - start1 + 1) // 2, 0))
        indices = list(range(end0 - n_left + 1, end0 + 1)) if n_left else []
        indices += list(range(start1, start1 + n_right)) if n_right else []
        count = len(indices)
        if count == 0:
            continue
        for step, index in enumerate(indices):
            blend = 0.0 if count == 1 else quintic_blend(step / (count - 1))
            sigma[index] = (1.0 - blend) * left + blend * right
    return np.maximum(sigma, 1.0)


def _tick_owner(phase: np.ndarray) -> np.ndarray:
    """Motion phase that owns each preview tick.

    Tick ``t`` commands the pose at ``t + 1``. A gripper tick belongs to the
    motion segment that arrived there, so a fast arrival raises that scale.
    """
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    owner = np.array(phase, copy=True)
    last = int(phase[0]) if len(phase) else 0
    motion = set(MOTION_PHASES)
    for index, value in enumerate(phase):
        if int(value) in motion:
            last = int(value)
        owner[index] = last
    if len(owner) <= 1:
        return owner
    dest = np.empty_like(owner)
    dest[:-1] = owner[1:]
    dest[-1] = owner[-1]
    return dest


def _scales_from_series(
    series: np.ndarray,
    owner: np.ndarray,
    v_max: np.ndarray,
    ratio: float,
    max_scale: float,
) -> dict[int, float]:
    limit = float(ratio) * np.asarray(v_max, dtype=float)
    scales: dict[int, float] = {}
    for pid in np.unique(owner):
        mask = owner == pid
        peak = np.max(series[mask], axis=0)
        need = float(np.max(peak / limit))
        if not np.isfinite(need):
            need = float(max_scale)
        scales[int(pid)] = float(np.clip(max(need, 1.0), 1.0, float(max_scale)))
    return scales


def _resample_profile(
    pos: np.ndarray,
    quat: np.ndarray,
    grip: np.ndarray,
    phase: np.ndarray,
    sigma: np.ndarray,
    events: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform control ticks along a variable slowdown. Events land on knots."""
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    grip = np.asarray(grip, dtype=float).reshape(-1)
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    n = len(pos)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1.0)
    if n < 2:
        return pos.copy(), quat.copy(), grip.copy(), phase.copy(), np.zeros(1)
    interval = 0.5 * (sigma[:-1] + sigma[1:])
    cursor = 0.0
    prev = 0
    for event in events:
        event = int(event)
        if event <= prev or event > n - 1:
            continue
        span = float(np.sum(interval[prev:event]))
        # Snap this segment's end onto an integer tick. The multiplier is the
        # ratio of the wanted duration to this segment alone; folding in the
        # time already accumulated would stretch every later segment.
        desired_end = max(float(np.round(cursor + span)), cursor + float(event - prev))
        delta = desired_end - cursor
        if span > 0.0:
            interval[prev:event] *= delta / span
        cursor = desired_end
        prev = event
    cumulative = np.concatenate((np.zeros(1), np.cumsum(interval)))
    total = float(cumulative[-1])
    if total > 0.0:
        snapped = float(np.round(total))
        cumulative *= snapped / total
        total = snapped
    n_new = int(total) + 1
    src_u = np.interp(np.arange(n_new, dtype=float), cumulative, np.arange(n, dtype=float))
    out_pos = np.empty((n_new, 3))
    out_quat = np.empty((n_new, 4))
    for axis in range(3):
        out_pos[:, axis] = np.interp(src_u, np.arange(n), pos[:, axis])
    out_grip = np.interp(src_u, np.arange(n), grip)
    out_phase = phase[np.clip(np.rint(src_u).astype(int), 0, n - 1)]
    for index, param in enumerate(src_u):
        lo = int(np.floor(param))
        hi = min(lo + 1, n - 1)
        out_quat[index] = slerp(quat[lo], quat[hi], float(param - lo))
    for event in events:
        knot = int(np.clip(np.rint(cumulative[int(event)]), 0, n_new - 1))
        out_pos[knot] = pos[int(event)]
        out_quat[knot] = quat[int(event)]
        out_grip[knot] = grip[int(event)]
        out_phase[knot] = phase[int(event)]
    return out_pos, out_quat, out_grip, out_phase, cumulative


def _named_scales(scales: dict[int, float]) -> dict[str, float]:
    named = {}
    for pid, value in scales.items():
        if 0 <= int(pid) < len(PHASE_NAMES):
            named[PHASE_NAMES[int(pid)]] = float(value)
    return named


def slow_until(
    ik: DlsIk,
    pos: np.ndarray,
    quat: np.ndarray,
    grip: np.ndarray,
    phase: np.ndarray,
    q_init: np.ndarray,
    v_max: np.ndarray,
    control_dt: float,
    *,
    ratio: float = 0.5,
    err_abort: float = 0.01,
    n_substeps: int = 25,
    a_max: np.ndarray | None = None,
    home_pos: np.ndarray | None = None,
    home_quat: np.ndarray | None = None,
    max_scale: float = 64.0,
    blend_ticks: int = 5,
    grasp: int | None = None,
    release: int | None = None,
    gripper: dict | None = None,
) -> dict:
    """Slow each motion segment by its own preview peak, then check the result.

    The slowdown blends across segment boundaries. When ``gripper`` is given,
    the fixed M4 wait / ramp / dwell is inserted after the resample and the
    acceptance check runs on that full sequence. The check repeats, raising
    only the segments that are still over, until
    ``simulate_stream(..., scheme="preview")`` reports a substep peak
    ``<= ratio * v_max`` and clear overspeed flags, or every offending
    segment is already at ``max_scale``. There is no iteration cap.

    Actions are quantized to float32 before the IK. A path whose IK position
    error already exceeds ``err_abort`` is returned at the current scale.
    """
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    grip = np.asarray(grip, dtype=float).reshape(-1)
    phase = np.asarray(phase, dtype=np.int8).reshape(-1)
    v_max = np.asarray(v_max, dtype=float).reshape(-1)
    if a_max is not None:
        a_max = np.asarray(a_max, dtype=float).reshape(-1)
    if home_pos is None:
        home_pos = np.asarray(pos[0], dtype=float)
    if home_quat is None:
        home_quat = np.asarray(quat[0], dtype=float)
    q_init = np.asarray(q_init, dtype=float).reshape(-1)
    limit = float(ratio) * v_max

    def solve(cur_pos, cur_quat, cur_grip):
        return _solve_stored(ik, cur_pos, cur_quat, cur_grip, q_init, home_pos, home_quat)

    def assemble(cur_pos, cur_quat, cur_grip, cur_phase, grasp_i, release_i):
        if gripper is None:
            return cur_pos, cur_quat, cur_grip, cur_phase, int(grasp_i), int(release_i)
        return insert_gripper_waits(
            cur_pos, cur_quat, cur_phase,
            grasp=int(grasp_i), release=int(release_i), **gripper,
        )

    def pack(cur, actions, q, err, scales, speed_ok, peak, flags, unlimited, iterations):
        values = list(scales.values()) if scales else [1.0]
        return {
            "pos": cur[0],
            "quat": cur[1],
            "grip": cur[2],
            "phase": cur[3],
            "actions": actions,
            "scale": float(max(values)),
            "segment_scales": _named_scales(scales),
            "q": q,
            "ik_pos_err": err,
            "ref_peak": peak,
            "ref_peak_unlimited": unlimited,
            "ref_flags": flags,
            "speed_ok": bool(speed_ok),
            "grasp_index": int(cur[4]),
            "release_index": int(cur[5]),
            "iterations": int(iterations),
        }

    events = [len(pos) - 1]
    if grasp is not None and release is not None:
        events = sorted({int(grasp), int(release), len(pos) - 1})
    grasp_i = 0 if grasp is None else int(grasp)
    release_i = 0 if release is None else int(release)
    actions, q, err = solve(pos, quat, grip)
    unlimited = unbraked_preview_peak(q, q_init, control_dt=control_dt, n_substeps=n_substeps)
    flags = np.ones(q_init.shape, dtype=bool)
    if float(np.max(err)) > float(err_abort):
        current = assemble(pos, quat, grip, phase, grasp_i, release_i)
        actions, q, err = solve(current[0], current[1], current[2])
        unlimited = unbraked_preview_peak(q, q_init, control_dt=control_dt, n_substeps=n_substeps)
        return pack(current, actions, q, err, {int(phase[0]): 1.0}, False, unlimited, flags, unlimited, 0)

    series = unbraked_preview_series(q, q_init, control_dt=control_dt, n_substeps=n_substeps)
    scales = _scales_from_series(series, _tick_owner(phase), v_max, ratio, max_scale)
    for _start, _end, pid in _segment_runs(phase):
        scales.setdefault(int(pid), 1.0)

    current = assemble(pos, quat, grip, phase, grasp_i, release_i)
    peak = unlimited
    iterations = 0
    motion_ids = {pid for _start, _end, pid in _segment_runs(phase)}
    while True:
        iterations += 1
        sigma = smooth_segment_scales(phase, scales, blend_ticks)
        sampled = _resample_profile(pos, quat, grip, phase, sigma, events)
        if grasp is None or release is None:
            mapped_grasp, mapped_release = 0, 0
        else:
            cumulative = sampled[4]
            mapped_grasp = int(np.clip(np.rint(cumulative[int(grasp)]), 0, len(sampled[0]) - 1))
            mapped_release = int(np.clip(np.rint(cumulative[int(release)]), 0, len(sampled[0]) - 1))
            if mapped_release < mapped_grasp:
                mapped_release = mapped_grasp
        current = assemble(sampled[0], sampled[1], sampled[2], sampled[3], mapped_grasp, mapped_release)
        actions, q, err = solve(current[0], current[1], current[2])
        series = unbraked_preview_series(q, q_init, control_dt=control_dt, n_substeps=n_substeps)
        unlimited = np.max(series, axis=0) if len(series) else np.zeros_like(q_init)
        peak = unlimited
        flags = np.ones(q_init.shape, dtype=bool)
        if float(np.max(err)) > float(err_abort):
            return pack(current, actions, q, err, scales, False, unlimited, flags, unlimited, iterations)
        owner = _tick_owner(current[3])
        tick_need = np.max(series / limit.reshape(1, -1), axis=1)
        if float(np.max(tick_need)) <= 1.0:
            peak, flags = _limited_peak(
                q, q_init, control_dt=control_dt, n_substeps=n_substeps, v_max=v_max, a_max=a_max,
            )
            if _under_ratio(peak, flags, v_max, ratio):
                return pack(current, actions, q, err, scales, True, peak, flags, unlimited, iterations)
            tick_need = np.maximum(tick_need, 1.02)
        bumped = False
        for pid in sorted(motion_ids):
            if not np.any(owner == pid):
                continue
            need = float(np.max(tick_need[owner == pid]))
            if not np.isfinite(need) or need <= 1.0:
                continue
            old = float(scales.get(pid, 1.0))
            new = min(float(max_scale), old * max(need, 1.02))
            if new > old + 1e-9:
                scales[pid] = new
                bumped = True
        if not bumped:
            return pack(current, actions, q, err, scales, False, peak, flags, unlimited, iterations)


def retarget_demo(
    eef_pos: np.ndarray,
    can_pos: np.ndarray,
    gripper_qpos: np.ndarray,
    *,
    m1: dict,
    m5: dict,
    home_pos: np.ndarray | None = None,
    home_quat: np.ndarray | None = None,
    ik: DlsIk | None = None,
    q_init: np.ndarray | None = None,
    v_max: np.ndarray | None = None,
    n_substeps: int = 25,
    a_max: np.ndarray | None = None,
) -> dict:
    """Retarget one demonstration. IK time scaling runs when ``ik`` is given."""
    opening = gripper_opening(gripper_qpos)
    source_grasp, source_release = grasp_release(
        opening,
        float(m5["gripper_close_width"]),
        min_frames=int(m5.get("min_close_frames", 1)),
    )
    xy, clamped = map_can_xy(
        np.asarray(can_pos[0, :2], dtype=float),
        np.asarray(m1["random_can"]["xy_low"], dtype=float),
        np.asarray(m1["random_can"]["xy_high"], dtype=float),
    )
    can_new = np.array([xy[0], xy[1], float(m1["can_rest_z"])])
    cell_new = np.asarray(m1["target_cells"]["fixed_1"]["center"], dtype=float)
    pos, phase = object_centric_positions(
        eef_pos,
        can_pos,
        grasp=source_grasp,
        release=source_release,
        can_new=can_new,
        cell_src=can_target_center(),
        cell_new=cell_new,
        carry_clearance_z=float(m5["carry_clearance_z"]),
    )
    # The source opens in the air. Finish the place on the cell, holding the
    # grasp offset, and open only after the descent.
    offset = np.asarray(eef_pos[source_grasp] - can_pos[source_grasp], dtype=float)
    place_tcp = cell_new.copy()
    place_tcp[:2] = cell_new[:2] + offset[:2]
    place_tcp[2] = float(cell_new[2]) + float(m5["place_gap"]) + float(offset[2])
    pos, phase, release = append_place_down(
        pos,
        phase,
        source_release,
        place_tcp,
        float(m5["retreat_z"]),
        float(m5["place_step"]),
    )
    yaw = choose_yaw(can_new[:2])
    quat = constant_quat(yaw, len(pos))
    dt = 1.0 / float(m1["control_hz"])
    grip = np.full(len(pos), float(m1["gripper_open"]))
    grasp_i = int(source_grasp)
    release_i = int(release)
    if home_pos is not None:
        n_home = max(2, int(round(float(m5["home_approach_s"]) / dt)))
        pos, quat, grip, phase = prepend_home(
            pos, quat, grip, phase, home_pos, home_quat, n_home
        )
        grasp_i += n_home
        release_i += n_home
    wait = grip_wait_ticks(int(m5["preview_lag_ticks"]), float(m5["settle_s"]), dt)
    dwell_close = seconds_to_ticks(float(m5["post_close_dwell_s"]), dt)
    dwell_open = seconds_to_ticks(float(m5["post_open_dwell_s"]), dt)
    gripper = {
        "gripper_open": float(m1["gripper_open"]),
        "gripper_close": float(m1["gripper_close"]),
        "wait_ticks": wait,
        "ramp_ticks": int(m5["grip_ramp_ticks"]),
        "dwell_close": dwell_close,
        "dwell_open": dwell_open,
    }
    scale = 1.0
    segment_scales: dict = {}
    iterations = 0
    q = None
    ik_err = None
    ref_peak = None
    ref_unlimited = None
    ref_flags = None
    speed_ok = None
    if ik is not None:
        scaled = slow_until(
            ik,
            pos,
            quat,
            grip,
            phase,
            q_init,
            v_max,
            dt,
            ratio=float(m5["speed_ratio"]),
            n_substeps=int(n_substeps),
            a_max=a_max,
            home_pos=home_pos if home_pos is not None else pos[0],
            home_quat=home_quat if home_quat is not None else quat[0],
            blend_ticks=int(m5.get("scale_blend_ticks", 5)),
            grasp=grasp_i,
            release=release_i,
            gripper=gripper,
        )
        pos = scaled["pos"]
        quat = scaled["quat"]
        grip = scaled["grip"]
        phase = scaled["phase"]
        actions = scaled["actions"]
        grasp_i = int(scaled["grasp_index"])
        release_i = int(scaled["release_index"])
        scale = float(scaled["scale"])
        segment_scales = dict(scaled["segment_scales"])
        iterations = int(scaled["iterations"])
        q = scaled["q"]
        ik_err = scaled["ik_pos_err"]
        ref_peak = scaled["ref_peak"]
        ref_unlimited = scaled["ref_peak_unlimited"]
        ref_flags = scaled["ref_flags"]
        speed_ok = bool(scaled["speed_ok"])
    else:
        pos, quat, grip, phase, grasp_i, release_i = insert_gripper_waits(
            pos, quat, phase, grasp=grasp_i, release=release_i, **gripper,
        )
        actions = lead_actions(pos, quat, grip)
    phase = lead_phase(phase)
    if len(pos) > 1:
        grasp_i = max(int(grasp_i) - 1, 0)
        release_i = max(int(release_i) - 1, 0)
    outside = bool(
        xy[0] < float(m1["random_can"]["xy_low"][0]) - 1e-9
        or xy[0] > float(m1["random_can"]["xy_high"][0]) + 1e-9
        or xy[1] < float(m1["random_can"]["xy_low"][1]) - 1e-9
        or xy[1] > float(m1["random_can"]["xy_high"][1]) + 1e-9
    )
    return {
        "pos": pos,
        "quat": quat,
        "grip": grip,
        "phase": phase,
        "actions": actions,
        "can_pos": can_new,
        "can_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "yaw": float(yaw),
        "slowdown": float(scale),
        "segment_scales": segment_scales,
        "iterations": int(iterations),
        "grasp_index": int(grasp_i),
        "release_index": int(release_i),
        "source_grasp": int(source_grasp),
        "source_release": int(source_release),
        "can_clamped": clamped,
        "can_out_of_bin": outside,
        "q": q,
        "ik_pos_err": ik_err,
        "ref_peak": ref_peak,
        "ref_peak_unlimited": ref_unlimited,
        "ref_flags": ref_flags,
        "speed_ok": speed_ok,
    }
