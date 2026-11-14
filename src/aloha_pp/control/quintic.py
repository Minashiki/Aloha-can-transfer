"""Joint references from one polynomial per 20 Hz segment.

``q``, ``qdot`` and ``qddot`` are derivatives of that same polynomial. Endpoint
acceleration is a boundary condition, not forced to zero. Two online schemes:

* ``lookahead`` caches one future knot and uses central velocity and central
  acceleration. The knot commanded at time ``k T`` is reached at ``(k+2) T``.
  Total lag is 100 ms: the original 50 ms pipeline plus one 50 ms buffer.
* ``causal`` uses only the current state and the knot that just arrived.
  Endpoint acceleration is the change in secant velocity, clipped to
  ``a_max``; it is not forced to zero. The knot is the end of the segment
  that starts when it arrives, so the pipeline lag stays 50 ms. A quintic
  can still ring between those ends, so ``a_max`` is not an interior bound.
  That is measured in the scheme comparison; the environment uses lookahead.

Velocity limits are unchanged. A segment that would exceed ``v_max`` is
shortened or braked. Braking matches the incoming velocity, so the chain stays
C1, and a start speed already sitting on the limit can still stop or reverse.
"""

from __future__ import annotations

import numpy as np

SCHEMES = ("lookahead", "causal")
_PEAK_SAMPLES = 401

# Ticks of delay from the 20 Hz command to the joint reference arriving on it.
SCHEME_LAG_TICKS = {"lookahead": 2, "causal": 1}


def scheme_lag_s(scheme: str, control_dt: float) -> float:
    if scheme not in SCHEME_LAG_TICKS:
        raise ValueError(f"unknown reference scheme {scheme}")
    return SCHEME_LAG_TICKS[scheme] * float(control_dt)


def eval_quintic(
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    duration: float,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic with position, velocity and acceleration fixed at both ends.

    ``s`` is the fraction of ``duration``. Returns ``q, qdot, qddot`` with
    shape ``(len(s), dof)``. Specifying ``a0 = a1 = 0`` is allowed, but it is
    not the default the stream uses.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    s = np.asarray(s, dtype=float).reshape(-1)
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    p0 = q0
    p1 = v0 * duration
    p2 = 0.5 * a0 * duration * duration
    target_ds = v1 * duration
    target_d2 = a1 * duration * duration
    coeff_a = q1 - p0 - p1 - p2
    coeff_b = target_ds - p1 - 2.0 * p2
    coeff_c = target_d2 - 2.0 * p2
    c3 = 10.0 * coeff_a - 4.0 * coeff_b + 0.5 * coeff_c
    c4 = -15.0 * coeff_a + 7.0 * coeff_b - coeff_c
    c5 = 6.0 * coeff_a - 3.0 * coeff_b + 0.5 * coeff_c
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    q = (
        p0
        + p1 * s[:, None]
        + p2 * s2[:, None]
        + c3 * s3[:, None]
        + c4 * s4[:, None]
        + c5 * s5[:, None]
    )
    qdot = (
        p1
        + 2.0 * p2 * s[:, None]
        + 3.0 * c3 * s2[:, None]
        + 4.0 * c4 * s3[:, None]
        + 5.0 * c5 * s4[:, None]
    ) / duration
    qddot = (
        2.0 * p2
        + 6.0 * c3 * s[:, None]
        + 12.0 * c4 * s2[:, None]
        + 20.0 * c5 * s3[:, None]
    ) / (duration * duration)
    return q, qdot, qddot


def _interior_peak(
    q0: float, v0: float, a0: float, q1: float, v1: float, a1: float, duration: float
) -> float:
    s = np.linspace(0.0, 1.0, _PEAK_SAMPLES)
    _, qdot, _ = eval_quintic(
        np.array([q0]),
        np.array([v0]),
        np.array([a0]),
        np.array([q1]),
        np.array([v1]),
        np.array([a1]),
        duration,
        s,
    )
    return float(np.max(np.abs(qdot)))


def _speed_ok(
    q0: float,
    v0: float,
    a0: float,
    q1: float,
    v1: float,
    a1: float,
    duration: float,
    vmax: float,
) -> bool:
    """True when the whole quintic, including both ends, stays inside ``vmax``.

    The incoming speed is compared with ``vmax`` itself. A start speed that
    already equals the limit is legal; it is not rejected by an interior
    margin.
    """
    if abs(v0) > vmax * (1.0 + 1e-8) or abs(v1) > vmax * (1.0 + 1e-8):
        return False
    return _interior_peak(q0, v0, a0, q1, v1, a1, duration) <= vmax * (1.0 + 1e-4)


def _brake(
    q0: float, v0: float, duration: float, vmax: float, amax: float
) -> tuple[float, float, float]:
    """Constant-acceleration stop, or the largest speed drop ``amax`` allows."""
    if abs(v0) < 1e-12:
        return q0, 0.0, 0.0
    a_brake = -float(np.sign(v0)) * min(amax, abs(v0) / duration)
    v1 = float(np.clip(v0 + a_brake * duration, -vmax, vmax))
    a1 = (v1 - v0) / duration
    q1 = q0 + 0.5 * (v0 + v1) * duration
    return q1, v1, a1


def _plan_scalar(
    q0: float,
    v0: float,
    a0: float,
    q_target: float,
    duration: float,
    vmax: float,
    amax: float,
) -> tuple[float, float, float, float, bool]:
    """One joint. Returns q1, v1, a1, a0 used by the polynomial, and the flag.

    The flag means the requested secant speed, or the accel-bounded profile
    toward the target, does not fit in ``vmax``. Missing the target because
    ``amax`` is tight is not a velocity-limit event. The polynomial always
    starts at ``v0``. ``a0`` is kept when that still fits; otherwise this
    segment becomes constant-acceleration so a speed already on the limit can
    brake. Position and velocity stay C1 either way.
    """
    if vmax <= 0.0 or amax <= 0.0:
        raise ValueError("v_max and a_max must be positive")
    secant = (q_target - q0) / duration
    moving_away = abs(v0) > 1e-8 and (q_target - q0) * v0 < -1e-12
    must_stop = abs(q_target - q0) <= 1e-9 and abs(v0) > 1e-8
    if moving_away or must_stop:
        q1, v1, a1 = _brake(q0, v0, duration, vmax, amax)
        # A small speed pointing the wrong way is a normal correction. The
        # flag is only for a request, or a state, that is already at v_max.
        limited = abs(v0) >= vmax * (1.0 - 1e-3) or abs(secant) > vmax + 1e-8
    else:
        a1 = float(np.clip((secant - v0) / duration, -amax, amax))
        v1 = v0 + a1 * duration
        q_coast = q0 + 0.5 * (v0 + v1) * duration
        toward = (q_target - q0) * (q_coast - q_target)
        if toward >= 0.0 and abs(q_target - q0) <= abs(q_coast - q0) + 1e-12:
            q1 = q_target
        else:
            q1 = q_coast
        # Clip only when this profile leaves the speed limit. Landing short of
        # the target under amax keeps the flag clear.
        limited = abs(v1) > vmax + 1e-8 or abs(secant) > vmax + 1e-8
        if limited:
            v1 = float(np.clip(v1, -vmax, vmax))
            a1 = float(np.clip((v1 - v0) / duration, -amax, amax))
            v1 = float(np.clip(v0 + a1 * duration, -vmax, vmax))
            a1 = (v1 - v0) / duration
            q_coast = q0 + 0.5 * (v0 + v1) * duration
            toward = (q_target - q0) * (q_coast - q_target)
            if toward >= 0.0 and abs(q_target - q0) <= abs(q_coast - q0) + 1e-12:
                q1 = q_target
            else:
                q1 = q_coast
    if _speed_ok(q0, v0, a0, q1, v1, a1, duration, vmax):
        return q1, v1, a1, a0, limited
    # Incoming acceleration would push the speed through the limit. Drop it
    # for this segment only; v0 itself is not rewritten.
    a_used = a1
    q_const = q0 + v0 * duration + 0.5 * a_used * duration * duration
    v_const = v0 + a_used * duration
    if abs(v_const) > vmax + 1e-8 or not _speed_ok(
        q0, v0, a_used, q_const, v_const, a_used, duration, vmax
    ):
        q_stop, v_stop, a_stop = _brake(q0, v0, duration, vmax, amax)
        return q_stop, v_stop, a_stop, a_stop, True
    return q_const, v_const, a_used, a_used, limited or abs(v_const) > abs(v1) + 1e-8


def plan_segment(
    q0: np.ndarray,
    v0: np.ndarray,
    q_target: np.ndarray,
    duration: float,
    v_max: np.ndarray,
    a0: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal one-step plan. Returns q_end, v_end, a_end, a0_used, flag.

    ``a0_used`` is the start acceleration of the polynomial. It equals ``a0``
    when that profile fits, and a feasible replacement when it does not.
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    q_target = np.asarray(q_target, dtype=float).reshape(-1)
    v_max = np.asarray(v_max, dtype=float).reshape(-1)
    if a0 is None:
        a0 = np.zeros_like(q0)
    else:
        a0 = np.asarray(a0, dtype=float).reshape(-1)
    if a_max is None:
        a_max = 2.0 * v_max / float(duration)
    else:
        a_max = np.asarray(a_max, dtype=float).reshape(-1)
    q_end = np.empty_like(q0)
    v_end = np.empty_like(q0)
    a_end = np.empty_like(q0)
    a_used = np.empty_like(q0)
    limited = np.zeros(q0.shape[0], dtype=bool)
    for i in range(q0.shape[0]):
        q_end[i], v_end[i], a_end[i], a_used[i], limited[i] = _plan_scalar(
            float(q0[i]),
            float(v0[i]),
            float(a0[i]),
            float(q_target[i]),
            float(duration),
            float(v_max[i]),
            float(a_max[i]),
        )
    return q_end, v_end, a_end, a_used, limited


class JointStream:
    """One joint target per control period, one sample per physics step."""

    def __init__(
        self,
        q0: np.ndarray,
        *,
        control_dt: float,
        n_substeps: int,
        v_max: np.ndarray,
        scheme: str = "lookahead",
        a_max: np.ndarray | None = None,
    ) -> None:
        if n_substeps < 1:
            raise ValueError("n_substeps must be positive")
        if scheme not in SCHEMES:
            raise ValueError(f"unknown reference scheme {scheme}")
        self.scheme = scheme
        self.control_dt = float(control_dt)
        self.n_substeps = int(n_substeps)
        self.v_max = np.asarray(v_max, dtype=float).copy()
        self.a_max = None if a_max is None else np.asarray(a_max, dtype=float).copy()
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self._pending: np.ndarray | None = None
        self._prev_knot = self.q.copy()
        self._active = False
        self._k = 0
        self._q0 = self.q.copy()
        self._v0 = self.v.copy()
        self._a0 = self.a.copy()
        self._q1 = self.q.copy()
        self._v1 = self.v.copy()
        self._a1 = self.a.copy()
        self._limited = np.zeros(self.q.shape[0], dtype=bool)

    def reset(self, q0: np.ndarray) -> None:
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self._pending = None
        self._prev_knot = self.q.copy()
        self._active = False
        self._k = 0
        self._limited = np.zeros(self.q.shape[0], dtype=bool)

    def push(self, q_target: np.ndarray) -> np.ndarray:
        """Open the segment played by the next ``n_substeps`` samples."""
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        if self.scheme == "causal":
            q1, v1, a1, a0_used, limited = plan_segment(
                self.q,
                self.v,
                q_target,
                self.control_dt,
                self.v_max,
                a0=self.a,
                a_max=self.a_max,
            )
            self._load(q1, v1, a1, a0_used, limited)
            return limited.copy()
        return self._push_lookahead(q_target)

    def _push_lookahead(self, q_target: np.ndarray) -> np.ndarray:
        if self._pending is None:
            # One-tick cache. This period holds the current reference.
            self._pending = q_target.copy()
            self._load(self.q, self.v, self.a, self.a, np.zeros(self.q.shape[0], dtype=bool))
            return self._limited.copy()
        q_prev = self._prev_knot
        q_mid = self._pending
        duration = self.control_dt
        v1 = (q_target - q_prev) / (2.0 * duration)
        a1 = (q_target - 2.0 * q_mid + q_prev) / (duration * duration)
        q1 = q_mid.copy()
        a0 = self.a.copy()
        limited = np.zeros(self.q.shape[0], dtype=bool)
        for i in range(self.q.shape[0]):
            fits = _speed_ok(
                float(self.q[i]),
                float(self.v[i]),
                float(a0[i]),
                float(q1[i]),
                float(v1[i]),
                float(a1[i]),
                duration,
                float(self.v_max[i]),
            )
            if fits:
                continue
            limited[i] = True
            scalar_amax = None if self.a_max is None else np.array([self.a_max[i]])
            qq, vv, aa, a0_used, _flag = plan_segment(
                np.array([self.q[i]]),
                np.array([self.v[i]]),
                np.array([q_mid[i]]),
                duration,
                np.array([self.v_max[i]]),
                a0=np.array([a0[i]]),
                a_max=scalar_amax,
            )
            q1[i] = qq[0]
            v1[i] = vv[0]
            a1[i] = aa[0]
            a0[i] = a0_used[0]
        self._prev_knot = q_mid.copy()
        self._pending = q_target.copy()
        self._load(q1, v1, a1, a0, limited)
        return limited.copy()

    def _load(
        self,
        q1: np.ndarray,
        v1: np.ndarray,
        a1: np.ndarray,
        a0: np.ndarray,
        limited: np.ndarray,
    ) -> None:
        self._q0 = self.q.copy()
        self._v0 = self.v.copy()
        self._a0 = np.asarray(a0, dtype=float).copy()
        self._q1 = np.asarray(q1, dtype=float).copy()
        self._v1 = np.asarray(v1, dtype=float).copy()
        self._a1 = np.asarray(a1, dtype=float).copy()
        self._limited = np.asarray(limited, dtype=bool).copy()
        self._active = True
        self._k = 0

    def advance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One physics step. Returns q_d, qd_d, qdd_d, overspeed flag."""
        if not self._active:
            flag = np.zeros(self.q.shape[0], dtype=bool)
            return self.q.copy(), self.v.copy(), self.a.copy(), flag
        self._k += 1
        s = np.array([self._k / self.n_substeps])
        q, qdot, qddot = eval_quintic(
            self._q0, self._v0, self._a0, self._q1, self._v1, self._a1, self.control_dt, s
        )
        q_s = q[0]
        qd_s = qdot[0]
        qdd_s = qddot[0]
        flag = self._limited.copy()
        if self._k >= self.n_substeps:
            q_s = self._q1.copy()
            qd_s = self._v1.copy()
            qdd_s = self._a1.copy()
            self.a = self._a1.copy()
            self._active = False
            self._limited = np.zeros_like(self._limited)
        self.q = np.asarray(q_s, dtype=float).copy()
        self.v = np.asarray(qd_s, dtype=float).copy()
        return self.q.copy(), self.v.copy(), np.asarray(qdd_s, dtype=float).copy(), flag


def simulate_stream(
    q_targets: np.ndarray,
    q0: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay targets. Returns qd, qdd, and the overspeed flag, each ``(N, dof)``."""
    stream = JointStream(
        q0,
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=v_max,
        scheme=scheme,
        a_max=a_max,
    )
    speeds = []
    accels = []
    flags = []
    for target in np.asarray(q_targets, dtype=float):
        stream.push(target)
        for _ in range(n_substeps):
            _q, qd, qdd, flag = stream.advance()
            speeds.append(qd)
            accels.append(qdd)
            flags.append(flag)
    return np.stack(speeds), np.stack(accels), np.stack(flags)


def stretch_joint_trajectory(
    q_knots: np.ndarray,
    *,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    max_scale: float = 64.0,
    scheme: str = "lookahead",
    a_max: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Lengthen a joint polyline, sampled at a fixed control period, until it fits."""
    q_knots = np.asarray(q_knots, dtype=float)
    if q_knots.ndim != 2 or q_knots.shape[0] < 2:
        raise ValueError("q_knots must have shape (N, dof) with N >= 2")
    v_max = np.asarray(v_max, dtype=float)
    scale = 1.0
    resampled = q_knots
    for _ in range(8):
        _speeds, _acc, flags = simulate_stream(
            resampled[1:],
            resampled[0],
            control_dt=control_dt,
            n_substeps=n_substeps,
            v_max=v_max,
            scheme=scheme,
            a_max=a_max,
        )
        ratio = _unlimited_ratio(
            resampled, control_dt, n_substeps, v_max, scheme=scheme, a_max=a_max
        )
        if ratio <= 1.0 + 1e-5 and not np.any(flags):
            duration = (resampled.shape[0] - 1) * control_dt
            return resampled, duration
        scale *= max(ratio, 1.01) * 1.02
        if scale > max_scale:
            break
        resampled = _resample(q_knots, scale)
    raise RuntimeError(f"joint trajectory still exceeds v_max at scale {scale:.3f}")


def _unlimited_ratio(
    q_knots: np.ndarray,
    control_dt: float,
    n_substeps: int,
    v_max: np.ndarray,
    *,
    scheme: str,
    a_max: np.ndarray | None,
) -> float:
    """Peak speed of the same scheme with the limit removed, over ``v_max``.

    The clamped stream cannot supply this ratio: once it is already at the
    limit the ratio stays near 1 and the duration would not grow.
    """
    huge = np.full_like(np.asarray(v_max, dtype=float), 1.0e6)
    speeds, _acc, _flags = simulate_stream(
        q_knots[1:],
        q_knots[0],
        control_dt=control_dt,
        n_substeps=n_substeps,
        v_max=huge,
        scheme=scheme,
        a_max=None,
    )
    peak = np.max(np.abs(speeds), axis=0)
    return float(np.max(peak / np.asarray(v_max, dtype=float)))


def _resample(q_knots: np.ndarray, scale: float) -> np.ndarray:
    n, dof = q_knots.shape
    new_n = int(np.ceil((n - 1) * scale)) + 1
    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, new_n)
    return np.column_stack([np.interp(new_t, old_t, q_knots[:, j]) for j in range(dof)])
