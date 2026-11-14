"""Waypoint and timing noise on top of the M4 planner.

Noise is drawn once per ``reset`` and added to the grasp / place waypoints,
the carry endpoints (a via-point offset), yaw, and each motion segment's
duration. It is not per-tick action jitter. Every amplitude defaults to 0.
With all amplitudes at 0 the schedule is the M4 schedule: ``reset`` returns
the superclass schedule, and ``_rebuild_carry`` calls the superclass, so no
extra arithmetic touches the ticks.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.kin.ik import top_down_quat
from aloha_pp.planning.planner import PickPlacePlanner
from aloha_pp.planning.schedule import CARRY

NOISE_KEYS = ("grasp_xy", "place_xy", "lift_z", "yaw", "time_scale", "via_xy")


def zero_noise() -> dict[str, float]:
    return {key: 0.0 for key in NOISE_KEYS}


class NoisyPickPlacePlanner(PickPlacePlanner):
    def __init__(self, m1: dict, m4: dict, m3: dict, speed: str, noise: dict, rng: np.random.Generator) -> None:
        super().__init__(m1, m4, m3, speed)
        self.noise = {key: float(noise.get(key, 0.0)) for key in NOISE_KEYS}
        self.rng = rng
        self._draw: dict | None = None
        self._lift_pos = np.zeros(3)

    def noise_is_zero(self) -> bool:
        return all(value == 0.0 for value in self.noise.values())

    def reset(self, env, *, mode: str, log=print) -> None:
        if self.noise_is_zero():
            self._draw = None
            return super().reset(env, mode=mode, log=log)
        super().reset(env, mode=mode, log=log)
        self._draw = self._sample()
        self._apply(self._waypoints, self._draw)
        self._lift_pos = self._waypoints[3][0].copy()
        self.motion_ticks = self._scale_ticks(self.motion_ticks, self._draw["time"])
        self.schedule = self._assemble(self._waypoints, self.motion_ticks)
        self.grasp_pos = self._waypoints[2][0].copy()
        self.place_tcp = self._waypoints[5][0].copy()
        self.k = 0

    def _rebuild_carry(self) -> None:
        if self._draw is None:
            return super()._rebuild_carry()
        super()._rebuild_carry()
        self._apply(self._waypoints, self._draw)
        # The prefix already ends on the lift point chosen at reset.
        self._waypoints[3] = (self._lift_pos.copy(), self._waypoints[3][1].copy())
        rebuilt = self._assemble(self._waypoints, self.motion_ticks)
        start = next(i for i, tick in enumerate(self.schedule) if tick.state == CARRY)
        rebuilt_start = next(i for i, tick in enumerate(rebuilt) if tick.state == CARRY)
        if start != rebuilt_start:
            raise RuntimeError("noisy rebuild moved the carry boundary")
        self.schedule = self.schedule[:start] + rebuilt[start:]
        self.place_tcp = self._waypoints[5][0].copy()

    def _sample(self) -> dict:
        def uniform(amp: float, size: int | None = None) -> np.ndarray:
            if amp == 0.0:
                return np.zeros(2) if size == 2 else np.zeros(7) if size == 7 else 0.0
            if size is None:
                return float(self.rng.uniform(-amp, amp))
            return self.rng.uniform(-amp, amp, size=size)

        span = float(self.noise["time_scale"])
        if span == 0.0:
            factors = np.ones(7)
        else:
            factors = self.rng.uniform(1.0 - span, 1.0 + span, size=7)
        return {
            "grasp_xy": np.asarray(uniform(self.noise["grasp_xy"], 2), dtype=float),
            "place_xy": np.asarray(uniform(self.noise["place_xy"], 2), dtype=float),
            "lift_z": float(uniform(self.noise["lift_z"])),
            "yaw": float(uniform(self.noise["yaw"])),
            "via": float(uniform(self.noise["via_xy"])),
            "time": factors,
        }

    def _apply(self, waypoints: list, draw: dict) -> None:
        for index in (1, 2, 3):
            waypoints[index][0][:2] = waypoints[index][0][:2] + draw["grasp_xy"]
        for index in (4, 5, 6):
            waypoints[index][0][:2] = waypoints[index][0][:2] + draw["place_xy"]
        waypoints[3][0][2] = float(waypoints[3][0][2]) + draw["lift_z"]
        waypoints[4][0][2] = float(waypoints[4][0][2]) + draw["lift_z"]
        direction = waypoints[4][0][:2] - waypoints[3][0][:2]
        norm = float(np.linalg.norm(direction))
        perp = np.array([1.0, 0.0]) if norm < 1e-8 else np.array([-direction[1], direction[0]]) / norm
        waypoints[3][0][:2] = waypoints[3][0][:2] + perp * draw["via"]
        waypoints[4][0][:2] = waypoints[4][0][:2] + perp * draw["via"] * 0.5
        quat = top_down_quat(float(self.yaw) + draw["yaw"], 0)
        for index in range(1, 7):
            waypoints[index] = (waypoints[index][0], quat.copy())

    @staticmethod
    def _scale_ticks(ticks: np.ndarray, factors: np.ndarray) -> np.ndarray:
        scaled = np.maximum(1, np.rint(np.asarray(ticks, dtype=float) * factors).astype(int))
        return scaled
