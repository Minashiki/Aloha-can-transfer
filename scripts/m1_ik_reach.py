"""Top-down reachability of can-center samples over both bin floors.

Sample bounds come from the bin geoms: inset by the wall thickness and the
can radius. The right bin also drops the cross divider plus the can radius.
Any target that misses the IK tolerance exits with status 1 after the full
statistics are printed.

Run: conda run -n AIControl python scripts/m1_ik_reach.py
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

from aloha_pp.kin.ik import DlsIk, top_down_quat
from aloha_pp.robot import BASE_BODY, scene_xml
from aloha_pp.sim.env import load_config


def _geom(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"geom not found: {name}")
    return geom_id


def _world_to_base(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray) -> np.ndarray:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    mat = data.xmat[base_id].reshape(3, 3)
    return mat.T @ (np.asarray(pos, dtype=float) - data.xpos[base_id])


def can_center_bounds(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[float, tuple[np.ndarray, np.ndarray], list[tuple[str, np.ndarray, np.ndarray]]]:
    """Resting can-center height and the valid xy rectangles, in the base frame.

    A wall geom is centered on the floor edge, so its thin half-size is the
    inset from the floor edge to the inner face. The can center must stay a
    radius inside that face. On the right bin the divider is the same kind of
    inset measured from the bin center.
    """
    mujoco.mj_forward(model, data)
    radius = float(model.geom_size[_geom(model, "can_collision")][0])
    floor = _geom(model, "bin_left_floor")
    floor_top = float(data.geom_xpos[floor][2] + model.geom_size[floor][2])
    half_height = float(model.geom_size[_geom(model, "can_collision")][1])
    z = float(_world_to_base(model, data, np.array([0.0, 0.0, floor_top + half_height]))[2])

    def outer_rect(floor_name: str, wall_x: str, wall_y: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        floor_id = _geom(model, floor_name)
        center = _world_to_base(model, data, data.geom_xpos[floor_id])
        half = np.asarray(model.geom_size[floor_id][:2], dtype=float)
        # x-wall is thin along x (size[0]); y-wall is thin along y (size[1]).
        inset = np.array(
            [
                float(model.geom_size[_geom(model, wall_x)][0]),
                float(model.geom_size[_geom(model, wall_y)][1]),
            ]
        )
        span = half - inset - radius
        return center[:2] - span, center[:2] + span, center[:2]

    left_low, left_high, _left_center = outer_rect(
        "bin_left_floor", "bin_left_wall_x+", "bin_left_wall_y+"
    )
    right_low, right_high, right_center = outer_rect(
        "bin_right_floor", "bin_right_wall_x+", "bin_right_wall_y+"
    )
    divider = np.array(
        [
            float(model.geom_size[_geom(model, "bin_right_div_x")][0]) + radius,
            float(model.geom_size[_geom(model, "bin_right_div_y")][1]) + radius,
        ]
    )
    x_parts = (
        ("nx", right_low[0], right_center[0] - divider[0]),
        ("px", right_center[0] + divider[0], right_high[0]),
    )
    y_parts = (
        ("ny", right_low[1], right_center[1] - divider[1]),
        ("py", right_center[1] + divider[1], right_high[1]),
    )
    cells: list[tuple[str, np.ndarray, np.ndarray]] = []
    for x_name, x0, x1 in x_parts:
        for y_name, y0, y1 in y_parts:
            cells.append((f"right_{x_name}{y_name}", np.array([x0, y0]), np.array([x1, y1])))
    return z, (left_low, left_high), cells


def _summarize(name: str, results: list) -> tuple[int, int]:
    total = len(results)
    ok = sum(int(result.success) for result in results)
    worst_pos = max((result.pos_err for result in results), default=0.0)
    worst_rot = max((result.rot_err for result in results), default=0.0)
    print(
        f"grid {name}: {ok}/{total} worst_pos={worst_pos:.4e} m "
        f"worst_rot={worst_rot:.4e} rad"
    )
    return ok, total


def main() -> None:
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    data = mujoco.MjData(model)
    ik = DlsIk(
        model,
        cfg["tcp_site"],
        damping=float(cfg["ik"]["damping"]),
        max_iters=int(cfg["ik"]["max_iters"]),
        pos_tol=float(cfg["ik"]["pos_tol"]),
        rot_tol=float(cfg["ik"]["rot_tol"]),
        rot_weight=float(cfg["ik"]["rot_weight"]),
    )
    home = np.asarray(cfg["home_q"], dtype=float)
    z, (left_low, left_high), cells = can_center_bounds(model, data)
    print(
        f"bounds left x[{left_low[0]:.5f},{left_high[0]:.5f}] "
        f"y[{left_low[1]:.5f},{left_high[1]:.5f}] z={z:.5f}"
    )
    for name, low, high in cells:
        print(
            f"bounds {name} x[{low[0]:.5f},{high[0]:.5f}] y[{low[1]:.5f},{high[1]:.5f}]"
        )

    failures = 0

    def solve_points(points: list[np.ndarray]) -> list:
        found = []
        for point in points:
            yaw = float(np.arctan2(point[1], point[0]))
            result = ik.solve(point, top_down_quat(yaw, 0), home)
            found.append(result)
        return found

    def grid_points(low: np.ndarray, high: np.ndarray, n: int = 7) -> list[np.ndarray]:
        xs = np.linspace(low[0], high[0], n)
        ys = np.linspace(low[1], high[1], n)
        return [np.array([x, y, z]) for x in xs for y in ys]

    left_results = solve_points(grid_points(left_low, left_high))
    ok, total = _summarize("left_pick", left_results)
    failures += total - ok
    regions = [(left_low, left_high)]
    for name, low, high in cells:
        results = solve_points(grid_points(low, high))
        ok, total = _summarize(name, results)
        failures += total - ok
        regions.append((low, high))

    rng = np.random.default_rng(0)
    pos_errs = []
    rot_errs = []
    successes = 0
    n = 200
    for i in range(n):
        low, high = regions[i % len(regions)]
        xy = rng.uniform(low, high)
        yaw = float(np.arctan2(xy[1], xy[0]))
        result = ik.solve(np.array([xy[0], xy[1], z]), top_down_quat(yaw, 0), home)
        successes += int(result.success)
        pos_errs.append(result.pos_err)
        rot_errs.append(result.rot_err)
    pos_errs = np.asarray(pos_errs)
    rot_errs = np.asarray(rot_errs)
    failures += n - successes
    print(
        f"random {n}: success {successes}/{n} "
        f"pos mean {pos_errs.mean():.4e} max {pos_errs.max():.4e} "
        f"rot mean {rot_errs.mean():.4e} max {rot_errs.max():.4e}"
    )
    print(f"ik_failures {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
