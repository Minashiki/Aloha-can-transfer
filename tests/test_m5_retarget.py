"""Synthetic retarget: frames, lead, gripper, and the 0.5 v_max slowdown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aloha_pp.data.frames import BIN1_POS, can_target_center
from aloha_pp.control.quintic import simulate_stream
from aloha_pp.control.reference import quintic_blend
from aloha_pp.data.retarget import (
    PHASE_CARRY,
    PHASE_GRIP_DWELL,
    PHASE_GRIP_RAMP,
    PHASE_GRIP_WAIT,
    PHASE_HUMAN_PREGRASP,
    PHASE_PLACE_SLIDE,
    PHASE_SYNTH_APPROACH,
    grasp_release,
    insert_gripper_waits,
    object_centric_positions,
    prepend_home,
    retarget_demo,
    smooth_segment_scales,
    unbraked_preview_peak,
)
from aloha_pp.kin.ik import DlsIk
from aloha_pp.robot import scene_xml
from aloha_pp.sim.env import load_config


def _configs():
    m1 = load_config()
    with (Path(__file__).resolve().parents[1] / "configs" / "m5_retarget.yaml").open(encoding="utf-8") as handle:
        m5 = yaml.safe_load(handle)
    return m1, m5


def _synthetic():
    """A can that stays at the start-bin center, and a hand that carries it to the cell."""
    n = 25
    can = np.repeat((BIN1_POS + np.array([0.0, 0.0, 0.05])).reshape(1, 3), n, axis=0)
    cell = can_target_center()
    eef = np.zeros((n, 3))
    # Approach from 15 cm above the can down to a 2 cm grasp offset.
    for t in range(6):
        eef[t] = can[0] + np.array([0.0, 0.0, 0.15 - 0.13 * t / 5.0])
    grasp = 5
    eef[grasp] = can[0] + np.array([0.0, 0.0, 0.02])
    release = 18
    for t in range(grasp, release + 1):
        u = (t - grasp) / (release - grasp)
        eef[t] = (1.0 - u) * eef[grasp] + u * (cell + np.array([0.0, 0.01, 0.12]))
        eef[t, 2] += 0.04 * np.sin(np.pi * u)
    for t in range(release, n):
        eef[t] = cell + np.array([0.0, 0.01, 0.12 + 0.02 * (t - release)])
    opening = np.full(n, 0.08)
    opening[grasp:release] = 0.01
    grip = np.stack((opening / 2.0, -opening / 2.0), axis=1)
    return eef, can, grip, grasp, release


def test_segments_frames_and_action_lead() -> None:
    m1, m5 = _configs()
    eef, can, grip, grasp, release = _synthetic()
    result = retarget_demo(eef, can, grip, m1=m1, m5=m5)
    low = np.asarray(m1["random_can"]["xy_low"])
    high = np.asarray(m1["random_can"]["xy_high"])
    assert np.all(result["can_pos"][:2] >= low - 1e-9)
    assert np.all(result["can_pos"][:2] <= high + 1e-9)
    # Before the inserted hold, the approach is the can-frame offset on our can.
    # Recompute the geometric path and check it directly.
    cell_new = np.asarray(m1["target_cells"]["fixed_1"]["center"], dtype=float)
    pos, phase = object_centric_positions(
        eef,
        can,
        grasp=grasp,
        release=release,
        can_new=result["can_pos"],
        cell_src=can_target_center(),
        cell_new=cell_new,
        carry_clearance_z=float(m5["carry_clearance_z"]),
    )
    assert phase[0] == PHASE_HUMAN_PREGRASP
    assert phase[grasp + 1] == PHASE_CARRY
    assert phase[-1] == PHASE_PLACE_SLIDE
    np.testing.assert_allclose(pos[2], result["can_pos"] + (eef[2] - can[2]))
    np.testing.assert_allclose(pos[-1], cell_new + (eef[-1] - can_target_center()))
    # Carry endpoints meet the approach and the place formulas.
    np.testing.assert_allclose(pos[grasp], result["can_pos"] + (eef[grasp] - can[grasp]))
    np.testing.assert_allclose(pos[release], cell_new + (eef[release] - can_target_center()))
    # Mid-carry is at least the clearance height.
    assert float(np.max(pos[grasp + 1 : release, 2])) >= float(m5["carry_clearance_z"]) - 1e-9
    actions = result["actions"]
    np.testing.assert_allclose(actions[:-1, :3], result["pos"][1:], atol=1e-8)
    np.testing.assert_allclose(actions[-1], actions[-2])
    assert np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1.0)
    # Gripper commands are the two openings, or the linear ramp between them.
    open_g = float(m1["gripper_open"])
    close_g = float(m1["gripper_close"])
    assert np.all(result["grip"] <= open_g + 1e-8)
    assert np.all(result["grip"] >= close_g - 1e-8)
    assert np.any(np.isclose(result["grip"], open_g))
    assert np.any(np.isclose(result["grip"], close_g))
    # The gripper stays closed through the inserted descent, and opens at the cell.
    cell = np.asarray(m1["target_cells"]["fixed_1"]["center"], dtype=float)
    closed = np.isclose(result["grip"], close_g)
    last_closed = int(np.where(closed)[0][-1])
    placed = result["pos"][last_closed]
    assert abs(placed[0] - cell[0]) < 0.03
    assert abs(placed[1] - cell[1]) < 0.03
    assert placed[2] < 0.10


def test_short_close_is_not_the_grasp() -> None:
    opening = np.full(40, 0.08)
    opening[:3] = 0.04
    opening[20:35] = 0.04
    grasp, release = grasp_release(opening, 0.055, min_frames=8)
    assert grasp == 20
    assert release == 35


def test_gripper_wait_then_ramp() -> None:
    pos = np.zeros((8, 3))
    pos[:, 0] = np.arange(8)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (8, 1))
    phase = np.zeros(8, dtype=np.int8)
    out_pos, _, grip, _, grasp_i, release_i = insert_gripper_waits(
        pos,
        quat,
        phase,
        grasp=2,
        release=5,
        gripper_open=0.037,
        gripper_close=0.015,
        wait_ticks=2,
        ramp_ticks=2,
        dwell_close=1,
        dwell_open=1,
    )
    # Two open holds, then a ramp that finishes on the close command.
    assert np.all(grip[grasp_i - 2 : grasp_i] == 0.037)
    assert grip[grasp_i + 1] == 0.015
    assert grip[release_i - 1] == 0.015
    assert grip[release_i + 1] == 0.037
    # The grasp pose (x = 2) is where the ramp happens.
    assert out_pos[grasp_i, 0] == 2.0


def test_time_scale_respects_half_speed() -> None:
    import mujoco

    m1, m5 = _configs()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    ik = DlsIk(
        model,
        m1["tcp_site"],
        damping=float(m1["ik"]["damping"]),
        max_iters=int(m1["ik"]["max_iters"]),
        pos_tol=float(m1["ik"]["pos_tol"]),
        rot_tol=float(m1["ik"]["rot_tol"]),
        rot_weight=float(m1["ik"]["rot_weight"]),
    )
    home = np.asarray(m1["home_q"], dtype=float)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home
    mujoco.mj_kinematics(model, data)
    home_pos, home_quat = ik.site_pose_in_base(data)
    eef, can, grip, _, _ = _synthetic()
    # Park the synthetic path on the home TCP so IK can reach it, keeping the shape.
    eef = eef - eef[0] + np.array([0.30, 0.12, 0.16])
    can = can - can[0] + np.array([0.30, 0.12, 0.14])
    # The source can is no longer at the robosuite bin, so skip retarget_demo's
    # bin map and call slow_until on an already-local fast path.
    from aloha_pp.data.retarget import constant_quat, slow_until

    pos = np.vstack((home_pos, home_pos + np.array([0.04, 0.04, -0.04]), home_pos))
    quat = constant_quat(0.0, len(pos))
    # Repeat the corners only a few times so the joint step per tick is large.
    pos = np.repeat(pos, 2, axis=0)
    quat = np.repeat(quat, 2, axis=0)
    grip_cmd = np.full(len(pos), 0.037)
    phase = np.zeros(len(pos), dtype=np.int8)
    v_max = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    scaled = slow_until(
        ik, pos, quat, grip_cmd, phase, home, v_max, 1.0 / 20.0,
        ratio=0.5, n_substeps=25, home_pos=home_pos, home_quat=home_quat,
    )
    assert float(np.max(scaled["ik_pos_err"])) < 0.01
    assert scaled["speed_ok"]
    assert np.all(scaled["ref_peak"] <= 0.5 * v_max)
    assert not np.any(scaled["ref_flags"])
    assert scaled["scale"] >= 1.0
    assert len(scaled["pos"]) >= len(pos)
    speeds, _acc, flags = simulate_stream(
        scaled["q"], home, control_dt=1.0 / 20.0, n_substeps=25, v_max=v_max, scheme="preview"
    )
    assert not np.any(flags)
    assert np.all(np.max(np.abs(speeds), axis=0) <= 0.5 * v_max)


def test_prepend_starts_at_home_with_quintic() -> None:
    pos = np.array([[0.30, 0.10, 0.20], [0.30, 0.10, 0.25]])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    grip = np.full(2, 0.037)
    phase = np.zeros(2, dtype=np.int8)
    home = np.array([0.20, 0.00, 0.30])
    home_quat = np.array([1.0, 0.0, 0.0, 0.0])
    out_pos, _quat, _grip, out_phase = prepend_home(pos, quat, grip, phase, home, home_quat, 10)
    np.testing.assert_allclose(out_pos[0], home)
    blend = quintic_blend(0.2)
    expected = (1.0 - blend) * home + blend * pos[0]
    np.testing.assert_allclose(out_pos[2], expected)
    linear = 0.8 * home + 0.2 * pos[0]
    assert float(np.linalg.norm(out_pos[2] - home)) < float(np.linalg.norm(linear - home))
    assert out_phase[0] == PHASE_SYNTH_APPROACH
    assert out_phase[10] == 0


def test_event_index_follows_the_close_ramp() -> None:
    import mujoco

    m1, m5 = _configs()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    ik = DlsIk(
        model, m1["tcp_site"], damping=0.05, max_iters=4, pos_tol=1e-4, rot_tol=1e-3, rot_weight=0.1
    )
    home = np.asarray(m1["home_q"], dtype=float)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home
    mujoco.mj_kinematics(model, data)
    home_pos, home_quat = ik.site_pose_in_base(data)
    eef, can, grip, _, _ = _synthetic()
    result = retarget_demo(eef, can, grip, m1=m1, m5=m5, home_pos=home_pos, home_quat=home_quat)
    np.testing.assert_allclose(result["pos"][0], home_pos, atol=1e-8)
    index = result["grasp_index"]
    assert result["phase"][index] == PHASE_GRIP_RAMP
    assert result["phase"][index - 1] == PHASE_GRIP_WAIT
    assert len(result["phase"]) == len(result["actions"])


def test_unbraked_peak_matches_unlimited_stream() -> None:
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=6) * 0.05
    q = q0 + np.cumsum(rng.normal(size=(24, 6)) * 0.001, axis=0)
    peak = unbraked_preview_peak(q, q0, control_dt=0.05, n_substeps=25)
    huge = np.full(6, 1.0e6)
    speeds, _acc, flags = simulate_stream(
        q, q0, control_dt=0.05, n_substeps=25, v_max=huge, scheme="preview"
    )
    np.testing.assert_allclose(peak, np.max(np.abs(speeds), axis=0), atol=1e-8)
    assert not np.any(flags)


def test_boundary_scale_follows_the_quintic() -> None:
    phase = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    sigma = smooth_segment_scales(phase, {0: 1.0, 1: 4.0}, blend_ticks=4)
    # Half of a length-10 run is 5, so the window is the requested 4 samples.
    np.testing.assert_allclose(sigma[:6], 1.0)
    np.testing.assert_allclose(sigma[14:], 4.0)
    indices = list(range(6, 14))
    for step, index in enumerate(indices):
        blend = quintic_blend(step / (len(indices) - 1))
        np.testing.assert_allclose(sigma[index], (1.0 - blend) * 1.0 + blend * 4.0)
    # The quintic leaves the flat region more slowly than a linear blend.
    linear = 1.0 + (4.0 - 1.0) * (1 / 7)
    assert float(sigma[7] - sigma[6]) < linear - 1.0


def test_an_early_slowdown_does_not_stretch_later_segments() -> None:
    from aloha_pp.data.retarget import _resample_profile

    phase = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
    sigma = np.array([4.0] * 10 + [1.0] * 20)
    pos = np.zeros((30, 3))
    pos[:, 0] = np.arange(30)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (30, 1))
    grip = np.zeros(30)
    _pos, _quat, _grip, out_phase, _cumulative = _resample_profile(
        pos, quat, grip, phase, sigma, [9, 19, 29]
    )
    assert int(np.count_nonzero(out_phase == 0)) > 30
    assert int(np.count_nonzero(out_phase == 1)) < 15
    assert int(np.count_nonzero(out_phase == 2)) < 15


def test_gripper_duration_ignores_the_slowdown() -> None:
    import mujoco

    from aloha_pp.data.retarget import constant_quat, slow_until

    m1, m5 = _configs()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    ik = DlsIk(
        model, m1["tcp_site"], damping=0.05, max_iters=8, pos_tol=1e-4, rot_tol=1e-3, rot_weight=0.1
    )
    home = np.asarray(m1["home_q"], dtype=float)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home
    mujoco.mj_kinematics(model, data)
    home_pos, home_quat = ik.site_pose_in_base(data)
    # A slow approach, then one fast carry step, then a slow retreat.
    approach = np.repeat(home_pos.reshape(1, 3), 8, axis=0)
    carry = home_pos + np.array([0.05, 0.04, -0.03])
    retreat = np.repeat(home_pos.reshape(1, 3), 6, axis=0)
    pos = np.vstack((approach, carry.reshape(1, 3), retreat))
    phase = np.array(
        [PHASE_SYNTH_APPROACH] * 8 + [PHASE_CARRY] + [PHASE_PLACE_SLIDE] * 6,
        dtype=np.int8,
    )
    quat = constant_quat(0.0, len(pos))
    grip = np.full(len(pos), float(m1["gripper_open"]))
    dt = 1.0 / 20.0
    wait = 1 + int(round(0.25 / dt))
    ramp = int(m5["grip_ramp_ticks"])
    dwell_close = int(round(0.40 / dt))
    dwell_open = int(round(0.20 / dt))
    v_max = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    scaled = slow_until(
        ik, pos, quat, grip, phase, home, v_max, dt,
        ratio=0.5, n_substeps=25, home_pos=home_pos, home_quat=home_quat,
        blend_ticks=int(m5["scale_blend_ticks"]),
        grasp=8, release=9,
        gripper={
            "gripper_open": float(m1["gripper_open"]),
            "gripper_close": float(m1["gripper_close"]),
            "wait_ticks": wait,
            "ramp_ticks": ramp,
            "dwell_close": dwell_close,
            "dwell_open": dwell_open,
        },
    )
    assert scaled["speed_ok"]
    assert np.all(scaled["ref_peak"] <= 0.5 * v_max + 1e-12)
    assert not np.any(scaled["ref_flags"])
    labels = scaled["phase"]
    assert int(np.count_nonzero(labels == PHASE_GRIP_RAMP)) == 2 * ramp
    assert int(np.count_nonzero(labels == PHASE_GRIP_WAIT)) == 2 * wait
    assert int(np.count_nonzero(labels == PHASE_GRIP_DWELL)) == dwell_close + dwell_open
    assert float(scaled["segment_scales"]["carry"]) > float(scaled["segment_scales"]["synth_approach"])
    assert scaled["phase"][scaled["grasp_index"]] == PHASE_GRIP_RAMP


def test_speed_check_is_not_capped_at_four(monkeypatch) -> None:
    import mujoco

    from aloha_pp.data.retarget import _limited_peak, slow_until

    calls = {"n": 0}
    real = _limited_peak

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        peak, flags = real(*args, **kwargs)
        if calls["n"] < 6:
            return peak * 2.0, np.ones_like(flags)
        return peak, flags

    monkeypatch.setattr("aloha_pp.data.retarget._limited_peak", wrapped)
    m1, _m5 = _configs()
    model = mujoco.MjModel.from_xml_path(str(scene_xml("position")))
    ik = DlsIk(
        model, m1["tcp_site"], damping=0.05, max_iters=8, pos_tol=1e-4, rot_tol=1e-3, rot_weight=0.1
    )
    home = np.asarray(m1["home_q"], dtype=float)
    data = mujoco.MjData(model)
    data.qpos[ik.qpos_adr] = home
    mujoco.mj_kinematics(model, data)
    home_pos, home_quat = ik.site_pose_in_base(data)
    from aloha_pp.data.retarget import constant_quat

    pos = np.vstack((home_pos, home_pos + np.array([0.03, 0.02, -0.02]), home_pos))
    pos = np.repeat(pos, 3, axis=0)
    quat = constant_quat(0.0, len(pos))
    grip = np.full(len(pos), 0.037)
    phase = np.zeros(len(pos), dtype=np.int8)
    v_max = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    scaled = slow_until(
        ik, pos, quat, grip, phase, home, v_max, 0.05,
        ratio=0.5, n_substeps=25, home_pos=home_pos, home_quat=home_quat,
    )
    assert calls["n"] >= 6
    assert scaled["speed_ok"]
    assert np.all(scaled["ref_peak"] <= 0.5 * v_max)
