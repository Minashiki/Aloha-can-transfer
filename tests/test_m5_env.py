"""Optional reset arguments. The default call is unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from aloha_pp.sim.env import AlohaCanEnv, load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from m4_run import _controller, _gains, _motor_cfg  # noqa: E402


def test_default_reset_ignores_absent_overrides() -> None:
    env = AlohaCanEnv()
    try:
        env.reset(seed=4, mode="fixed_1", render=False)
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        can = env.can_center_base().copy()
        env.reset(seed=4, mode="fixed_1", render=False, can_pose=None, state=None)
        np.testing.assert_allclose(env.data.qpos, qpos)
        np.testing.assert_allclose(env.data.qvel, qvel)
        np.testing.assert_allclose(env.can_center_base(), can)
    finally:
        env.close()


def test_can_pose_and_state_restore() -> None:
    env = AlohaCanEnv()
    try:
        requested = np.array([0.31, 0.14, 0.03663, 1.0, 0.0, 0.0, 0.0])
        env.reset(seed=5, mode="random", can_pose=requested, render=False)
        # Settle can shift the can by a few millimetres. It should stay near the request.
        np.testing.assert_allclose(env.can_center_base()[:2], requested[:2], atol=0.01)
        state = env.capture_state()
        assert "qacc_warmstart" in state
        assert state["time"] > 0.0
        env.reset(seed=0, mode="fixed_2", render=False)
        env.reset(mode="random", state=state, render=False)
        np.testing.assert_allclose(env.data.qpos, state["qpos"])
        np.testing.assert_allclose(env.data.qvel, state["qvel"])
        np.testing.assert_allclose(env.data.ctrl, state["ctrl"])
        np.testing.assert_allclose(env.data.qacc_warmstart, state["qacc_warmstart"])
        assert env.data.time == state["time"]
    finally:
        env.close()


def test_stored_action_replay_matches_bit_exact() -> None:
    """Same initial state and the same stored actions replay to a zero qpos diff."""
    root = Path(__file__).resolve().parents[1]
    m1 = load_config()
    with (root / "configs" / "m3_track.yaml").open(encoding="utf-8") as handle:
        m3 = yaml.safe_load(handle)
    with (root / "configs" / "m4_task.yaml").open(encoding="utf-8") as handle:
        m4 = yaml.safe_load(handle)
    env = AlohaCanEnv(_motor_cfg(m1))
    env.controller = _controller(env, m3, *_gains(m3, m4))
    try:
        env.reset(seed=3, mode="random", render=False)
        state = env.capture_state()
        env.controller.reset(env.model, env.data)
        tcp = env.get_obs(render=False)["tcp_pose"]
        actions = []
        for k in range(8):
            command = np.array(tcp, dtype=np.float64)
            command[0] += 0.004 * (k + 1)
            command[2] += 0.002 * (k + 1)
            command = np.concatenate((command[:7], np.array([float(m1["gripper_open"])])))
            actions.append(np.asarray(np.asarray(command, dtype=np.float32), dtype=np.float64))
        actions = np.stack(actions)
        nxt = np.vstack((actions[1:], actions[-1:]))

        def run() -> np.ndarray:
            rows = []
            for t in range(len(actions)):
                rows.append(env.data.qpos.copy())
                env.step(actions[t], render=False, next_action=nxt[t])
            return np.stack(rows)

        first = run()
        env.reset(mode="random", state=state, render=False)
        second = run()
    finally:
        env.close()
    assert float(np.max(np.abs(first - second))) == 0.0
    assert np.array_equal(first.astype(np.float32), second.astype(np.float32))
