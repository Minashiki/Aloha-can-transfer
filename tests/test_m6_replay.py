"""The shared C3 env replays stored demonstrations.

The pixel check (maximum absolute difference at most 1) is a property of this
machine's GPU renderer. It was measured with MuJoCo's EGL backend on an
NVIDIA GeForce RTX 5080. Another GL driver can disagree by much more than one
level; do not loosen the threshold to make a different machine pass.
"""

from __future__ import annotations

import numpy as np

from aloha_pp.data.dataset import list_demos, load_episode
from aloha_pp.policy.rollout import task_configs
from aloha_pp.robot import repo_root
from aloha_pp.sim.c3 import make_c3_env, reset_controller_after_settle
from aloha_pp.sim.env import AlohaCanEnv


def _state(episode: dict) -> dict:
    return {
        "qpos": episode["qpos0"],
        "qvel": episode["qvel0"],
        "ctrl": episode["ctrl0"],
        "time": episode["time0"],
        "qacc_warmstart": episode["qacc_warmstart0"],
        "qacc": episode["qacc0"],
    }


def _match(obs: dict, episode: dict, t: int) -> None:
    """Low-dimensional observations are the stored float32 values.

    Offscreen pixels can move by one level between renders of the same state.
    A larger change would mean the camera or the state diverged.
    """
    stored = episode["obs"]
    for camera in ("agentview", "wrist_cam_left"):
        delta = np.abs(obs[camera].astype(np.int16) - stored[camera][t].astype(np.int16))
        worst = int(delta.max())
        assert worst <= 1, f"{camera} t={t} max pixel delta {worst}"
    np.testing.assert_array_equal(np.asarray(obs["tcp_pose"], dtype=np.float32), stored["tcp_pose"][t])
    np.testing.assert_array_equal(
        np.asarray(obs["gripper_opening"], dtype=np.float32).reshape(1),
        stored["gripper_opening"][t].reshape(1),
    )
    np.testing.assert_array_equal(np.asarray(obs["q"], dtype=np.float32), stored["q"][t])
    np.testing.assert_array_equal(np.asarray(obs["qvel"], dtype=np.float32), stored["qvel"][t])


def test_two_demos_replay_successfully_and_match_observations() -> None:
    m1, m3, m4, _m5 = task_configs()
    path = repo_root() / "outputs/m5/dataset/demos.hdf5"
    names = list_demos(path)[:2]
    env = make_c3_env(m1, m3, m4)
    try:
        assert env.controller.kind == "C3"
        for name in names:
            episode = load_episode(path, name)
            obs = env.reset(mode="random", state=_state(episode), render=True)
            success = False
            actions = episode["actions"]
            nxt = episode["next_actions"]
            for t in range(len(actions)):
                _match(obs, episode, t)
                obs, success, _info = env.step(
                    np.asarray(actions[t], dtype=np.float64),
                    render=True,
                    next_action=np.asarray(nxt[t], dtype=np.float64),
                )
            assert success, name
    finally:
        env.close()


def test_renderer_cache_keeps_each_resolution() -> None:
    with AlohaCanEnv() as env:
        env.reset(seed=0, render=False)
        small = env.render("agentview")
        large = env.render("agentview", 256, 256)
        assert small.shape == (96, 96, 3)
        assert large.shape == (256, 256, 3)
        assert set(env._renderers) == {(96, 96), (256, 256)}
        again = env.render("agentview")
        assert int(np.abs(again.astype(np.int16) - small.astype(np.int16)).max()) == 0
        assert env._get_renderer(96, 96) is env._renderers[(96, 96)]
        assert env._get_renderer(256, 256) is env._renderers[(256, 256)]
    assert env._renderer is None
    assert env._renderers == {}


def _controller_view(env) -> dict:
    ctrl = env.controller
    return {
        "home_q": np.array(ctrl.home_q, dtype=float, copy=True),
        "seed": np.array(ctrl._seed, dtype=float, copy=True),
        "stream_q": np.array(ctrl.stream.q, dtype=float, copy=True),
        "stream_v": np.array(ctrl.stream.v, dtype=float, copy=True),
        "reference_mode": ctrl.reference_mode,
        "arm_q": np.array(env.data.qpos[env.arm_qpos_adr], dtype=float, copy=True),
    }


def test_eval_reset_matches_m5_controller_initial_state() -> None:
    """M6 calls the same post-settle controller.reset that M5 uses.

    ``env.reset`` itself is left as it is: it still resets the controller
    before the settle loop.
    """
    m1, m3, m4, _m5 = task_configs()
    reference = make_c3_env(m1, m3, m4)
    followed = make_c3_env(m1, m3, m4)
    try:
        reference.reset(seed=3, mode="random", render=False)
        reference.controller.reset(reference.model, reference.data)
        followed.reset(seed=3, mode="random", render=False)
        reset_controller_after_settle(followed)
        left = _controller_view(reference)
        right = _controller_view(followed)
        for key in ("home_q", "seed", "stream_q", "stream_v", "arm_q"):
            np.testing.assert_array_equal(left[key], right[key])
        assert left["reference_mode"] == right["reference_mode"] == "wait"
        np.testing.assert_array_equal(right["home_q"], right["arm_q"])
    finally:
        reference.close()
        followed.close()
