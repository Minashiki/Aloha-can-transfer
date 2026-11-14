"""DSRL noise bound, frozen features, chunk env, SAC, and seed separation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch import nn

from aloha_pp.policy.checkpoint import policy_from_config
from aloha_pp.policy.normalizer import LimitNormalizer
from aloha_pp.policy.quaternion import to_env_action
from aloha_pp.policy.rollout import rollout_episode, task_configs
from aloha_pp.policy.checkpoint import rng_state
from aloha_pp.rl.checkpoint import assert_resume_config, build_models, load_actor, restore_rng, save_training
from aloha_pp.rl.chunk_env import ChunkEnv, make_chunk_env
from aloha_pp.rl.compare import build_report, mcnemar_p, transitions
from aloha_pp.rl.evaluate import chunk_noise_fn, evaluate_latent
from aloha_pp.rl.nets import SquashedGaussianActor, TwinCritic
from aloha_pp.rl.trainer import _validate
from aloha_pp.rl.noise import clip_standard_normal, seeded_generator, zero_noise
from aloha_pp.rl.replay import Replay
from aloha_pp.rl.reward import StageTracker, step_reward, td_target
from aloha_pp.rl.sac import SAC, alpha_loss, soft_update
from aloha_pp.rl.spec import (
    ACTOR_FORBIDDEN_FIELDS,
    ACTOR_OBS_KEYS,
    PRIVILEGED_FIELDS,
    SeedStream,
    assert_seed_ranges,
    eval_seed_set,
    pack_privileged,
    privileged_dim,
    val_seed_set,
)
from aloha_pp.rl.vec import SubprocChunkEnv
from aloha_pp.robot import repo_root
from aloha_pp.sim.c3 import make_c3_env


def _small_cfg() -> dict:
    return {
        "n_obs_steps": 2,
        "pred_horizon": 16,
        "exec_horizon": 8,
        "action_dim": 8,
        "state_dim": 8,
        "image_size": 32,
        "denoising_steps": 10,
        "ddim_steps": 4,
        "denoised_clip": 1.0,
        "ema_decay": 0.995,
        "augment": True,
        "shift_pad": 4,
        "vit": {"depth": 1, "embed_dim": 32, "num_heads": 4, "embed_norm": 0},
        "unet": {
            "diffusion_step_embed_dim": 16,
            "dim": 32,
            "dim_mults": [1, 2],
            "kernel_size": 5,
            "n_groups": 8,
            "cond_predict_scale": True,
            "spatial_emb": 32,
            "smaller_encoder": True,
            "dropout": 0.0,
        },
    }


def _obs(image: int = 32) -> dict:
    tcp = np.zeros((2, 7), dtype=np.float32)
    tcp[:, 3] = 1.0
    return {
        "agentview": np.zeros((2, image, image, 3), dtype=np.uint8),
        "wrist_cam_left": np.zeros((2, image, image, 3), dtype=np.uint8),
        "tcp_pose": tcp,
        "gripper_opening": np.full((2, 1), 0.02, dtype=np.float32),
    }


def _policy():
    rng = np.random.default_rng(0)
    action = LimitNormalizer().fit(rng.normal(size=(8, 8)).astype(np.float32))
    state = LimitNormalizer().fit(rng.normal(size=(8, 8)).astype(np.float32))
    return policy_from_config(_small_cfg(), action, state)


def _home_chunk(gripper: float = 0.037) -> np.ndarray:
    chunk = np.zeros((16, 8), dtype=np.float32)
    chunk[:, 0] = 0.30
    chunk[:, 2] = 0.20
    chunk[:, 3] = 1.0
    chunk[:, 7] = np.float32(gripper)
    return chunk


def _rewards() -> dict:
    return {
        "both_contact": 0.1,
        "lifted": 0.2,
        "above_cell": 0.3,
        "released_in_cell": 0.4,
    }


def _staged(enabled: bool) -> dict:
    return {"enabled": enabled, **_rewards()}


def _cell():
    return np.array([0.336, -0.08, 0.03604]), np.array([0.0324, 0.0414, 0.02])


class _Controller:
    def reset(self, model, data) -> None:
        del model, data


class _Fake:
    def __init__(self, succeed_at: int | None) -> None:
        self.cfg = {"control_hz": 20.0}
        self.model = None
        self.data = None
        self.controller = _Controller()
        self.success = False
        self.n = 0
        self.succeed_at = succeed_at

    def reset(self, seed=None, mode="random", render=True):
        del mode, render
        self.n = 0
        self.success = False
        return self.get_obs()

    def step(self, action, on_substep=None, render=True, next_action=None):
        self.n += 1
        self.last_action = np.asarray(action)
        self.last_next = np.asarray(next_action)
        if on_substep is not None:
            on_substep(self)
        if self.succeed_at is not None and self.n >= self.succeed_at:
            self.success = True
        return self.get_obs(), self.success, {}

    def get_obs(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return {
            "agentview": image,
            "wrist_cam_left": image.copy(),
            "tcp_pose": np.array([0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            "gripper_opening": np.array([0.037]),
            "q": np.zeros(6),
            "qvel": np.zeros(6),
        }

    def get_privileged(self):
        return {"can_pose": np.zeros(7), "can_linvel": np.zeros(3)}

    def finger_can_contacts(self):
        return np.array([True, True])

    def can_center_base(self):
        return np.zeros(3)

    def close(self):
        return None


def _wrap(fake, *, timeout_s: float, staged: bool, trace: bool = False) -> ChunkEnv:
    center, half = _cell()
    return ChunkEnv(
        fake,
        n_obs=2,
        pred_horizon=16,
        exec_horizon=8,
        action_dim=8,
        timeout_s=timeout_s,
        stages=StageTracker(
            enabled=staged,
            rewards=_rewards(),
            lift_thresh=0.02,
            cell_center=center,
            cell_half=half,
        ),
        gripper_close=0.015,
        grip_tol=1e-4,
        trace=trace,
    )


def test_encode_does_not_change_sample_and_ignores_privileged_keys() -> None:
    policy = _policy()
    policy.train()
    obs = _obs()
    noise = torch.randn(1, 16, 8)
    first = policy.sample(obs, noise=noise.clone())
    feat = policy.encode_actor_obs(obs)
    second = policy.sample(obs, noise=noise.clone())
    assert torch.equal(first, second)
    assert feat.shape == (1, policy.actor_feature_dim())
    assert policy.actor_feature_dim() == 32 * 2 + 8 * 2
    again = policy.encode_actor_obs(obs)
    assert torch.equal(feat, again)
    dirty = dict(obs)
    dirty["can_pose"] = np.ones((2, 7), dtype=np.float32)
    dirty["can_linvel"] = np.ones((2, 3), dtype=np.float32)
    dirty["q"] = np.ones((2, 6), dtype=np.float32)
    dirty["qvel"] = np.ones((2, 6), dtype=np.float32)
    assert torch.equal(feat, policy.encode_actor_obs(dirty))
    third = policy.sample(obs, noise=noise.clone())
    assert torch.equal(first, third)


def test_actor_fields_exclude_object_truth_and_critic_includes_it() -> None:
    assert set(ACTOR_FORBIDDEN_FIELDS).isdisjoint(ACTOR_OBS_KEYS)
    names = [name for name, _width in PRIVILEGED_FIELDS]
    for name in ACTOR_FORBIDDEN_FIELDS:
        assert name in names
    assert privileged_dim() == 30
    obs = {
        "tcp_pose": np.arange(7, dtype=np.float32),
        "gripper_opening": np.array([0.2], dtype=np.float32),
        "q": np.arange(6, dtype=np.float32),
        "qvel": np.arange(6, dtype=np.float32) + 10,
    }
    privileged = {"can_pose": np.arange(7, dtype=np.float32) + 20, "can_linvel": np.array([1, 2, 3], dtype=np.float32)}
    packed = pack_privileged(obs, privileged)
    assert packed.shape == (30,)
    assert np.allclose(packed[:7], np.arange(7) + 20)
    assert np.allclose(packed[7:10], [1, 2, 3])
    actor = SquashedGaussianActor(8, 4, hidden=8, n_layers=2, noise_scale=1.5)
    critic = TwinCritic(privileged_dim(), 8, 4, hidden=8, n_layers=2)
    assert actor.trunk[0].in_features == 8
    assert critic.priv_dim == 30
    assert critic.in_dim == 30 + 8 + 4
    assert critic.trunk1[0].in_features == critic.in_dim


def test_tanh_noise_stays_inside_the_scale() -> None:
    actor = SquashedGaussianActor(4, 6, hidden=8, n_layers=2, noise_scale=1.5)
    feat = torch.randn(5, 4)
    action, _logp = actor.sample(feat)
    assert torch.all(action.abs() <= 1.5 + 1e-5)
    assert torch.all(actor.deterministic(feat).abs() <= 1.5 + 1e-5)
    actor.mean_head.weight.data.zero_()
    actor.mean_head.bias.data.fill_(40.0)
    saturated = actor.deterministic(torch.zeros(2, 4))
    assert torch.allclose(saturated, torch.full_like(saturated, 1.5), atol=1e-4)
    clipped = clip_standard_normal(np.random.default_rng(0), (1000, 8), 1.5)
    assert np.all(np.abs(clipped) <= 1.5)


def test_same_noise_makes_the_same_chunk() -> None:
    policy = _policy()
    obs = _obs()
    noise = torch.randn(1, 16, 8)
    assert torch.equal(policy.sample(obs, noise=noise.clone()), policy.sample(obs, noise=noise.clone()))


def test_chunk_is_eight_control_steps_and_previews_the_ninth() -> None:
    chunk = _home_chunk()
    env = make_chunk_env(timeout_s=0.4, staged=_staged(False), trace=True)
    try:
        env.reset(seed=1)
        step = env.step(chunk)
        assert step["info"]["n_control"] == 8
        assert step["truncated"] and not step["terminated"]
        assert step["reward"] == 0.0
        assert np.allclose(step["info"]["actions"][7], to_env_action(chunk[7]))
        assert np.allclose(step["info"]["next_actions"][7], to_env_action(chunk[8]))
    finally:
        env.close()
    longer = make_chunk_env(timeout_s=0.5, staged=_staged(False), trace=True)
    try:
        longer.reset(seed=2)
        first = longer.step(chunk)
        second = longer.step(chunk)
        assert first["info"]["n_control"] == 8 and not first["truncated"] and not first["terminated"]
        assert second["info"]["n_control"] == 2 and second["truncated"] and not second["terminated"]
        assert first["reward"] == 0.0 and second["reward"] == 0.0
        assert first["info"]["n_control"] + second["info"]["n_control"] == 10
    finally:
        longer.close()


def test_success_is_terminal_and_timeout_is_truncated() -> None:
    success = _wrap(_Fake(succeed_at=3), timeout_s=5.0, staged=False, trace=False)
    success.reset(seed=0)
    step = success.step(_home_chunk())
    assert step["info"]["n_control"] == 3
    assert step["terminated"] and not step["truncated"]
    assert step["reward"] == 1.0
    try:
        success.step(_home_chunk())
    except RuntimeError:
        pass
    else:
        raise AssertionError("a finished episode must be reset before the next step")
    timeout = _wrap(_Fake(None), timeout_s=0.4, staged=False)
    timeout.reset(seed=0)
    done = timeout.step(_home_chunk())
    assert done["info"]["n_control"] == 8
    assert done["truncated"] and not done["terminated"]
    assert done["reward"] == 0.0


def test_stage_bonus_is_once_per_episode_and_off_means_zero_or_one() -> None:
    center, half = _cell()
    tracker = StageTracker(enabled=True, rewards=_rewards(), lift_thresh=0.02, cell_center=center, cell_half=half)
    assert tracker.update(grasped_contact=True, contact_z=0.03, max_z_after_contact=0.03, end_pos=np.array([0.3, 0.1, 0.03]), end_released=False) == 0.1
    assert tracker.update(grasped_contact=False, contact_z=None, max_z_after_contact=0.03, end_pos=np.array([0.3, 0.1, 0.03]), end_released=False) == 0.0
    assert tracker.update(grasped_contact=True, contact_z=0.03, max_z_after_contact=0.03, end_pos=np.array([0.3, 0.1, 0.03]), end_released=False) == 0.0
    assert tracker.update(grasped_contact=True, contact_z=None, max_z_after_contact=0.06, end_pos=np.array([0.3, 0.1, 0.06]), end_released=False) == 0.2
    assert tracker.update(grasped_contact=False, contact_z=None, max_z_after_contact=0.03, end_pos=np.array([0.3, 0.1, 0.03]), end_released=False) == 0.0
    assert tracker.update(grasped_contact=False, contact_z=None, max_z_after_contact=0.08, end_pos=np.array([0.336, -0.08, 0.08]), end_released=False) == 0.3
    assert tracker.update(grasped_contact=False, contact_z=None, max_z_after_contact=0.04, end_pos=np.array([0.336, -0.08, 0.036]), end_released=True) == 0.4
    assert tracker.update(grasped_contact=False, contact_z=None, max_z_after_contact=0.04, end_pos=np.array([0.336, -0.08, 0.036]), end_released=True) == 0.0
    disabled = StageTracker(enabled=False, rewards=_rewards(), lift_thresh=0.02, cell_center=center, cell_half=half)
    assert disabled.update(grasped_contact=True, contact_z=0.03, max_z_after_contact=0.2, end_pos=np.array([0.336, -0.08, 0.036]), end_released=True) == 0.0
    assert not any(disabled.paid.values())
    assert step_reward(False, 5.0, staged=False) == 0.0
    assert step_reward(True, 5.0, staged=False) == 1.0
    env = _wrap(_Fake(None), timeout_s=5.0, staged=True)
    env.reset(seed=0)
    first = env.step(_home_chunk(0.015))
    second = env.step(_home_chunk(0.015))
    assert first["reward"] == 0.1
    assert second["reward"] == 0.0
    assert not first["terminated"] and not second["terminated"]


def test_parallel_matches_one_env_on_the_same_seed_and_chunk() -> None:
    setup = {
        "timeout_s": 2.0,
        "staged": _staged(False),
        "n_obs": 2,
        "pred_horizon": 16,
        "exec_horizon": 8,
        "action_dim": 8,
        "video_fps": 10,
        "trace": True,
    }
    seeds = [11, 12]
    chunk = _home_chunk()
    vec = SubprocChunkEnv(2, setup)
    try:
        vec.reset(seeds)
        out = vec.step(np.stack([chunk, chunk]))
    finally:
        vec.close()
    singles = []
    for seed in seeds:
        env = make_chunk_env(timeout_s=2.0, staged=_staged(False), trace=True)
        try:
            env.reset(seed=seed)
            singles.append(env.step(chunk))
        finally:
            env.close()
    for index, single in enumerate(singles):
        for camera in ("agentview", "wrist_cam_left"):
            delta = np.abs(out["obs"][camera][index].astype(np.int16) - single["obs"][camera].astype(np.int16))
            assert int(delta.max()) <= 1, f"{camera} max pixel delta {int(delta.max())}"
        for key in ("tcp", "q", "can_pose"):
            assert np.allclose(out["info"][index][key], single["info"][key], atol=1e-5, rtol=0.0)


def test_td_target_bootstraps_timeout_and_not_success() -> None:
    reward = torch.tensor([[0.0], [1.0]])
    terminated = torch.tensor([[0.0], [1.0]])
    nxt = torch.tensor([[5.0], [5.0]])
    assert torch.allclose(td_target(reward, terminated, 0.5, nxt), torch.tensor([[2.5], [1.0]]))


def test_critic_loss_falls_target_tracks_and_alpha_moves_the_right_way() -> None:
    torch.manual_seed(0)
    feat_dim, priv_dim, act_dim = 6, 4, 3
    actor = SquashedGaussianActor(feat_dim, act_dim, hidden=32, n_layers=2, noise_scale=1.0)
    critic = TwinCritic(priv_dim, feat_dim, act_dim, hidden=32, n_layers=2)
    agent = SAC(actor, critic, lr=3e-3, gamma=0.99, tau=0.005, target_entropy=0.0)
    rng = np.random.default_rng(0)
    n = 256
    action = rng.standard_normal((n, act_dim)).astype(np.float32)
    reward = action.sum(axis=1).astype(np.float32)
    feat = rng.standard_normal((n, feat_dim)).astype(np.float32)
    priv = rng.standard_normal((n, priv_dim)).astype(np.float32)
    replay = Replay(n, feat_dim, priv_dim, act_dim, rng=np.random.default_rng(1))
    replay.add(feat, priv, action, reward, feat, priv, np.ones(n, np.float32), np.zeros(n, np.float32))
    before = [parameter.detach().clone() for parameter in agent.target.parameters()]
    losses = [agent.update(replay.sample(128, "cpu"))["critic_loss"] for _ in range(60)]
    assert losses[-1] < losses[0] * 0.5
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.target.parameters()))
    online = nn.Linear(3, 1)
    target = nn.Linear(3, 1)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(1.0)
        for parameter in target.parameters():
            parameter.zero_()
    soft_update(online, target, 0.25)
    for parameter in target.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.25))
    peaked = nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([peaked], lr=0.2)
    loss = alpha_loss(peaked, torch.full((16, 1), 4.0), 0.0)
    loss.backward()
    opt.step()
    assert float(peaked.detach()) > 0.0
    flat = nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([flat], lr=0.2)
    loss = alpha_loss(flat, torch.full((16, 1), -4.0), 0.0)
    loss.backward()
    opt.step()
    assert float(flat.detach()) < 0.0


def test_replay_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    replay = Replay(4, 2, 3, 1, rng=np.random.default_rng(0))
    for index in range(6):
        replay.add(
            np.full((1, 2), index, np.float32),
            np.zeros((1, 3), np.float32),
            np.full((1, 1), index, np.float32),
            np.array([index], np.float32),
            np.full((1, 2), index, np.float32),
            np.zeros((1, 3), np.float32),
            np.array([0.0], np.float32),
            np.array([1.0], np.float32),
        )
    assert replay.size == 4
    assert set(replay.action[: replay.size, 0].tolist()) == {2.0, 3.0, 4.0, 5.0}
    path = tmp_path / "replay.pt"
    replay.save(path)
    loaded = Replay.load(path)
    assert loaded.size == 4
    assert np.array_equal(loaded.action[: loaded.size], replay.action[: replay.size])
    assert np.array_equal(loaded.truncated[: loaded.size], replay.truncated[: replay.size])
    cfg = {
        "noise_scale": 1.5,
        "sac": {
            "layer_size": 8,
            "num_layers": 2,
            "log_std_init": 0.0,
            "log_std_min": -20.0,
            "log_std_max": 2.0,
            "use_layer_norm": True,
            "actor_lr": 1e-3,
            "discount": 0.99,
            "tau": 0.005,
            "target_entropy": 0.0,
        },
    }
    actor, agent = build_models(4, 3, cfg, torch.device("cpu"))
    with torch.no_grad():
        actor.mean_head.bias.fill_(0.25)
    ckpt = tmp_path / "ckpt.pt"
    save_training(ckpt, actor=actor, agent=agent, cfg=cfg, extra={"feat_dim": 4, "action_dim": 3, "env_steps": 7})
    restored, blob = load_actor(ckpt, "cpu")
    assert torch.equal(restored.mean_head.bias, actor.mean_head.bias)
    assert blob["env_steps"] == 7


def test_restore_rng_accepts_a_cuda_byte_tensor() -> None:
    if not torch.cuda.is_available():
        return
    state = rng_state()
    state["torch"] = state["torch"].to(device="cuda")
    if state.get("cuda") is not None:
        state["cuda"] = [item.to(device="cuda") for item in state["cuda"]]
    restore_rng(state)
    assert torch.is_tensor(torch.get_rng_state())


def test_train_val_and_eval_seeds_are_disjoint() -> None:
    with (repo_root() / "configs" / "m7_dsrl.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["noise_scale"] == 1.5
    assert cfg["sac"]["utd"] == 20
    assert cfg["sac"]["layer_size"] == 2048
    assert cfg["sac"]["num_layers"] == 3
    assert cfg["sac"]["target_entropy"] == 0.0
    assert cfg["sac"]["discount"] == 0.99
    assert cfg["n_val_episodes"] == 16
    assert cfg["staged_reward"]["enabled"] is False
    assert cfg["checkpoint_select"] == "best_val"
    evaluation = eval_seed_set(cfg["eval_seed_start"], cfg["eval_seed_count"])
    validation = val_seed_set(cfg["val_seed_start"], cfg["n_val_episodes"])
    assert_seed_ranges(cfg["train_seed_start"], cfg["val_seed_start"], cfg["n_val_episodes"], cfg["eval_seed_start"], cfg["eval_seed_count"])
    assert evaluation.isdisjoint(validation)
    stream = SeedStream(int(cfg["train_seed_start"]), evaluation | validation)
    taken = stream.take(10020)
    assert set(taken).isdisjoint(evaluation)
    assert set(taken).isdisjoint(validation)
    assert 19999 in taken and 20000 not in taken and 20016 in taken
    assert transitions({0: False, 1: True, 2: False}, {0: True, 1: True, 2: False})["right_only"] == [0]


def test_bc_zero_is_zero_and_bc_seeded_repeats(tmp_path: Path) -> None:
    assert np.all(zero_noise((16, 8)) == 0)
    first = seeded_generator(4).standard_normal((2, 16, 8))
    second = seeded_generator(4).standard_normal((2, 16, 8))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, seeded_generator(5).standard_normal((2, 16, 8)))

    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []
            self.noise_shape = (16, 8)
            self.n_obs_steps = 2
            self.pred_horizon = 16
            self.exec_horizon = 8
            self.action_dim = 8
            self.action_lo = torch.zeros(8)

        def sample(self, obs, noise=None, ddim_steps=None):
            del obs, ddim_steps
            array = np.asarray(noise, dtype=np.float32).reshape(16, 8)
            self.seen.append(array.copy())
            chunk = np.zeros((1, 16, 8), dtype=np.float32)
            chunk[..., 0] = 0.30
            chunk[..., 1] = 0.02 * np.tanh(array[None, :, 0])
            chunk[..., 2] = 0.20
            chunk[..., 3] = 1.0
            chunk[..., 7] = 0.037
            return torch.as_tensor(chunk)

    m1, m3, m4, m5 = task_configs()
    env = make_c3_env(m1, m3, m4)
    try:
        zero_policy = _Recorder()
        rollout_episode(
            env, zero_policy, seed=3, mode="random", timeout_s=0.8, live_viewer=False, record=False,
            video_path=None, video_fps=10, video_max_mb=2.0, m1=m1, m4=m4, m5=m5,
            chunk_noise=chunk_noise_fn("bc_zero", zero_policy, None, 3),
        )
        assert zero_policy.seen and all(np.all(item == 0) for item in zero_policy.seen)
        left = _Recorder()
        right = _Recorder()
        kwargs = dict(seed=3, mode="random", timeout_s=0.8, live_viewer=False, record=False, video_path=None, video_fps=10, video_max_mb=2.0, m1=m1, m4=m4, m5=m5)
        episode_a = rollout_episode(env, left, chunk_noise=chunk_noise_fn("bc_seeded", left, None, 3), **kwargs)
        episode_b = rollout_episode(env, right, chunk_noise=chunk_noise_fn("bc_seeded", right, None, 3), **kwargs)
    finally:
        env.close()
    assert len(left.seen) >= 2
    assert all(np.array_equal(a, b) for a, b in zip(left.seen, right.seen))
    assert episode_a["n_control_steps"] == episode_b["n_control_steps"]
    assert episode_a["success"] == episode_b["success"]
    assert episode_a["failure"] == episode_b["failure"]
    assert episode_a["place_xy"] == episode_b["place_xy"]


def _episode(seed: int, success: bool) -> dict:
    return {
        "seed": seed,
        "mode": "random",
        "success": success,
        "failure": None if success else "not_grasped",
        "ik_current_fail_ticks": 0,
        "ik_preview_fail_ticks": 0,
        "ik_fail_ticks": 0,
        "ik_fail_fraction": 0.0,
        "ik_any": False,
        "place_xy": 0.1,
        "place_xyz": 0.2,
        "place_vs_place_pos": 0.3,
        "duration_s": 1.0,
        "sim_time_s": 1.0,
        "n_control_steps": 8,
        "timed_out": not success,
        "infer_s": [0.01],
        "infer_s_median": 0.01,
        "infer_s_max": 0.01,
        "n_chunks": 1,
        "video": None,
    }


def test_videos_none_does_not_render(tmp_path, monkeypatch) -> None:
    renders = {"n": 0}
    records: list[bool] = []

    class _Env:
        def render(self, *args, **kwargs):
            renders["n"] += 1
            return np.zeros((256, 256, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    def fake_rollout(env, policy, **kwargs):
        del policy
        record = bool(kwargs["record"])
        records.append(record)
        if record:
            env.render("agentview", 256, 256)
        return _episode(int(kwargs["seed"]), success=False)

    monkeypatch.setattr("aloha_pp.rl.evaluate.make_c3_env", lambda *args, **kwargs: _Env())
    monkeypatch.setattr("aloha_pp.rl.evaluate.render_environment", lambda env: {"driver": "test", "gpu": "test"})
    monkeypatch.setattr("aloha_pp.rl.evaluate.task_configs", lambda: ({}, {}, {}, {}))
    monkeypatch.setattr("aloha_pp.rl.evaluate.rollout_episode", fake_rollout)
    policy = SimpleNamespace(noise_shape=(16, 8), exec_horizon=8)
    summary = evaluate_latent(
        policy, [0, 1, 2], out_dir=tmp_path, timeout_s=1.0, method="bc_zero", checkpoint="x", videos="none",
    )
    assert records == [False, False, False]
    assert renders["n"] == 0
    assert summary["videos"] == "none"
    assert summary["video_seeds"] == []
    text = (tmp_path / "eval.json").read_text(encoding="utf-8")
    assert '"videos": "none"' in text
    assert '"video_seeds": []' in text


def test_encode_actor_obs_rejects_state_and_missing_tcp() -> None:
    policy = _policy()
    obs = _obs()
    bad = dict(obs)
    bad["state"] = np.zeros((2, 8), dtype=np.float32)
    try:
        policy.encode_actor_obs(bad)
    except ValueError as exc:
        assert "state" in str(exc)
    else:
        raise AssertionError("a state key must be rejected")
    missing = dict(obs)
    del missing["tcp_pose"]
    try:
        policy.encode_actor_obs(missing)
    except ValueError as exc:
        assert "tcp_pose" in str(exc)
    else:
        raise AssertionError("missing tcp_pose must be rejected")
    kept = dict(obs)
    kept["can_pose"] = np.ones((2, 7), dtype=np.float32)
    assert torch.equal(policy.encode_actor_obs(obs), policy.encode_actor_obs(kept))


def test_actor_moves_the_rewarded_component_positive() -> None:
    """One-step bandit. Reward is the sign of w[0]. Every transition terminates."""
    previous = torch.get_num_threads()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    try:
        feat_dim, priv_dim, action_dim = 2, 1, 1
        actor = SquashedGaussianActor(
            feat_dim, action_dim, hidden=32, n_layers=2, noise_scale=1.0, use_layer_norm=True,
        )
        critic = TwinCritic(priv_dim, feat_dim, action_dim, hidden=32, n_layers=2, use_layer_norm=True)
        agent = SAC(actor, critic, lr=3e-3, gamma=0.99, tau=0.005, target_entropy=0.0)
        feat = torch.zeros(1, feat_dim)
        with torch.no_grad():
            before = float(actor.deterministic(feat)[0, 0])
        assert abs(before) < 0.05, before
        rng = np.random.default_rng(0)
        n = 2048
        actions = rng.uniform(-1.0, 1.0, size=(n, action_dim)).astype(np.float32)
        rewards = (actions[:, 0] > 0.0).astype(np.float32)
        feats = np.zeros((n, feat_dim), dtype=np.float32)
        priv = np.zeros((n, priv_dim), dtype=np.float32)
        replay = Replay(n, feat_dim, priv_dim, action_dim, rng=np.random.default_rng(0))
        replay.add(feats, priv, actions, rewards, feats, priv, np.ones(n, np.float32), np.zeros(n, np.float32))
        for _ in range(500):
            stats = agent.update(replay.sample(128, "cpu"))
        assert stats["logp_mean"] == stats["logp_mean"]
        assert "q_mean" in stats and "target_q_mean" in stats and "actor_abs_mean" in stats
        with torch.no_grad():
            after = float(actor.deterministic(feat)[0, 0])
        assert after > 0.3, after
    finally:
        torch.set_num_threads(previous)


def test_resume_rejects_changed_noise_scale_and_allows_n_envs() -> None:
    saved = {
        "noise_scale": 1.5,
        "seed": 0,
        "train_seed_start": 10000,
        "val_seed_start": 20000,
        "n_val_episodes": 16,
        "eval_seed_start": 0,
        "eval_seed_count": 50,
        "n_envs": 8,
        "sac": {"actor_lr": 3e-4, "batch_size": 256, "tau": 0.005, "discount": 0.99, "utd": 20,
                 "layer_size": 2048, "num_layers": 3, "use_layer_norm": True, "n_critics": 2,
                 "target_entropy": 0.0, "log_std_init": 0.0, "log_std_min": -20.0, "log_std_max": 2.0,
                 "critic_include_actor_feature": True},
    }
    current = dict(saved)
    current["sac"] = dict(saved["sac"])
    current["n_envs"] = 4
    assert_resume_config(saved, current)
    changed = dict(current)
    changed["noise_scale"] = 2.0
    try:
        assert_resume_config(saved, changed)
    except SystemExit as exc:
        assert "noise_scale" in str(exc)
    else:
        raise AssertionError("a different noise_scale must refuse to resume")
    wider = dict(current)
    wider["sac"] = dict(current["sac"])
    wider["sac"]["layer_size"] = 512
    try:
        assert_resume_config(saved, wider)
    except SystemExit as exc:
        assert "layer_size" in str(exc)
    else:
        raise AssertionError("a different network width must refuse to resume")


def test_paired_comparison_uses_mcnemar_and_marks_the_declared_result() -> None:
    assert abs(mcnemar_p(8, 2) - (112 / 1024)) < 1e-12
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 3) == 1.0

    def summary(flags: dict[int, bool], videos: str = "none") -> dict:
        episodes = []
        for seed, success in flags.items():
            episode = _episode(seed, success)
            episodes.append(episode)
        n = len(episodes)
        k = sum(1 for episode in episodes if episode["success"])
        return {
            "n": n,
            "n_success": k,
            "success_rate": k / n,
            "wilson_95": [0.0, 1.0],
            "failure_counts": {},
            "ik_current_fail_ticks_sum": 0,
            "ik_preview_fail_ticks_sum": 0,
            "ik_fail_ticks_sum": 0,
            "place_xy_median": 0.1,
            "duration_s_median": 1.0,
            "episodes": episodes,
            "videos": videos,
            "video_seeds": [],
        }

    seeded = {0: False, 1: True, 2: False, 3: True}
    zero = {0: False, 1: False, 2: True, 3: True}
    best = {0: True, 1: True, 2: False, 3: False}
    report = build_report(
        [
            ("bc", summary(seeded)),
            ("bc_zero", summary(zero)),
            ("bc_seeded", summary(seeded)),
            ("bc_dsrl_best", summary(best)),
            ("bc_dsrl_last", summary(zero)),
        ],
        primary="bc_dsrl_best",
    )
    roles = {row["method"]: row["role"] for row in report["rows"]}
    assert roles["bc"] == "仅作参考"
    assert roles["bc_dsrl_best"] == "主结果（事先声明）"
    assert roles["bc_dsrl_last"] == ""
    assert "随机噪声（未固定）" in next(row["noise"] for row in report["rows"] if row["method"] == "bc")
    assert next(row["recorded"] for row in report["rows"] if row["method"] == "bc_seeded") == "不含录像"
    paired = {(item["left"], item["right"]): item for item in report["paired"]}
    seeded_best = paired[("bc_seeded", "bc_dsrl_best")]
    assert seeded_best["improve"] == 1
    assert seeded_best["regress"] == 1
    assert "bc_vs_bc_dsrl_best" not in report["transfers"]
    assert mcnemar_p(seeded_best["improve"], seeded_best["regress"]) == seeded_best["mcnemar_p"]


def test_validate_without_record_does_not_write_video(tmp_path, monkeypatch) -> None:
    written = []
    monkeypatch.setattr("aloha_pp.rl.trainer.write_video", lambda frames, path, fps, max_mb: written.append(path))

    class _Policy:
        noise_shape = (2, 2)

        def encode_actor_obs(self, obs):
            return torch.zeros(obs["agentview"].shape[0], 3)

        def sample(self, obs, noise=None, ddim_steps=None):
            del obs, ddim_steps
            return torch.zeros(noise.shape[0], 2, 2)

    class _Actor:
        def deterministic(self, feat):
            return torch.zeros(feat.shape[0], 4)

    class _Vec:
        def reset(self, seeds, record_index=None, indices=None):
            del record_index, indices
            n = len(seeds)
            image = np.zeros((n, 2, 4, 4, 3), dtype=np.uint8)
            return {
                "obs": {
                    "agentview": image,
                    "wrist_cam_left": image,
                    "tcp_pose": np.zeros((n, 2, 7), dtype=np.float32),
                    "gripper_opening": np.zeros((n, 2, 1), dtype=np.float32),
                },
                "priv": np.zeros((n, 1), dtype=np.float32),
            }

        def step(self, chunks, active=None):
            del active
            n = int(chunks.shape[0])
            image = np.zeros((n, 2, 4, 4, 3), dtype=np.uint8)
            return {
                "obs": {
                    "agentview": image,
                    "wrist_cam_left": image,
                    "tcp_pose": np.zeros((n, 2, 7), dtype=np.float32),
                    "gripper_opening": np.zeros((n, 2, 1), dtype=np.float32),
                },
                "priv": np.zeros((n, 1), dtype=np.float32),
                "reward": np.zeros(n, dtype=np.float32),
                "terminated": np.ones(n, dtype=bool),
                "truncated": np.zeros(n, dtype=bool),
                "info": [{"frames": np.zeros((2, 8, 8, 3), dtype=np.uint8)} for _ in range(n)],
            }

    quiet = _validate(
        _Policy(), _Actor(), _Vec(), [1, 2],
        device=torch.device("cpu"), video_path=tmp_path / "quiet.mp4", video_fps=10, video_max_mb=2.0, record=False,
    )
    assert written == []
    assert quiet["video"] is None
    assert quiet["n"] == 2 and quiet["n_success"] == 2
    recorded = _validate(
        _Policy(), _Actor(), _Vec(), [3],
        device=torch.device("cpu"), video_path=tmp_path / "video.mp4", video_fps=10, video_max_mb=2.0, record=True,
    )
    assert recorded["video"] == str(tmp_path / "video.mp4")
    assert len(written) == 1
