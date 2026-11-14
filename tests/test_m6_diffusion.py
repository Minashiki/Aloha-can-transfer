"""Diffusion schedule, DDIM determinism, network shapes, and a tiny overfit."""

from __future__ import annotations

import numpy as np
import torch

from aloha_pp.policy.checkpoint import load_checkpoint, policy_from_config, save_checkpoint
from aloha_pp.policy.chunk import ChunkExecutor
from aloha_pp.policy.diffusion import DiffusionModel, cosine_beta_schedule, ddim_indices
from aloha_pp.policy.normalizer import LimitNormalizer
from aloha_pp.policy.failure import classify_physics
from aloha_pp.policy.quaternion import canonicalize_quaternions, pose_state, to_env_action
from aloha_pp.policy.sequences import action_window, observation_window, select_indices
from aloha_pp.data.dataset import list_demos, load_episode
from aloha_pp.policy.unet1d import Unet1D, VisionUnet1D
from aloha_pp.policy.vit import VitEncoder
from aloha_pp.robot import repo_root


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


def _fitted() -> tuple[LimitNormalizer, LimitNormalizer]:
    rng = np.random.default_rng(0)
    action = LimitNormalizer().fit(rng.normal(size=(32, 8)).astype(np.float32))
    state = LimitNormalizer().fit(rng.normal(size=(32, 8)).astype(np.float32))
    return action, state


def test_cosine_schedule_and_ddim_index_count() -> None:
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas > 0) and torch.all(betas < 1)
    alphas = torch.cumprod(1 - betas, dim=0)
    for steps in (5, 10):
        t, alpha, prev = ddim_indices(alphas, steps)
        assert t.shape == alpha.shape == prev.shape == (steps,)
        assert float(prev[-1]) == 1.0
        assert int(t[-1]) == 0


def test_ddim_eta0_is_deterministic_and_shape_stable() -> None:
    torch.manual_seed(0)
    net = Unet1D(action_dim=8, cond_dim=8, dim=32, dim_mults=(1, 2), n_groups=8, smaller_encoder=True)
    model = DiffusionModel(net, horizon_steps=16, action_dim=8, denoising_steps=20, ddim_steps=5)
    model.eval()
    cond = {"state": torch.zeros(2, 1, 8)}
    noise = torch.randn(2, 16, 8)
    first = model.ddim_sample(cond, noise, ddim_steps=5)
    second = model.ddim_sample(cond, noise.clone(), ddim_steps=5)
    assert torch.equal(first, second)
    other = model.ddim_sample(cond, noise, ddim_steps=10)
    assert other.shape == first.shape == (2, 16, 8)
    assert model.eta == 0.0


def test_training_loss_decreases_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    net = Unet1D(action_dim=8, cond_dim=8, dim=32, dim_mults=(1, 2), n_groups=8, smaller_encoder=True)
    model = DiffusionModel(net, horizon_steps=16, action_dim=8, denoising_steps=16, ddim_steps=4)
    action = torch.zeros(16, 16, 8)
    action[:, :, 0] = torch.linspace(-0.8, 0.8, 16)
    cond = {"state": torch.zeros(16, 1, 8)}
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(action, cond)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.75, losses


def test_unet_and_encoder_shapes_for_two_batch_sizes() -> None:
    encoder = VitEncoder(num_channel=6, img_h=96, img_w=96, embed_dim=128, num_heads=4, depth=1)
    features = encoder(torch.rand(2, 6, 96, 96) * 255)
    assert features.shape == (2, encoder.num_patch, 128)
    net = VisionUnet1D(
        action_dim=8,
        cond_dim=16,
        img_cond_steps=2,
        img_h=96,
        img_w=96,
        dim=40,
        dim_mults=(1, 2),
        n_groups=8,
        spatial_emb=128,
        augment=False,
    )
    net.eval()
    for batch in (1, 3):
        x = torch.randn(batch, 16, 8)
        cond = {
            "rgb": torch.rand(batch, 2, 6, 96, 96) * 255,
            "state": torch.randn(batch, 2, 8),
        }
        y = net(x, torch.zeros(batch, dtype=torch.long), cond)
        assert y.shape == x.shape


def test_sample_is_bitwise_for_the_same_noise() -> None:
    torch.manual_seed(0)
    policy = policy_from_config(_small_cfg(), *_fitted())
    policy.eval()
    obs = _obs()
    noise = torch.randn(1, *policy.noise_shape)
    first = policy.sample(obs, noise)
    second = policy.sample(obs, noise.clone())
    assert tuple(noise.shape) == (1, 16, 8)
    assert policy.noise_shape == (16, 8)
    assert policy.eta == 0.0
    assert torch.equal(first, second)
    changed = policy.sample(obs, noise + 0.5)
    assert changed.shape == first.shape
    assert not torch.equal(first, changed)


def test_sample_is_bitwise_on_cuda() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    policy = policy_from_config(_small_cfg(), *_fitted()).cuda()
    obs = _obs()
    noise = torch.randn(1, *policy.noise_shape, device="cuda")
    first = policy.sample(obs, noise)
    second = policy.sample(obs, noise.clone())
    assert torch.equal(first, second)


def test_checkpoint_roundtrip_is_bitwise(tmp_path) -> None:
    torch.manual_seed(0)
    action, state = _fitted()
    cfg = _small_cfg()
    policy = policy_from_config(cfg, action, state)
    obs = _obs()
    noise = torch.randn(1, *policy.noise_shape)
    expected = policy.sample(obs, noise)
    extra = {
        "config": cfg,
        "normalizer": {
            "eps": 1e-6,
            "action_lo": action.lo,
            "action_hi": action.hi,
            "state_lo": state.lo,
            "state_hi": state.hi,
        },
        "train_source_ids": [1, 2],
        "epoch": 0,
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, policy, None, extra)
    torch.manual_seed(1)
    restored, _blob = load_checkpoint(path, "cpu")
    got = restored.sample(obs, noise)
    assert torch.equal(expected, got)


def test_windows_pad_the_ends_and_subset_is_reproducible() -> None:
    series = np.arange(6, dtype=np.float32).reshape(6, 1)
    window = observation_window(series, 0, 2)
    assert np.all(window[0] == series[0]) and np.all(window[1] == series[0])
    assert np.all(observation_window(series, 3, 2)[:, 0] == np.array([2, 3], dtype=np.float32))
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)
    tail = action_window(actions, 2, 4)
    assert tail.shape == (4, 8)
    assert np.all(tail[0] == actions[2]) and np.all(tail[-1] == actions[2])
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    norm = LimitNormalizer().fit(values)
    back = norm.unnormalize(norm.normalize(values))
    np.testing.assert_allclose(back, values, atol=1e-6)
    flat = LimitNormalizer().fit(np.ones((4, 2), dtype=np.float32))
    assert np.all(flat.normalize(np.ones((2, 2), dtype=np.float32)) == -1)
    assert np.array_equal(select_indices(120, 30, 0), select_indices(120, 30, 0))
    assert not np.array_equal(select_indices(120, 30, 0), select_indices(120, 30, 1))
    names = list_demos(repo_root() / "outputs/m5/dataset/demos.hdf5")
    pick = select_indices(len(names), 30, 0)
    assert len(pick) == 30 and len(set(pick.tolist())) == 30


def test_stored_window_matches_online_condition_and_sign_flip() -> None:
    path = repo_root() / "outputs/m5/dataset/demos.hdf5"
    episode = load_episode(path, "demo_0004")
    tcp = np.asarray(episode["obs"]["tcp_pose"], dtype=np.float32)
    negative = np.flatnonzero(tcp[:, 3] < 0)
    t = int(negative[0]) if negative.size else 0
    raw_tcp = observation_window(tcp, t, 2)
    grip = observation_window(
        np.asarray(episode["obs"]["gripper_opening"], dtype=np.float32).reshape(-1, 1), t, 2
    )
    images = {
        "agentview": np.zeros((2, 32, 32, 3), dtype=np.uint8),
        "wrist_cam_left": np.zeros((2, 32, 32, 3), dtype=np.uint8),
    }
    state = pose_state(raw_tcp, grip)
    normalizer = LimitNormalizer().fit(np.concatenate([state, state + np.float32(0.25)], axis=0))
    action, _unused = _fitted()
    policy = policy_from_config(_small_cfg(), action, normalizer)
    trained = policy._cond({**images, "state": normalizer.normalize(state)}, state_is_normalized=True)
    online = policy._cond({**images, "tcp_pose": raw_tcp, "gripper_opening": grip}, state_is_normalized=False)
    assert torch.equal(trained["state"].cpu(), online["state"].cpu())
    flipped = raw_tcp.copy()
    flipped[:, 3:7] *= np.float32(-1.0)
    signed = policy._cond({**images, "tcp_pose": flipped, "gripper_opening": grip}, state_is_normalized=False)
    assert torch.equal(online["state"].cpu(), signed["state"].cpu())


def _task_reason(closed: np.ndarray, contact: np.ndarray, height: np.ndarray) -> str | None:
    n = len(closed)
    can = np.zeros((n, 3), dtype=float)
    can[:, 2] = height
    return classify_physics(
        success=False,
        contact=contact,
        closed=closed,
        can=can,
        vel=np.zeros((n, 3), dtype=float),
        center=np.zeros(3),
        half_extents=np.ones(3),
        vel_thresh=0.01,
        drop_z=0.03,
        lift_thresh=0.02,
        lost_steps=3,
    )


def test_failure_uses_every_grasp_attempt_and_not_ik() -> None:
    idle = np.zeros((4, 2), dtype=bool)
    assert _task_reason(np.zeros(4, dtype=bool), idle, np.zeros(4)) == "not_grasped"
    closed = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    contact = np.zeros((8, 2), dtype=bool)
    contact[:2] = True
    contact[4:] = True
    height = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.03])
    # The first closed run never lifts. The second one does, and the fingers
    # are still on the can at the end, so the episode is not_released.
    assert _task_reason(closed, contact, height) == "not_released"


def test_chunk_preview_and_quaternion() -> None:
    chunk = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    player = ChunkExecutor(16, 8, 8)
    player.push(chunk)
    for k in range(8):
        action, nxt = player.pop()
        np.testing.assert_array_equal(action, chunk[k])
        np.testing.assert_array_equal(nxt, chunk[k + 1])
    assert player.needs_chunk()
    quat = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    fixed, info = canonicalize_quaternions(quat)
    assert fixed[0, 0] > 0
    assert info["hemisphere"] == 1
    assert float(np.dot(fixed[0], fixed[1])) >= 0
    raw = np.array([0.1, 0.2, 0.3, 0.0, 2.0, 0.0, 0.0, 0.037], dtype=np.float64)
    env_action = to_env_action(raw)
    assert env_action.dtype == np.float64
    assert abs(float(np.linalg.norm(env_action[3:7])) - 1.0) < 1e-6
    assert env_action[3] >= 0
    assert env_action[0] == np.float32(0.1)
