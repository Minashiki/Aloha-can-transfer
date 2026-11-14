"""Image diffusion policy. ``sample(obs, noise)`` is the M7 noise-space interface.

``noise is None`` draws a standard normal initial noise of shape
``(B, pred_horizon, action_dim)``. A provided tensor of that shape is the
initial noise and, with DDIM ``eta = 0``, no later noise is drawn. The returned
chunk is in environment units (metres, unit quaternion wxyz with ``w >= 0``,
gripper opening). ``noise_shape`` is the unbatched shape ``(pred_horizon, action_dim)``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from aloha_pp.policy.diffusion import DiffusionModel
from aloha_pp.policy.normalizer import LimitNormalizer
from aloha_pp.policy.quaternion import pose_state
from aloha_pp.policy.unet1d import VisionUnet1D


def _as_batch_images(images) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(images)
    if tensor.ndim == 4:
        return tensor.unsqueeze(0), True
    if tensor.ndim != 5:
        raise ValueError(f"images must be (T,H,W,3) or (B,T,H,W,3), got {tuple(tensor.shape)}")
    return tensor, False


def _nhwc_to_nchw(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.shape[-1] == 3:
        images = images.permute(0, 1, 4, 2, 3)
    return images.contiguous()


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        n_obs_steps: int,
        pred_horizon: int,
        action_dim: int,
        state_dim: int,
        image_size: int = 96,
        denoising_steps: int = 100,
        ddim_steps: int = 8,
        denoised_clip: float = 1.0,
        ema_decay: float = 0.995,
        augment: bool = True,
        shift_pad: int = 4,
        exec_horizon: int = 8,
        vit: dict | None = None,
        unet: dict | None = None,
        action_normalizer: LimitNormalizer | None = None,
        state_normalizer: LimitNormalizer | None = None,
    ) -> None:
        super().__init__()
        vit = vit or {}
        unet = unet or {}
        self.n_obs_steps = int(n_obs_steps)
        self.pred_horizon = int(pred_horizon)
        self.exec_horizon = int(exec_horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.ddim_steps = int(ddim_steps)
        self.ema_decay = float(ema_decay)
        self.eta = 0.0
        cond_dim = self.state_dim * self.n_obs_steps
        network = VisionUnet1D(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            img_cond_steps=self.n_obs_steps,
            num_img=2,
            img_h=int(image_size),
            img_w=int(image_size),
            vit_depth=int(vit.get("depth", 1)),
            vit_embed_dim=int(vit.get("embed_dim", 128)),
            vit_heads=int(vit.get("num_heads", 4)),
            vit_embed_norm=int(vit.get("embed_norm", 0)),
            diffusion_step_embed_dim=int(unet.get("diffusion_step_embed_dim", 32)),
            dim=int(unet.get("dim", 40)),
            dim_mults=tuple(unet.get("dim_mults", (1, 2))),
            smaller_encoder=bool(unet.get("smaller_encoder", False)),
            kernel_size=int(unet.get("kernel_size", 5)),
            n_groups=int(unet.get("n_groups", 8)),
            cond_predict_scale=bool(unet.get("cond_predict_scale", True)),
            spatial_emb=int(unet.get("spatial_emb", 128)),
            dropout=float(unet.get("dropout", 0.0)),
            augment=augment,
            shift_pad=int(shift_pad),
        )
        self.diffusion = DiffusionModel(
            network,
            horizon_steps=self.pred_horizon,
            action_dim=self.action_dim,
            denoising_steps=int(denoising_steps),
            ddim_steps=self.ddim_steps,
            denoised_clip_value=float(denoised_clip),
        )
        self.ema = copy.deepcopy(self.diffusion)
        self.ema.eval()
        self.ema.requires_grad_(False)
        self.register_buffer("action_lo", torch.zeros(self.action_dim))
        self.register_buffer("action_hi", torch.ones(self.action_dim))
        self.register_buffer("state_lo", torch.zeros(self.state_dim))
        self.register_buffer("state_hi", torch.ones(self.state_dim))
        if action_normalizer is not None:
            self.set_normalizer(action_normalizer, state_normalizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.ema.eval()
        return self

    def set_normalizer(self, action: LimitNormalizer, state: LimitNormalizer) -> None:
        self.action_lo.copy_(torch.as_tensor(action.lo, dtype=torch.float32))
        self.action_hi.copy_(torch.as_tensor(action.hi, dtype=torch.float32))
        self.state_lo.copy_(torch.as_tensor(state.lo, dtype=torch.float32))
        self.state_hi.copy_(torch.as_tensor(state.hi, dtype=torch.float32))

    @property
    def noise_shape(self) -> tuple[int, int]:
        """Initial noise for one sample, without the batch axis."""
        return (self.pred_horizon, self.action_dim)

    def actor_feature_dim(self) -> int:
        """Width of ``encode_actor_obs``: both cameras' spatial embeddings plus proprio."""
        net = self.ema.network
        if not hasattr(net, "compress"):
            raise TypeError("actor features require the vision UNet")
        proj = int(net.compress[0].input_proj[0].out_features)
        return proj * len(net.compress) + self.state_dim * self.n_obs_steps

    @torch.no_grad()
    def encode_actor_obs(self, obs: dict) -> torch.Tensor:
        """Frozen features for a DSRL actor. Does not draw noise or change ``sample``.

        The vector is the EMA ViT + SpatialEmb output of both cameras, with
        augmentation off, concatenated with the normalized proprio state.
        Proprio is ``pose_state(tcp_pose, gripper_opening)`` over
        ``n_obs_steps`` frames, the same tensor ``sample`` conditions on.
        Required keys are the two images, ``tcp_pose``, and ``gripper_opening``.
        A ``state`` key is an error. Other keys are ignored, so a privileged
        field cannot change this vector. ``_cond`` is unchanged and still
        accepts ``state`` for the M6 training path.
        """
        if "state" in obs:
            raise ValueError("encode_actor_obs does not accept a state key; pass tcp_pose and gripper_opening")
        missing = [key for key in ("agentview", "wrist_cam_left", "tcp_pose", "gripper_opening") if key not in obs]
        if missing:
            raise ValueError("encode_actor_obs requires " + ", ".join(missing))
        was = self.training
        self.eval()
        try:
            cond = self._cond(obs, state_is_normalized=False)
            net = self.ema.network
            net.eval()
            flat = cond["state"].reshape(cond["state"].shape[0], -1)
            feats = []
            for view, compress in zip(net._images(cond["rgb"]), net.compress):
                feats.append(compress(net.backbone(view), flat))
            visual = torch.cat(feats, dim=-1)
            return torch.cat([visual, flat], dim=-1)
        finally:
            self.train(was)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.diffusion.parameters())

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return 2.0 * (state - self.state_lo) / (self.state_hi - self.state_lo) - 1.0

    def _normalize_raw_state(self, state: np.ndarray) -> np.ndarray:
        """Same float32 map ``LimitNormalizer.normalize`` used on the training windows."""
        norm = LimitNormalizer()
        norm.lo = self.state_lo.detach().cpu().numpy()
        norm.hi = self.state_hi.detach().cpu().numpy()
        return norm.normalize(np.asarray(state, dtype=np.float32))

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) * 0.5 * (self.action_hi - self.action_lo) + self.action_lo

    def _cond(self, obs: dict, *, state_is_normalized: bool) -> dict[str, torch.Tensor]:
        agent, unbatched = _as_batch_images(obs["agentview"])
        wrist, wrist_unbatched = _as_batch_images(obs["wrist_cam_left"])
        if unbatched != wrist_unbatched:
            raise ValueError("agent and wrist batch ranks differ")
        if "state" in obs and "tcp_pose" not in obs:
            state = torch.as_tensor(obs["state"], dtype=torch.float32)
            if unbatched:
                state = state.unsqueeze(0)
        else:
            tcp = np.asarray(obs["tcp_pose"], dtype=np.float32)
            grip = np.asarray(obs["gripper_opening"], dtype=np.float32)
            if unbatched:
                tcp = tcp[None, ...]
                grip = grip[None, ...]
            state = torch.as_tensor(self._normalize_raw_state(pose_state(tcp, grip)))
            state_is_normalized = True
        device = self.action_lo.device
        agent = _nhwc_to_nchw(agent).to(device)
        wrist = _nhwc_to_nchw(wrist).to(device)
        state = state.to(device)
        if not state_is_normalized:
            state = self.normalize_state(state)
        if state.shape[1] != self.n_obs_steps:
            raise ValueError(f"expected {self.n_obs_steps} observation frames, got {state.shape[1]}")
        rgb = torch.cat([agent, wrist], dim=2)
        return {"rgb": rgb, "state": state}

    def loss(self, batch: dict) -> torch.Tensor:
        cond = self._cond(batch, state_is_normalized=True)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.action_lo.device)
        return self.diffusion.loss(action, cond)

    @torch.no_grad()
    def update_ema(self) -> None:
        decay = self.ema_decay
        for online, shadow in zip(self.diffusion.parameters(), self.ema.parameters()):
            shadow.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
        for online, shadow in zip(self.diffusion.buffers(), self.ema.buffers()):
            shadow.copy_(online)

    def _finalize(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clone()
        quat = action[..., 3:7]
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        quat = quat / norm.clamp_min(1e-8)
        quat = torch.where(quat[..., :1] < 0, -quat, quat)
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        action[..., 3:7] = torch.where(norm >= 1e-8, quat, identity)
        return action

    @torch.no_grad()
    def sample(
        self,
        obs: dict,
        noise: torch.Tensor | None = None,
        *,
        ddim_steps: int | None = None,
    ) -> torch.Tensor:
        """Deterministic when ``noise`` is passed and ``eta`` is 0.

        Returns ``(B, pred_horizon, action_dim)`` float32 on the policy device.
        """
        if self.eta != 0.0 or self.ema.eta != 0.0:
            raise RuntimeError("DDIM eta must stay 0")
        was = self.training
        flags = (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        self.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            cond = self._cond(obs, state_is_normalized=False)
            batch = cond["state"].shape[0]
            if noise is None:
                noise = torch.randn(batch, self.pred_horizon, self.action_dim, device=self.action_lo.device)
            else:
                noise = torch.as_tensor(noise, dtype=torch.float32, device=self.action_lo.device)
                if noise.ndim == 2:
                    noise = noise.unsqueeze(0)
                expected = (batch, self.pred_horizon, self.action_dim)
                if tuple(noise.shape) != expected:
                    raise ValueError(f"noise shape {tuple(noise.shape)} != {expected}")
            steps = self.ddim_steps if ddim_steps is None else int(ddim_steps)
            normalized = self.ema.ddim_sample(cond, noise, ddim_steps=steps)
            return self._finalize(self.unnormalize_action(normalized))
        finally:
            self.train(was)
            (
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            ) = flags
