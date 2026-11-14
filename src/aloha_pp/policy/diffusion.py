"""Gaussian diffusion training loss and DDIM sampling.

Source: https://github.com/irom-princeton/dppo
Commit: cc7234ad7ff39a8f32de3af903606723a16f0648
Paths: model/diffusion/diffusion.py, model/diffusion/sampling.py
License: MIT (see THIRD_PARTY_LICENSES). The cosine schedule is the one
DPPO copies from Diffuser / Nichol & Dhariwal.

Changes: DDIM tables are built at sample time so the step count can change
without rebuilding the module. ``eta`` is fixed at 0; the sigma term is then
zero and no extra noise is drawn. The initial noise is an argument. The source
``forward`` also drew a noise tensor and multiplied it by a zero std; skipping
that draw keeps the RNG unchanged and the sample identical. A ``deterministic``
keyword that the source passed into ``p_mean_var`` but did not accept is not copied.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Cosine schedule from https://openreview.net/forum?id=-NEXDKk8gZ ."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas, dtype=dtype)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def ddim_indices(alphas_cumprod: torch.Tensor, ddim_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading-style DDIM timesteps, flipped so index 0 is the noisiest step.

    Returns ``(t, alpha, alpha_prev)`` each of length ``ddim_steps``.
    ``alpha_prev`` of the last index is 1.
    """
    denoising_steps = int(alphas_cumprod.shape[0])
    if ddim_steps < 1 or ddim_steps > denoising_steps:
        raise ValueError(f"ddim_steps must be in 1..{denoising_steps}, got {ddim_steps}")
    step_ratio = denoising_steps // int(ddim_steps)
    t = torch.arange(0, int(ddim_steps), device=alphas_cumprod.device) * step_ratio
    alphas = alphas_cumprod[t]
    alpha_prev = torch.cat([alphas_cumprod.new_tensor([1.0]), alphas_cumprod[t[:-1]]])
    return torch.flip(t, [0]), torch.flip(alphas, [0]), torch.flip(alpha_prev, [0])


class DiffusionModel(nn.Module):
    """Epsilon-predicting DDPM loss and eta = 0 DDIM sampling."""

    def __init__(
        self,
        network: nn.Module,
        horizon_steps: int,
        action_dim: int,
        denoising_steps: int = 100,
        ddim_steps: int = 8,
        denoised_clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.denoising_steps = int(denoising_steps)
        self.ddim_steps = int(ddim_steps)
        self.denoised_clip_value = float(denoised_clip_value)
        self.eta = 0.0
        self.predict_epsilon = True
        betas = cosine_beta_schedule(self.denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x_start: torch.Tensor, cond: dict) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.randint(0, self.denoising_steps, (batch,), device=x_start.device)
        return self.p_losses(x_start, cond, t)

    def p_losses(self, x_start: torch.Tensor, cond: dict, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(pred, noise)
        return F.mse_loss(pred, x_start)

    def _ddim_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict,
        alpha: torch.Tensor,
        alpha_prev: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.network(x, t, cond=cond)
        sqrt_one_minus = (1.0 - alpha).sqrt()
        x_recon = (x - sqrt_one_minus * eps) / alpha.sqrt()
        if self.denoised_clip_value is not None:
            x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            eps = (x - alpha.sqrt() * x_recon) / sqrt_one_minus
        # eta = 0, so the stochastic coefficient is zero and mu is deterministic.
        direction = (1.0 - alpha_prev).sqrt() * eps
        return alpha_prev.sqrt() * x_recon + direction

    @torch.no_grad()
    def ddim_sample(self, cond: dict, noise: torch.Tensor, ddim_steps: int | None = None) -> torch.Tensor:
        if self.eta != 0.0:
            raise RuntimeError("this sampler is eta = 0 only")
        steps = self.ddim_steps if ddim_steps is None else int(ddim_steps)
        t_all, alphas, alpha_prev = ddim_indices(self.alphas_cumprod, steps)
        x = noise
        if x.ndim != 3 or x.shape[1:] != (self.horizon_steps, self.action_dim):
            raise ValueError(
                f"noise shape {tuple(x.shape)} != (B, {self.horizon_steps}, {self.action_dim})"
            )
        batch = x.shape[0]
        for i in range(steps):
            t = torch.full((batch,), int(t_all[i].item()), device=x.device, dtype=torch.long)
            index = torch.full((batch,), i, device=x.device, dtype=torch.long)
            alpha = extract(alphas, index, x.shape)
            prev = extract(alpha_prev, index, x.shape)
            x = self._ddim_step(x, t, cond, alpha, prev)
        return x
