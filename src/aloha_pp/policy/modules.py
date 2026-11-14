"""Building blocks for the 1D UNet and the image encoder.

Source: https://github.com/irom-princeton/dppo
Commit: cc7234ad7ff39a8f32de3af903606723a16f0648
Paths: model/diffusion/modules.py, model/common/modules.py
License: MIT (see THIRD_PARTY_LICENSES in this directory)

Changes: ``Rearrange`` from einops is replaced with ``unsqueeze`` / ``squeeze``.
``Conv1dBlock`` builds a fresh activation module for each block. The random-shift
grid and ``SpatialEmb`` match the source, including the patch/feature transpose.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _activation(name: str) -> nn.Module:
    if name == "Mish":
        return nn.Mish()
    if name == "ReLU":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}")


class Conv1dBlock(nn.Module):
    """Conv1d, optional GroupNorm, then Mish or ReLU.

    GroupNorm is applied on ``(batch, channels, 1, horizon)`` so the grouping
    matches the einops layout in the source.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int | None = None,
        activation_type: str = "Mish",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_groups is not None and out_channels % n_groups != 0:
            raise ValueError(f"out_channels {out_channels} is not divisible by n_groups {n_groups}")
        layers: list[nn.Module] = [
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        ]
        if n_groups is not None:
            layers.append(_GroupNorm1d(n_groups, out_channels, eps=eps))
        layers.append(_activation(activation_type))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _GroupNorm1d(nn.Module):
    def __init__(self, n_groups: int, channels: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.unsqueeze(2)).squeeze(2)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int | None = None,
        cond_predict_scale: bool = False,
        larger_encoder: bool = False,
        activation_type: str = "Mish",
        groupnorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                    activation_type=activation_type,
                    eps=groupnorm_eps,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                    activation_type=activation_type,
                    eps=groupnorm_eps,
                ),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        if larger_encoder:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, cond_channels),
                _activation(activation_type),
                nn.Linear(cond_channels, cond_channels),
                _activation(activation_type),
                nn.Linear(cond_channels, cond_channels),
            )
        else:
            self.cond_encoder = nn.Sequential(
                _activation(activation_type),
                nn.Linear(cond_dim, cond_channels),
            )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            out = embed[:, 0] * out + embed[:, 1]
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class RandomShiftsAug:
    """Pad-and-crop shift. ``pad`` pixels, matching DPPO's ``RandomShiftsAug``."""

    def __init__(self, pad: int) -> None:
        self.pad = int(pad)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, _c, h, w = x.size()
        if h != w:
            raise ValueError(f"RandomShiftsAug expects a square image, got {h}x{w}")
        padding = (self.pad, self.pad, self.pad, self.pad)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class SpatialEmb(nn.Module):
    """IBRL spatial embedding as vendored by DPPO.

    ``feat`` is ``(batch, num_patch, patch_dim)``. The source transposes it
    before the linear layer, so the linear features are the patches and the
    sequence axis is the embedding dimension. That layout is kept.
    """

    def __init__(self, num_patch: int, patch_dim: int, prop_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_patch + prop_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, patch_dim, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def forward(self, feat: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        feat = feat.transpose(1, 2)
        if self.prop_dim > 0:
            repeated = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated), dim=-1)
        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        return self.dropout(z)
