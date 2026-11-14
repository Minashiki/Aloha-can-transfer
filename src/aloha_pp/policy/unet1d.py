"""Conditional 1D UNet, with and without images.

Source: https://github.com/irom-princeton/dppo
Commit: cc7234ad7ff39a8f32de3af903606723a16f0648
Path: model/diffusion/unet.py
License: MIT (see THIRD_PARTY_LICENSES)

Changes: einops rearranges are transposes. The empty ``h_local`` skip is
removed. ``cond_mlp`` / ResidualMLP is not copied (the Can image config does
not set it). Random shift runs only while the module is in train mode, so
``sample`` is deterministic. Image history uses the sliced time length.
Upsampling still runs on every up stage, matching DPPO: that pairs with the
downsample skipped on the last down stage.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from aloha_pp.policy.modules import (
    Downsample1d,
    RandomShiftsAug,
    ResidualBlock1D,
    SinusoidalPosEmb,
    SpatialEmb,
    Upsample1d,
    Conv1dBlock,
)
from aloha_pp.policy.vit import VitEncoder


def _time_feature(time: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(time):
        time = torch.tensor([time], dtype=torch.long, device=ref.device)
    elif time.ndim == 0:
        time = time[None].to(device=ref.device)
    return time.expand(ref.shape[0])


class _UNetTrunk(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cond_block_dim: int,
        diffusion_step_embed_dim: int,
        dim: int,
        dim_mults: tuple[int, ...],
        kernel_size: int,
        n_groups: int | None,
        activation_type: str,
        cond_predict_scale: bool,
        groupnorm_eps: float,
        smaller_encoder: bool,
    ) -> None:
        super().__init__()
        dims = [action_dim, *tuple(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self._n_down = len(in_out)
        larger = not smaller_encoder
        dsed = diffusion_step_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        block = dict(
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            larger_encoder=larger,
            activation_type=activation_type,
            groupnorm_eps=groupnorm_eps,
        )
        mid = dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock1D(mid, mid, cond_dim=cond_block_dim, **block),
                ResidualBlock1D(mid, mid, cond_dim=cond_block_dim, **block),
            ]
        )
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(dim_in, dim_out, cond_dim=cond_block_dim, **block),
                        ResidualBlock1D(dim_out, dim_out, cond_dim=cond_block_dim, **block),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_block_dim, **block),
                        ResidualBlock1D(dim_in, dim_in, cond_dim=cond_block_dim, **block),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, n_groups=n_groups, activation_type=activation_type, eps=groupnorm_eps),
            nn.Conv1d(dim, action_dim, 1),
        )

    def forward_trunk(self, x: torch.Tensor, time: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        time = _time_feature(time, x)
        global_feature = torch.cat([self.time_mlp(time), global_cond], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        return self.final_conv(x)


class Unet1D(nn.Module):
    """Low-dimensional conditional UNet. ``cond['state']`` is ``(B, T, D)``."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        diffusion_step_embed_dim: int = 32,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2),
        smaller_encoder: bool = False,
        kernel_size: int = 5,
        n_groups: int | None = 8,
        activation_type: str = "Mish",
        cond_predict_scale: bool = True,
        groupnorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.trunk = _UNetTrunk(
            action_dim=action_dim,
            cond_block_dim=diffusion_step_embed_dim + cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            dim=dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            n_groups=n_groups,
            activation_type=activation_type,
            cond_predict_scale=cond_predict_scale,
            groupnorm_eps=groupnorm_eps,
            smaller_encoder=smaller_encoder,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        del kwargs
        x = x.transpose(1, 2).contiguous()
        state = cond["state"].reshape(x.shape[0], -1)
        if state.shape[-1] != self.cond_dim:
            raise ValueError(f"flattened state {state.shape[-1]} != cond_dim {self.cond_dim}")
        x = self.trunk.forward_trunk(x, time, state)
        return x.transpose(1, 2).contiguous()


class VisionUnet1D(nn.Module):
    """Image-conditioned UNet. Two cameras are encoded by a shared ViT."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        img_cond_steps: int = 2,
        num_img: int = 2,
        img_h: int = 96,
        img_w: int = 96,
        vit_depth: int = 1,
        vit_embed_dim: int = 128,
        vit_heads: int = 4,
        vit_embed_norm: int = 0,
        diffusion_step_embed_dim: int = 32,
        dim: int = 40,
        dim_mults: tuple[int, ...] = (1, 2),
        smaller_encoder: bool = False,
        kernel_size: int = 5,
        n_groups: int = 8,
        activation_type: str = "Mish",
        cond_predict_scale: bool = True,
        groupnorm_eps: float = 1e-5,
        spatial_emb: int = 128,
        dropout: float = 0.0,
        augment: bool = True,
        shift_pad: int = 4,
    ) -> None:
        super().__init__()
        self.img_cond_steps = int(img_cond_steps)
        self.num_img = int(num_img)
        self.cond_dim = int(cond_dim)
        self.augment = bool(augment)
        self.aug = RandomShiftsAug(pad=shift_pad)
        self.backbone = VitEncoder(
            depth=vit_depth,
            embed_dim=vit_embed_dim,
            num_heads=vit_heads,
            embed_norm=vit_embed_norm,
            num_channel=3 * self.img_cond_steps,
            img_h=img_h,
            img_w=img_w,
        )
        if spatial_emb <= 1:
            raise ValueError("spatial_emb must be the projection dimension")
        self.compress = nn.ModuleList(
            [
                SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=self.cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                for _ in range(self.num_img)
            ]
        )
        # Independent weights per camera, matching DPPO's deepcopy of SpatialEmb.
        if self.num_img == 2:
            self.compress[1] = deepcopy(self.compress[0])
        visual_feature_dim = spatial_emb * self.num_img
        self.trunk = _UNetTrunk(
            action_dim=action_dim,
            cond_block_dim=diffusion_step_embed_dim + self.cond_dim + visual_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            dim=dim,
            dim_mults=tuple(dim_mults),
            kernel_size=kernel_size,
            n_groups=n_groups,
            activation_type=activation_type,
            cond_predict_scale=cond_predict_scale,
            groupnorm_eps=groupnorm_eps,
            smaller_encoder=smaller_encoder,
        )

    def _images(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        rgb = rgb[:, -self.img_cond_steps :].float()
        b, t_rgb, channels, h, w = rgb.shape
        if channels != self.num_img * 3:
            raise ValueError(f"expected {self.num_img * 3} image channels, got {channels}")
        rgb = rgb.reshape(b, t_rgb, self.num_img, 3, h, w)
        rgb = rgb.permute(0, 2, 1, 3, 4, 5).contiguous()
        rgb = rgb.reshape(b, self.num_img, t_rgb * 3, h, w)
        frames = []
        for index in range(self.num_img):
            view = rgb[:, index]
            if self.augment and self.training:
                view = self.aug(view)
            frames.append(view)
        return frames

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        del kwargs
        b = x.shape[0]
        x = x.transpose(1, 2).contiguous()
        state = cond["state"].reshape(b, -1)
        if state.shape[-1] != self.cond_dim:
            raise ValueError(f"flattened state {state.shape[-1]} != cond_dim {self.cond_dim}")
        feats = []
        for view, compress in zip(self._images(cond["rgb"]), self.compress):
            feats.append(compress(self.backbone(view), state))
        visual = torch.cat(feats, dim=-1)
        global_cond = torch.cat([visual, state], dim=-1)
        x = self.trunk.forward_trunk(x, time, global_cond)
        return x.transpose(1, 2).contiguous()
