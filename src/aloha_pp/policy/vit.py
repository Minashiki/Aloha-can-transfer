"""Small ViT image encoder.

Source: https://github.com/irom-princeton/dppo
Commit: cc7234ad7ff39a8f32de3af903606723a16f0648
Path: model/common/vit.py
License: MIT (see THIRD_PARTY_LICENSES). DPPO attributes this encoder to
IBRL (https://github.com/hengyuan-hu/ibrl).

Changes: einops rearranges are ``flatten`` / ``permute`` / ``reshape``.
Attention uses the math SDPA kernel so a fixed input produces a fixed output.
The unused ``act_layer`` attribute and the ``__main__`` smoke checks are omitted.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import trunc_normal_


class PatchEmbed2(nn.Module):
    def __init__(self, embed_dim: int, use_norm: bool, num_channel: int = 3, img_h: int = 96, img_w: int = 96) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(num_channel, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
        ]
        layers.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2))
        self.embed = nn.Sequential(*layers)
        h1 = math.ceil((img_h - 8) / 4) + 1
        w1 = math.ceil((img_w - 8) / 4) + 1
        h2 = math.ceil((h1 - 3) / 2) + 1
        w2 = math.ceil((w1 - 3) / 2) + 1
        self.num_patch = h2 * w2
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x)
        if y.shape[2] * y.shape[3] != self.num_patch:
            raise RuntimeError(f"patch grid {y.shape[2:]} does not match num_patch {self.num_patch}")
        return y.flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_head: int) -> None:
        super().__init__()
        if embed_dim % num_head != 0:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_head {num_head}")
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        head_dim = self.embed_dim // self.num_head
        qkv = self.qkv_proj(x).view(b, t, 3, self.num_head, head_dim).permute(0, 2, 3, 1, 4)
        q, k, v = qkv.unbind(dim=1)
        with sdpa_kernel(SDPBackend.MATH):
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(b, t, self.embed_dim)
        return self.out_proj(attn)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(nn.functional.gelu(self.linear1(x)))


class MinVit(nn.Module):
    def __init__(
        self,
        embed_style: str,
        embed_dim: int,
        embed_norm: int,
        num_head: int,
        depth: int,
        num_channel: int = 3,
        img_h: int = 96,
        img_w: int = 96,
    ) -> None:
        super().__init__()
        if embed_style != "embed2":
            raise ValueError("only embed2 is used; embed1 was not copied")
        self.patch_embed = PatchEmbed2(
            embed_dim,
            use_norm=bool(embed_norm),
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patch, embed_dim))
        self.net = nn.Sequential(*[TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)


class VitEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        embed_style: str = "embed2",
        embed_norm: int = 0,
        num_channel: int = 3,
        img_h: int = 96,
        img_w: int = 96,
    ) -> None:
        super().__init__()
        self.vit = MinVit(
            embed_style=embed_style,
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            num_head=num_heads,
            depth=depth,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
        )
        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = embed_dim
        self.repr_dim = embed_dim * self.vit.num_patches

    def forward(self, obs: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        feats = self.vit.forward(obs)
        if flatten:
            feats = feats.flatten(1, 2)
        return feats


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    del name
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn, module: nn.Module, name: str = "", depth_first: bool = True, include_root: bool = False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
