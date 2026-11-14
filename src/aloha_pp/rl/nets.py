"""Squashed-Gaussian actor and twin Q networks.

The MLP matches the DSRL training script's SB3 policy kwargs: ``num_layers``
hidden widths, a LayerNorm after each linear, then tanh. The actor's mean
head is initialized small so an untrained deterministic action sits near 0,
which is the ``bc_zero`` noise.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: int, n_layers: int, use_layer_norm: bool) -> tuple[nn.Sequential, int]:
    gain = nn.init.calculate_gain("tanh")
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(n_layers)):
        linear = nn.Linear(dim, int(hidden))
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if use_layer_norm:
            layers.append(nn.LayerNorm(int(hidden)))
        layers.append(nn.Tanh())
        dim = int(hidden)
    return nn.Sequential(*layers), dim


def _head(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(int(in_dim), int(out_dim))
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class SquashedGaussianActor(nn.Module):
    """Maps the actor feature to a latent noise ``w = tanh(u) * noise_scale``.

    The feature is the frozen observation encoding. This module has no input
    for the privileged state.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        *,
        hidden: int,
        n_layers: int,
        noise_scale: float,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.noise_scale = float(noise_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk, width = _mlp(self.feat_dim, hidden, n_layers, use_layer_norm)
        self.mean_head = _head(width, self.action_dim, gain=0.01)
        self.log_std_head = _head(width, self.action_dim, gain=1.0)
        nn.init.constant_(self.log_std_head.bias, float(log_std_init))

    def _gaussian(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"actor feature {feat.shape[-1]} != {self.feat_dim}")
        hidden = self.trunk(feat)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized ``w`` and the tanh-corrected log probability, shape ``(B, 1)``."""
        mean, log_std = self._gaussian(feat)
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.noise_scale
        log_prob = dist.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, feat: torch.Tensor) -> torch.Tensor:
        """``tanh(mean) * noise_scale``. Validation and eval use this."""
        mean, _log_std = self._gaussian(feat)
        return torch.tanh(mean) * self.noise_scale


class TwinCritic(nn.Module):
    """Two Q functions of ``(privileged, actor feature, latent action)``."""

    def __init__(
        self,
        priv_dim: int,
        feat_dim: int,
        action_dim: int,
        *,
        hidden: int,
        n_layers: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.priv_dim = int(priv_dim)
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.in_dim = self.priv_dim + self.feat_dim + self.action_dim
        self.trunk1, width = _mlp(self.in_dim, hidden, n_layers, use_layer_norm)
        self.trunk2, _width2 = _mlp(self.in_dim, hidden, n_layers, use_layer_norm)
        self.head1 = _head(width, 1, gain=1.0)
        self.head2 = _head(width, 1, gain=1.0)

    def _cat(self, priv: torch.Tensor, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if priv.shape[-1] != self.priv_dim:
            raise ValueError(f"privileged width {priv.shape[-1]} != {self.priv_dim}")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"actor feature width {feat.shape[-1]} != {self.feat_dim}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"latent action width {action.shape[-1]} != {self.action_dim}")
        return torch.cat([priv, feat, action], dim=-1)

    def forward(self, priv: torch.Tensor, feat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self._cat(priv, feat, action)
        return self.head1(self.trunk1(flat)), self.head2(self.trunk2(flat))

    def min_q(self, priv: torch.Tensor, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(priv, feat, action)
        return torch.min(q1, q2)
