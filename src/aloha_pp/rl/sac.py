"""SAC update used by DSRL.

Critic, actor, then temperature, then a Polyak update of both Q networks.
The backup uses the minimum target Q and stops the bootstrap on terminal
transitions only. ``target_entropy`` is the value the official can config
sets (0), not ``-action_dim``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from aloha_pp.rl.nets import SquashedGaussianActor, TwinCritic
from aloha_pp.rl.reward import td_target


def alpha_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Temperature loss. Too-peaked ``log_prob`` (above ``-target``) raises alpha."""
    return -(log_alpha * (log_prob.detach() + float(target_entropy))).mean()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """``target = (1 - tau) * target + tau * source``."""
    with torch.no_grad():
        for online, shadow in zip(source.parameters(), target.parameters()):
            shadow.mul_(1.0 - float(tau)).add_(online, alpha=float(tau))


class SAC:
    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: TwinCritic,
        *,
        lr: float,
        gamma: float,
        tau: float,
        target_entropy: float,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target = copy.deepcopy(critic)
        self.target.requires_grad_(False)
        self.target.eval()
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = float(target_entropy)
        self.log_alpha = nn.Parameter(torch.zeros((), device=next(actor.parameters()).device))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(lr))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(lr))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        reward = batch["reward"]
        terminated = batch["terminated"]
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(batch["next_feat"])
            next_q = self.target.min_q(batch["next_priv"], batch["next_feat"], next_action)
            next_q = next_q - self.alpha.detach() * next_logp
            backup = td_target(reward, terminated, self.gamma, next_q)

        q1, q2 = self.critic(batch["priv"], batch["feat"], batch["action"])
        critic_loss = torch.nn.functional.mse_loss(q1, backup) + torch.nn.functional.mse_loss(q2, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        action, log_prob = self.actor.sample(batch["feat"])
        actor_loss = (self.alpha.detach() * log_prob - self.critic.min_q(batch["priv"], batch["feat"], action)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        temperature_loss = alpha_loss(self.log_alpha, log_prob, self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.alpha_opt.step()

        soft_update(self.critic, self.target, self.tau)
        with torch.no_grad():
            q_all = torch.cat([q1, q2], dim=0)
            target_q = backup.detach()
            deterministic = self.actor.deterministic(batch["feat"])
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(temperature_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "logp_mean": float(log_prob.detach().mean()),
            "q_mean": float(q_all.mean()),
            "q_max": float(q_all.max()),
            "target_q_mean": float(target_q.mean()),
            "target_q_max": float(target_q.max()),
            "actor_abs_mean": float(deterministic.abs().mean()),
        }
