"""Minimal PPO for vectorised environments. / 벡터 환경용 최소 PPO.

Self-contained (torch only) so that skill fine-tuning does not pull in an RL framework. The
rollout buffer stores one fixed-length horizon over all environments; advantages use GAE with
bootstrapping masked at episode boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from factor.skills.residual import ResidualPolicy


@dataclass(frozen=True)
class PPOConfig:
    horizon: int = 32
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0


class Rollout:
    """Fixed-horizon storage ``(T, N, ...)`` on the policy device. / 고정 길이 롤아웃 저장소."""

    def __init__(self, horizon: int, num_envs: int, feature_dim: int, device):
        t, n = horizon, num_envs
        self.features = torch.zeros(t, n, feature_dim, device=device)
        self.actions = torch.zeros(t, n, 3, device=device)
        self.log_probs = torch.zeros(t, n, device=device)
        self.values = torch.zeros(t, n, device=device)
        self.rewards = torch.zeros(t, n, device=device)
        self.dones = torch.zeros(t, n, device=device)
        self.advantages = torch.zeros(t, n, device=device)
        self.returns = torch.zeros(t, n, device=device)
        self.t = 0

    def push(self, features, actions, log_probs, values, rewards, dones) -> None:
        i = self.t
        self.features[i] = features
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = values
        self.rewards[i] = rewards
        self.dones[i] = dones.float()
        self.t += 1

    def compute(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE over the horizon; ``dones[t]`` ends the transition taken at ``t``."""
        adv = torch.zeros_like(last_value)
        for t in reversed(range(self.t)):
            next_value = last_value if t == self.t - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values
        self.t = 0


def ppo_update(policy: ResidualPolicy, optimiser, rollout: Rollout, config: PPOConfig) -> dict[str, float]:
    """Clipped-objective update over the whole buffer. / 버퍼 전체에 대한 클리핑 목적 갱신."""
    t, n = rollout.features.shape[:2]
    flat = lambda x: x.reshape(t * n, *x.shape[2:])  # noqa: E731
    features, actions = flat(rollout.features), flat(rollout.actions)
    old_log_probs, returns = flat(rollout.log_probs), flat(rollout.returns)
    advantages = flat(rollout.advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    count = 0
    batch = max(1, t * n // config.minibatches)
    for _ in range(config.epochs):
        perm = torch.randperm(t * n, device=features.device)
        for start in range(0, t * n, batch):
            idx = perm[start : start + batch]
            dist, value = policy.distribution(features[idx])
            log_prob = dist.log_prob(actions[idx]).sum(1)
            ratio = (log_prob - old_log_probs[idx]).exp()
            adv = advantages[idx]
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv)
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (value - returns[idx]).pow(2).mean()
            entropy = dist.entropy().sum(1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimiser.step()
            stats["policy"] += float(policy_loss)
            stats["value"] += float(value_loss)
            stats["entropy"] += float(entropy)
            stats["clip_frac"] += float(((ratio - 1).abs() > config.clip).float().mean())
            count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}
