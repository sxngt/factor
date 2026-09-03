"""Residual skill policy: a learned correction on top of the funnel planner.

The rule-based skill (funnel planner + contact search) is the initialisation; reinforcement
learning fine-tunes a small residual displacement added to each planner step. The policy sees
only quantities that exist on the robot too, so the same network runs in the simulation twin
(batched, torch) and in the real-robot loop (one environment).

잔차 스킬 정책. 규칙 스킬(깔때기 플래너 + 접촉 탐색)을 초기값으로 두고, 각 스텝에 더하는 작은 잔차
이동량을 강화학습으로 미세조정한다. 입력은 실기에서도 얻을 수 있는 값만 쓰므로 트윈(배치 torch)과
실기(env 하나)에서 같은 망이 돈다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

FEATURE_DIM = 15
NUDGE_STEPS = 30.0  # last contact-search index / 접촉 탐색 마지막 인덱스
POSITION_SCALE = 0.05  # metres per unit of the relative-position features / 상대 위치 피처 스케일


def skill_features(
    grip: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    holding: torch.Tensor,
    inserting: torch.Tensor,
    nudge_index: torch.Tensor,
    nudge_stall: torch.Tensor,
    last_residual: torch.Tensor,
    hover_height: float,
    zone_top: float,
    max_step: float,
) -> torch.Tensor:
    """Build the ``(N, FEATURE_DIM)`` policy input. / 정책 입력 ``(N, FEATURE_DIM)`` 구성.

    All positions are in the same (believed) frame as the planner targets.
    """
    z = grip[:, 2]
    return torch.cat(
        [
            (grip - object_pos) / POSITION_SCALE,
            (grip - goal_pos) / POSITION_SCALE,
            holding.float().unsqueeze(1),
            inserting.float().unsqueeze(1),
            ((z - zone_top) / POSITION_SCALE).unsqueeze(1),
            ((z - hover_height) / POSITION_SCALE).unsqueeze(1),
            (nudge_index.float() / NUDGE_STEPS).unsqueeze(1),
            (nudge_stall.float() / 3.0).unsqueeze(1),
            last_residual / max_step,
        ],
        1,
    )


@dataclass(frozen=True)
class ResidualConfig:
    feature_dim: int = FEATURE_DIM
    hidden: int = 128
    residual_scale: float = 0.002  # metres at a saturated action / 포화 행동에서의 잔차 (m)
    init_log_std: float = -1.0


class ResidualPolicy(nn.Module):
    """Gaussian policy with a tanh-squashed mean and a value head. / 가우시안 정책 + 가치 머리.

    ``forward`` returns ``(mean, log_std, value)``; ``residual`` converts a normalised action
    into metres. The mean head is zero-initialised so that the untrained policy reproduces the
    rule-based skill exactly.
    """

    def __init__(self, config: ResidualConfig | None = None):
        super().__init__()
        self.config = config or ResidualConfig()
        c = self.config
        self.trunk = nn.Sequential(
            nn.Linear(c.feature_dim, c.hidden), nn.Tanh(), nn.Linear(c.hidden, c.hidden), nn.Tanh()
        )
        self.mean = nn.Linear(c.hidden, 3)
        self.value = nn.Linear(c.hidden, 1)
        self.log_std = nn.Parameter(torch.full((3,), c.init_log_std))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(features)
        return torch.tanh(self.mean(h)), self.log_std.expand(h.shape[0], 3), self.value(h).squeeze(1)

    def distribution(self, features: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, log_std, value = self(features)
        return torch.distributions.Normal(mean, log_std.exp()), value

    def residual(self, action: torch.Tensor) -> torch.Tensor:
        """Normalised action → displacement in metres. / 정규화 행동 → 이동량 (m)."""
        return action.clamp(-1.0, 1.0) * self.config.residual_scale

    @torch.no_grad()
    def act(self, features: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std, _ = self(features)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * log_std.exp()


def save_policy(policy: ResidualPolicy, path: str, extra: dict | None = None) -> None:
    torch.save({"config": policy.config.__dict__, "state_dict": policy.state_dict(), **(extra or {})}, path)


def load_policy(path: str, device="cpu") -> ResidualPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    policy = ResidualPolicy(ResidualConfig(**ck["config"])).to(device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    return policy


def export_numpy(policy: ResidualPolicy, path: str) -> None:
    """Save the deterministic policy as plain arrays for torch-free inference. / NumPy 추론용 내보내기.

    Loaded by :class:`factor.skills.residual_np.NumpyResidualPolicy` in the real-robot loop.
    """
    import numpy as np

    arrays = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items() if k != "log_std"}
    np.savez(path, residual_scale=np.float64(policy.config.residual_scale), **arrays)
