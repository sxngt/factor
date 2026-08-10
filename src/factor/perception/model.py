"""Estimator network for canonical views.

The network maps a canonical RGB view plus optional kinematic render channels (robot mask and
depth rendered from the URDF at the measured joint angles) to:

- a pose correction of the anchor frame (``residual``), in normalised units,
- an interaction state over ``n_states`` classes (free / between fingers / grasped / seated / jammed),
- the object centre relative to the grip point (``offset``), which is the quantity that matters
  while the tool occludes the object,
- a log-variance for the residual, used to weight temporal fusion.

Images are standardised per sample inside the network so that training buffers, simulation and
the real camera all feed raw 0..1 pixels.

정규화 시야(+ URDF 로 렌더한 로봇 마스크·깊이 채널)에서 앵커 포즈 보정량, 상호작용 상태, 불확실도를
출력하는 네트워크. 표준화는 네트워크 안에서 하므로 버퍼·시뮬·실기 모두 0~1 원본을 넣는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EstimatorConfig:
    resolution: int = 128
    in_channels: int = 3
    render_channels: int = 0
    n_residual: int = 3
    n_states: int = 5
    n_offset: int = 3
    offset_scale: float = 0.1  # metres per unit of the offset output / 오프셋 출력 단위당 미터
    pos_scale: float = 0.01  # metres per unit of the (dx, dy) correction / 보정 (dx, dy) 단위당 미터
    yaw_scale: float = 0.08726646259971647  # radians per unit of the yaw correction (5°) / yaw 단위당 rad
    features: int = 256
    belief_fingers: bool = False  # belief render draws both fingers (must match at inference)
    label: str = "global"  # "global": one anchor correction; "local": translation at the active peg
    view_width: float = 0.14  # metres across the canonical view (sets the virtual camera) / 시야 폭
    render_mode: str = "mask"  # which render channels the model expects (old checkpoints: mask)
    proprio_dim: int = 0  # believed grip point in the anchor frame (robot's own belief) / 믿는 그립점
    proprio_scale: float = 0.1  # metres per unit of the proprioceptive input
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    encoder: str = "coordconv"  # "coordconv" or "keypoints" (spatial soft-argmax) / 인코더 종류
    keypoints: int = 32


class CoordConvEncoder(nn.Module):
    """Strided convolutional encoder with coordinate channels. / 좌표 채널을 더한 합성곱 인코더.

    Coordinate channels make it easy for the network to express *where* a feature is, which is
    what a pose regressor needs.
    """

    def __init__(self, resolution: int, in_channels: int, features: int):
        super().__init__()
        channels = in_channels + 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        side = math.ceil(resolution / 16)
        self.fc = nn.Sequential(nn.Linear(128 * side * side, features), nn.ELU())

        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, resolution), torch.linspace(-1.0, 1.0, resolution), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xs, ys], dim=0).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        return self.fc(self.conv(torch.cat([x, grid], dim=1)))


def _mlp(n_in: int, hidden: tuple[int, ...], n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(n_in, width), nn.ELU()]
        n_in = width
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """Spatial soft-argmax encoder: K learned keypoints with sub-pixel image coordinates.

    Precise manipulation needs *where* things are to a fraction of a pixel; a dense layer over a
    flattened feature map is poor at that, the expected coordinate of a softmax heat map is good
    at it. The keypoints (x, y and their peak strength) plus a pooled appearance vector feed the
    heads. 공간 소프트-argmax: 학습된 K 개 키포인트의 부화소 좌표를 특징으로 쓴다.
    """

    def __init__(self, resolution: int, in_channels: int, features: int, keypoints: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ELU(),
        )
        self.heat = nn.Conv2d(64, keypoints, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        side = math.ceil(resolution / 4)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, side), torch.linspace(-1.0, 1.0, side), indexing="ij"
        )
        self.register_buffer("xs", xs.reshape(1, 1, -1))
        self.register_buffer("ys", ys.reshape(1, 1, -1))
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, resolution), torch.linspace(-1.0, 1.0, resolution), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([gx, gy], dim=0).unsqueeze(0))
        self.fc = nn.Sequential(nn.Linear(3 * keypoints + 64, features), nn.ELU())

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(torch.cat([x, self.grid.expand(x.shape[0], -1, -1, -1)], dim=1))
        logits = self.heat(h).flatten(2) / self.log_temperature.exp()
        p = torch.softmax(logits, dim=-1)
        kx = (p * self.xs).sum(-1)
        ky = (p * self.ys).sum(-1)
        strength = logits.amax(-1).tanh()
        pooled = h.mean(dim=(2, 3))
        return self.fc(torch.cat([kx, ky, strength, pooled], dim=1))


class StateEstimator(nn.Module):
    """Pose-correction and interaction-state estimator. / 포즈 보정·상호작용 상태 추정기.

    Inputs
    ------
    rgb
        ``(N, 3, H, W)`` canonical view in 0..1.
    render
        ``(N, C, H, W)`` kinematic render channels in 0..1, or ``None`` when the model was
        configured without them.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__()
        self.config = config
        channels = config.in_channels + config.render_channels
        if config.encoder == "keypoints":
            self.encoder = KeypointEncoder(config.resolution, channels, config.features, config.keypoints)
        else:
            self.encoder = CoordConvEncoder(config.resolution, channels, config.features)
        self.dropout = nn.Dropout(config.dropout)
        # Proprioception: where the robot *believes* its grip point is (anchor frame). The image
        # shows where it really is relative to the tower; the belief error is their difference,
        # so neither the camera pose nor any calibration is needed.
        # 고유수용 입력: 로봇이 믿는 그립점. 영상이 보여주는 실제 위치와의 차이가 믿음의 오차다.
        width = config.features
        if config.proprio_dim:
            self.proprio = nn.Sequential(
                nn.Linear(config.proprio_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU()
            )
            width += 64
        self.residual_head = _mlp(width, config.hidden, config.n_residual)
        self.logvar_head = _mlp(width, config.hidden, config.n_residual)
        self.state_head = _mlp(config.features, config.hidden, config.n_states)
        self.offset_head = _mlp(config.features, config.hidden, config.n_offset)

    @staticmethod
    def standardise(x: Tensor) -> Tensor:
        """Per-sample, per-channel standardisation. / 샘플·채널별 표준화."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + 0.05)

    def forward(
        self, rgb: Tensor, render: Tensor | None = None, proprio: Tensor | None = None
    ) -> dict[str, Tensor]:
        x = self.standardise(rgb)
        if render is not None:
            x = torch.cat([x, render], dim=1)
        features = self.dropout(self.encoder(x))
        fused = features
        if self.config.proprio_dim:
            if proprio is None:
                raise ValueError("this estimator needs the believed grip point (proprio)")
            fused = torch.cat(
                [features, self.proprio(proprio[:, : self.config.proprio_dim] / self.config.proprio_scale)], 1
            )
        return {
            "residual": self.residual_head(fused),
            "logvar": self.logvar_head(fused),
            "state_logits": self.state_head(features),
            "offset": self.offset_head(features),
        }


def estimator_loss(
    output: dict[str, Tensor], residual: Tensor, state: Tensor | None = None, offset: Tensor | None = None
) -> Tensor:
    """Heteroscedastic regression loss plus optional state cross-entropy.

    이분산 회귀 손실(불확실도 학습) + 상호작용 상태 교차 엔트로피(라벨이 있을 때).
    """
    logvar = output["logvar"].clamp(-6.0, 4.0)
    nll = 0.5 * (torch.exp(-logvar) * (output["residual"] - residual) ** 2 + logvar)
    loss = nll.mean()
    if state is not None:
        loss = loss + nn.functional.cross_entropy(output["state_logits"], state)
    if offset is not None:
        loss = loss + ((output["offset"] - offset) ** 2).mean()
    return loss
