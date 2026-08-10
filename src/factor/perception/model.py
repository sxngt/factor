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


