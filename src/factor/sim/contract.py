"""What a simulation twin must provide. / 시뮬 트윈이 제공해야 하는 것.

This module is dependency-free so that the contract can be type-checked and documented without
a simulator installed. A concrete twin (Isaac Lab) implements :class:`TwinContract`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Randomisation:
    """Per-episode randomisation ranges. / 에피소드별 무작위화 범위.

    ``anchor_xy`` / ``anchor_yaw`` are the sampled *belief* errors of the anchor pose; they are the
    estimator's labels. ``camera_xyz`` perturbs the camera position around its nominal pose.
    Appearance terms scale image gain, add bias and Gaussian noise.
    """

    anchor_xy: float = 0.008
    anchor_yaw: float = np.radians(1.0)
    anchor_z: float = 0.0  # belief error in height (anchor z, tool length) / 높이 믿음 오차
    camera_xyz: float = 0.08  # the camera may be re-placed between runs / 카메라 위치는 매번 달라질 수 있다
    base_xy: float = 0.01
    base_yaw: float = np.radians(3.0)
    gain: float = 0.4
    bias: float = 0.08
    noise: float = 0.05
    aim_jitter: float = 0.0015  # fiducial-detection noise on the view anchor (m) / 마커 검출 잡음
    blur: float = 0.3  # probability of a mild defocus blur per frame / 흐림 확률


@dataclass(frozen=True)
class Frame:
    """One labelled observation. / 라벨이 있는 관측 하나."""

    rgb: np.ndarray
    render: np.ndarray | None
    correction: np.ndarray
    interaction_state: int
    grip: np.ndarray
    holding: bool


class TwinContract(Protocol):
    """Interface of a simulation twin. / 시뮬 트윈 인터페이스."""

    def reset(self) -> list[Frame]: ...

    def step(self, steps: np.ndarray, close: np.ndarray) -> list[Frame]: ...

    @property
    def num_envs(self) -> int: ...
