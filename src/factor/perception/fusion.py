"""Temporal fusion of per-frame pose corrections.

The estimator returns a correction for every frame. The scene it corrects (the anchor pose
error) is constant within an episode, so the corrections are averaged over time. Updates are
suspended while the gripper is below the hover height, where the tool occludes the target and
single-frame estimates degrade; the last good value is latched until the gripper rises again.

프레임마다 나오는 보정량을 시간 축으로 융합한다. 보정 대상(앵커 포즈 오차)은 에피소드 안에서
상수이므로 지수이동평균으로 잡음을 줄이고, 그리퍼가 hover 높이 아래로 내려가 대상을 가리는
동안은 갱신을 멈추고 마지막 값을 고정(latch)한다.
"""

from __future__ import annotations

import numpy as np


class CorrectionFilter:
    """Exponential moving average with a height latch. / 높이 latch 가 있는 지수이동평균.

    Parameters
    ----------
    alpha
        Weight of the previous estimate (0 disables smoothing).
    latch_below
        Updates are frozen when the grip point is more than this far below ``hover_height``.
    hover_height
        Height of the grip point at which the target is fully visible.
    """

    def __init__(self, alpha: float, latch_below: float, hover_height: float):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = alpha
        self.latch_below = latch_below
        self.hover_height = hover_height
        self._state: np.ndarray | None = None

    @property
    def value(self) -> np.ndarray | None:
        """Current fused correction, or ``None`` before the first update. / 현재 융합값."""
        return None if self._state is None else self._state.copy()

    def reset(self) -> None:
        """Forget the history, e.g. when the believed anchor pose changes. / 믿음이 바뀌면 초기화."""
        self._state = None

    def update(self, correction: np.ndarray, grip_height: float) -> np.ndarray:
        """Fuse one frame. / 한 프레임 융합.

        The first frame after a reset is taken as is. Later frames are averaged only while the
        grip point is high enough to see the target.
        """
        correction = np.asarray(correction, dtype=float)
        if self._state is None:
            self._state = correction.copy()
        elif grip_height > self.hover_height - self.latch_below:
            self._state = self.alpha * self._state + (1.0 - self.alpha) * correction
        return self._state.copy()
