"""Safety envelope for commanded grip points. / 명령 그립점의 안전 상자.

The box is axis-aligned in a frame of choice (usually the anchor frame so that it moves with
the task). Every command is clamped into it before inverse kinematics; a command that had to be
clamped is reported so the caller can log or abort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceBox:
    """Axis-aligned limits ``lo <= p <= hi``. / 축 정렬 한계."""

    lo: np.ndarray
    hi: np.ndarray

    def clamp(self, point: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the clamped point and whether clamping changed it. / 클램프된 점과 변경 여부."""
        point = np.asarray(point, dtype=float)
        clamped = np.minimum(np.maximum(point, self.lo), self.hi)
        return clamped, bool(np.any(clamped != point))

    def contains(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        return bool(np.all(point >= self.lo) and np.all(point <= self.hi))
