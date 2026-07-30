"""Contact search for insertion (NUDGE).

When a held object is lowered onto its goal and stops making downward progress, the estimate
is off by more than the mechanical clearance. Rather than pushing harder, the goal is shifted
along an outward spiral until the object drops in.
The search is state-based so that it works identically in simulation and on the robot.

삽입 접촉 탐색. 들고 있는 물체가 목표 위에서 더 내려가지 않으면 추정 오차가 기계적 여유보다 크다는
뜻이므로, 목표를 바깥쪽 나선을 따라 옮겨 가며 재시도한다. 상태 기반이라 시뮬과 실기에서
동일하게 동작한다.
"""

from __future__ import annotations

import math

import numpy as np

# Search offsets in units of the amplitude: an outward spiral whose successive points are
# neighbours, so the held object is never dragged across the peg. 6 points per turn, one
# amplitude of radius per turn, out to 5 amplitudes (31 points).
# 탐색 오프셋(진폭 단위): 인접점으로 이어지는 바깥 나선. 한 바퀴에 6점, 한 바퀴마다 반경 +1, 반경 5 까지.


def _spiral(points_per_turn: int = 6, turns: int = 5) -> tuple[tuple[float, float], ...]:
    out = [(0.0, 0.0)]
    for k in range(1, points_per_turn * turns + 1):
        r = k / points_per_turn
        a = 2.0 * math.pi * k / points_per_turn
        out.append((round(r * math.cos(a), 4), round(r * math.sin(a), 4)))
    return tuple(out)


PATTERN = _spiral()


class NudgeSearch:
    """Stall detector plus lateral search pattern. / 정체 감지 + 옆 탐색 패턴.

    Parameters
    ----------
    amplitude
        Base offset of the pattern in metres (1.5 mm is a good start for a 1 mm clearance).
    stall_steps
        Consecutive non-progressing steps before advancing the pattern.
    progress
        Minimum downward progress per step that counts as "moving", in metres.
    """

    def __init__(
        self,
        amplitude: float = 0.0007,
        stall_steps: int = 3,
        progress: float = 0.0005,
        blocked_lag: float = 0.0015,
    ):
        self.amplitude = amplitude
        self.blocked_lag = blocked_lag
        self.stall_steps = stall_steps
        self.progress = progress
        self._retry = False
        self._stall = 0
        self._index = 0
        self._prev_height: float | None = None

    @property
    def offset(self) -> np.ndarray:
        """Current lateral offset to add to the goal. / 목표에 더할 현재 옆 오프셋."""
        dx, dy = PATTERN[self._index]
        return np.array([dx * self.amplitude, dy * self.amplitude])

    @property
    def index(self) -> int:
        """Current pattern index. / 현재 패턴 인덱스."""
        return self._index

    @property
    def stall(self) -> int:
        """Consecutive non-progressing steps. / 연속 정체 스텝 수."""
        return self._stall

    @property
    def retry(self) -> bool:
        """Pattern exhausted: lift, let the estimate refresh, descend again. / 재시도 요청."""
        return self._retry

    def clear_retry(self) -> None:
        self._retry = False

    def reset(self) -> None:
        self._stall = 0
        self._index = 0
        self._prev_height = None
        self._retry = False

    def update(self, grip_height: float, inserting: bool, commanded: float | None = None) -> np.ndarray:
        """Advance the detector with the current grip height. / 현재 그립 높이로 상태 갱신.

        ``inserting`` is true while the planner is trying to lower the held object into the
        goal; outside that phase the search is idle.
        """
        if not inserting:
            self._stall = 0
            self._prev_height = grip_height
            return self.offset

        moved = self._prev_height is not None and (self._prev_height - grip_height) >= self.progress
        # Not blocked if the arm already sits at the height it was told: the planner itself is holding
        # or creeping (lateral alignment, slow insert). Only a lag behind the command is a stall.
        # 명령한 높이에 이미 있으면 막힌 것이 아니다 (계획기가 정렬 중이거나 천천히 내리는 중).
        if commanded is not None and grip_height - commanded < self.blocked_lag:
            moved = True
        self._stall = 0 if moved else self._stall + 1
        self._prev_height = grip_height

        if self._stall >= self.stall_steps:
            self._stall = 0
            if self._index == len(PATTERN) - 1:
                self._index = 0
                self._retry = True
            else:
                self._index += 1
        return self.offset
