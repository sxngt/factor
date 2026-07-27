"""Funnel planner for a single pick-and-place of a stacked object.

The planner is deliberately simple and continuous so that a learned student can imitate it and
a learned corrector can shift its targets. Approach is a funnel: while the lateral error is
large the tool stays at hover height; inside the funnel it descends while correcting laterally,
but stops a safety margin above the object until the lateral error is within the pick tolerance.
Placing mirrors this above the goal with the insertion tolerance.

하나의 물체를 집어 옮기는 깔때기 플래너. 옆 오차가 크면 hover 높이에서 이동, 깔때기 안에서는
옆을 맞추며 하강하되 정렬 전엔 물체 위 여유 높이에서 멈춘다. 놓기는 목표 위에서 같은 규칙을
삽입 허용치로 적용한다. 학습 층은 이 플래너가 쓰는 목표점만 보정한다.

All positions are 3-vectors in one common frame (the robot base or the anchor frame); the
planner never needs to know which.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FunnelTolerances:
    """Distances in metres. / 거리(m)."""

    pick_xy: float = 0.003
    place_xy: float = 0.002
    z: float = 0.002
    funnel_xy: float = 0.012
    funnel_z: float = 0.012
    hover_rise: float = 0.010
    freeze_xy_below: float = 0.0075  # above the object centre: fingers are around it, no lateral motion


@dataclass(frozen=True)
class PlannerCommand:
    """One step of the planner. / 플래너 한 스텝 출력.

    ``step`` is the commanded grip-point displacement (already saturated to ``max_step``) and
    ``close`` the desired gripper state.
    """

    step: np.ndarray
    close: bool
    target: np.ndarray


class FunnelPlanner:
    """Stateless funnel planner. / 상태 없는 깔때기 플래너.

    Parameters
    ----------
    hover_height
        Height of the grip point when travelling.
    zone_top
        Height above which a held object clears every obstacle (for example the peg tops).
    max_step
        Saturation of the commanded displacement per call, in metres.
    """

    def __init__(
        self, hover_height: float, zone_top: float, max_step: float, tol: FunnelTolerances | None = None
    ):
        self.hover_height = hover_height
        self.zone_top = zone_top
        self.max_step = max_step
        self.tol = tol or FunnelTolerances()

    def plan(
        self, grip: np.ndarray, object_pos: np.ndarray, goal_pos: np.ndarray, holding: bool
    ) -> PlannerCommand:
        """Compute the next command. / 다음 명령 계산.

        ``object_pos`` is the believed centre of the object to pick and ``goal_pos`` the believed
        centre where it should rest. ``holding`` says whether the object is currently held.
        """
        grip = np.asarray(grip, dtype=float)
        target_pos, floor = (goal_pos, self.zone_top) if holding else (object_pos, None)
        target_pos = np.asarray(target_pos, dtype=float)
        tol_xy = self.tol.place_xy if holding else self.tol.pick_xy

        lateral = float(np.linalg.norm(grip[:2] - target_pos[:2]))
        aligned = lateral <= tol_xy
        # Once the fingers are down around the object, lateral correction would push it:
        # finish the descent straight down. / 핑거가 물체를 감싼 높이에선 옆 이동 금지, 수직 하강만.
        around = (not holding) and grip[2] < target_pos[2] + self.tol.freeze_xy_below
        # A held object below the peg top has entered it: slide down only. / 페그에 들어간 뒤엔 하강만.
        entered = holding and lateral <= self.tol.funnel_xy and grip[2] < self.zone_top - 0.002
        if around or entered:
            aligned = True
        far = lateral > self.tol.funnel_xy
        low = grip[2] < self.hover_height - (0.005 if holding else self.tol.hover_rise)

        if far and low:
            target = np.array([grip[0], grip[1], self.hover_height])
        elif far:
            target = np.array([target_pos[0], target_pos[1], self.hover_height])
        else:
            if aligned:
                z = target_pos[2]
            elif floor is None:
                z = max(target_pos[2] + self.tol.funnel_z, min(grip[2], self.hover_height))
            else:
                z = max(floor + self.tol.funnel_z, min(grip[2], self.hover_height))
            target = np.array([target_pos[0], target_pos[1], z])

        at_height = abs(grip[2] - target_pos[2]) < self.tol.z
        close = (not holding and aligned and at_height) or (holding and not (aligned and at_height))

        if around or entered:
            target = np.array([grip[0], grip[1], target[2]])
        step = np.clip(target - grip, -self.max_step, self.max_step)
        return PlannerCommand(step=step, close=close, target=target)
