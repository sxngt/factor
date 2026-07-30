"""Rule-based pick-and-place skill with learned pose correction.

Each tick the skill receives the grip point, the nominal object and goal positions computed from
the scene model, the current gripper state and, optionally, a per-frame pose correction from the
estimator. It fuses the correction over time, applies it to the nominal targets, runs the funnel
planner and the contact search, and returns a displacement plus a gripper command.

규칙 기반 집기·놓기 스킬. 매 틱 그립점, 장면 모델에서 계산한 공칭 물체·목표 위치, 그리퍼 상태,
(선택) 추정기의 프레임별 포즈 보정을 받아 시간 융합 → 공칭 목표 보정 → 깔때기 플래너 → 접촉 탐색
순으로 처리하고 이동량과 그리퍼 명령을 돌려준다.

The correction is a rigid 2-D transform of the anchor frame: ``(dx, dy, dyaw)``. It is applied
to every nominal point about the anchor origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factor.perception.fusion import CorrectionFilter
from factor.planning.funnel import FunnelPlanner
from factor.planning.nudge import NudgeSearch
from factor.skills.residual_np import skill_features


@dataclass(frozen=True)
class SkillObservation:
    grip: np.ndarray
    object_nominal: np.ndarray
    goal_nominal: np.ndarray
    holding: bool
    correction: np.ndarray | None = None
    object_offset: np.ndarray | None = None  # held object centre minus grip point / 손 안 오프셋 추정


@dataclass(frozen=True)
class SkillCommand:
    step: np.ndarray
    close: bool
    object_corrected: np.ndarray
    goal_corrected: np.ndarray
    target: np.ndarray  # the point the skill is heading for this tick / 이번 틱의 목표점


def apply_correction(point: np.ndarray, correction: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Rotate ``point`` about ``origin`` by ``dyaw`` and translate by ``(dx, dy)``.

    앵커 원점 기준 회전 + 평행이동. z 는 유지한다.
    """
    dx, dy, dyaw = correction[:3]
    dz = float(correction[3]) if len(correction) > 3 else 0.0  # height correction / 높이 보정
    c, s = np.cos(dyaw), np.sin(dyaw)
    rel = point[:2] - origin[:2]
    return np.array(
        [origin[0] + c * rel[0] - s * rel[1] + dx, origin[1] + s * rel[0] + c * rel[1] + dy, point[2] + dz]
    )


class PickPlaceSkill:
    """Pick one object and place it on its goal. / 물체 하나를 집어 목표에 놓는다.

    Parameters
    ----------
    planner
        Funnel planner configured for the scene heights.
    anchor_origin
        Anchor-frame origin expressed in the planner's frame; corrections rotate about it.
    fusion, nudge
        Temporal fusion of corrections and the insertion search. Both are optional.
    residual
        Learned residual policy (features → displacement in metres), e.g.
        :class:`factor.skills.residual_np.NumpyResidualPolicy`. Optional; without it the skill
        is purely rule-based.
    """

    def __init__(
        self,
        planner: FunnelPlanner,
        anchor_origin: np.ndarray,
        fusion: CorrectionFilter | None = None,
        nudge: NudgeSearch | None = None,
        residual=None,
    ):
        self.planner = planner
        self.anchor_origin = np.asarray(anchor_origin, dtype=float)
        self.fusion = fusion
        self.nudge = nudge
        self.residual = residual
        self._was_holding = False
        self._last_residual = np.zeros(3)
        self._retry_hold = 0
        self.retry_hold_steps = 5
        self.hop_height = 0.003  # lift before each lateral search shift / 옆 이동 전 들어 올리는 높이
        self._hop_until_z = -1.0
        self._last_target_z: float | None = None

    def reset(self) -> None:
        if self.fusion is not None:
            self.fusion.reset()
        if self.nudge is not None:
            self.nudge.reset()
        self._was_holding = False
        self._last_residual = np.zeros(3)
        self._retry_hold = 0
        self._hop_until_z = -1.0
        self._last_target_z = None

    def _lift_for_retry(self, step: np.ndarray, grip_z: float) -> np.ndarray:
        """Exhausted contact search: rise to hover, wait, descend again. / 탐색 소진 시 재시도."""
        if self.nudge is None or not self.nudge.retry:
            return step
        if grip_z > self.planner.hover_height - 0.005:
            self._retry_hold += 1
            if self._retry_hold >= self.retry_hold_steps:
                self._retry_hold = 0
                self.nudge.clear_retry()
                return step
        else:
            self._retry_hold = 0
        lift = np.zeros(3)
        lift[2] = np.clip(self.planner.hover_height - grip_z, -self.planner.max_step, self.planner.max_step)
        return lift

    def step(self, obs: SkillObservation) -> SkillCommand:
        correction = np.zeros(3)
        if obs.correction is not None:
            correction = (
                obs.correction if self.fusion is None else self.fusion.update(obs.correction, obs.grip[2])
            )

        object_pos = apply_correction(
            np.asarray(obs.object_nominal, dtype=float), correction, self.anchor_origin
        )
        goal_pos = apply_correction(np.asarray(obs.goal_nominal, dtype=float), correction, self.anchor_origin)
        if obs.holding and obs.object_offset is not None:
            goal_pos = goal_pos - np.array([obs.object_offset[0], obs.object_offset[1], 0.0])

        zone_top = self.planner.zone_top
        near_goal = (
            float(np.linalg.norm(np.asarray(obs.grip[:2]) - goal_pos[:2])) <= self.planner.tol.funnel_xy
        )
        inserting = bool(
            obs.holding
            and near_goal  # not while lifting off the source stack / 출발 탑에서 들어 올릴 때는 아님
            and obs.grip[2] < zone_top + self.planner.tol.funnel_z + 0.004
            and obs.grip[2] > zone_top - 0.002
        )
        hop = False
        if self.nudge is not None:
            if not obs.holding and self._was_holding:
                self.nudge.reset()
            before = self.nudge.index
            goal_pos = goal_pos + np.append(
                self.nudge.update(obs.grip[2], inserting, self._last_target_z), 0.0
            )
            if self.nudge.index != before and inserting:
                # New search offset: lift a little first so the object is not dragged across the peg.
                # 새 탐색 오프셋: 끌지 않도록 먼저 조금 들어 올린다.
                self._hop_until_z = float(obs.grip[2]) + self.hop_height
            hop = obs.holding and obs.grip[2] < self._hop_until_z
            if not hop:
                self._hop_until_z = -1.0  # one-shot / 한 번만
        self._was_holding = obs.holding

        cmd = self.planner.plan(obs.grip, object_pos, goal_pos, obs.holding)
        step = self._lift_for_retry(cmd.step, float(obs.grip[2])) if obs.holding else cmd.step
        if hop and step is cmd.step:
            step = np.array([0.0, 0.0, min(self._hop_until_z - float(obs.grip[2]), self.planner.max_step)])
        target = cmd.target if step is cmd.step else np.asarray(obs.grip, dtype=float) + step
        if self.residual is not None:
            features = skill_features(
                obs.grip,
                object_pos,
                goal_pos,
                obs.holding,
                inserting,
                0 if self.nudge is None else self.nudge.index,
                0 if self.nudge is None else self.nudge.stall,
                self._last_residual,
                self.planner.hover_height,
                zone_top,
                self.planner.max_step,
            )
            self._last_residual = np.asarray(self.residual(features), dtype=float)
            step = np.clip(step + self._last_residual, -self.planner.max_step, self.planner.max_step)
            target = np.asarray(obs.grip, dtype=float) + step
        self._last_target_z = float(target[2])
        return SkillCommand(
            step=step, close=cmd.close, object_corrected=object_pos, goal_corrected=goal_pos, target=target
        )
