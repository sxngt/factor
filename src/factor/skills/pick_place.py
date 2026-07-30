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


