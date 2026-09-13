"""Live telemetry of the real-robot loop. / 실기 루프 라이브 텔레메트리.

Every tick the loop hands over its signals (fiducial pose, raw and fused corrections, grasp
probability, skill phase, contact-search state, commanded target) and the images involved. This
module draws what the system *believes* over the camera frame and over the canonical view, and
writes everything to a directory that :mod:`factor.real.viewer` serves in a browser:

- ``camera.jpg``   the raw frame with detected markers, the anchor axes, the believed scene
  geometry (plate, pegs, every ring) and the corrected object / goal / held-ring estimate;
- ``canonical.jpg`` the estimator input with the believed and corrected outlines, like the
  simulation overlay videos;
- ``state.json``   the latest tick; ``log.jsonl`` every tick appended.

매 틱 루프가 넘겨주는 신호(앵커 포즈, 원시·융합 보정, 파지 확률, 스킬 단계, 접촉 탐색 상태, 목표)와
영상을 받아, 시스템이 믿고 있는 기하와 추정 기하를 원본 프레임·정규화 시야 위에 그리고, 뷰어가
읽는 디렉터리에 기록한다. Overlays reuse the drawing code of the simulation videos so that sim and
real are read the same way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from factor.perception.canonical import VirtualCamera
from factor.perception.overlay import (
    COLOUR_ESTIMATE,
    COLOUR_GOAL,
    camera_projector,
    canonical_projector,
    draw_peg,
    draw_polyline,
    draw_ring,
    label,
)

COLOUR_BELIEF = (0, 200, 255)  # believed (nominal) geometry, BGR / 믿는 기하
COLOUR_GRIP = (255, 255, 255)
COLOUR_MARKER = (255, 120, 0)


@dataclass
class SceneGeometry:
    """Static geometry to draw, in the anchor frame. / 앵커 프레임의 정적 기하."""

    plate: np.ndarray  # (4, 3) corners of the plate top / 판 윗면 모서리
    pegs: list[np.ndarray]  # peg base points / 페그 밑점
    peg_height: float
    rings: list[tuple[np.ndarray, float, float]] = field(default_factory=list)  # (centre, radius, height)


@dataclass
class MoveGeometry:
    """Geometry of the move in progress, in the anchor frame. / 진행 중인 수의 기하."""

    object_center: np.ndarray
    goal_center: np.ndarray
    radius: float
    height: float
    name: str = ""


