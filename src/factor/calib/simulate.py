"""Synthetic check of the active self-calibration (no robot, no camera).

A ground-truth camera, anchor and board mount are drawn around the scene file's values; the
"camera" returns noisy corner detections of what would be visible. Strategies are compared on the
error of the peg points in the base frame against simulated wall-clock time.

합성 검증: 참값 카메라·앵커·보드 장착을 장면 값 근처에서 뽑고, 보이는 코너만 잡음과 함께 돌려준다.
전략별로 (모의 경과 시간 대비) 베이스 기준 기둥 점 오차를 비교한다.

    python -m factor.calib.simulate --scene configs/hanoi.yaml --seeds 20 --budget 60
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from factor.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    average_poses,
    choose_next,
    move_time,
    perturb,
    project,
)
from factor.calib.poses import generate, search_poses


def peg_points(spacing: float = 0.06, plate: float = 0.007, height: float = 0.055) -> np.ndarray:
    """Peg feet and tops in the anchor frame. / 기둥 아래·위 점 (앵커 좌표)."""
    return np.array([[x, 0.0, z] for x in (-spacing, 0.0, spacing) for z in (plate, plate + height)])


def mat_corners(scene) -> np.ndarray:
    out = []
    for m in scene.fiducials.markers:
        cx, cy, cz = m.center
        h = m.size / 2
        out += [[cx - h, cy + h, cz], [cx + h, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]]
    return np.array(out)


