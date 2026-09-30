"""Marker-free self-calibration from the wrist LED ring.

The FR3 wrist carries a ring of status LEDs around the last joint axis. Whatever their count or
angle, every LED lies on one circle: centre on the flange z axis at offset ``d``, radius ``r``,
normal along the axis. With forward kinematics that circle is known in the base frame up to
``(d, r)``, so each detected LED says "this pixel lies on the projected circle". Stacked over
poses this fixes the camera in the base frame ``X`` without any marker on the robot.

The LEDs rotate with (or around) the axis, so their identity is never needed: the residual is the
image distance from each LED blob to the projected circle, measured along the curve normal.

손목 LED 링을 이용한 무표식 자기보정. LED 개수·각도와 상관없이 모든 LED 는 플랜지 z 축을 중심으로
한 원(축 위치 d, 반지름 r) 위에 있다. 정기구학으로 그 원을 베이스 좌표에서 (d, r) 만 빼고 알 수 있으므로,
검출된 LED 마다 "이 픽셀은 투영된 원 위에 있다"는 제약이 생긴다. 여러 자세를 모으면 로봇에 표식을 붙이지
않고도 카메라-베이스 X 가 정해진다. LED 식별은 필요 없다 (원까지의 법선 거리만 쓴다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factor.calib.handeye import Prediction, perturb, pose_error, project, skew

_THETA = np.linspace(0.0, 2.0 * math.pi, 361)[:-1]


def detect_leds(rgb: np.ndarray, roi=None, min_area: int = 15, max_area: int = 4000) -> np.ndarray:
    """Centroids (n, 2) of saturated blue blobs, optionally inside ``roi`` = (x0, y0, x1, y1).

    채도 높은 파란 덩어리의 중심 (roi 안에서만).
    """
    import cv2

    x0, y0 = 0, 0
    img = rgb
    if roi is not None:
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
        x1, y1 = min(w, int(roi[2])), min(h, int(roi[3]))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 2))
        img = rgb[y0:y1, x0:x1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (92, 110, 140), (132, 255, 255))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    if not keep:
        return np.zeros((0, 2))
    return cents[keep] + np.array([x0, y0])


def _project_many(k: np.ndarray, dist: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``project`` for a (..., 3) array of camera-frame points. / 카메라 좌표 점 배열 투영."""
    shape = pts.shape[:-1]
    uv = project(k, dist, np.eye(4), pts.reshape(-1, 3))
    return uv.reshape(*shape, 2)


@dataclass
class RingObservation:
    flange_in_base: np.ndarray
    points: np.ndarray  # (n, 2) LED centroids kept by the gate (px) / 게이트를 통과한 LED 중심
    tag: str = ""
    raw: np.ndarray | None = None  # every blue blob in the ROI / ROI 안의 모든 파란 덩어리


