"""Belief render channel: where the robot *thinks* its gripper is, drawn into the canonical view.

The canonical view is anchored by the fiducial detection, so it shows the true scene. The robot's
belief about the anchor enters only through its own grip point expressed in the believed anchor
frame. Drawing that point (and a short segment up the finger axis) into the view as an extra
channel turns "where is the belief wrong" into a local image comparison: the network sees the
real fingers and the drawn belief side by side. Needs no camera calibration — only the virtual
canonical camera and the detected anchor.

믿음 렌더 채널: 로봇이 믿는 그립점(과 핑거 방향의 짧은 선분)을 정규화 시야에 그린 채널. 실제 핑거와
믿음의 위치를 같은 영상에서 비교하게 만든다. 카메라 보정이 필요 없다.

The simulation twin uses a batched torch version with the same constants
(:meth:`factor.sim.isaaclab.twin.HanoiTwin.belief_render`).
"""

from __future__ import annotations

import numpy as np

SEGMENT_HEIGHT = 0.04  # metres drawn up the finger axis from the grip point / 그립점 위로 그리는 길이
SEGMENT_POINTS = 9
SIGMA_PX = 1.5


def segment_points(grip_in_anchor: np.ndarray, half_gap: float | None = None) -> np.ndarray:
    """Points of the believed gripper, anchor frame ``(K, 3)``.

    Without ``half_gap``: one segment up the tool axis from the grip point. With it: that segment
    plus the two fingers at ``±half_gap`` along the finger axis (anchor y), a coarse silhouette
    that gives the network more pixels to compare with the real fingers.
    ``half_gap`` 가 있으면 핑거 두 개(앵커 y 축으로 ±반개구)까지 그려 실제 핑거와 비교할 픽셀을 늘린다.
    """
    t = np.linspace(0.0, SEGMENT_HEIGHT, SEGMENT_POINTS)
    up = t[:, None] * np.array([0.0, 0.0, 1.0])
    g = np.asarray(grip_in_anchor, dtype=float)[None, :]
    if half_gap is None:
        return g + up
    side = np.array([0.0, float(half_gap), 0.0])
    return np.concatenate([g + up, g + side + up, g - side + up])


def render(pixels: np.ndarray, resolution: int, sigma: float = SIGMA_PX) -> np.ndarray:
    """Max of Gaussians at ``pixels`` ``(K, 2)`` → ``(resolution, resolution)`` float in 0..1."""
    v, u = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    d2 = (u[None] - pixels[:, 0, None, None]) ** 2 + (v[None] - pixels[:, 1, None, None]) ** 2
    return np.exp(-d2.min(axis=0) / (2.0 * sigma * sigma)).astype(np.float32)
