"""Overlays of estimated 3-D geometry on canonical views.

The estimator reports *where the planner should believe the objects are*; this module draws
that belief as projected 3-D outlines (ring rims, peg axes, goal slot) so that the estimate is
visible even where the tool occludes the object. In simulation the true outline is drawn in a
second colour, which makes the residual error directly readable.

추정된 3D 기하를 정규화 시야에 윤곽으로 겹쳐 그린다. 그리퍼가 물체를 가린 구간에서도 모델이 어디를
믿고 있는지 보이며, 시뮬에선 참값 윤곽을 다른 색으로 겹쳐 오차가 바로 읽힌다. 시뮬과 실기가 같은
함수를 쓴다.

Geometry is expressed in the anchor frame (z up). The canonical camera is defined by its
position in that frame and the look-at point, with the same conventions as
:class:`factor.perception.canonical.CanonicalView`.
"""

from __future__ import annotations

import numpy as np

from factor.perception.canonical import VirtualCamera

STATE_NAMES = ("free", "between fingers", "grasped", "seated", "jammed")
# OpenCV BGR. / OpenCV 는 BGR 순서.
COLOUR_ESTIMATE = (60, 255, 60)
COLOUR_TRUTH = (255, 80, 60)
COLOUR_GOAL = (40, 220, 255)


def canonical_projector(camera_pos: np.ndarray, look_at: np.ndarray, virtual: VirtualCamera):
    """Return a function mapping anchor-frame points ``(M, 3)`` to canonical pixels ``(M, 2)``.

    앵커 프레임 3D 점 → 정규화 시야 픽셀. 가상 카메라는 ``camera_pos`` 에서 ``look_at`` 을 보고,
    영상 위쪽은 앵커 z 축이다.
    """
    camera_pos = np.asarray(camera_pos, dtype=float)
    forward = np.asarray(look_at, dtype=float) - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 0.0, -1.0]), forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward], axis=0)
    k = virtual.matrix

    def project(points: np.ndarray) -> np.ndarray:
        p = (np.asarray(points, dtype=float) - camera_pos) @ rot.T
        uvw = p @ k.T
        return uvw[:, :2] / uvw[:, 2:3]

    return project


def camera_projector(anchor_in_camera: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray):
    """Return a function mapping anchor-frame points to pixels of the *real* camera image.

    앵커 프레임 3D 점 → 실제 카메라 영상 픽셀 (왜곡 포함). 라이브 뷰어가 원본 프레임 위에 믿는
    기하와 추정 기하를 겹쳐 그릴 때 쓴다.
    """
    import cv2

    anchor_in_camera = np.asarray(anchor_in_camera, dtype=float)
    rvec, _ = cv2.Rodrigues(anchor_in_camera[:3, :3])
    tvec = anchor_in_camera[:3, 3].copy()
    k = np.asarray(camera_matrix, dtype=float)
    dist = np.asarray(distortion, dtype=float).ravel()

    def project(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
        pixels, _ = cv2.projectPoints(pts, rvec, tvec, k, dist)
        return pixels.reshape(-1, 2)

    return project


def ring_outline(center: np.ndarray, radius: float, height: float, n: int = 48) -> np.ndarray:
    """Points on the top and bottom rims of a ring. / 링 위·아래 테두리 점."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    top = circle + center + np.array([0.0, 0.0, height / 2.0])
    bottom = circle + center - np.array([0.0, 0.0, height / 2.0])
    return np.concatenate([top, bottom], axis=0)


def draw_polyline(
    image: np.ndarray, pixels: np.ndarray, colour, closed: bool = True, thickness: int = 1
) -> None:
    import cv2

    pts = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], closed, colour, thickness, lineType=cv2.LINE_AA)


def draw_ring(image: np.ndarray, project, center: np.ndarray, radius: float, height: float, colour) -> None:
    """Draw both rims and two side edges of a ring. / 링의 두 테두리와 측면 선."""
    pts = project(ring_outline(center, radius, height))
    n = len(pts) // 2
    draw_polyline(image, pts[:n], colour)
    draw_polyline(image, pts[n:], colour)
    for i in (0, n // 2):
        draw_polyline(image, pts[[i, n + i]], colour, closed=False)


def draw_peg(image: np.ndarray, project, base: np.ndarray, height: float, colour) -> None:
    pts = project(np.stack([base, base + np.array([0.0, 0.0, height])]))
    draw_polyline(image, pts, colour, closed=False)


def label(image: np.ndarray, text: str, origin=(4, 14), colour=(255, 255, 255)) -> None:
    import cv2

    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
