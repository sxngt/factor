"""Fiducial-anchored canonical views.

A canonical view is the image a *virtual* camera would see if it shared the real camera's
optical centre but looked straight at a chosen point of the anchor frame with the anchor's
z axis as image "up". Because only the orientation changes, the mapping from the real image is
a pure rotation homography ``H = K_v R K^-1`` and is exact for every depth. The simulation twin
reproduces the same convention by aiming its camera at the same point, so a network trained in
simulation sees the same geometry on the real robot regardless of where the camera or the anchor
board were placed.

정규화 시야. 실제 카메라와 광학 중심을 공유하되 앵커 프레임의 지정 점을 정면으로 보는 가상
카메라의 영상이다. 회전만 바뀌므로 실제 영상에서의 변환은 깊이에 무관한 순수 회전 호모그래피이고,
시뮬 트윈은 카메라를 같은 점으로 조준해 같은 규약을 재현한다.

Two view scales are supported through :class:`VirtualCamera`: a coarse view that captures the
whole workspace around the anchor and a fine view zoomed on the active target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VirtualCamera:
    """Square virtual camera. / 정사각 가상 카메라.

    ``resolution`` is the edge length in pixels and ``view_width`` the width of the scene (in
    metres) that fills the image at ``distance`` metres from the look-at point.
    """

    resolution: int
    view_width: float
    distance: float

    @property
    def hfov(self) -> float:
        """Horizontal field of view in radians. / 수평 화각(rad)."""
        return 2.0 * math.atan(self.view_width / (2.0 * self.distance))

    @property
    def focal_px(self) -> float:
        return self.resolution / (2.0 * math.tan(self.hfov / 2.0))

    @property
    def matrix(self) -> np.ndarray:
        f = self.focal_px
        c = self.resolution / 2.0
        return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])

    @property
    def metres_per_pixel(self) -> float:
        """Ground sampling distance at the look-at point. / 시선점에서의 1 px 당 길이."""
        return self.view_width / self.resolution


class CanonicalView:
    """Rotation-homography warp from a real camera to a virtual camera. / 회전 호모그래피 워프.

    Parameters
    ----------
    camera_matrix, distortion
        Real camera intrinsics (OpenCV convention).
    virtual
        The virtual camera that defines the canonical view.
    """

    def __init__(self, camera_matrix: np.ndarray, distortion: np.ndarray, virtual: VirtualCamera):
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.distortion = np.asarray(distortion, dtype=float).ravel()
        self.virtual = virtual
        self._camera_matrix_inv = np.linalg.inv(self.camera_matrix)

    @staticmethod
    def rotation(anchor_in_camera: np.ndarray, look_at: np.ndarray) -> np.ndarray:
        """Rotation ``R`` that maps real-camera coordinates to virtual-camera coordinates.

        The virtual optical axis passes through ``look_at`` (a point in the anchor frame) and the
        virtual image "up" is the anchor z axis. Both cameras use the OpenCV convention
        (z forward, y down).

        실카메라 좌표 → 가상카메라 좌표 회전. 가상 광축은 앵커 프레임 점 ``look_at`` 을 지나고
        영상 위쪽은 앵커 z 축이다.
        """
        anchor_in_camera = np.asarray(anchor_in_camera, dtype=float)
        rot, trans = anchor_in_camera[:3, :3], anchor_in_camera[:3, 3]

        forward = trans + rot @ np.asarray(look_at, dtype=float)
        forward = forward / np.linalg.norm(forward)
        up_world = rot @ np.array([0.0, 0.0, 1.0])

        right = np.cross(-up_world, forward)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward], axis=0)

    def homography(self, anchor_in_camera: np.ndarray, look_at: np.ndarray) -> np.ndarray:
        """``H = K_v R K^-1`` for the undistorted real image. / 왜곡 보정된 실영상 기준 호모그래피."""
        rot = self.rotation(anchor_in_camera, look_at)
        return self.virtual.matrix @ rot @ self._camera_matrix_inv

    def warp(self, image: np.ndarray, anchor_in_camera: np.ndarray, look_at: np.ndarray) -> np.ndarray:
        """Undistort ``image`` and warp it into the canonical view. / 왜곡 보정 후 정규화 시야로 워프.

        Requires OpenCV. The return value is ``resolution x resolution x channels``.
        """
        import cv2

        undistorted = cv2.undistort(image, self.camera_matrix, self.distortion)
        size = (self.virtual.resolution, self.virtual.resolution)
        return cv2.warpPerspective(
            undistorted, self.homography(anchor_in_camera, look_at), size, flags=cv2.INTER_AREA
        )

    def project(
        self, anchor_in_camera: np.ndarray, look_at: np.ndarray, point_camera: np.ndarray
    ) -> np.ndarray:
        """Project a 3-D point given in real-camera coordinates into canonical pixels.

        Useful for tests and for drawing the nominal geometry over the view.
        실카메라 좌표의 3차원 점을 정규화 시야 픽셀로 투영한다(검증·시각화용).
        """
        rot = self.rotation(anchor_in_camera, look_at)
        p = rot @ np.asarray(point_camera, dtype=float)
        uvw = self.virtual.matrix @ p
        return uvw[:2] / uvw[2]
