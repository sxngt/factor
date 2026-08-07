"""Fiducial board detection: anchor pose from the fixed camera.

The anchor board is a planar set of ArUco markers whose centres are known in the anchor frame
(:class:`factor.scene.FiducialLayout`). A single frame gives the board pose in the camera frame
by PnP over all detected corners; with the camera extrinsics this becomes the anchor pose in
the robot base frame. Low-light frames are retried after contrast enhancement.

고정 카메라 프레임에서 ArUco 보드를 검출해 앵커 포즈를 구한다. 검출된 모든 코너로 PnP 를 풀고,
카메라 외부 파라미터가 있으면 로봇 베이스 기준 앵커 포즈까지 계산한다. 어두운 프레임은 대비를
올려 재시도한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factor.scene.spec import CameraSpec, FiducialLayout


@dataclass(frozen=True)
class FiducialDetection:
    """Result of one detection. / 검출 결과 하나."""

    anchor_in_camera: np.ndarray
    marker_ids: tuple[int, ...]
    reprojection_rms_px: float
    corners: tuple = ()  # detected marker corners (image px), for overlays / 오버레이용 마커 모서리


class FiducialDetector:
    """ArUco board detector for the anchor frame. / 앵커 프레임용 ArUco 보드 검출기."""

    def __init__(self, layout: FiducialLayout, camera: CameraSpec, min_markers: int = 2):
        import cv2

        self.layout = layout
        self.camera = camera
        self.min_markers = min_markers
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionary))
        corners = np.stack([self._corners(m.center, m.size) for m in layout.markers]).astype(np.float32)
        self.board = cv2.aruco.Board(corners, dictionary, np.array(layout.ids, dtype=np.int32))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    @staticmethod
    def _corners(center, size: float) -> np.ndarray:
        """Marker corners in the anchor frame, OpenCV order (top-left, clockwise)."""
        cx, cy, cz = center
        h = size / 2.0
        return np.array(
            [[cx - h, cy + h, cz], [cx + h, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]]
        )

    def detect(self, rgb: np.ndarray) -> FiducialDetection | None:
        """Detect the board in an RGB frame. / RGB 프레임에서 보드 검출.

        Returns ``None`` when fewer than ``min_markers`` markers of the board are visible.
        """
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids = self._detect_markers(gray)
        if ids is None or len(ids) < self.min_markers:
            enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            lut = np.array([255.0 * (i / 255.0) ** 0.5 for i in range(256)], dtype=np.uint8)
            corners, ids = self._detect_markers(cv2.LUT(enhanced, lut))
        if ids is None or len(ids) < self.min_markers:
            return None

        object_points, image_points = self.board.matchImagePoints(corners, ids)
        k, dist = self.camera.matrix, self.camera.distortion
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, k, dist, rvec, tvec)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, dist)
        rms = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1)))
        )

        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.ravel()
        return FiducialDetection(
            anchor_in_camera=pose,
            marker_ids=tuple(int(i) for i in ids.ravel()),
            reprojection_rms_px=rms,
            corners=tuple(np.asarray(c).reshape(-1, 2) for c in corners),
        )

    def _detect_markers(self, gray: np.ndarray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return [], None
        keep = [i for i, k in enumerate(ids.ravel()) if int(k) in self.layout.ids]
        if not keep:
            return [], None
        return [corners[i] for i in keep], ids[keep]


def anchor_in_base(detection: FiducialDetection, camera_in_base: np.ndarray) -> np.ndarray:
    """Compose the camera extrinsics with a detection. / 외부 파라미터와 검출을 합성."""
    return np.asarray(camera_in_base, dtype=float) @ detection.anchor_in_camera


def calibrate_camera_in_base(
    detections: list[FiducialDetection], anchor_in_base_known: np.ndarray
) -> np.ndarray:
    """Camera extrinsics from frames taken while the anchor pose in the base frame is known.

    The board is placed at a pose that has been measured by the robot itself (for example by
    touching two reference points); averaging over frames suppresses detection noise.
    로봇이 직접 잰 앵커 포즈로 카메라 외부 파라미터를 구한다 (여러 프레임 평균).
    """
    poses = [
        np.asarray(anchor_in_base_known, dtype=float) @ np.linalg.inv(d.anchor_in_camera) for d in detections
    ]
    mean = np.mean(poses, axis=0)
    u, _, vt = np.linalg.svd(mean[:3, :3])
    mean[:3, :3] = u @ vt
    return mean
