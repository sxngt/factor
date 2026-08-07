"""Fiducial detection on a synthetic board image. / 합성 보드 영상에서의 검출 검증."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from factor.real.fiducial import FiducialDetector  # noqa: E402
from factor.scene.spec import CameraSpec, FiducialLayout, FiducialMarker  # noqa: E402

PIX_PER_M = 4000.0  # synthetic top-down image scale / 합성 영상 축척


def make_layout() -> FiducialLayout:
    markers = []
    k = 0
    for y in (0.045, -0.045):
        for x in (-0.075, 0.0, 0.075):
            markers.append(FiducialMarker(marker_id=k, center=(x, y, 0.0), size=0.04))
            k += 1
    return FiducialLayout(dictionary="DICT_4X4_50", markers=tuple(markers))


def render_board(layout: FiducialLayout, width: int, height: int) -> np.ndarray:
    """Top-down orthographic render: anchor x right, y up, origin at the image centre."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((height, width), 255, dtype=np.uint8)
    for m in layout.markers:
        size_px = int(round(m.size * PIX_PER_M))
        tile = cv2.aruco.generateImageMarker(dictionary, m.marker_id, size_px)
        u = int(round(width / 2 + (m.center[0] - m.size / 2) * PIX_PER_M))
        v = int(round(height / 2 - (m.center[1] + m.size / 2) * PIX_PER_M))
        img[v : v + size_px, u : u + size_px] = tile
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_detects_pose_of_synthetic_board():
    layout = make_layout()
    width, height = 1000, 600
    rgb = render_board(layout, width, height)
    distance = 1.0
    f = PIX_PER_M * distance
    camera = CameraSpec(width=width, height=height, fx=f, fy=f, cx=width / 2, cy=height / 2)
    det = FiducialDetector(layout, camera).detect(rgb)
    assert det is not None
    assert len(det.marker_ids) == 6
    assert det.reprojection_rms_px < 1.0
    t = det.anchor_in_camera[:3, 3]
    assert abs(t[2] - distance) < 0.01
    assert np.linalg.norm(t[:2]) < 0.002
    # anchor y up in the image means anchor y is opposite to camera y (OpenCV y down)
    assert det.anchor_in_camera[1, 1] < -0.99
