"""Canonical view geometry. / 정규화 시야 기하 검증.

The look-at point must land at the image centre and the anchor z axis must point up in the
image, for any camera placement.
"""

import numpy as np

from factor.perception.canonical import CanonicalView, VirtualCamera


def anchor_in_camera(cam_pos_anchor):
    """Camera looking roughly at the anchor origin from ``cam_pos_anchor``. / 앵커 원점을 향한 카메라."""
    cam_pos_anchor = np.asarray(cam_pos_anchor, float)
    forward = -cam_pos_anchor / np.linalg.norm(cam_pos_anchor)
    right = np.cross(np.array([0.0, 0.0, -1.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot_cam_in_anchor = np.stack([right, down, forward], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = rot_cam_in_anchor.T
    pose[:3, 3] = -rot_cam_in_anchor.T @ cam_pos_anchor
    return pose


def test_look_at_lands_at_center_and_up_is_up():
    virtual = VirtualCamera(resolution=128, view_width=0.14, distance=0.81)
    view = CanonicalView(
        np.array([[1979.0, 0, 1296.0], [0, 1979.0, 972.0], [0, 0, 1.0]]), np.zeros(5), virtual
    )
    for cam_pos in ([0.1, -0.4, 0.7], [-0.3, -0.5, 0.5], [0.2, -0.6, 0.9]):
        pose = anchor_in_camera(cam_pos)
        look = np.array([0.0, 0.0, 0.037])
        center = pose[:3, :3] @ look + pose[:3, 3]
        uv = view.project(pose, look, center)
        assert np.allclose(uv, [64.0, 64.0], atol=1e-6)

        above = pose[:3, :3] @ (look + np.array([0.0, 0.0, 0.02])) + pose[:3, 3]
        uv_above = view.project(pose, look, above)
        assert uv_above[1] < 64.0
        assert abs(uv_above[0] - 64.0) < 1e-6


def test_metres_per_pixel():
    virtual = VirtualCamera(resolution=128, view_width=0.14, distance=0.81)
    assert abs(virtual.metres_per_pixel - 0.14 / 128) < 1e-12
    assert abs(np.degrees(virtual.hfov) - 9.87) < 0.05
