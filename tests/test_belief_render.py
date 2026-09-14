"""Belief render: the drawn point lands where the canonical projector says. / 믿음 렌더 위치 검증."""

import numpy as np

from factor.perception import belief_render
from factor.perception.canonical import VirtualCamera
from factor.perception.overlay import canonical_projector


def test_peak_at_projected_grip_point():
    virtual = VirtualCamera(128, 0.14, 0.81)
    project = canonical_projector(np.array([0.07, 0.39, 0.71]), np.array([0.0, 0.0, 0.037]), virtual)
    grip = np.array([0.03, 0.0, 0.06])
    image = belief_render.render(project(belief_render.segment_points(grip)), 128)
    u, v = project(grip[None])[0]
    assert image.max() <= 1.0 and image[int(round(v)), int(round(u))] > 0.8
    assert image[0, 0] < 1e-6
