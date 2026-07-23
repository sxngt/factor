"""Scene specification loading. / 장면 명세 로딩 검증."""

from pathlib import Path

import numpy as np
import pytest

from factor.scene import load_scene

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hanoi.yaml"


def test_load_example_scene():
    scene = load_scene(CONFIG)
    assert scene.robot.joint_names == ("j1", "j2", "j3", "j4", "j5", "j6")
    assert scene.camera.matrix[0, 0] == pytest.approx(1979.1, abs=0.1)
    assert scene.fiducials.ids == (0, 1, 2, 3, 4, 5)
    assert len(scene.objects) == 6
    assert scene.object("ring5").grasp_half_gap == 0.0135
    # The believed anchor is a session value (updated when the tower moves); check it is plausible.
    x, y, z = scene.anchor_in_base[:3, 3]
    assert -0.15 < x < 0.05 and -0.55 < y < -0.35 and 0.0 < z < 0.05
    yaw = np.degrees(np.arctan2(scene.anchor_in_base[1, 0], scene.anchor_in_base[0, 0]))
    assert abs(yaw - (-178.96)) < 1e-6
    assert scene.extras["task"] == "hanoi"
