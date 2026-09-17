"""Hindsight labels: pick frames from the grasp point, place frames only when seated."""

import numpy as np

from factor.real.recorder import MoveRecorder


def test_labels_from_grasp_and_release(tmp_path):
    rec = MoveRecorder(str(tmp_path))
    for _ in range(3):
        rec.frame(np.zeros((8, 8, 4), np.uint8), np.zeros(3), holding=False)
    rec.grasped(np.array([0.062, 0.003, 0.05]))
    rec.verified = True  # the loop sets this when the vision check confirms the grasp
    for _ in range(2):
        rec.frame(np.zeros((8, 8, 4), np.uint8), np.zeros(3), holding=True)
    rec.released(np.array([0.001, -0.002, 0.02]))
    d = np.load(rec.save(np.array([0.06, 0.0, 0.05]), np.array([0.0, 0.0, 0.02]), seated=True))
    assert np.allclose(d["labels"][:3], [0.002, 0.003], atol=1e-6)
    assert np.allclose(d["labels"][3:], [0.001, -0.002], atol=1e-6)
    d2 = np.load(rec.save(np.array([0.06, 0.0, 0.05]), np.array([0.0, 0.0, 0.02]), seated=False, tag="_b"))
    assert np.isnan(d2["labels"][3:]).all()


def test_unverified_grasp_is_not_saved(tmp_path):
    rec = MoveRecorder(str(tmp_path))
    rec.frame(np.zeros((8, 8, 4), np.uint8), np.zeros(3), holding=False)
    rec.grasped(np.array([0.06, 0.0, 0.05]))
    assert rec.save(np.zeros(3), np.zeros(3), seated=False) is None
