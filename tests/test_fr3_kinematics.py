"""FR3 kinematics against controller-reported poses and IK round trips. / FR3 기구학 검증.

The reference joint/flange pair was read from a real FR3 controller (firmware v3.8.6) with the
calibrated DH set below.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from factor.robots.fr3 import FR3Kinematics, FR3Params, top_down_rotation  # noqa: E402

CALIBRATED = FR3Params(
    d1=0.14, a2=-0.280491, a3=-0.239929, d4=0.102452, d5=0.102123, d6=0.101, tool_length=0.2045
)


def test_forward_matches_controller_flange():
    kin = FR3Kinematics(CALIBRATED)
    q = np.radians([61.48, -63.42, 56.75, -83.32, -90.0, -119.55])
    flange = kin.flange(q)[:3, 3] * 1000.0
    assert np.allclose(flange, [-132.4, -458.3, 317.7], atol=1.0)


def test_inverse_round_trip_top_down():
    kin = FR3Kinematics(CALIBRATED)
    rot = top_down_rotation("y", 1, yaw=np.radians(-178.96))
    branch = (0, 0, 1)
    for tip in ([-0.0744 + 0.058, -0.4572, 0.12], [-0.0744, -0.4572, 0.09], [-0.0744 - 0.058, -0.4572, 0.06]):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = tip
        q = kin.inverse(pose, branch)
        assert q is not None
        back = kin.tool(q)
        assert np.allclose(back[:3, 3], tip, atol=1e-6)
        assert np.allclose(back[:3, :3], rot, atol=1e-6)


def test_batch_reports_unreachable():
    kin = FR3Kinematics(CALIBRATED)
    rot = top_down_rotation()
    poses = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    poses[:, :3, :3] = torch.as_tensor(rot)
    poses[0, :3, 3] = torch.tensor([0.0, -0.45, 0.1])
    poses[1, :3, 3] = torch.tensor([0.0, -1.50, 0.1])
    _, valid = kin.inverse_batch(poses, (0, 0, 1))
    assert bool(valid[0]) and not bool(valid[1])


def test_numpy_inverse_matches_batch():
    kin = FR3Kinematics(CALIBRATED)
    rot = top_down_rotation("y", 1, yaw=np.radians(-178.96))
    poses = []
    for tip in ([-0.0744 + 0.058, -0.4572, 0.12], [-0.0744, -0.4572, 0.09], [-0.03, -0.42, 0.07]):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = tip
        poses.append(pose)
    q_batch, valid = kin.inverse_batch(torch.as_tensor(np.stack(poses)), (0, 0, 1))
    for pose, qb, ok in zip(poses, q_batch, valid, strict=True):
        q = kin.inverse(pose, (0, 0, 1))
        assert bool(ok) and q is not None
        assert np.allclose(q, qb.numpy(), atol=1e-9)
