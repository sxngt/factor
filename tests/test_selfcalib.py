"""Self-calibration: the estimator recovers a known camera and the pose choice is sane."""

import math

import numpy as np

from factor.calib.handeye import ActiveHandEye, GripperBoard, choose_next, perturb, pose_error
from factor.calib.poses import generate, search_poses
from factor.calib.simulate import Rig, peg_points
from factor.robots.fr3.kinematics import FR3Kinematics, FR3Params, top_down_rotation
from factor.robots.fr3.ros_adapter import tool_yaw_from_anchor
from factor.scene import load_scene


def _setup(seed=0):
    scene = load_scene("configs/hanoi.yaml")
    board = GripperBoard.from_json("configs/gripper_board.json")
    kin = FR3Kinematics(FR3Params.from_mapping(scene.robot.kinematics))
    anchor = np.asarray(scene.anchor_in_base, dtype=float)
    rot = top_down_rotation("y", 1, yaw=tool_yaw_from_anchor(anchor))
    grip = np.asarray(scene.robot.grip_offset, dtype=float)
    centre = anchor[:3, 3] + [0, 0, 0.122]
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = centre - rot @ grip
    q0 = kin.inverse(pose, (0, 0, 1))
    rig = Rig(scene, board, kin, q0, np.random.default_rng(seed), fk_noise=(0.0, 0.0))
    t_ca = rig.detect_mat()
    est = ActiveHandEye(
        scene.camera.matrix, scene.camera.distortion, board, rig.anchor_belief @ np.linalg.inv(t_ca)
    )
    return scene, board, kin, rot, grip, centre, q0, rig, t_ca, est


def _first(est, board, kin, rot, grip, centre, q0, rig):
    import cv2

    for c in search_poses(kin, (0, 0, 1), rot, centre, grip, q0):
        hit = rig.detect_board(c.joints)
        if hit is None:
            continue
        ids, uv = hit
        _, rvec, tvec = cv2.solvePnP(
            board.object_points(ids), uv, est.camera_matrix, est.distortion, flags=cv2.SOLVEPNP_IPPE
        )
        t = np.eye(4)
        t[:3, :3] = cv2.Rodrigues(rvec)[0]
        t[:3, 3] = tvec.ravel()
        est.initialise_mount(kin.flange(c.joints), t)
        est.add(kin.flange(c.joints), ids, uv)
        est.solve()
        return c.joints
    raise AssertionError("board never visible")


def test_se3_helpers_roundtrip():
    t = perturb(np.eye(4), np.array([0.3, -0.2, 2.9, 0.1, 0.2, -0.3]))
    d = pose_error(np.eye(4), t)
    assert np.allclose(perturb(np.eye(4), d), t, atol=1e-9)


def test_active_calibration_recovers_anchor():
    scene, board, kin, rot, grip, centre, q0, rig, t_ca, est = _setup()
    q = _first(est, board, kin, rot, grip, centre, q0, rig)
    cands = generate(
        kin, (0, 0, 1), rot, centre, grip, q0, float(rig.anchor_belief[2, 3]) + 0.097, n=120, seed=1
    )
    pts = peg_points()
    sigmas = []
    for _ in range(8):
        c, info = choose_next(est, cands, q, t_ca, pts)
        assert c is not None and info["sigma_after_mm"] <= info["sigma_now_mm"]
        c.used = True
        q = c.joints
        hit = rig.detect_board(q)
        if hit is None:
            c.failed += 1
            continue
        est.add(kin.flange(q), *hit)
        est.solve()
        sigmas.append(est.task_uncertainty(est.information(), t_ca, pts))
    assert len(sigmas) >= 5 and sigmas[-1] < sigmas[0]
    got = est.X @ t_ca
    truth = rig.anchor_true
    diff = (got[:3, :3] @ pts.T).T + got[:3, 3] - ((truth[:3, :3] @ pts.T).T + truth[:3, 3])
    err = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    assert err < 0.001, f"peg-point error {err * 1e3:.2f} mm"
    assert math.degrees(np.linalg.norm(pose_error(truth, got)[:3])) < 0.3
