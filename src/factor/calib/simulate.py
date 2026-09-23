"""Synthetic check of the active self-calibration (no robot, no camera).

A ground-truth camera, anchor and board mount are drawn around the scene file's values; the
"camera" returns noisy corner detections of what would be visible. Strategies are compared on the
error of the peg points in the base frame against simulated wall-clock time.

합성 검증: 참값 카메라·앵커·보드 장착을 장면 값 근처에서 뽑고, 보이는 코너만 잡음과 함께 돌려준다.
전략별로 (모의 경과 시간 대비) 베이스 기준 기둥 점 오차를 비교한다.

    python -m factor.calib.simulate --scene configs/hanoi.yaml --seeds 20 --budget 60
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from factor.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    average_poses,
    choose_next,
    move_time,
    perturb,
    project,
)
from factor.calib.poses import generate, search_poses


def peg_points(spacing: float = 0.06, plate: float = 0.007, height: float = 0.055) -> np.ndarray:
    """Peg feet and tops in the anchor frame. / 기둥 아래·위 점 (앵커 좌표)."""
    return np.array([[x, 0.0, z] for x in (-spacing, 0.0, spacing) for z in (plate, plate + height)])


def mat_corners(scene) -> np.ndarray:
    out = []
    for m in scene.fiducials.markers:
        cx, cy, cz = m.center
        h = m.size / 2
        out += [[cx - h, cy + h, cz], [cx + h, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]]
    return np.array(out)


class Rig:
    """Ground truth + noisy detections. / 참값과 잡음 검출."""

    def __init__(
        self,
        scene,
        board,
        kin,
        q_nominal,
        rng,
        px_noise=0.3,
        fk_noise=(0.0002, 0.00035),  # (m, rad ≈ 0.02°)
        mat_noise=0.3,
    ):
        self.scene, self.board, self.kin, self.rng = scene, board, kin, rng
        self.k, self.dist = scene.camera.matrix, scene.camera.distortion
        self.px_noise, self.fk_noise, self.mat_noise = px_noise, fk_noise, mat_noise
        belief = np.asarray(scene.anchor_in_base, dtype=float)
        # The tower (with the mat) moved since the scene file was written, and the camera was moved too.
        # 장면 파일 이후 탑(매트 포함)과 카메라가 모두 옮겨졌다고 가정.
        self.anchor_true = perturb(
            belief, np.r_[rng.normal(0, math.radians(2), 3) * [0.2, 0.2, 1], rng.normal(0, 0.02, 2), 0.0]
        )
        cam_moved = perturb(
            np.asarray(scene.camera_in_anchor, dtype=float),
            np.r_[rng.normal(0, math.radians(3), 3), rng.normal(0, 0.04, 3)],
        )
        self.X_true = self.anchor_true @ cam_moved
        self.anchor_belief = belief
        # Board taped flat on the gripper body side that faces the camera, ~10 cm above the grip point.
        # 보드: 카메라 쪽 그리퍼 몸체 옆면에 평평하게, 그립점 위 약 10 cm.
        f0 = kin.flange(q_nominal)
        tip = kin.tool(q_nominal)[:3, 3]
        to_cam = self.X_true[:3, 3] - tip
        horiz = np.array([to_cam[0], to_cam[1], 0.0])
        horiz /= np.linalg.norm(horiz)
        z = horiz
        y = np.array([0.0, 0.0, 1.0])
        x = np.cross(y, z)
        board_in_base = np.eye(4)
        board_in_base[:3, :3] = np.column_stack([x, y, z])
        board_in_base[:3, 3] = tip + np.array([0, 0, 0.10]) + 0.045 * horiz
        self.Y_true = np.linalg.inv(f0) @ board_in_base

    def detect_board(self, q, flange=None):
        """Detected (ids, corners) at joints q, or None. / 관절 q 에서 보드 검출.

        ``flange`` overrides the true flange pose (to inject kinematic error).
        실제 플랜지 포즈를 바꿔 넣는다 (기구학 오차 주입용).
        """
        flange = self.kin.flange(q) if flange is None else flange
        t = np.linalg.inv(self.X_true) @ flange @ self.Y_true
        to_cam = -t[:3, 3] / np.linalg.norm(t[:3, 3])
        if t[:3, 2] @ to_cam < math.cos(math.radians(72)):  # detector limit / 검출 한계
            return None
        uv = project(self.k, self.dist, t, self.board.object_points())
        if np.any(uv < 0) or np.any(uv[:, 0] > 2592) or np.any(uv[:, 1] > 1944):
            return None
        return list(self.board.ids), uv + self.rng.normal(0, self.px_noise, uv.shape)

    ring_radius = 0.052
    ring_offset = -0.035
    n_leds = 8

    def detect_leds(self, q, flange=None):
        """Blue LED centroids on the wrist ring facing the camera, plus a stray blob. / 손목 LED 검출."""
        flange = self.kin.flange(q) if flange is None else flange
        ang = np.arange(self.n_leds) * 2 * math.pi / self.n_leds + 0.3
        pts = np.stack(
            [[self.ring_radius * math.cos(a), self.ring_radius * math.sin(a), self.ring_offset] for a in ang]
        )
        nrm = np.stack([[math.cos(a), math.sin(a), 0.0] for a in ang])
        t = np.linalg.inv(self.X_true) @ flange
        p3 = pts @ t[:3, :3].T + t[:3, 3]
        n3 = nrm @ t[:3, :3].T
        vis = np.einsum("ij,ij->i", n3, -p3 / np.linalg.norm(p3, axis=1, keepdims=True)) > 0.3
        uv = project(self.k, self.dist, t, pts[vis]) + self.rng.normal(0, 1.0, (int(vis.sum()), 2))
        stray = project(self.k, self.dist, t, np.array([[0.07, 0.03, -0.06]]))  # cable connector glint
        return np.vstack([uv, stray])

    def fk_reading(self, q):
        """Forward kinematics with a small model error. / 작은 모델 오차가 섞인 정기구학."""
        tp, rot = self.fk_noise
        return perturb(self.kin.flange(q), np.r_[self.rng.normal(0, rot, 3), self.rng.normal(0, tp, 3)])

    def detect_mat(self):
        import cv2

        obj = mat_corners(self.scene)
        t = np.linalg.inv(self.X_true) @ self.anchor_true
        uv = project(self.k, self.dist, t, obj) + self.rng.normal(0, self.mat_noise, (len(obj), 2))
        _, rvec, tvec = cv2.solvePnP(obj, uv, self.k, self.dist, flags=cv2.SOLVEPNP_IPPE)
        out = np.eye(4)
        out[:3, :3] = cv2.Rodrigues(rvec)[0]
        out[:3, 3] = tvec.ravel()
        return out


def run(
    strategy: str,
    scene,
    board,
    kin,
    branch,
    base_rot,
    grip_offset,
    q0,
    centre,
    min_tip_z,
    seed: int,
    budget: float,
    frames_per_pose: int = 2,
    noise=None,
):
    rng = np.random.default_rng(seed)
    rig = Rig(scene, board, kin, q0, rng, **(noise or {}))
    mats = [rig.detect_mat() for _ in range(3)]
    t_ca = average_poses(mats)
    prior = rig.anchor_belief @ np.linalg.inv(t_ca)
    est = ActiveHandEye(scene.camera.matrix, scene.camera.distortion, board, prior)
    pts = peg_points()
    cands = generate(kin, branch, base_rot, centre, grip_offset, q0, min_tip_z, seed=seed)
    q = q0.copy()
    clock = 0.0
    curve = []
    # First sighting / 첫 관측
    for c in search_poses(kin, branch, base_rot, centre, grip_offset, q0):
        clock += move_time(c.joints - q) + 1.4
        q = c.joints
        d = rig.detect_board(q)
        if d is not None:
            ids, uv = d
            import cv2

            _, rvec, tvec = cv2.solvePnP(
                board.object_points(ids), uv, est.camera_matrix, est.distortion, flags=cv2.SOLVEPNP_IPPE
            )
            tcg = np.eye(4)
            tcg[:3, :3] = cv2.Rodrigues(rvec)[0]
            tcg[:3, 3] = tvec.ravel()
            fk = rig.fk_reading(q)
            est.initialise_mount(fk, tcg)
            est.add(fk, ids, uv)
            break
    if not est.ready:
        return None
    est.solve()
    while clock < budget:
        if strategy == "active":
            c, info = choose_next(est, cands, q, t_ca, pts)
        else:
            pool = [
                c
                for c in cands
                if not c.used and (strategy == "blind" or est.predict(c.flange_in_base).visible)
            ]
            c = pool[rng.integers(len(pool))] if pool else None
        if c is None:
            break
        dt = move_time(c.joints - q) + 1.4
        if clock + dt > budget:
            break
        clock += dt
        q = c.joints
        c.used = True
        d = rig.detect_board(q)
        mats.append(rig.detect_mat())
        if d is None:
            c.failed += 1
            continue
        fk = rig.fk_reading(q)
        ids, uv = d
        for _ in range(frames_per_pose - 1):  # averaged frames / 프레임 평균
            uv = (uv + rig.detect_board(q)[1]) / 2
        est.add(fk, ids, uv)
        est.solve()
        t_ca = average_poses(mats)
        truth = (rig.anchor_true[:3, :3] @ pts.T).T + rig.anchor_true[:3, 3]
        a = est.X @ t_ca
        got = (a[:3, :3] @ pts.T).T + a[:3, 3]
        err = float(np.sqrt(np.mean(np.sum((got - truth) ** 2, axis=1))))
        sig = est.task_uncertainty(est.information(), t_ca, pts)
        curve.append((clock, len(est.observations), err * 1e3, sig * 1e3))
    return curve


