"""Active eye-to-hand calibration with a gripper-mounted ArUco board.

A fixed camera ``C`` looks at the robot. A small board ``G`` is taped to the gripper body. For a
flange pose ``T_BF`` (forward kinematics) the board corners project to

    u = π(K, dist, X⁻¹ · T_BF · Y · p_G)

with the unknowns ``X = T_BC`` (camera in base) and ``Y = T_FG`` (board on flange). Both are
estimated together by Gauss-Newton on corner reprojection, with a weak prior on ``X`` from the
previous session. What the task needs is not ``X`` itself but the anchor board in the robot base,
``T_BA = X · T_CA``, at the pegs. So every candidate pose is scored by how much it would shrink
the covariance of the peg points in the base frame, per second of motion. The robot then goes to
the best pose, observes, re-solves and repeats until the peg points are known well enough or the
time budget runs out.

Because the gripper board and the anchor board are seen by the same camera at similar depth, most
of the camera model error cancels in ``X · T_CA``. That is why the score targets the peg points
and not the camera pose.

그리퍼 몸체에 붙인 ArUco 보드로 고정 카메라의 외부 파라미터 X=T_BC 와 보드 장착 Y=T_FG 를 함께
재투영 오차로 푼다. 작업에 필요한 것은 X 자체가 아니라 기둥 위치(베이스 기준) T_BA = X·T_CA 이므로,
후보 자세마다 "이 자세를 보면 기둥 점들의 공분산이 얼마나 줄어드는가 / 이동 시간"을 예측해 가장
좋은 자세로 간다. 관측 → 재추정 → 다음 자세 선택을 수렴하거나 시간 예산이 끝날 때까지 반복한다.
같은 카메라가 비슷한 깊이에서 두 보드를 보므로 카메라 모델 오차의 대부분이 X·T_CA 에서 상쇄된다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- small SE(3) helpers
def rotvec_to_matrix(r: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector -> matrix. / 회전벡터 → 회전행렬."""
    r = np.asarray(r, dtype=float)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        k = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        return np.eye(3) + k
    k = r / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues. / 회전행렬 → 회전벡터."""
    c = max(-1.0, min(1.0, (np.trace(m) - 1.0) / 2.0))
    a = math.acos(c)
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if a < 1e-9:
        return w / 2.0
    if math.pi - a < 1e-6:
        # Near π: take the axis from the symmetric part. / π 근처: 대칭부에서 축을 구한다.
        b = (m + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(b), 0, None))
        i = int(np.argmax(axis))
        axis = b[:, i] / max(axis[i], 1e-12)
        return a * axis / np.linalg.norm(axis)
    return a * w / (2.0 * math.sin(a))


def perturb(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Right perturbation ``T · [R(d_r), d_t]``. / 오른쪽 섭동."""
    e = np.eye(4)
    e[:3, :3] = rotvec_to_matrix(d[:3])
    e[:3, 3] = d[3:6]
    return t @ e


def pose_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tangent difference ``[rotvec(R_aᵀR_b), R_aᵀ(t_b − t_a)]``. / 두 포즈 차이 (접공간)."""
    d = np.linalg.inv(a) @ b
    return np.concatenate([matrix_to_rotvec(d[:3, :3]), d[:3, 3]])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def average_poses(poses, weights=None) -> np.ndarray:
    """Weighted mean of rigid poses (chordal rotation mean). / 강체 포즈 가중 평균."""
    poses = [np.asarray(p, dtype=float) for p in poses]
    w = np.ones(len(poses)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    m = sum(wi * p[:3, :3] for wi, p in zip(w, poses, strict=True))
    u, _, vt = np.linalg.svd(m)
    r = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = sum(wi * p[:3, 3] for wi, p in zip(w, poses, strict=True))
    return out


