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


def project(k: np.ndarray, dist: np.ndarray, t_cam_obj: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project object points with OpenCV's distortion model (k1 k2 p1 p2 k3), NumPy only.

    OpenCV 왜곡 모델 투영 (NumPy 전용, 후보 수백 개를 빠르게 평가하기 위함).
    """
    p = pts @ t_cam_obj[:3, :3].T + t_cam_obj[:3, 3]
    z = p[:, 2:3]
    xy = p[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    d = np.zeros(5)
    d[: min(5, len(dist))] = np.asarray(dist, dtype=float)[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]], axis=1)


# ----------------------------------------------------------------------------- the gripper board
@dataclass(frozen=True)
class GripperBoard:
    """Marker corners of the gripper board in its own frame. / 그리퍼 보드의 마커 모서리 (보드 좌표).

    Frame: origin at the board centre, z = board normal (markers face), corners in OpenCV order.
    """

    dictionary: str
    size: float
    centers: dict  # id -> (x, y)

    @classmethod
    def from_json(cls, path: str) -> GripperBoard:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return cls(m["dict"], float(m["marker_size_m"]), {int(k): tuple(v) for k, v in m["markers"].items()})

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.centers))

    def corners(self, marker_id: int) -> np.ndarray:
        cx, cy = self.centers[marker_id]
        h = self.size / 2.0
        return np.array([[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]])

    def object_points(self, ids=None) -> np.ndarray:
        ids = self.ids if ids is None else ids
        return np.concatenate([self.corners(int(i)) for i in ids])


# ----------------------------------------------------------------------------- the estimator
@dataclass
class Observation:
    flange_in_base: np.ndarray  # T_BF from forward kinematics / 정기구학 플랜지 포즈
    object_points: np.ndarray  # (n, 3) board-frame corners / 보드 좌표 코너
    image_points: np.ndarray  # (n, 2) detected corners (px) / 검출 코너
    tag: str = ""


@dataclass
class Prediction:
    """What the camera should see at a candidate pose. / 후보 자세에서 보일 것의 예측."""

    image_points: np.ndarray
    facing_cos: float  # cosine between board normal and the ray to the camera / 보드 법선과 시선의 코사인
    marker_px: float  # smallest projected marker edge (px) / 가장 작은 마커 변 (px)
    visible: bool


