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


@dataclass
class ActiveHandEye:
    """Joint estimate of camera-in-base ``X`` and board-on-flange ``Y`` with active pose choice.

    카메라-베이스 X 와 보드-플랜지 Y 의 공동 추정 + 능동 자세 선택.

    Parameters
    ----------
    camera_matrix, distortion
        Intrinsics of the fixed camera.
    board
        Gripper board geometry.
    camera_in_base_prior
        Starting guess of ``X`` (for example previous anchor pose · current anchor detection).
    prior_sigma
        (rotation rad, translation m) standard deviation of that guess; weak by default.
    pixel_sigma
        Corner noise floor (px); the working value is re-estimated from the residuals.
    """

    camera_matrix: np.ndarray
    distortion: np.ndarray
    board: GripperBoard
    camera_in_base_prior: np.ndarray
    prior_sigma: tuple[float, float] = (math.radians(10.0), 0.05)
    pixel_sigma: float = 0.35
    fk_sigma: tuple[float, float] = (
        math.radians(0.03),
        0.0003,
    )  # per-pose kinematic error (rad, m) / 자세별 기구학 오차
    image_size: tuple[int, int] = (2592, 1944)
    min_facing_cos: float = math.cos(math.radians(78.0))
    min_marker_px: float = 18.0
    margin_px: float = 40.0
    observations: list = field(default_factory=list)

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=float)
        self.distortion = np.asarray(self.distortion, dtype=float)
        self.X = np.asarray(self.camera_in_base_prior, dtype=float).copy()
        self.Y: np.ndarray | None = None
        self.sigma = self.pixel_sigma
        self._prior_w = np.array([1 / self.prior_sigma[0]] * 3 + [1 / self.prior_sigma[1]] * 3)
        self._fk_var0 = np.array([self.fk_sigma[0] ** 2] * 3 + [self.fk_sigma[1] ** 2] * 3)
        self._fk_var = self._fk_var0.copy()
        self._mount_w = np.array([1 / math.radians(90.0)] * 3 + [1 / 0.5] * 3)  # keeps Y regular / Y 정칙화

    # ---- state
    @property
    def ready(self) -> bool:
        return self.Y is not None

    def board_in_camera(self, flange_in_base: np.ndarray, x=None, y=None) -> np.ndarray:
        x = self.X if x is None else x
        y = self.Y if y is None else y
        return np.linalg.inv(x) @ flange_in_base @ y

    def initialise_mount(self, flange_in_base: np.ndarray, board_in_camera: np.ndarray) -> None:
        """First sighting: ``Y = T_BF⁻¹ · X · T_CG`` with the current ``X``. / 첫 관측으로 Y 초기화."""
        self.Y = np.linalg.inv(flange_in_base) @ self.X @ board_in_camera
        self._mount_prior = self.Y.copy()

    def add(self, flange_in_base, ids, image_points, tag: str = "") -> None:
        """Add one averaged detection (marker ids, (n·4, 2) corners). / 관측 하나 추가."""
        self.observations.append(
            Observation(
                np.asarray(flange_in_base, dtype=float),
                self.board.object_points(ids),
                np.asarray(image_points, dtype=float).reshape(-1, 2),
                tag,
            )
        )

    # ---- residuals and Jacobians (12 local parameters: δX, δY)
    def _data_residual(self, x, y, obs=None) -> np.ndarray:
        obs = self.observations if obs is None else obs
        if not obs:
            return np.zeros(0)
        out = []
        for o in obs:
            uv = project(
                self.camera_matrix,
                self.distortion,
                self.board_in_camera(o.flange_in_base, x, y),
                o.object_points,
            )
            out.append((uv - o.image_points).ravel())
        return np.concatenate(out)

    def _whitener(self, o: Observation) -> np.ndarray:
        """``W`` with ``WᵀW = (σ²I + J_f Σ_f J_fᵀ)⁻¹`` for one pose.

        A kinematic error moves all corners of a pose together, so corners are not independent.
        ``J_f`` maps a flange perturbation to the corners; ``Σ_f`` is the kinematic error model.
        기구학 오차는 한 자세의 모든 코너를 함께 움직이므로 코너 잡음은 독립이 아니다.
        """
        base = project(
            self.camera_matrix, self.distortion, self.board_in_camera(o.flange_in_base), o.object_points
        ).ravel()
        jac = np.zeros((base.size, 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-6
            moved = np.linalg.inv(self.X) @ perturb(o.flange_in_base, d) @ self.Y
            jac[:, i] = (
                project(self.camera_matrix, self.distortion, moved, o.object_points).ravel() - base
            ) / 1e-6
        cov = (self.sigma**2) * np.eye(base.size) + jac @ np.diag(self._fk_var) @ jac.T
        return np.linalg.cholesky(np.linalg.inv(cov)).T

    def _prior_residual(self, x, y) -> np.ndarray:
        r = [self._prior_w * pose_error(self.camera_in_base_prior, x)]
        r.append(self._mount_w * pose_error(self._mount_prior, y))
        return np.concatenate(r)

    def _jacobian(self, fn, h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        r0 = fn(self.X, self.Y)
        jac = np.zeros((r0.size, 12))
        for i in range(12):
            d = np.zeros(12)
            d[i] = h
            jac[:, i] = (fn(perturb(self.X, d[:6]), perturb(self.Y, d[6:])) - r0) / h
        return r0, jac

    def _block_whiten(self, obs) -> np.ndarray:
        """Block-diagonal whitening matrix over poses. / 자세별 백색화 행렬의 블록 대각."""
        ws = [self._whitener(o) for o in obs]
        n = sum(w.shape[0] for w in ws)
        out = np.zeros((n, n))
        i = 0
        for w in ws:
            k = w.shape[0]
            out[i : i + k, i : i + k] = w
            i += k
        return out

    def solve(self, iterations: int = 12, huber: float = 3.0) -> dict:
        """Levenberg-Marquardt over all poses, whitened per pose. / 자세별 백색화 비선형 최소제곱."""
        if not self.ready or not self.observations:
            return {}
        lam = 1e-4
        cost = None
        for _ in range(iterations):
            wb = self._block_whiten(self.observations)
            rd, jd = self._jacobian(self._data_residual)
            rw, jw = wb @ rd, wb @ jd
            rp, jp = self._jacobian(self._prior_residual)
            a = np.abs(rw)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))  # Huber IRLS
            h = (jw * w[:, None]).T @ jw + jp.T @ jp
            g = (jw * w[:, None]).T @ rw + jp.T @ rp
            cost0 = float((w * rw * rw).sum() + (rp * rp).sum())
            step = np.zeros(12)
            for _try in range(6):
                step = -np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), g)
                x1, y1 = perturb(self.X, step[:6]), perturb(self.Y, step[6:])
                rw1 = wb @ self._data_residual(x1, y1)
                rp1 = self._prior_residual(x1, y1)
                a1 = np.abs(rw1)
                w1 = np.where(a1 <= huber, 1.0, huber / np.maximum(a1, 1e-12))
                cost1 = float((w1 * rw1 * rw1).sum() + (rp1 * rp1).sum())
                if cost1 <= cost0:
                    self.X, self.Y, cost = x1, y1, cost1
                    lam = max(lam / 4, 1e-9)
                    break
                lam *= 8
            else:
                break
            if np.linalg.norm(step) < 1e-8:
                break
        rd = self._data_residual(self.X, self.Y)
        rw = self._block_whiten(self.observations) @ rd
        rms = float(np.sqrt(np.mean(rd**2))) if rd.size else float("nan")
        # Inflate the noise model when the data disagree more than it says (never deflate).
        # 데이터가 잡음 모델보다 더 어긋나면 모델을 키운다 (줄이지는 않음).
        dof = rd.size - 12
        chi = float((rw**2).sum() / dof) if dof >= 24 else 1.0
        if chi > 1.0:
            self.sigma = self.pixel_sigma * math.sqrt(chi)
            self._fk_var = self._fk_var0 * chi
        return {
            "reproj_rms_px": rms,
            "chi2_per_dof": chi,
            "sigma_px": self.sigma,
            "cost": cost,
            "n_obs": len(self.observations),
        }

    def information(self) -> np.ndarray:
        """Posterior information (12×12) at the current estimate. / 현재 추정의 정보 행렬."""
        _, jp = self._jacobian(self._prior_residual)
        h = jp.T @ jp
        if self.observations:
            _, jd = self._jacobian(self._data_residual)
            jw = self._block_whiten(self.observations) @ jd
            h = h + jw.T @ jw
        return h

    def pose_information(self, flange_in_base: np.ndarray, ids=None) -> np.ndarray:
        """Predicted information of observing the board at ``flange_in_base``. / 후보 자세의 예측 정보."""
        obj = self.board.object_points(ids)
        pseudo = Observation(flange_in_base, obj, np.zeros((len(obj), 2)))
        _, jac = self._jacobian(lambda x, y: self._data_residual(x, y, [pseudo]))
        jw = self._whitener(pseudo) @ jac
        return jw.T @ jw

    def predict(self, flange_in_base: np.ndarray) -> Prediction:
        """Where the board will appear and whether it will be detectable. / 보드가 어디에 보일지."""
        t = self.board_in_camera(flange_in_base)
        uv = project(self.camera_matrix, self.distortion, t, self.board.object_points())
        to_cam = -t[:3, 3] / max(np.linalg.norm(t[:3, 3]), 1e-9)
        facing = float(t[:3, 2] @ to_cam)
        quads = uv.reshape(-1, 4, 2)
        edge = float(min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for q in quads for i in range(4)))
        w, h = self.image_size
        m = self.margin_px
        inside = bool(np.all((uv[:, 0] > m) & (uv[:, 0] < w - m) & (uv[:, 1] > m) & (uv[:, 1] < h - m)))
        visible = inside and t[2, 3] > 0.1 and facing > self.min_facing_cos and edge > self.min_marker_px
        return Prediction(uv, facing, edge, visible)

    # ---- what the task needs: peg points in the base frame
    @staticmethod
    def point_covariance(
        cov_x: np.ndarray, x: np.ndarray, anchor_in_camera: np.ndarray, points_anchor
    ) -> np.ndarray:
        """Covariance (k,3,3) of anchor points mapped to the base by ``X · T_CA``. / 기둥 점의 베이스 공분산.

        With ``X = X̂ · [R(δr), δt]``: ``p_B = R_X (R(δr) p_C + δt) + t_X`` so
        ``∂p_B/∂δr = −R_X [p_C]×`` and ``∂p_B/∂δt = R_X``.
        """
        pts = np.asarray(points_anchor, dtype=float)
        p_c = pts @ anchor_in_camera[:3, :3].T + anchor_in_camera[:3, 3]
        r = x[:3, :3]
        out = []
        for p in p_c:
            jac = np.hstack([-r @ skew(p), r])
            out.append(jac @ cov_x @ jac.T)
        return np.stack(out)

    def task_uncertainty(self, information: np.ndarray, anchor_in_camera, points_anchor) -> float:
        """RMS 1σ (m) of the peg points in the base frame. / 기둥 점들의 1σ RMS (m)."""
        cov = np.linalg.inv(information)[:6, :6]
        c = self.point_covariance(cov, self.X, anchor_in_camera, points_anchor)
        return float(np.sqrt(np.mean(np.trace(c, axis1=1, axis2=2))))

    def mount_consistency(self) -> dict:
        """Per-pose implied mount ``Y_i = T_BF⁻¹ X T_CG,i`` spread. / 자세별 장착 변환 일관성."""
        import cv2

        devs, angs = [], []
        for o in self.observations:
            ok, rvec, tvec = cv2.solvePnP(
                o.object_points, o.image_points, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_IPPE
            )
            if not ok:
                continue
            t = np.eye(4)
            t[:3, :3] = cv2.Rodrigues(rvec)[0]
            t[:3, 3] = tvec.ravel()
            yi = np.linalg.inv(o.flange_in_base) @ self.X @ t
            e = pose_error(self.Y, yi)
            angs.append(math.degrees(np.linalg.norm(e[:3])))
            devs.append(1000 * np.linalg.norm(e[3:]))
        if not devs:
            return {}
        return {
            "pos_rms_mm": float(np.sqrt(np.mean(np.square(devs)))),
            "rot_rms_deg": float(np.sqrt(np.mean(np.square(angs)))),
        }


# ----------------------------------------------------------------------------- choosing the next pose
@dataclass
class Candidate:
    """A reachable pose the robot may visit. / 로봇이 갈 수 있는 후보 자세."""

    joints: np.ndarray
    flange_in_base: np.ndarray
    label: str = ""
    used: bool = False
    failed: int = 0


def move_time(
    dq: np.ndarray, v_max: float = math.radians(24.0), a_max: float = math.radians(36.0), floor: float = 0.8
) -> float:
    """Duration of a minimum-jerk joint move within speed and acceleration limits.

    최소 저크 관절 이동 시간: 최고 속도 1.875Δ/T ≤ v, 최고 가속도 5.77Δ/T² ≤ a.
    """
    d = float(np.max(np.abs(dq)))
    return max(floor, 1.875 * d / v_max, math.sqrt(5.7735 * d / a_max))


def choose_next(
    est: ActiveHandEye,
    candidates: list,
    current_joints: np.ndarray,
    anchor_in_camera: np.ndarray,
    points_anchor,
    dwell_s: float = 1.4,
) -> tuple[Candidate | None, dict]:
    """Greedy information-per-second choice. / 초당 정보 이득이 가장 큰 후보.

    Score = log(σ_now² / σ_after²) / (move time + dwell), where σ is the RMS 1σ of the peg points
    in the base frame. Unseen-board or blocked candidates are skipped.
    """
    h = est.information()
    now = est.task_uncertainty(h, anchor_in_camera, points_anchor)
    best, best_score, best_info = None, -1.0, {}
    for c in candidates:
        if c.used or c.failed >= 1:
            continue
        pred = est.predict(c.flange_in_base)
        if not pred.visible:
            continue
        after = est.task_uncertainty(
            h + est.pose_information(c.flange_in_base), anchor_in_camera, points_anchor
        )
        t = move_time(c.joints - current_joints) + dwell_s
        score = 2.0 * math.log(now / max(after, 1e-12)) / t
        if score > best_score:
            best, best_score = c, score
            best_info = {
                "sigma_now_mm": now * 1e3,
                "sigma_after_mm": after * 1e3,
                "seconds": t,
                "score": score,
                "facing": pred.facing_cos,
                "marker_px": pred.marker_px,
            }
    return best, best_info
