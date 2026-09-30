"""Marker-free self-calibration from the wrist LED ring.

The FR3 wrist carries a ring of status LEDs around the last joint axis. Whatever their count or
angle, every LED lies on one circle: centre on the flange z axis at offset ``d``, radius ``r``,
normal along the axis. With forward kinematics that circle is known in the base frame up to
``(d, r)``, so each detected LED says "this pixel lies on the projected circle". Stacked over
poses this fixes the camera in the base frame ``X`` without any marker on the robot.

The LEDs rotate with (or around) the axis, so their identity is never needed: the residual is the
image distance from each LED blob to the projected circle, measured along the curve normal.

손목 LED 링을 이용한 무표식 자기보정. LED 개수·각도와 상관없이 모든 LED 는 플랜지 z 축을 중심으로
한 원(축 위치 d, 반지름 r) 위에 있다. 정기구학으로 그 원을 베이스 좌표에서 (d, r) 만 빼고 알 수 있으므로,
검출된 LED 마다 "이 픽셀은 투영된 원 위에 있다"는 제약이 생긴다. 여러 자세를 모으면 로봇에 표식을 붙이지
않고도 카메라-베이스 X 가 정해진다. LED 식별은 필요 없다 (원까지의 법선 거리만 쓴다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from factor.calib.handeye import Prediction, perturb, pose_error, project, skew

_THETA = np.linspace(0.0, 2.0 * math.pi, 361)[:-1]


def detect_leds(rgb: np.ndarray, roi=None, min_area: int = 15, max_area: int = 4000) -> np.ndarray:
    """Centroids (n, 2) of saturated blue blobs, optionally inside ``roi`` = (x0, y0, x1, y1).

    채도 높은 파란 덩어리의 중심 (roi 안에서만).
    """
    import cv2

    x0, y0 = 0, 0
    img = rgb
    if roi is not None:
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
        x1, y1 = min(w, int(roi[2])), min(h, int(roi[3]))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 2))
        img = rgb[y0:y1, x0:x1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (92, 110, 140), (132, 255, 255))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    if not keep:
        return np.zeros((0, 2))
    return cents[keep] + np.array([x0, y0])


def _project_many(k: np.ndarray, dist: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``project`` for a (..., 3) array of camera-frame points. / 카메라 좌표 점 배열 투영."""
    shape = pts.shape[:-1]
    uv = project(k, dist, np.eye(4), pts.reshape(-1, 3))
    return uv.reshape(*shape, 2)


@dataclass
class RingObservation:
    flange_in_base: np.ndarray
    points: np.ndarray  # (n, 2) LED centroids kept by the gate (px) / 게이트를 통과한 LED 중심
    tag: str = ""
    raw: np.ndarray | None = None  # every blue blob in the ROI / ROI 안의 모든 파란 덩어리


@dataclass
class RingHandEye:
    """Camera-in-base ``X`` plus ring ``(r, d)`` from LED-on-circle constraints.

    Same interface as :class:`factor.calib.handeye.ActiveHandEye`, so :func:`choose_next` drives it.
    보드 추정기와 같은 인터페이스 (능동 자세 선택을 그대로 쓴다).
    """

    camera_matrix: np.ndarray
    distortion: np.ndarray
    camera_in_base_prior: np.ndarray
    ring_radius: float = 0.05
    ring_offset: float = -0.03
    prior_sigma: tuple[float, float] = (math.radians(10.0), 0.05)
    ring_sigma: tuple[float, float] = (0.008, 0.015)  # (radius, axial offset) m / 반지름·축 위치
    pixel_sigma: float = 1.0
    fk_sigma: tuple[float, float] = (math.radians(0.03), 0.0003)
    image_size: tuple[int, int] = (2592, 1944)
    min_facing_cos: float = 0.0  # unused for the ring; kept for the shared interface / 인터페이스 호환
    expected_leds: int = 3  # LEDs typically visible per frame / 한 프레임에 보이는 LED 수
    observations: list = field(default_factory=list)

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=float)
        self.distortion = np.asarray(self.distortion, dtype=float)
        self.X = np.asarray(self.camera_in_base_prior, dtype=float).copy()
        self.ring = np.array([self.ring_radius, self.ring_offset])
        self._ring0 = self.ring.copy()
        self.sigma = self.pixel_sigma
        self._prior_w = np.array([1 / self.prior_sigma[0]] * 3 + [1 / self.prior_sigma[1]] * 3)
        self._fk_var0 = np.array([self.fk_sigma[0] ** 2] * 3 + [self.fk_sigma[1] ** 2] * 3)
        self._fk_var = self._fk_var0.copy()
        self.aligned = False

    @property
    def ready(self) -> bool:
        return True

    # ---- geometry
    def circle(self, ring=None, theta=_THETA) -> np.ndarray:
        r, d = self.ring if ring is None else ring
        return np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, d)], axis=1)

    def project_circle(self, flange_in_base, x=None, ring=None) -> np.ndarray:
        x = self.X if x is None else x
        return project(
            self.camera_matrix, self.distortion, np.linalg.inv(x) @ flange_in_base, self.circle(ring)
        )

    def visible_arc(self, flange_in_base, x=None, ring=None, threshold: float = 0.25) -> np.ndarray:
        """Mask of circle samples facing the camera (outward normal toward it). / 카메라를 향한 원호."""
        x = self.X if x is None else x
        t = np.linalg.inv(x) @ flange_in_base
        pts = self.circle(ring) @ t[:3, :3].T + t[:3, 3]
        normals = np.stack([np.cos(_THETA), np.sin(_THETA), np.zeros_like(_THETA)], axis=1) @ t[:3, :3].T
        view = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return np.einsum("ij,ij->i", normals, view) > threshold

    def roi(self, flange_in_base, margin: float = 120.0):
        """Image box around the predicted ring. / 예측 링 주변 영상 영역."""
        uv = self.project_circle(flange_in_base)
        return (
            uv[:, 0].min() - margin,
            uv[:, 1].min() - margin,
            uv[:, 0].max() + margin,
            uv[:, 1].max() + margin,
        )

    # ---- residuals
    def _point_residual(self, x, ring, obs: list) -> np.ndarray:
        """Signed distance of each LED to the projected circle (px). / 각 LED 와 투영 원 사이 법선 거리."""
        out = []
        for o in obs:
            cu = self.project_circle(o.flange_in_base, x, ring)
            # Only the arc facing the camera can carry a visible LED (edge-on LEDs included).
            # 카메라를 향한 원호에만 보이는 LED 가 있을 수 있다 (가장자리 포함).
            front = np.flatnonzero(self.visible_arc(o.flange_in_base, x, ring, threshold=-0.05))
            if len(front) == 0:
                front = np.arange(len(cu))
            for p in o.points:
                j = int(front[np.argmin(np.sum((cu[front] - p) ** 2, axis=1))])
                a, b = cu[j - 1], cu[(j + 1) % len(cu)]
                t = b - a
                n = np.array([-t[1], t[0]]) / max(np.linalg.norm(t), 1e-9)
                out.append(float((p - cu[j]) @ n))
        return np.array(out)

    def _params(self, v):
        return perturb(self.X, v[:6]), self.ring + v[6:8]

    def _jacobian(self, fn, h: float = 1e-6):
        r0 = fn(np.zeros(8))
        jac = np.zeros((r0.size, 8))
        for i in range(8):
            d = np.zeros(8)
            d[i] = h
            jac[:, i] = (fn(d) - r0) / h
        return r0, jac

    def _prior(self, v) -> np.ndarray:
        x, ring = self._params(v)
        return np.concatenate(
            [
                self._prior_w * pose_error(self.camera_in_base_prior, x),
                (ring - self._ring0) / np.asarray(self.ring_sigma),
            ]
        )

    def _obs_whitening(self, o: RingObservation) -> np.ndarray:
        """Per-pose whitening for pixel noise plus a shared kinematic error. / 자세별 백색화."""

        def f(dfl):
            moved = RingObservation(perturb(o.flange_in_base, dfl), o.points)
            return self._point_residual(self.X, self.ring, [moved])

        base = f(np.zeros(6))
        jf = np.zeros((base.size, 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-6
            jf[:, i] = (f(d) - base) / 1e-6
        cov = self.sigma**2 * np.eye(base.size) + jf @ np.diag(self._fk_var) @ jf.T
        return np.linalg.cholesky(np.linalg.inv(cov)).T

    def _used(self) -> list:
        return [o for o in self.observations if len(o.points)]

    def _stack(self, v, whiten):
        rows = []
        for o, w in zip(self._used(), whiten, strict=True):
            x, ring = self._params(v)
            rows.append(w @ self._point_residual(x, ring, [o]))
        return np.concatenate(rows) if rows else np.zeros(0)

    def add(self, flange_in_base, points, tag: str = "") -> int:
        """Add one pose's blue blobs; the gate keeps those near the predicted ring.

        한 자세의 파란 덩어리를 추가한다 (예측 링 근처만 사용).
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        o = RingObservation(np.asarray(flange_in_base, dtype=float), pts, tag, pts.copy())
        self.observations.append(o)
        self._regate()
        return len(o.points)

    def _gate_px(self) -> float:
        return 40.0 if not self.aligned else max(20.0, 5.0 * self.sigma)

    def _regate(self) -> None:
        for o in self.observations:
            if o.raw is None or len(o.raw) == 0:
                o.points = np.zeros((0, 2))
                continue
            probe = RingObservation(o.flange_in_base, o.raw)
            d = np.abs(self._point_residual(self.X, self.ring, [probe]))
            o.points = o.raw[d < self._gate_px()]

    def align(self, span: float = 0.08, step: float = 0.01) -> dict:
        """Coarse alignment over all poses so far: grid over the camera position (base frame) and the
        ring's axial offset, maximising LEDs that fall on the front arc. Vectorised over the grid.

        장면 사전값이 수 cm 틀릴 수 있어, 지금까지의 모든 자세로 카메라 위치(격자)와 링 축 위치를 훑어
        LED 가 앞쪽 원호에 가장 많이 오는 값을 먼저 찾는다 (격자 전체를 한 번에 계산).
        """
        obs = [o for o in self.observations if o.raw is not None and len(o.raw)]
        if not obs:
            return {}
        g = np.arange(-span, span + 1e-9, step)
        deltas = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        k, dist = self.camera_matrix, self.distortion
        r_x = self.X[:3, :3]
        best = (-1.0, None, None)
        for d_off in np.arange(-0.055, -0.0149, 0.005):
            ring = np.array([self.ring[0], d_off])
            score = np.zeros(len(deltas))
            for o in obs:
                t = np.linalg.inv(self.X) @ o.flange_in_base
                front = self.visible_arc(o.flange_in_base, None, ring, threshold=-0.05)
                pc = self.circle(ring)[front] @ t[:3, :3].T + t[:3, 3]  # (m, 3) camera frame
                shift = deltas @ r_x  # camera moves by δ (base) -> points move by -R_Xᵀ δ
                pts = pc[None, :, :] - shift[:, None, :]
                uv = _project_many(k, dist, pts)  # (G, m, 2)
                dd = np.linalg.norm(uv[:, None, :, :] - o.raw[None, :, None, :], axis=3).min(axis=2)  # (G, n)
                score += np.clip(1.0 - (dd / 25.0) ** 2, 0.0, None).sum(axis=1)
            i_best = int(np.argmax(score))
            if score[i_best] > best[0]:
                best = (float(score[i_best]), deltas[i_best], d_off)
        self.X = self.X.copy()
        self.X[:3, 3] = self.X[:3, 3] + best[1]
        self.ring = np.array([self.ring[0], best[2]])
        self.aligned = True
        self._regate()
        return {
            "shift_mm": (best[1] * 1e3).round(0).tolist(),
            "ring_offset_mm": round(best[2] * 1e3),
            "score": round(best[0], 2),
        }

    def solve(self, iterations: int = 15, huber: float = 3.0) -> dict:
        if not self._used():
            return {}
        for _round in range(2):  # solve, re-gate with the better estimate, solve again / 풀고 다시 게이트
            info = self._solve_once(iterations, huber)
            self._regate()
        return info

    def _solve_once(self, iterations: int, huber: float) -> dict:
        lam = 1e-3
        for _ in range(iterations):
            whiten = [self._obs_whitening(o) for o in self._used()]
            rd, jd = self._jacobian(lambda v, w=whiten: self._stack(v, w))
            rp, jp = self._jacobian(self._prior)
            a = np.abs(rd)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
            h = (jd * w[:, None]).T @ jd + jp.T @ jp
            g = (jd * w[:, None]).T @ rd + jp.T @ rp
            cost0 = float((w * rd * rd).sum() + (rp * rp).sum())
            step = np.zeros(8)
            for _try in range(6):
                step = -np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), g)
                r1 = self._stack(step, whiten)
                p1 = self._prior(step)
                a1 = np.abs(r1)
                w1 = np.where(a1 <= huber, 1.0, huber / np.maximum(a1, 1e-12))
                if float((w1 * r1 * r1).sum() + (p1 * p1).sum()) <= cost0:
                    self.X, self.ring = self._params(step)
                    self._shift_prior(step)
                    lam = max(lam / 4, 1e-9)
                    break
                lam *= 8
            else:
                break
            if np.linalg.norm(step) < 1e-9:
                break
        raw = self._point_residual(self.X, self.ring, self._used())
        whiten = [self._obs_whitening(o) for o in self._used()]
        rw = self._stack(np.zeros(8), whiten)
        dof = raw.size - 8
        chi = float((rw**2).sum() / dof) if dof >= 16 else 1.0
        if chi > 1.0:
            self.sigma = self.pixel_sigma * math.sqrt(chi)
            self._fk_var = self._fk_var0 * chi
        return {
            "reproj_rms_px": float(np.sqrt(np.mean(raw**2))) if raw.size else float("nan"),
            "chi2_per_dof": chi,
            "sigma_px": self.sigma,
            "n_obs": len(self.observations),
            "n_leds": int(raw.size),
            "ring_mm": (self.ring * 1e3).round(2).tolist(),
        }

    def _shift_prior(self, step) -> None:
        """Priors are anchored in absolute terms; nothing to re-express. / 사전은 절대값 기준."""

    # ---- information for the active choice
    def information(self) -> np.ndarray:
        _, jp = self._jacobian(self._prior)
        h = jp.T @ jp
        if self._used():
            whiten = [self._obs_whitening(o) for o in self._used()]
            _, jd = self._jacobian(lambda v, w=whiten: self._stack(v, w))
            h = h + jd.T @ jd
        return h

    def pose_information(self, flange_in_base: np.ndarray) -> np.ndarray:
        """Expected information: ``expected_leds`` points spread over the visible arc. / 예상 정보."""
        vis = np.flatnonzero(self.visible_arc(flange_in_base))
        if len(vis) < 3:
            return np.zeros((8, 8))
        pick = vis[np.linspace(len(vis) * 0.15, len(vis) * 0.85, self.expected_leds).astype(int)]
        pts = self.project_circle(flange_in_base)[pick]
        o = RingObservation(flange_in_base, pts)
        w = self._obs_whitening(o)

        def f(v):
            x, ring = self._params(v)
            return w @ self._point_residual(x, ring, [o])

        _, jac = self._jacobian(f)
        return jac.T @ jac

    def predict(self, flange_in_base: np.ndarray) -> Prediction:
        uv = self.project_circle(flange_in_base)
        vis = self.visible_arc(flange_in_base)
        w, h = self.image_size
        inside = bool(np.all((uv[:, 0] > 40) & (uv[:, 0] < w - 40) & (uv[:, 1] > 40) & (uv[:, 1] < h - 40)))
        frac = float(vis.mean())
        # Before the coarse alignment the prior may put the ring off-image; do not trust that.
        # 거친 정렬 전에는 사전값이 링을 화면 밖에 둘 수 있으니 화면 안 검사를 하지 않는다.
        return Prediction(uv, frac, float(np.ptp(uv[:, 0])), (inside or not self.aligned) and frac > 0.2)

    def task_uncertainty(self, information, anchor_in_camera, points_anchor) -> float:
        cov = np.linalg.inv(information)[:6, :6]
        pts = np.asarray(points_anchor, dtype=float)
        p_c = pts @ anchor_in_camera[:3, :3].T + anchor_in_camera[:3, 3]
        r = self.X[:3, :3]
        tr = 0.0
        for p in p_c:
            jac = np.hstack([-r @ skew(p), r])
            tr += float(np.trace(jac @ cov @ jac.T))
        return math.sqrt(tr / len(p_c))

    def mount_consistency(self) -> dict:
        raw = self._point_residual(self.X, self.ring, self._used())
        return {"led_residual_rms_px": float(np.sqrt(np.mean(raw**2))), "ring_mm": (self.ring * 1e3).tolist()}
