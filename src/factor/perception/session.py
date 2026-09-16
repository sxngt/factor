"""Session-level belief estimation. / 세션 단위 믿음 오차 추정.

On the robot the belief error is a *constant* of the session (the anchor constants are wrong
by a fixed amount), while each frame's estimate carries roughly independent image noise. So
rather than trusting one frame, the loop probes: it looks at each peg from several hover poses,
collects estimates, and fuses them with inverse-variance weights from the estimator's own
uncertainty and the ensemble disagreement. The fused value is what the skill uses, and its
standard error gates the action: an estimate that stays uncertain is refused, not executed.

로봇에서 믿음의 오차는 세션 동안 상수이고, 프레임별 추정에는 영상 잡음이 섞인다. 여러 자세에서
본 추정을 불확실성 가중 평균으로 합치고, 그 표준오차로 행동 여부를 판단한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusedEstimate:
    correction: np.ndarray  # (dx, dy, dyaw) in metres / radians
    stderr: float  # standard error of the translation (m) / 평행이동 표준오차
    count: int


@dataclass
class SessionEstimator:
    """Inverse-variance fusion per key (a peg, or the whole anchor). / 키별 역분산 융합.

    Parameters
    ----------
    floor
        Per-frame noise that is always assumed (m); keeps a confident but wrong model from
        dominating. / 항상 가정하는 프레임 잡음.
    outlier
        Samples further than this many robust standard deviations from the median are dropped.
    """

    floor: float = 0.0008
    outlier: float = 3.0
    samples: dict[str, list[tuple[np.ndarray, float]]] = field(default_factory=dict)

    def add(self, key: str, correction: np.ndarray, sigma: float = 0.0, spread: float = 0.0) -> None:
        var = self.floor**2 + float(sigma) ** 2 + float(spread) ** 2
        self.samples.setdefault(key, []).append((np.asarray(correction, dtype=float), var))

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self.samples.clear()
        else:
            self.samples.pop(key, None)

    def fused(self, key: str) -> FusedEstimate | None:
        data = self.samples.get(key)
        if not data:
            return None
        values = np.stack([c for c, _ in data])
        var = np.array([v for _, v in data])
        median = np.median(values[:, :2], axis=0)
        dev = np.linalg.norm(values[:, :2] - median, axis=1)
        mad = float(np.median(dev)) * 1.4826 + 1e-6
        keep = dev <= self.outlier * max(mad, self.floor)
        values, var = values[keep], var[keep]
        w = 1.0 / var
        mean = (values * w[:, None]).sum(0) / w.sum()
        model_se = float(np.sqrt(1.0 / w.sum()))
        # Frames from one pose are correlated: the empirical scatter between samples bounds
        # the error from below as well. / 같은 자세 프레임은 상관되므로 경험적 퍼짐도 반영.
        scatter = float(np.sqrt((w * ((values[:, :2] - mean[:2]) ** 2).sum(1)).sum() / w.sum()))
        empirical_se = scatter / np.sqrt(max(len(values) / 6.0, 1.0))
        return FusedEstimate(mean, max(model_se, empirical_se), len(values))
