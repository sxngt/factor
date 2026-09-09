"""NumPy inference for the residual skill policy. / 잔차 스킬 정책의 NumPy 추론.

The real-robot loop runs without PyTorch, so the deterministic policy (two tanh layers and a
tanh-squashed mean head) is evaluated here from arrays written by
:func:`factor.skills.residual.export_numpy`. :func:`skill_features` mirrors the batched torch
version feature for feature.
"""

from __future__ import annotations

import numpy as np

FEATURE_DIM = 15
NUDGE_STEPS = 30.0  # last contact-search index / 접촉 탐색 마지막 인덱스
POSITION_SCALE = 0.05


def skill_features(
    grip: np.ndarray,
    object_pos: np.ndarray,
    goal_pos: np.ndarray,
    holding: bool,
    inserting: bool,
    nudge_index: int,
    nudge_stall: int,
    last_residual: np.ndarray,
    hover_height: float,
    zone_top: float,
    max_step: float,
) -> np.ndarray:
    """Single-environment counterpart of ``factor.skills.residual.skill_features``."""
    grip = np.asarray(grip, dtype=float)
    return np.concatenate(
        [
            (grip - np.asarray(object_pos, dtype=float)) / POSITION_SCALE,
            (grip - np.asarray(goal_pos, dtype=float)) / POSITION_SCALE,
            [
                float(holding),
                float(inserting),
                (grip[2] - zone_top) / POSITION_SCALE,
                (grip[2] - hover_height) / POSITION_SCALE,
                nudge_index / NUDGE_STEPS,
                nudge_stall / 3.0,
            ],
            np.asarray(last_residual, dtype=float) / max_step,
        ]
    )


class NumpyResidualPolicy:
    """Deterministic residual policy from exported arrays. / 내보낸 배열로 만든 결정적 정책."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.w0, self.b0 = arrays["trunk.0.weight"], arrays["trunk.0.bias"]
        self.w1, self.b1 = arrays["trunk.2.weight"], arrays["trunk.2.bias"]
        self.wm, self.bm = arrays["mean.weight"], arrays["mean.bias"]
        self.residual_scale = float(arrays["residual_scale"])

    @classmethod
    def from_file(cls, path: str) -> NumpyResidualPolicy:
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Features → residual displacement in metres. / 피처 → 잔차 이동량 (m)."""
        h = np.tanh(self.w0 @ np.asarray(features, dtype=float) + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        return np.tanh(self.wm @ h + self.bm) * self.residual_scale
