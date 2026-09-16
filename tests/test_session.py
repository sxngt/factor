"""Session estimator: fusion shrinks the error, outliers are rejected. / 세션 추정 검증."""

import numpy as np

from factor.perception.session import SessionEstimator


def test_fusion_converges_and_rejects_outliers():
    rng = np.random.default_rng(0)
    truth = np.array([0.004, -0.003, 0.0])
    session = SessionEstimator()
    for _ in range(60):
        session.add("middle", truth + np.r_[rng.normal(0, 0.0013, 2), 0.0], sigma=0.0013)
    session.add("middle", truth + np.array([0.03, 0.03, 0.0]), sigma=0.0013)  # a wild frame
    fused = session.fused("middle")
    assert np.linalg.norm(fused.correction[:2] - truth[:2]) < 0.0006
    assert fused.stderr < 0.001 and fused.count == 60


def test_uncertain_model_is_down_weighted():
    session = SessionEstimator()
    for _ in range(10):
        session.add("k", np.array([0.002, 0.0, 0.0]), sigma=0.001)
        session.add("k", np.array([0.010, 0.0, 0.0]), sigma=0.010)
    assert session.fused("k").correction[0] < 0.003
