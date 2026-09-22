"""Candidate calibration poses around the work area, filtered for reach and clearance.

작업 영역 주변의 보정 후보 자세 (도달·여유 높이 검사 포함).
"""

from __future__ import annotations

import math

import numpy as np

from factor.calib.handeye import Candidate


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def near_within_limits(kin, q, q_ref, margin_deg: float = 10.0):
    """Joint solution wrapped closest to ``q_ref`` that keeps ``margin_deg`` from every limit, or None.

    ``q_ref`` 에 가장 가깝게 감되 모든 관절 한계에서 ``margin_deg`` 떨어진 해 (없으면 None).
    """
    lo = kin.limits[:, 0] + math.radians(margin_deg)
    hi = kin.limits[:, 1] - math.radians(margin_deg)
    out = np.array(q, dtype=float)
    for i in range(len(out)):
        options = [out[i] + k * 2 * math.pi for k in (-1, 0, 1)]
        options = [v for v in options if lo[i] <= v <= hi[i]]
        if not options:
            return None
        out[i] = min(options, key=lambda v: abs(v - q_ref[i]))
    return out


def path_clear(kin, q0, q1, min_tip_z: float, tilt_drop: float, min_link_z: float, samples: int = 12) -> bool:
    """Joint-space straight path keeps the tool tip and the flange above their floors.

    관절공간 직선 경로에서 툴 끝(+기울기에 따른 핑거 하강)과 플랜지가 바닥 높이 위에 있는지.
    """
    for s in np.linspace(0.0, 1.0, samples):
        q = q0 + (q1 - q0) * s
        tool = kin.tool(q)
        if tool[2, 3] - tilt_drop < min_tip_z or kin.flange(q)[2, 3] < min_link_z:
            return False
    return True


