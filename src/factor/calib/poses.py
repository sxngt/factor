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


def generate(
    kin,
    branch,
    base_rotation: np.ndarray,
    centre: np.ndarray,
    grip_offset: np.ndarray,
    q_start: np.ndarray,
    min_tip_z: float,
    finger_reach: float = 0.05,
    n: int = 400,
    span_xy: float = 0.07,
    span_z: tuple[float, float] = (0.0, 0.09),
    max_tilt_deg: float = 28.0,
    max_yaw_deg: float = 60.0,
    max_joint_step_deg: float = 100.0,
    seed: int = 0,
) -> list[Candidate]:
    """Random reachable poses: grip point near ``centre`` (base), tool tilted about two world axes
    and turned about world z, so that the relative rotations span all three axes.

    무작위 후보: 그립점은 centre 근처, 툴은 세계 x/y 축 기울기 + z 회전 (상대 회전이 세 축을 모두 덮도록).
    """
    rng = np.random.default_rng(seed)
    out: list[Candidate] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        dx, dy = rng.uniform(-span_xy, span_xy, 2)
        dz = rng.uniform(*span_z)
        tx, ty = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
        yaw = math.radians(rng.uniform(-max_yaw_deg, max_yaw_deg))
        rot = _rz(yaw) @ _rx(tx) @ _ry(ty) @ base_rotation
        grip = np.asarray(centre, dtype=float) + np.array([dx, dy, dz])
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = grip - rot @ grip_offset
        q = kin.inverse(pose, branch)
        if q is None:
            continue
        # Same joint wrap as the start, away from the limits (FR3 stops with error 29 at a limit).
        # 시작과 같은 감김, 관절 한계에서 여유 (한계에 닿으면 FR3 가 에러 29 로 멈춘다).
        q = near_within_limits(kin, q, q_start)
        if q is None:
            continue
        if np.degrees(np.abs(q - q_start)).max() > max_joint_step_deg:
            continue
        tilt = math.acos(max(-1.0, min(1.0, float((rot @ base_rotation.T)[2, 2]))))
        drop = finger_reach * math.sin(tilt)
        if pose[2, 3] - drop < min_tip_z:
            continue
        out.append(
            Candidate(
                q,
                kin.flange(q),
                f"d({dx * 1e3:+.0f},{dy * 1e3:+.0f},{dz * 1e3:+.0f})mm "
                f"t({math.degrees(tx):+.0f},{math.degrees(ty):+.0f}) yaw{math.degrees(yaw):+.0f}",
            )
        )
    return out


def search_poses(kin, branch, base_rotation, centre, grip_offset, q_start, step_deg: int = 30):
    """Poses for the first sighting when the board mount is unknown: turn about the vertical.

    Sweeps one way from the start until a joint limit, then the other way until a full turn is
    covered. Each pose is wrapped next to the previous one, so the wrist never swings across.
    보드 장착을 모를 때 첫 관측 자세: 수직축 회전. 한쪽으로 관절 한계까지, 반대쪽으로 한 바퀴가
    덮일 때까지 이어서 돌린다 (각 자세는 직전 자세 옆으로 감아 손목이 크게 휘돌지 않게).
    """

    def at(yaw_deg, q_ref):
        rot = _rz(math.radians(yaw_deg)) @ base_rotation
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = np.asarray(centre, dtype=float) - rot @ grip_offset
        q = kin.inverse(pose, branch)
        if q is None:
            return None
        q = q_ref + (q - q_ref + math.pi) % (2 * math.pi) - math.pi
        lo = kin.limits[:, 0] + math.radians(10.0)
        hi = kin.limits[:, 1] - math.radians(10.0)
        return q if np.all(q >= lo) and np.all(q <= hi) else None

    out = []
    q0 = at(0, np.asarray(q_start, dtype=float))
    if q0 is None:
        return out
    out.append(Candidate(q0, kin.flange(q0), "search yaw+0"))
    reach = {1: 0, -1: 0}
    for sign in (1, -1):
        q_prev = q0
        y = 0
        while reach[1] + reach[-1] < 360 - step_deg:
            y += step_deg
            q = at(sign * y, q_prev)
            if q is None:
                break
            out.append(Candidate(q, kin.flange(q), f"search yaw{sign * y:+d}"))
            reach[sign] = y
            q_prev = q
    return out
