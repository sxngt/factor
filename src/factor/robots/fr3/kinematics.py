"""FR3 kinematics: classic-DH forward kinematics and analytic fixed-branch inverse kinematics.

The FR3 shares the UR structure (joints 2-4 parallel, joint 5 perpendicular). The inverse
kinematics follows Hawkins (2013); FACTOR only ever needs *one* branch (shoulder, elbow,
wrist) because the grasp posture is fixed, so the solver returns that branch directly and
reports reachability instead of enumerating all eight solutions. A vectorised torch version
serves the simulation twin.

FR3 기구학. 고정된 파지 자세만 쓰므로 해석적 IK 는 지정한 가지 하나만 돌려주고 도달 가능 여부를
함께 반환한다. torch 벡터화 버전은 시뮬 트윈이 쓴다.

Tool frame convention (shared with the real-robot scripts): approach axis = tool -y, finger
opening axis = tool x, origin = flange origin + ``tool_length`` along the flange normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FR3Params:
    """Calibrated DH parameters and limits. / 보정된 DH 파라미터와 리밋.

    Defaults are the nominal FR3 values; a calibrated set lives in the scene configuration.
    """

    d1: float = 0.140
    a2: float = -0.280
    a3: float = -0.240
    d4: float = 0.102
    d5: float = 0.102
    d6: float = 0.100
    tool_length: float = 0.2045
    mount_yaw: float = 0.0
    limits_deg: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: ((-175, 175), (-265, 85), (-150, 150), (-265, 85), (-175, 175), (-175, 175))
    )

    @classmethod
    def from_mapping(cls, m: dict) -> FR3Params:
        """Build from a scene's ``robot.kinematics`` mapping. / 장면 명세의 kinematics 사전에서 생성."""
        kwargs = {k: float(m[k]) for k in ("d1", "a2", "a3", "d4", "d5", "d6", "tool_length") if k in m}
        if "mount_yaw_deg" in m:
            kwargs["mount_yaw"] = math.radians(float(m["mount_yaw_deg"]))
        if "limits_deg" in m:
            kwargs["limits_deg"] = tuple(tuple(float(v) for v in pair) for pair in m["limits_deg"])
        return cls(**kwargs)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def top_down_rotation(finger_axis: str = "y", sign: int = 1, yaw: float = 0.0) -> np.ndarray:
    """Tool rotation for a top-down grasp. / 탑다운 파지의 툴 회전.

    The approach axis (tool -y) points down, so tool y is world +z. The finger axis (tool x)
    is world ±y or ±x, rotated by ``yaw`` about world z.
    """
    y = np.array([0.0, 0.0, 1.0])
    x = np.array([0.0, float(sign), 0.0]) if finger_axis == "y" else np.array([float(sign), 0.0, 0.0])
    rot = np.column_stack([x, y, np.cross(x, y)])
    return _rot_z(yaw)[:3, :3] @ rot


class FR3Kinematics:
    """Forward and fixed-branch inverse kinematics. / 정기구학과 고정 가지 역기구학."""

    def __init__(self, params: FR3Params | None = None):
        self.p = params or FR3Params()
        self.limits = np.radians(np.asarray(self.p.limits_deg, dtype=float))

    def tool_transform(self) -> np.ndarray:
        """Flange -> tool. / 플랜지 → 툴."""
        m = self.p.mount_yaw
        rot = np.array([[math.cos(m), 0.0, -math.sin(m)], [math.sin(m), 0.0, math.cos(m)], [0.0, -1.0, 0.0]])
        t = np.eye(4)
        t[:3, :3] = rot
        t[2, 3] = self.p.tool_length
        return t

    def flange(self, q: np.ndarray) -> np.ndarray:
        """Base -> flange pose. / 베이스 → 플랜지."""
        p = self.p
        q = np.asarray(q, dtype=float)
        return (
            _rot_z(q[0])
            @ _trans(0, 0, p.d1)
            @ _rot_x(math.pi / 2)
            @ _rot_z(q[1])
            @ _trans(p.a2, 0, 0)
            @ _rot_z(q[2])
            @ _trans(p.a3, 0, 0)
            @ _rot_z(q[3])
            @ _trans(0, 0, p.d4)
            @ _rot_x(math.pi / 2)
            @ _rot_z(q[4])
            @ _trans(0, 0, p.d5)
            @ _rot_x(-math.pi / 2)
            @ _rot_z(q[5])
            @ _trans(0, 0, p.d6)
        )

    def tool(self, q: np.ndarray) -> np.ndarray:
        """Base -> tool pose. / 베이스 → 툴."""
        return self.flange(q) @ self.tool_transform()

    def inverse(self, tool_pose: np.ndarray, branch: tuple[int, int, int]) -> np.ndarray | None:
        """Joint angles for ``tool_pose`` on ``branch``, or ``None`` if unreachable (NumPy only).

        ``branch`` is (shoulder, elbow, wrist) with each entry 0 or 1. The same derivation as
        :meth:`inverse_batch`, kept dependency-free for the robot-side process.
        지정 가지의 관절각, 도달 불가면 ``None``. torch 없이 동작한다.
        """
        p = self.p
        sh, el, wr = branch
        t06 = np.asarray(tool_pose, dtype=float) @ np.linalg.inv(self.tool_transform())
        p06 = t06[:3, 3]
        p05 = p06 - p.d6 * t06[:3, 2]
        r05 = math.hypot(p05[0], p05[1])
        if r05 <= abs(p.d4):
            return None
        phi = math.atan2(p05[1], p05[0])
        psi = math.acos(max(-1.0, min(1.0, p.d4 / r05)))
        th1 = phi + (psi if sh == 0 else -psi) + math.pi / 2
        s1, c1 = math.sin(th1), math.cos(th1)
        c5 = (p06[0] * s1 - p06[1] * c1 - p.d4) / p.d6
        if abs(c5) > 1.0:
            return None
        th5 = math.acos(max(-1.0, min(1.0, c5)))
        th5 = th5 if wr == 0 else -th5
        s5 = math.sin(th5)
        t60 = np.linalg.inv(t06)
        if abs(s5) < 1e-6:
            th6 = 0.0
        else:
            th6 = math.atan2((-t60[1, 0] * s1 + t60[1, 1] * c1) / s5, (t60[0, 0] * s1 - t60[0, 1] * c1) / s5)
        t01 = _rot_z(th1) @ _trans(0, 0, p.d1) @ _rot_x(math.pi / 2)
        t45 = _rot_z(th5) @ _trans(0, 0, p.d5) @ _rot_x(-math.pi / 2)
        t56 = _rot_z(th6) @ _trans(0, 0, p.d6)
        t14 = np.linalg.inv(t01) @ t06 @ np.linalg.inv(t45 @ t56)
        p14 = t14[:3, 3]
        r14 = math.hypot(p14[0], p14[1])
        c3 = (r14 * r14 - p.a2 * p.a2 - p.a3 * p.a3) / (2 * p.a2 * p.a3)
        if abs(c3) > 1.0:
            return None
        th3 = math.acos(max(-1.0, min(1.0, c3)))
        th3 = th3 if el == 0 else -th3
        th2 = math.atan2(p14[1], p14[0]) - math.atan2(p.a3 * math.sin(th3), p.a2 + p.a3 * math.cos(th3))
        t34 = np.linalg.inv(_rot_z(th2) @ _trans(p.a2, 0, 0) @ _rot_z(th3) @ _trans(p.a3, 0, 0)) @ t14
        th4 = math.atan2(t34[1, 0], t34[0, 0])
        q = np.array([th1, th2, th3, th4, th5, th6])
        q = (q + math.pi) % (2 * math.pi) - math.pi
        for i in range(6):
            if q[i] < self.limits[i, 0]:
                q[i] += 2 * math.pi
            if q[i] > self.limits[i, 1]:
                q[i] -= 2 * math.pi
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return None
        return q

    def inverse_batch(self, tool_poses, branch: tuple[int, int, int]):
        """Vectorised fixed-branch IK. / 벡터화 고정 가지 IK.

        ``tool_poses`` is a ``(N, 4, 4)`` torch tensor of base -> tool poses. Returns
        ``(q, valid)`` with ``q`` wrapped into the joint limits where possible.
        """
        import torch

        p = self.p
        sh, el, wr = branch
        t = tool_poses
        dev, dt = t.device, t.dtype
        n = t.shape[0]

        tool_inv = torch.as_tensor(np.linalg.inv(self.tool_transform()), device=dev, dtype=dt)
        t06 = t @ tool_inv
        p06 = t06[:, :3, 3]
        p05 = p06 - p.d6 * t06[:, :3, 2]
        r05 = torch.hypot(p05[:, 0], p05[:, 1])
        valid = r05 > abs(p.d4)

        phi = torch.atan2(p05[:, 1], p05[:, 0])
        psi = torch.acos((p.d4 / r05.clamp(min=1e-9)).clamp(-1.0, 1.0))
        th1 = phi + (psi if sh == 0 else -psi) + math.pi / 2
        s1, c1 = torch.sin(th1), torch.cos(th1)

        c5 = (p06[:, 0] * s1 - p06[:, 1] * c1 - p.d4) / p.d6
        valid &= c5.abs() <= 1.0
        th5 = torch.acos(c5.clamp(-1.0, 1.0))
        th5 = th5 if wr == 0 else -th5
        s5 = torch.sin(th5)
        safe_s5 = torch.where(s5.abs() < 1e-6, torch.ones_like(s5), s5)
        t60 = torch.linalg.inv(t06)
        th6 = torch.atan2(
            (-t60[:, 1, 0] * s1 + t60[:, 1, 1] * c1) / safe_s5,
            (t60[:, 0, 0] * s1 - t60[:, 0, 1] * c1) / safe_s5,
        )
        th6 = torch.where(s5.abs() < 1e-6, torch.zeros_like(th6), th6)

        def rot_z(a):
            c, s = torch.cos(a), torch.sin(a)
            m = torch.eye(4, device=dev, dtype=dt).repeat(a.shape[0], 1, 1)
            m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -s, s, c
            return m

        def const(m):
            return torch.as_tensor(m, device=dev, dtype=dt).repeat(n, 1, 1)

        t01 = rot_z(th1) @ const(_trans(0, 0, p.d1) @ _rot_x(math.pi / 2))
        t45 = rot_z(th5) @ const(_trans(0, 0, p.d5) @ _rot_x(-math.pi / 2))
        t56 = rot_z(th6) @ const(_trans(0, 0, p.d6))
        t14 = torch.linalg.inv(t01) @ t06 @ torch.linalg.inv(t45 @ t56)
        p14 = t14[:, :3, 3]
        r14 = torch.hypot(p14[:, 0], p14[:, 1])

        c3 = (r14 * r14 - p.a2 * p.a2 - p.a3 * p.a3) / (2 * p.a2 * p.a3)
        valid &= c3.abs() <= 1.0
        th3 = torch.acos(c3.clamp(-1.0, 1.0))
        th3 = th3 if el == 0 else -th3
        th2 = torch.atan2(p14[:, 1], p14[:, 0]) - torch.atan2(
            p.a3 * torch.sin(th3), p.a2 + p.a3 * torch.cos(th3)
        )
        t34 = (
            torch.linalg.inv(rot_z(th2) @ const(_trans(p.a2, 0, 0)) @ rot_z(th3) @ const(_trans(p.a3, 0, 0)))
            @ t14
        )
        th4 = torch.atan2(t34[:, 1, 0], t34[:, 0, 0])

        q = torch.stack([th1, th2, th3, th4, th5, th6], dim=1)
        q = (q + math.pi) % (2 * math.pi) - math.pi
        lim = torch.as_tensor(self.limits, device=dev, dtype=dt)
        for i in range(6):
            q[:, i] = torch.where(q[:, i] < lim[i, 0], q[:, i] + 2 * math.pi, q[:, i])
            q[:, i] = torch.where(q[:, i] > lim[i, 1], q[:, i] - 2 * math.pi, q[:, i])
        valid &= ((q >= lim[:, 0]) & (q <= lim[:, 1])).all(dim=1)
        return q, valid
