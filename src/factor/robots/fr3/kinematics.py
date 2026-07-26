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


