"""Kinematic stand-ins for the robot and gripper. / 로봇·그리퍼의 운동학 대역.

``factor-run --fake`` drives the whole real-robot loop (camera, fiducials, estimator, skill,
workspace clamp, IK, telemetry) against these instead of the hardware: streamed joint targets
are taken as reached instantly, the gripper reports what it was told. Useful for validating a
new estimator or a loop change with the live camera before any motion.

실기 루프 전체를 하드웨어 없이 돌려 보기 위한 대역. 스트리밍한 관절 목표는 즉시 도달한 것으로
치고, 그리퍼는 명령값을 그대로 보고한다.
"""

from __future__ import annotations

import numpy as np

from factor.robots.fr3.kinematics import FR3Kinematics


class FakeRobot:
    """Instantly-reaching robot with the FR3 kinematics. / 즉시 도달하는 FR3 운동학 로봇."""

    def __init__(
        self,
        kinematics: FR3Kinematics,
        branch: tuple[int, int, int],
        grip_offset: np.ndarray,
        rotation: np.ndarray,
        joint_positions: np.ndarray,
    ):
        self.kin = kinematics
        self.branch = branch
        self.grip_offset = np.asarray(grip_offset, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.q = np.asarray(joint_positions, dtype=float)
        self.streamed = 0

    def joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def grip_point(self) -> np.ndarray:
        tool = self.kin.tool(self.q)
        return tool[:3, 3] + tool[:3, :3] @ self.grip_offset

    def solve_ik(self, grip_point: np.ndarray) -> np.ndarray | None:
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = np.asarray(grip_point, dtype=float) - self.rotation @ self.grip_offset
        return self.kin.inverse(pose, self.branch)

    def stream_target(self, joint_positions: np.ndarray) -> None:
        self.q = np.asarray(joint_positions, dtype=float)
        self.streamed += 1

    def start_streaming(self) -> None:
        pass

    def stop_streaming(self) -> None:
        pass


class FakeGripper:
    """Gripper that reports its last command. / 마지막 명령을 보고하는 그리퍼."""

    def __init__(self, half_gap: float = 0.03):
        self._half_gap = half_gap
        self.commands: list[float] = []

    def command(self, half_gap: float, blocking: bool = False) -> None:
        self._half_gap = float(half_gap)
        self.commands.append(self._half_gap)

    def half_gap(self) -> float:
        return self._half_gap
