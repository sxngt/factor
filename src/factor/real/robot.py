"""Robot and gripper interfaces. / 로봇·그리퍼 인터페이스.

The skill loop needs only three things from a robot: read the joint state, stream a joint target
at a fixed rate, and stop. Inverse kinematics is the adapter's responsibility because the
grasp posture is robot-specific.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Gripper(Protocol):
    """Parallel gripper. / 평행 그리퍼."""

    def command(self, half_gap: float, blocking: bool = False) -> None: ...

    def half_gap(self) -> float: ...


class Robot(Protocol):
    """Position-controlled arm with a streaming interface. / 스트리밍 위치 제어 팔."""

    def joint_positions(self) -> np.ndarray: ...

    def grip_point(self) -> np.ndarray:
        """Grip point in the base frame (forward kinematics plus tool offset). / 베이스 프레임 그립점."""
        ...

    def solve_ik(self, grip_point: np.ndarray) -> np.ndarray | None:
        """Joint target for a grasp posture at ``grip_point``, or ``None`` if unreachable."""
        ...

    def stream_target(self, joint_positions: np.ndarray) -> None:
        """Send one joint target; the adapter rate-limits and safety-checks it."""
        ...

    def start_streaming(self) -> None: ...

    def stop_streaming(self) -> None: ...
