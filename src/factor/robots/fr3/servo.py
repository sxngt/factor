"""Rate-limited joint servo stream, independent of ROS. / ROS 와 무관한 속도 제한 관절 서보 스트림.

The driver receives sparse joint targets (the FACTOR loop runs at ~10 Hz) and must turn them
into a smooth ServoJ stream (every ``cmd_t``, 16 ms on the FR3). Each tick the commanded joints
move towards the latest target with three limits:

* speed: at most ``max_joint_speed_deg * cmd_t`` per tick;
* acceleration: the per-tick step changes by at most ``max_joint_accel_deg * cmd_t²`` (a step
  change of speed shakes the base);
* braking: the step never exceeds the speed from which the arm can still stop within the
  remaining distance ``d`` when it slows by ``a`` per tick, ``v (v + a) / 2a <= d``, so it
  slows down smoothly into the target. (The continuous-time bound ``sqrt(2 a d)`` lets the
  discrete loop overshoot by up to ~0.2° at the default limits; this one stays below ``a``.)

The stream ends on request, when no target arrived for ``servo_timeout`` seconds, when a target
jumps more than ``servo_max_jump_deg`` from the command, on a safety trip, or on an SDK error.

드라이버는 드문 관절 목표(~10 Hz)를 ``cmd_t`` (16 ms) 마다의 부드러운 ServoJ 스트림으로 바꾼다.
매 틱 속도·가속도·제동(틱당 a 씩 줄여 남은 거리 안에 멈출 수 있는 속도) 한계로 목표에 다가간다.
요청, 목표 무갱신(``servo_timeout``), 목표 급변(``servo_max_jump_deg``), 안전 트립, SDK 오류에서 끝난다.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServoParams:
    """Servo stream parameters (the driver's ROS parameters). / 서보 스트림 파라미터."""

    cmd_t: float = 0.016  # ServoJ period (s) / ServoJ 주기
    max_joint_speed_deg: float = 30.0  # deg/s
    max_joint_accel_deg: float = 60.0  # deg/s²
    servo_timeout: float = 1.5  # s without a new target / 목표 무갱신 한계
    servo_max_jump_deg: float = 15.0  # target vs command / 목표-명령 차 한계
    start_tol_deg: float = 2.0  # safety monitor tracking deviation / 안전 감시 추종 편차

    @property
    def max_step(self) -> float:
        """Largest joint step per tick (rad). / 틱당 최대 관절 이동(rad)."""
        return math.radians(self.max_joint_speed_deg) * self.cmd_t

    @property
    def max_dv(self) -> float:
        """Largest change of the per-tick step (rad/tick). / 틱당 이동량의 최대 변화."""
        return math.radians(self.max_joint_accel_deg) * self.cmd_t * self.cmd_t

    @property
    def max_jump(self) -> float:
        return math.radians(self.servo_max_jump_deg)


def servo_step(
    q_cmd: np.ndarray, v_cmd: np.ndarray, target: np.ndarray, max_step: float, max_dv: float
) -> tuple[np.ndarray, np.ndarray]:
    """One tick of the limiter: new command and new per-tick step. / 한 틱: 새 명령과 틱당 이동량.

    All quantities are per joint, in radians (``v_cmd`` in rad per tick).
    모든 값은 관절별 rad (``v_cmd`` 는 rad/틱).
    """
    q_cmd = np.asarray(q_cmd, dtype=float)
    v_cmd = np.asarray(v_cmd, dtype=float)
    gap = np.asarray(target, dtype=float) - q_cmd

    # Speed limit, then the braking speed for the remaining distance: the largest v with
    # v + (v - a) + (v - 2a) + ... <= d, i.e. v (v + a) / 2a <= d.
    # 속도 한계, 그다음 남은 거리에서 틱당 a 씩 줄여 멈출 수 있는 최대 속도.
    want = np.clip(gap, -max_step, max_step)
    brake = 0.5 * (np.sqrt(max_dv * max_dv + 8.0 * max_dv * np.abs(gap)) - max_dv)
    want = np.clip(want, -brake, brake)

    # Acceleration limit on the change of the step. / 이동량 변화의 가속도 한계.
    v_new = np.clip(want, v_cmd - max_dv, v_cmd + max_dv)
    return q_cmd + v_new, v_new


