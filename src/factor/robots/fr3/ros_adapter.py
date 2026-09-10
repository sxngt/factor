"""FR3 adapter over ROS 2 for the real-robot loop.

Implements :class:`factor.real.robot.Robot` on top of a driver that publishes ``/joint_states``
and accepts streamed joint targets on ``/fr/servo_target`` between ``/fr/servo_start`` and
``/fr/servo_stop`` (``std_srvs/Trigger``). The driver owns rate limiting and the safety monitor;
this adapter only does kinematics and message plumbing. The gripper goes through the vendor SDK
(:class:`SdkGripper`) because the FR3 controller drives it over the tool-flange RS-485 bus.

ROS 2 위의 FR3 어댑터. 드라이버가 ``/joint_states`` 를 내고 ``/fr/servo_target`` 으로 스트리밍 목표를
받는다(속도 제한·안전 감시는 드라이버 몫). 이 어댑터는 기구학과 메시지 연결만 담당한다.
"""

from __future__ import annotations

import math
import time

import numpy as np

from factor.robots.fr3.kinematics import FR3Kinematics, FR3Params, top_down_rotation

JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")


