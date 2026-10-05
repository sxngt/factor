"""ROS 2 servo-stream driver for the FAIRINO FR3 with an AG-95 gripper (``factor-fr3-driver``).

Interface (what :mod:`factor.robots.fr3.ros_adapter` uses):

* publishes ``/joint_states`` (``sensor_msgs/JointState``, 50 Hz): j1..j6 (rad, rad/s, Nm) plus
  the two AG-95 finger joints (half gap, m), from the controller's real-time state packet;
* publishes ``/fr/status`` (``std_msgs/String``, JSON, 2 Hz): decoded controller status;
* subscribes ``/fr/servo_target`` (``sensor_msgs/JointState``, j1..j6 rad): the latest target of
  the servo stream;
* services (``std_srvs/Trigger``): ``/fr/servo_start``, ``/fr/servo_stop``, ``/fr/reset_error``,
  ``/fr/stop``, ``/fr/enable``, ``/fr/set_mode_auto``;
* action ``/ag95_gripper_controller/follow_joint_trajectory``
  (``control_msgs/FollowJointTrajectory``): the last point's ``ag95_left_finger_joint`` value is
  the half gap to move to.

Parameters (``--ros-args -p name:=value`` or ``ros2 param set``): ``start_tol_deg`` (safety
tracking deviation), ``max_joint_speed_deg``, ``max_joint_accel_deg``, ``servo_timeout``,
``servo_max_jump_deg``, ``cmd_t``; see :mod:`factor.robots.fr3.servo` for the limiter.

The driver never commands motion on its own: the arm moves only inside a servo session, which
starts only in AUTO mode without safety stop or error, and every SDK call goes through one lock
(the SDK's XML-RPC proxy is not thread safe; a shared proxy once ended the stream exactly at a
gripper release). ``--fake`` runs without the SDK on a first-order arm model.

FR3 + AG-95 용 ROS 2 서보 스트림 드라이버. 드라이버는 스스로 모션을 내지 않는다: 팔은 서보 세션 안에서만
움직이고, 세션은 AUTO·안전 정상·에러 없음에서만 시작한다. 모든 SDK 호출은 하나의 잠금을 거친다.
``--fake`` 는 SDK 없이 1차 지연 팔 모델로 돈다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np

try:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError as exc:  # pragma: no cover - needs a ROS 2 install
    raise ImportError(
        "factor-fr3-driver needs ROS 2 (rclpy, control_msgs, std_srvs); source the ROS setup file first"
    ) from exc

from factor.robots.fr3 import sdk
from factor.robots.fr3.gripper import STROKE_M, ControllerGripper, FakeGripper
from factor.robots.fr3.servo import FakeArm, ServoParams, ServoStream

JOINTS = list(sdk.JOINTS)
FINGERS = ["ag95_left_finger_joint", "ag95_right_finger_joint"]
GRIP_ACTION = "/ag95_gripper_controller/follow_joint_trajectory"
GRIP_JOINT = "ag95_left_finger_joint"
# A pose above the reference workspace, for --fake. / --fake 시작 자세(기준 작업영역 위).
FAKE_START_DEG = (61.48, -60.45, 40.49, -70.02, -89.99, -119.55)


