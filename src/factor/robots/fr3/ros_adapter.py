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


class FR3Ros:
    """Streaming FR3 over the FACTOR ROS 2 driver. / ROS 2 드라이버 위의 FR3.

    Parameters
    ----------
    node
        An ``rclpy`` node owned by the caller (spun by the caller between ticks).
    params, branch
        Kinematic parameters and the IK branch of the grasp posture.
    grip_offset
        Tool-frame vector from the tool tip to the grip point.
    tool_yaw
        Rotation of the finger axis about world z (the task row direction).
    """

    def __init__(
        self,
        node,
        params: FR3Params,
        branch: tuple[int, int, int],
        grip_offset: np.ndarray,
        tool_yaw: float,
        finger_axis: str = "y",
        finger_sign: int = 1,
    ):
        import rclpy  # noqa: F401
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger

        self.node = node
        self.kin = FR3Kinematics(params)
        self.branch = branch
        self.grip_offset = np.asarray(grip_offset, dtype=float)
        self.rotation = top_down_rotation(finger_axis, finger_sign, yaw=tool_yaw)
        self._q: np.ndarray | None = None
        self._msg = JointState
        node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self._pub = node.create_publisher(JointState, "/fr/servo_target", 10)
        self._start = node.create_client(Trigger, "/fr/servo_start")
        self._stop = node.create_client(Trigger, "/fr/servo_stop")
        self._reset = node.create_client(Trigger, "/fr/reset_error")

    def _on_joints(self, msg) -> None:
        pos = dict(zip(msg.name, msg.position, strict=False))
        if all(j in pos for j in JOINTS):
            self._q = np.array([pos[j] for j in JOINTS], dtype=float)

    def _spin(self, seconds: float = 0.0) -> None:
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=seconds)

    def wait_ready(self, timeout: float = 10.0) -> np.ndarray:
        t0 = time.time()
        while self._q is None and time.time() - t0 < timeout:
            self._spin(0.1)
        if self._q is None:
            raise RuntimeError("no /joint_states (is the driver running?)")
        return self._q.copy()

    # ---- Robot protocol
    def joint_positions(self) -> np.ndarray:
        self._spin(0.0)
        if self._q is None:
            raise RuntimeError("no joint state yet")
        return self._q.copy()

    def grip_point(self) -> np.ndarray:
        tool = self.kin.tool(self.joint_positions())
        return tool[:3, 3] + tool[:3, :3] @ self.grip_offset

    def solve_ik(self, grip_point: np.ndarray) -> np.ndarray | None:
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = np.asarray(grip_point, dtype=float) - self.rotation @ self.grip_offset
        return self.kin.inverse(pose, self.branch)

    def stream_target(self, joint_positions: np.ndarray) -> None:
        msg = self._msg()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(v) for v in joint_positions]
        self._pub.publish(msg)

    def _call(self, client, timeout: float = 5.0):
        import rclpy

        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{client.srv_name} unavailable (driver without servo stream?)")
        future = client.call_async(client.srv_type.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        return future.result()

    def start_streaming(self) -> None:
        # A stopped controller ignores ServoJ until its error state is reset (FR3 v3.8.6).
        # 정지 상태의 제어기는 에러 리셋 전까지 ServoJ 를 무시한다.
        # The driver's error flag lags the reset by a status cycle: retry briefly.
        # 드라이버의 에러 플래그는 리셋보다 한 상태 주기 늦게 풀린다: 잠시 재시도.
        result = None
        for _ in range(4):
            self._call(self._reset)
            time.sleep(0.5)
            result = self._call(self._start)
            if result is not None and result.success:
                break
        if result is None or not result.success:
            raise RuntimeError(f"servo stream refused: {None if result is None else result.message}")

    def stop_streaming(self) -> None:
        self._call(self._stop)


def open_sdk_rpc(ip: str, wait_state: float = 1.5):
    """Open a FAIRINO SDK handle that works next to the ROS driver. / 드라이버 옆에서 쓰는 SDK 핸들.

    The controller serves its real-time state port (20004) to one client only, and the driver
    holds it. The SDK then loops forever trying to reconnect, and every XML-RPC call blocks
    while that loop runs. So: wait briefly for state packets and, if none arrive, stop the
    state thread and make the handle XML-RPC only (gripper and error-reset commands still work).

    제어기는 실시간 상태 포트(20004)를 한 클라이언트에만 주고 드라이버가 잡고 있다. SDK 는 그동안
    재접속 루프를 돌며 모든 XML-RPC 호출을 막으므로, 상태 패킷이 안 오면 상태 스레드를 끄고
    XML-RPC 전용으로 쓴다.
    """
    import contextlib
    import io

    from fairino import Robot  # vendor SDK / 제조사 SDK

    with contextlib.redirect_stdout(io.StringIO()):
        rpc = Robot.RPC(ip)
    if not Robot.RPC.is_conect:
        Robot.RPC.is_conect = True
        raise ConnectionError(f"{ip}: XML-RPC (20003) connection failed")

    def has_state() -> bool:
        pkg = rpc.robot_state_pkg
        return not isinstance(pkg, type) and int(pkg.frame_head) == 0x5A5A

    t0 = time.time()
    while time.time() - t0 < wait_state and not has_state():
        time.sleep(0.05)
    if not has_state():
        rpc.closeRPC_state = True
        rpc.robot_realstate_exit = True
        try:
            rpc.stop_event.set()
        except AttributeError:
            pass
        rpc.connect_to_robot = lambda: True
        rpc.reconnect_flag = False
    return rpc


class SdkGripper:
    """Parallel gripper through the FAIRINO SDK (controller RS-485). / 제어기 경유 그리퍼.

    ``half_gap`` values are metres; the controller takes a percentage of the stroke where
    ``open_pct`` is fully open.
    """

    def __init__(
        self, rpc, stroke: float = 0.095, open_pct: int = 100, index: int = 1, vel: int = 30, force: int = 30
    ):
        self.rpc = rpc
        self.stroke = stroke
        self.open_pct = open_pct
        self.index = index
        self.vel = vel
        self.force = force
        self._last = stroke / 2.0

    def command(self, half_gap: float, blocking: bool = False) -> None:
        """Send the width; the XML-RPC round trip can take over a second, so the loop must not
        wait for it: non-blocking calls run in a worker thread while the servo stream continues.

        XML-RPC 왕복이 1 초를 넘길 수 있어 루프가 기다리면 서보 스트림이 끊긴다. 비차단 호출은
        작업 스레드에서 보낸다.
        """
        import threading

        width = max(0.0, min(self.stroke, 2.0 * float(half_gap)))
        pct = width / self.stroke * 100.0
        pos = int(round(pct if self.open_pct == 100 else 100.0 - pct))
        self._last = width / 2.0

        def send() -> None:
            err = self.rpc.MoveGripper(self.index, pos, self.vel, self.force, 5000, 1, 0, 0.0, 0, 0)
            if err != 0:
                print(f"[gripper] MoveGripper error {err}", flush=True)

        if blocking:
            send()
            time.sleep(1.5)
        else:
            threading.Thread(target=send, daemon=True).start()

    def half_gap(self) -> float:
        return self._last


class RosGripper:
    """Gripper through the driver's MoveIt action. / 드라이버의 MoveIt 액션 경유 그리퍼.

    The driver owns the only SDK handle that can drive the AG-95 reliably (it activates the
    gripper after a controller power cycle and clamps the width); the loop sends the half-gap
    as a one-point trajectory and never waits for it, so the servo stream keeps flowing.

    드라이버가 그리퍼를 책임진다(전원 재투입 후 활성화, 폭 클램프). 루프는 반개구를 한 점짜리
    궤적으로 보내고 기다리지 않는다.
    """

    def __init__(self, node, joint: str = "ag95_left_finger_joint", action: str | None = None):
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient

        self.node = node
        self.joint = joint
        self._msg = FollowJointTrajectory
        self._client = ActionClient(
            node, FollowJointTrajectory, action or "/ag95_gripper_controller/follow_joint_trajectory"
        )
        self._last = 0.03
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("gripper action server unavailable (driver running?)")

    def command(self, half_gap: float, blocking: bool = False) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self._msg.Goal()
        goal.trajectory.joint_names = [self.joint]
        point = JointTrajectoryPoint()
        point.positions = [float(half_gap)]
        point.time_from_start = Duration(sec=1)
        goal.trajectory.points = [point]
        self._last = float(half_gap)
        future = self._client.send_goal_async(goal)
        if blocking:
            import rclpy

            rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
            handle = future.result()
            if handle is not None and handle.accepted:
                rclpy.spin_until_future_complete(self.node, handle.get_result_async(), timeout_sec=8.0)

    def half_gap(self) -> float:
        return self._last


def tool_yaw_from_anchor(anchor_in_base: np.ndarray) -> float:
    """Finger-axis yaw that aligns with the anchor's x axis. / 앵커 x 축에 맞춘 핑거축 yaw."""
    rot = np.asarray(anchor_in_base, dtype=float)[:3, :3]
    return math.atan2(rot[1, 0], rot[0, 0])
