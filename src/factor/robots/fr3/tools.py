"""FR3 command-line helpers: power/mode, preflight check, servo probe. / FR3 명령행 도구.

* ``factor-fr3-power``: controller state, enable, AUTO/MANUAL, drag, stop, error reset, speed,
  and ``recover`` (wait for the e-stop release, reset, enable, AUTO);
* ``factor-fr3-check``: read-only preflight of the controller (exit 0 = READY);
* ``factor-fr3-probe``: through the running driver, wait for ``/joint_states``, set driver
  parameters, and check that the servo stream really moves the arm (j6 ±0.3°).

The first two talk to the controller through the vendor SDK. The state port serves one client:
while the driver runs, they get an XML-RPC-only handle, so commands work but status does not.
앞의 두 도구는 SDK 로 제어기와 통신한다. 드라이버가 상태 포트를 잡고 있으면 명령은 되지만 상태 읽기는 안 된다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from factor.robots.fr3 import sdk

# ----------------------------------------------------------------------------- preflight


@dataclass
class Report:
    """Preflight result: ``(level, name, detail)`` checks. / 사전 점검 결과."""

    checks: list = field(default_factory=list)  # level in PASS / WARN / FAIL / INFO

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.checks.append((level, name, detail))

    @property
    def ready(self) -> bool:
        return not any(level == "FAIL" for level, _, _ in self.checks)

    def text(self) -> str:
        width = max((len(name) for _, name, _ in self.checks), default=10)
        lines = [f"[{level:4s}] {name:<{width}s}  {detail}" for level, name, detail in self.checks]
        lines.append(f"==> {'READY' if self.ready else 'NOT READY'}")
        return "\n".join(lines)


def preflight(
    status: sdk.RobotStatus, expect_mode: str | None = "AUTO", versions: dict | None = None
) -> Report:
    """Checks before streaming motion (ServoJ needs AUTO, no error, no safety stop).

    모션 전 점검 (ServoJ 는 AUTO·에러 없음·안전정지 없음 필요).
    """
    r = Report()
    if not status.ok:
        r.add(
            "FAIL",
            "controller",
            f"{status.ip}: {status.error} (power, cable, IP; driver holding the state port?)",
        )
        return r
    r.add("PASS", "controller", f"{status.ip} real-time state OK (frame {status.frame_cnt})")
    if versions is not None:
        software = versions.get("software")
        firmware = software[1] if isinstance(software, list) and len(software) > 1 else "?"
        level = "PASS" if firmware == sdk.TESTED_FIRMWARE else "WARN"
        note = "" if level == "PASS" else f" (tested with firmware {sdk.TESTED_FIRMWARE}; match the SDK tag)"
        r.add(level, "versions", f"robot {software} sdk {versions.get('sdk')}{note}")
    r.add(
        "FAIL" if status.estop else "PASS",
        "e-stop",
        "pressed: release, then factor-fr3-power reset" if status.estop else "released",
    )
    if status.safety_stop0 or status.safety_stop1:
        r.add("FAIL", "safety stop", f"SI0={status.safety_stop0} SI1={status.safety_stop1}")
    else:
        r.add("PASS", "safety stop", "SI0/SI1 clear")
    r.add(
        "FAIL" if status.collision else "PASS",
        "collision",
        "flag set: factor-fr3-power reset" if status.collision else "none",
    )
    if status.has_error:
        r.add("FAIL", "error code", f"main {status.main_code} sub {status.sub_code}: factor-fr3-power reset")
    else:
        r.add("PASS", "error code", "0/0")
    if status.jnt_ang and float(np.abs(status.jnt_ang).max()) < 1e-6:
        r.add("FAIL", "encoders", "all joints read 0: state packet broken")
    r.add(
        "PASS" if status.in_limits else "FAIL",
        "soft limits",
        "inside" if status.in_limits else f"outside {np.round(status.jnt_ang, 1).tolist()}",
    )
    r.add(
        "INFO",
        "mode/state",
        f"{status.mode} robot_state={sdk.ROBOT_STATE.get(status.robot_state, status.robot_state)}",
    )
    if expect_mode and status.mode != expect_mode:
        r.add("FAIL", "mode", f"{status.mode}, need {expect_mode}: factor-fr3-power auto")
    if status.robot_state == 4:
        r.add("FAIL", "drag", "drag mode on: factor-fr3-power drag off")
    if status.enabled == 0:
        r.add("FAIL", "servo enable", "disabled: factor-fr3-power enable")
    elif status.enabled == 1:
        r.add("PASS", "servo enable", "enabled")
    hot = [f"{sdk.JOINTS[i]}={t:.1f}Nm" for i, t in enumerate(status.jnt_tor) if abs(t) > 20]
    if hot:
        r.add("WARN", "joint torque", ", ".join(hot))
    r.add("INFO", "joints (deg)", str(np.round(status.jnt_ang, 2).tolist()))
    if status.gripper_active:
        r.add(
            "INFO", "gripper", f"active, position {status.gripper_position} %, fault {status.gripper_fault}"
        )
    return r


def check_main(argv=None) -> None:
    """``factor-fr3-check``: read-only controller preflight. / 읽기 전용 제어기 사전 점검."""
    ap = argparse.ArgumentParser(description="FR3 preflight (read only); exit 0 = READY")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument("--any-mode", action="store_true", help="do not require AUTO mode")
    ap.add_argument("--no-versions", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=float, default=0.0, help="repeat every N seconds")
    args = ap.parse_args(argv)
    while True:
        status = sdk.read_status(args.ip)
        versions = None if (args.no_versions or not status.ok) else sdk.versions(args.ip)
        report = preflight(status, None if args.any_mode else "AUTO", versions)
        if args.json:
            print(
                json.dumps(
                    {"ready": report.ready, "checks": report.checks, "status": status.to_dict()}, default=str
                )
            )
        else:
            print(time.strftime("%H:%M:%S") + "\n" + report.text(), flush=True)
        if not args.watch:
            sys.exit(0 if report.ready else 1)
        time.sleep(args.watch)


# ----------------------------------------------------------------------------- power / mode


def _show(ip: str) -> sdk.RobotStatus:
    st = sdk.read_status(ip, timeout=1.5)
    if not st.ok:
        print(f"status unavailable: {st.error}")
        return st
    print(
        f"mode {st.mode}  state {sdk.ROBOT_STATE.get(st.robot_state, st.robot_state)}  "
        f"error {st.main_code}/{st.sub_code}  estop {st.estop}  collision {st.collision}  "
        f"safety {st.safety_stop0}/{st.safety_stop1}  motion_done {st.motion_done}\n"
        f"  q(deg) {[round(v, 2) for v in st.jnt_ang]}  flange(mm,deg) {[round(v, 1) for v in st.flange_pos]}"
    )
    return st


def _report(what: str, err: int) -> bool:
    print(f"{what} -> {'OK' if err == 0 else f'error {err}'}")
    return err == 0


def power_main(argv=None) -> None:
    """``factor-fr3-power``: controller power, mode and recovery. / 제어기 전원·모드·복구."""
    ap = argparse.ArgumentParser(description="FR3 controller power/mode helper")
    ap.add_argument(
        "cmd",
        choices=[
            "status",
            "enable",
            "disable",
            "auto",
            "manual",
            "drag",
            "stop",
            "reset",
            "speed",
            "recover",
        ],
    )
    ap.add_argument("arg", nargs="?", default="on", help="drag: on|off; speed: 0..100")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument(
        "--wait-estop", type=float, default=120.0, help="recover: seconds to wait for the e-stop release"
    )
    args = ap.parse_args(argv)

    if args.cmd == "status":
        st = _show(args.ip)
        if st.ok:
            print("versions:", sdk.versions(args.ip))
        sys.exit(0 if st.ok else 1)

    rpc = sdk.connect(args.ip)
    ok = True
    if args.cmd == "enable":
        ok = _report("RobotEnable(1)", rpc.RobotEnable(1))
        time.sleep(1.0)
    elif args.cmd == "disable":
        ok = _report("RobotEnable(0)", rpc.RobotEnable(0))
        time.sleep(1.0)
    elif args.cmd == "auto":
        ok = _report("Mode(0)", rpc.Mode(0))
        time.sleep(0.5)
    elif args.cmd == "manual":
        ok = _report("Mode(1)", rpc.Mode(1))
        time.sleep(0.5)
    elif args.cmd == "drag":
        on = args.arg.lower() in ("on", "1", "true")
        if on:
            # Drag teaching needs MANUAL mode. / 드래그 교시는 수동 모드에서.
            _report("Mode(1)", rpc.Mode(1))
            time.sleep(0.5)
        ok = _report(f"DragTeachSwitch({int(on)})", rpc.DragTeachSwitch(1 if on else 0))
        time.sleep(1.0)
    elif args.cmd == "stop":
        ok = _report("StopMotion", sdk.stop_motion(rpc))
    elif args.cmd == "reset":
        ok = _report("ResetAllError", rpc.ResetAllError())
        time.sleep(1.0)
    elif args.cmd == "speed":
        ok = _report(f"SetSpeed({args.arg})", rpc.SetSpeed(int(float(args.arg))))
    elif args.cmd == "recover":
        ok = _recover(args.ip, rpc, args.wait_estop)
    _show(args.ip)
    sys.exit(0 if ok else 1)


def _recover(ip: str, rpc, wait_estop: float) -> bool:
    """Wait for the e-stop release, then reset, enable, AUTO. / E-stop 해제 대기 후 리셋·enable·AUTO."""
    print("[recover] 1/3 waiting for the e-stop release (turn the button)...", flush=True)
    t0 = time.time()
    while True:
        st = sdk.read_status(ip, timeout=1.5)
        if st.ok and st.safety_ok:
            break
        if not st.ok and "state" in st.error:
            print("[recover] cannot read the state port (driver running?); stop the driver first")
            return False
        if time.time() - t0 > wait_estop:
            print(f"[recover] still stopped: estop={st.estop} SI={st.safety_stop0}/{st.safety_stop1}")
            return False
        time.sleep(1.0)
    print("[recover] 2/3 ResetAllError", flush=True)
    _report("ResetAllError", rpc.ResetAllError())
    time.sleep(1.0)
    print("[recover] 3/3 RobotEnable(1) + Mode(0)", flush=True)
    ok = _report("RobotEnable(1)", rpc.RobotEnable(1))
    time.sleep(1.0)
    ok = _report("Mode(0)", rpc.Mode(0)) and ok
    time.sleep(0.5)
    return ok


# ----------------------------------------------------------------------------- servo probe


def probe_offsets(duration: float = 2.0, amplitude_deg: float = 0.3, rate: float = 50.0) -> np.ndarray:
    """j6 offsets (rad) of the probe: 0 → amplitude → 0, triangular. / 탐색 j6 오프셋(삼각형)."""
    n = max(2, int(round(duration * rate)))
    s = np.linspace(0.0, 1.0, n + 1)
    return np.radians(amplitude_deg) * (1.0 - np.abs(2.0 * s - 1.0))


def parse_param(text: str) -> tuple[str, object]:
    """``name=value`` with numbers as floats. / ``이름=값`` (숫자는 float)."""
    name, _, raw = text.partition("=")
    if not name or not raw:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    try:
        return name.strip(), float(raw)
    except ValueError:
        return name.strip(), raw.strip()


def probe_main(argv=None) -> None:
    """``factor-fr3-probe``: driver reachability, parameters and a servo motion check.

    드라이버 응답, 파라미터 설정, 서보 동작 확인.
    """
    ap = argparse.ArgumentParser(description="probe the FR3 driver's servo stream (j6 ±0.3 deg)")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to wait for /joint_states")
    ap.add_argument("--set", type=parse_param, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--node", default="/fr_ros2_driver", help="driver node for --set")
    ap.add_argument("--no-motion", action="store_true", help="only check /joint_states and set parameters")
    ap.add_argument("--amplitude-deg", type=float, default=0.3)
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    rclpy.init()
    node = Node("factor_fr3_probe")
    latest: dict = {}

    def on_joints(msg) -> None:
        d = dict(zip(msg.name, msg.position, strict=False))
        if all(j in d for j in sdk.JOINTS):
            latest["q"] = np.array([d[j] for j in sdk.JOINTS], dtype=float)

    node.create_subscription(JointState, "/joint_states", on_joints, 10)

    def spin_for(seconds: float) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def call(name: str) -> tuple[bool, str]:
        client = node.create_client(Trigger, name)
        if not client.wait_for_service(timeout_sec=3.0):
            return False, f"{name} unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        return (False, f"{name} timed out") if result is None else (result.success, result.message)

    code = 1
    try:
        t0 = time.time()
        while "q" not in latest and time.time() - t0 < args.wait:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "q" not in latest:
            print(f"no /joint_states within {args.wait:.0f} s (driver not running?)")
            return
        if args.set and not _set_parameters(node, args.node, args.set):
            return
        if args.no_motion:
            print("driver up")
            code = 0
            return

        call("/fr/reset_error")
        spin_for(0.5)
        ok, message = call("/fr/servo_start")
        if not ok:
            print(f"servo start refused: {message}")
            return
        pub = node.create_publisher(JointState, "/fr/servo_target", 10)
        q0 = latest["q"].copy()
        peak = 0.0
        for offset in probe_offsets(amplitude_deg=args.amplitude_deg):
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(sdk.JOINTS)
            q = q0.copy()
            q[5] += offset
            msg.position = [float(v) for v in q]
            pub.publish(msg)
            spin_for(0.02)
            peak = max(peak, abs(math.degrees(latest["q"][5] - q0[5])))
        spin_for(0.5)
        call("/fr/servo_stop")
        moved = peak > 0.1
        print(f"j6 peak excursion {peak:.3f} deg -> {'MOVED' if moved else 'NOT MOVED'}")
        code = 0 if moved else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(code)


def _set_parameters(node, target: str, params: list[tuple[str, object]]) -> bool:
    """Set parameters of another node through its ``set_parameters`` service. / 다른 노드 파라미터 설정."""
    import rclpy
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter

    client = node.create_client(SetParameters, f"{target}/set_parameters")
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"{target}/set_parameters unavailable")
        return False
    request = SetParameters.Request()
    request.parameters = [Parameter(name, value=value).to_parameter_msg() for name, value in params]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    response = future.result()
    if response is None:
        print("set_parameters timed out")
        return False
    ok = True
    for (name, value), result in zip(params, response.results, strict=False):
        if not result.successful:
            print(f"parameter {name}={value} refused: {result.reason}")
            ok = False
    if ok:
        print("parameters set: " + ", ".join(f"{n}={v}" for n, v in params))
    return ok
