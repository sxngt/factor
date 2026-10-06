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


