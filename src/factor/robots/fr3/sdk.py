"""FAIRINO FR3 controller access through the vendor SDK. / 제조사 SDK 를 통한 FR3 제어기 접근.

The vendor SDK is the ``fairino`` Python package (``from fairino import Robot``). It is **not on
PyPI**: install it from FAIRINO's ``fairino-python-sdk`` repository at the tag that matches the
controller firmware (tested: SDK v2.1.6 with firmware v3.8.6). It is imported lazily, so the rest
of FACTOR (and the ``--fake`` driver) works without it.

The controller exposes two ports: XML-RPC commands (20003) and a real-time state stream (20004).
The state stream is served to **one client only**. The process that holds it (normally the
driver) reads joints and flags from the packet; any other process gets an XML-RPC-only handle
(:func:`connect` silences the SDK's reconnect loop so commands such as ``StopMotion`` and
``ResetAllError`` still go through).

제조사 SDK 는 ``fairino`` 파이썬 패키지이며 PyPI 에 없다: 제어기 펌웨어에 맞는 태그의
``fairino-python-sdk`` 를 설치한다(검증: SDK v2.1.6 + 펌웨어 v3.8.6). 지연 import 하므로 나머지
FACTOR 와 ``--fake`` 드라이버는 SDK 없이 동작한다. 실시간 상태 포트(20004)는 한 클라이언트만
받으므로, 다른 프로세스는 XML-RPC 전용 핸들을 쓴다.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_IP = os.environ.get("ROBOT_IP", "192.168.58.2")  # controller factory default / 제어기 공장 기본값
JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
TESTED_FIRMWARE = "v3.8.6"

ROBOT_STATE = {1: "Stopped", 2: "Running", 3: "Paused", 4: "Drag"}
ROBOT_MODE = {0: "AUTO", 1: "MANUAL"}
PROGRAM_STATE = {1: "stopped", 2: "running", 3: "paused"}
# Controller soft limits (deg), from GetJointSoftLimitDeg. / 제어기 소프트 리밋(deg).
SOFT_LIMITS_DEG = ((-175, 175), (-265, 85), (-150, 150), (-265, 85), (-175, 175), (-175, 175))

_HANDLES: dict[str, object] = {}
_HANDLES_LOCK = threading.Lock()


def import_sdk():
    """Import the vendor ``Robot`` module or explain where it comes from. / SDK import 또는 안내."""
    try:
        from fairino import Robot
    except ImportError as exc:
        raise ImportError(
            "the FAIRINO SDK ('fairino' package) is not installed; it is not on PyPI. "
            "Install fairino-python-sdk from FAIRINO at the tag matching the controller firmware, "
            "or run the driver with --fake."
        ) from exc
    return Robot


def has_state(rpc) -> bool:
    """Whether this handle receives real-time state packets. / 실시간 상태 패킷 수신 여부."""
    pkg = rpc.robot_state_pkg
    return not isinstance(pkg, type) and int(pkg.frame_head) == 0x5A5A


def silence_state_thread(rpc) -> None:
    """Make a handle XML-RPC only. / 핸들을 XML-RPC 전용으로 만든다.

    Without the state port the SDK loops on reconnect, and every XML-RPC call (``StopMotion``
    included) blocks while that loop runs. Pretend the reconnect succeeded and end the thread.
    상태 포트 없이는 SDK 가 재접속 루프를 돌며 모든 XML-RPC 호출을 막는다. 재접속을 성공으로 위장하고
    스레드를 끝낸다.
    """
    rpc.closeRPC_state = True
    rpc.robot_realstate_exit = True
    with contextlib.suppress(AttributeError):
        rpc.stop_event.set()
    rpc.connect_to_robot = lambda: True
    rpc.reconnect_flag = False


def connect(ip: str = DEFAULT_IP, wait_state: float = 1.5, require_state: bool = False):
    """SDK handle for ``ip``, cached per process. / 프로세스당 IP 별로 캐시되는 SDK 핸들.

    Waits up to ``wait_state`` seconds for state packets; if none arrive (another process holds
    the state port) the handle becomes XML-RPC only. ``require_state`` turns that case into a
    ``ConnectionError``.
    상태 패킷을 ``wait_state`` 초 기다리고, 오지 않으면 XML-RPC 전용 핸들이 된다.
    """
    with _HANDLES_LOCK:
        rpc = _HANDLES.get(ip)
        if rpc is None:
            robot = import_sdk()
            # The SDK prints connection chatter on construction. / 생성 시 SDK 로그를 숨긴다.
            with contextlib.redirect_stdout(io.StringIO()):
                rpc = robot.RPC(ip)
            if not robot.RPC.is_conect:
                robot.RPC.is_conect = True  # reset the SDK's global flag for the next try / 전역 플래그 복구
                raise ConnectionError(f"{ip}: XML-RPC (port 20003) connection failed")
            t0 = time.time()
            while time.time() - t0 < wait_state and not has_state(rpc):
                time.sleep(0.05)
            if not has_state(rpc):
                silence_state_thread(rpc)
            _HANDLES[ip] = rpc
    if require_state and not has_state(rpc):
        raise ConnectionError(f"{ip}: no real-time state (port 20004 is held by another process)")
    return rpc


