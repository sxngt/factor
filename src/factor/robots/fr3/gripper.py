"""DH-Robotics AG-95 gripper behind the FR3 controller, as driven by the driver.

The AG-95 hangs on the tool-flange RS-485 bus and the controller drives it (configure the
controller's gripper as DH, device 0, once in its web UI). Commands are a percentage of the
95 mm stroke; the driver takes the finger joint value, i.e. the half gap in metres.

Safety: force and speed are capped, and the width never goes below ``min_width`` (no full
close). After a controller power cycle the gripper must be activated (``ActGripper``), which
sweeps the fingers once over the full stroke.

FR3 제어기 뒤의 AG-95 그리퍼(툴 플랜지 RS-485). 명령은 95 mm 스트로크의 %, 드라이버 입력은 핑거 관절값
(반개구, m). 힘·속도 상한, 최소 폭(전폐 금지). 제어기 전원 재투입 후에는 활성화가 필요하며 핑거가 한 번
전체 스트로크를 움직인다.
"""

from __future__ import annotations

import time
from collections.abc import Callable

STROKE_M = 0.095
MIN_WIDTH_M = 0.010
MAX_FORCE_PCT = 30
MAX_VEL_PCT = 50


def half_gap_to_percent(
    half_gap: float, stroke: float = STROKE_M, open_pct: int = 100, min_width: float = MIN_WIDTH_M
) -> tuple[int, float]:
    """Controller position (%) and the clamped width (m) for a half gap. / 반개구 → 위치 %, 폭.

    ``open_pct`` is the percentage the controller calls fully open (100 on the reference setup;
    0 if the gripper is mounted the other way round).
    ``open_pct`` 는 제어기가 완전 개방이라 부르는 % (기준 셋업 100).
    """
    width = min(stroke, max(min_width, 2.0 * float(half_gap)))
    pct = width / stroke * 100.0
    pos = int(round(pct if open_pct == 100 else 100.0 - pct))
    return pos, width


def percent_to_half_gap(pos: float, stroke: float = STROKE_M, open_pct: int = 100) -> float:
    """Half gap (m) for a controller position (%). / 위치 % → 반개구(m)."""
    pct = max(0.0, min(100.0, float(pos)))
    if open_pct != 100:
        pct = 100.0 - pct
    return stroke * pct / 100.0 / 2.0


class ControllerGripper:
    """AG-95 through the controller of an :class:`factor.robots.fr3.sdk.SdkArm`. / 제어기 경유 AG-95."""

    def __init__(
        self,
        arm,
        index: int = 1,
        vel: int = 30,
        force: int = 30,
        open_pct: int = 100,
        stroke: float = STROKE_M,
        min_width: float = MIN_WIDTH_M,
        log: Callable[[str], None] = print,
    ):
        self.arm = arm
        self.index = index
        self.vel = min(int(vel), MAX_VEL_PCT)
        self.force = max(20, min(int(force), MAX_FORCE_PCT))
        self.open_pct = open_pct
        self.stroke = stroke
        self.min_width = min_width
        self.log = log
        self._last = stroke / 2.0

    def ensure_active(self, timeout: float = 10.0) -> None:
        """Activate after a power cycle or a fault (fingers sweep once). / 비활성·고장 시 재활성화."""
        st = self.arm.status()
        if st.gripper_active == 1 and st.gripper_fault == 0:
            return
        self.log(f"gripper inactive (active {st.gripper_active}, fault {st.gripper_fault}): activating")
        if st.gripper_fault:
            self.arm.call("ActGripper", self.index, 0)
            time.sleep(0.5)
        self.arm.call("ActGripper", self.index, 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.5)
            st = self.arm.status()
            if st.gripper_active == 1 and st.gripper_fault == 0:
                return
        raise RuntimeError("gripper activation failed (controller gripper config DH/device 0, wiring, 24 V)")

    def move(self, half_gap: float, timeout: float = 5.0) -> float:
        """Move to ``half_gap``; return the measured width (m). / 이동 후 측정 폭 반환.

        Waits until the position is within 2 % of the goal, stops changing (an object is held),
        or ``timeout`` passes. The controller's own blocking flag returns immediately.
        목표 ±2 % 도달, 위치 정지(물체 파지), 시간 초과 중 하나까지 기다린다.
        """
        pos, width = half_gap_to_percent(half_gap, self.stroke, self.open_pct, self.min_width)
        if 2.0 * half_gap < self.min_width:
            self.log(f"gripper width {2000 * half_gap:.1f} mm clamped to {self.min_width * 1000:.0f} mm")
        self.ensure_active()
        self._last = width / 2.0
        err = self.arm.call(
            "MoveGripper", self.index, pos, self.vel, self.force, int(timeout * 1000), 0, 0, 0.0, 0, 0
        )
        if err != 0:
            raise RuntimeError(f"MoveGripper error {err}")

        t0 = time.time()
        last_pos = None
        last_change = t0
        while time.time() - t0 < timeout:
            time.sleep(0.05)
            now_pos = self.arm.status().gripper_position
            if abs(now_pos - pos) <= 2:
                break
            if now_pos != last_pos:
                last_pos = now_pos
                last_change = time.time()
            elif time.time() - t0 > 0.3 and time.time() - last_change > 0.5:
                break
        return 2.0 * percent_to_half_gap(self.arm.status().gripper_position, self.stroke, self.open_pct)

    def half_gap(self) -> float:
        """Measured half gap if the gripper is active, else the last command. / 측정 또는 마지막 명령."""
        st = self.arm.status()
        if st.gripper_active:
            return percent_to_half_gap(st.gripper_position, self.stroke, self.open_pct)
        return self._last


class FakeGripper:
    """Gripper stand-in for ``--fake``. / ``--fake`` 용 가짜 그리퍼."""

    def __init__(self, stroke: float = STROKE_M, min_width: float = MIN_WIDTH_M):
        self.stroke = stroke
        self.min_width = min_width
        self._half_gap = stroke / 2.0

    def move(self, half_gap: float, timeout: float = 5.0) -> float:
        _, width = half_gap_to_percent(half_gap, self.stroke, 100, self.min_width)
        time.sleep(0.3)
        self._half_gap = width / 2.0
        return width

    def half_gap(self) -> float:
        return self._half_gap
