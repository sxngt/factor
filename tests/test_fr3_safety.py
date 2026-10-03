"""Safety monitor, gripper conversion and preflight report (no SDK). / 안전 감시·그리퍼·사전 점검."""

from types import SimpleNamespace

import numpy as np
import pytest

from factor.robots.fr3 import sdk
from factor.robots.fr3.gripper import half_gap_to_percent, percent_to_half_gap
from factor.robots.fr3.safety import SafetyMonitor
from factor.robots.fr3.tools import parse_param, preflight, probe_offsets


def status(**kw):
    base = dict(
        ok=True,
        ip="test",
        mode="AUTO",
        jnt_ang=[10.0, -60.0, 40.0, -70.0, -90.0, 0.0],
        jnt_tor=[0.0] * 6,
        enabled=1,
        frame_cnt=1,
    )
    base.update(kw)
    return sdk.RobotStatus(**base)


def test_controller_flags_trip_and_stop():
    calls = []
    mon = SafetyMonitor(
        lambda: None, stop=lambda: calls.append("stop"), on_trip=calls.append, log=lambda s: None
    )
    why = mon.check(status(estop=1))
    assert why and "estop=1" in why
    mon.trip(why)
    assert calls == [why, "stop"]
    # Latched: a second trip does nothing. / 래치: 두 번째 트립은 무시.
    mon.trip("again")
    assert mon.tripped == why and len(calls) == 2


def test_error_code_trips():
    mon = SafetyMonitor(lambda: None, log=lambda s: None)
    assert "error=10/2" in mon.check(status(main_code=10, sub_code=2))


def test_tracking_deviation():
    mon = SafetyMonitor(lambda: None, dev_deg=2.0, log=lambda s: None)
    assert mon.check(status(jnt_ang=[0, 0, 0, 0, 0, 5.0])) is None  # no target yet / 목표 없음
    mon.set_target([0, 0, 0, 0, 0, 4.0])
    assert mon.check(status(jnt_ang=[0, 0, 0, 0, 0, 5.0])) is None
    why = mon.check(status(jnt_ang=[0, 0, 0, 0, 0, 1.5]))
    assert why and "j6" in why
    assert mon.max_dev == pytest.approx(2.5)


def test_torque_jump_needs_consecutive_samples():
    mon = SafetyMonitor(lambda: None, warmup=0.0, tor_hold=3, log=lambda s: None)
    t = 1000.0
    mon._t0 = t
    for k in range(10):
        assert mon.check(status(jnt_tor=[1.0] * 6), now=t + 0.02 * k) is None
    t += 0.2
    jump = [20.0, 1, 1, 1, 1, 1]
    assert mon.check(status(jnt_tor=jump), now=t) is None
    assert mon.check(status(jnt_tor=jump), now=t + 0.02) is None
    why = mon.check(status(jnt_tor=jump), now=t + 0.04)
    assert why and "torque jump" in why and "j1" in why


def test_monitor_thread_trips_on_read():
    stops = []
    feed = iter([status(), status(collision=1)] + [status()] * 100)
    mon = SafetyMonitor(lambda: next(feed), stop=lambda: stops.append(1), rate=200.0, log=lambda s: None)
    import time

    with mon:
        t0 = time.time()
        while mon.tripped is None and time.time() - t0 < 2.0:
            time.sleep(0.01)
    assert mon.tripped and "collision=1" in mon.tripped
    assert stops == [1]


def test_status_from_packet_decodes_fields():
    pkg = SimpleNamespace(
        robot_mode=0,
        robot_state=2,
        EmergencyStop=0,
        collisionState=1,
        main_code=0,
        sub_code=0,
        jt_cur_pos=[1, 2, 3, 4, 5, 6],
        jt_cur_tor=[0] * 6,
        gripper_active=1,
        gripper_position=40,
        frame_cnt=7,
    )
    st = sdk.status_from_packet("ip", pkg)
    assert st.mode == "AUTO" and st.robot_state == 2
    assert st.jnt_ang == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert not st.safety_ok and not st.has_error
    assert st.enabled == -1 and st.motion_done == 1


def test_gripper_percent_round_trip_and_clamps():
    pos, width = half_gap_to_percent(0.02)
    assert width == pytest.approx(0.04)
    assert pos == round(0.04 / 0.095 * 100)
    assert percent_to_half_gap(pos) == pytest.approx(0.02, abs=0.0005)
    # Never below the minimum width, never above the stroke. / 최소 폭 아래·스트로크 위 금지.
    assert half_gap_to_percent(0.0)[1] == pytest.approx(0.010)
    assert half_gap_to_percent(1.0) == (100, pytest.approx(0.095))
    # Reversed mounting. / 반대 장착.
    assert half_gap_to_percent(0.0475, open_pct=0)[0] == 0
    assert percent_to_half_gap(0, open_pct=0) == pytest.approx(0.0475)


def test_preflight_ready_and_failures():
    assert preflight(status()).ready
    report = preflight(status(mode="MANUAL", estop=1))
    assert not report.ready
    names = {name for level, name, _ in report.checks if level == "FAIL"}
    assert names == {"e-stop", "mode"}
    assert preflight(status(mode="MANUAL"), expect_mode=None).ready
    assert not preflight(sdk.RobotStatus(ok=False, error="no state")).ready
    report = preflight(status(), versions={"software": ["x", "v0.0.1"], "sdk": "V2"})
    assert report.ready and any(level == "WARN" for level, _, _ in report.checks)


def test_probe_profile_and_param_parsing():
    offsets = probe_offsets(duration=2.0, amplitude_deg=0.3, rate=50.0)
    assert offsets[0] == 0.0 and offsets[-1] == pytest.approx(0.0)
    assert np.degrees(offsets.max()) == pytest.approx(0.3)
    assert parse_param("start_tol_deg=8") == ("start_tol_deg", 8.0)
    assert parse_param("name=abc") == ("name", "abc")
