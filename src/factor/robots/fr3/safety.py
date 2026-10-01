"""Safety monitor for streamed FR3 motion. / 스트리밍 FR3 모션의 안전 감시.

A background thread reads the controller state (20 Hz) and trips on:

* controller flags: emergency stop, collision, safety stops SI0/SI1, non-zero error code;
* tracking deviation: ``max |target - measured| > dev_deg`` while a target is set (the arm was
  pushed, blocked, or is not following);
* torque jumps: ``|tau - median(last tor_win s)| > tor_abs`` per joint for ``tor_hold`` samples
  in a row (impacts; gravity torque changes slowly and is not flagged).

A trip calls ``on_trip`` (e.g. end the servo stream), then ``stop`` (``StopMotion``). It latches:
a tripped monitor never re-arms, a person has to look first.

백그라운드 스레드가 제어기 상태를 20 Hz 로 읽고 제어기 플래그(비상정지·충돌·안전정지·에러),
추종 편차, 토크 급변 중 하나라도 걸리면 ``on_trip`` → ``stop`` 순으로 멈춘다. 한 번 걸리면 다시
켜지지 않는다(사람이 확인 후 재실행).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
DEFAULT_TOR_ABS = (8.0, 8.0, 6.0, 3.0, 3.0, 3.0)  # Nm per joint / 관절별 허용 토크 변화


