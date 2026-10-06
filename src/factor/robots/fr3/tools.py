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


