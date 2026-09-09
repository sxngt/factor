"""Client for :mod:`factor.perception.server` (NumPy only). / 추론 서버 클라이언트 (NumPy 만 사용).

The client launches the server in another Python interpreter and exchanges frames over pipes.
It is the only perception entry point the real-robot loop needs.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """One estimator reply. / 추론 결과 하나."""

    correction: np.ndarray  # (dx, dy, dyaw) in metres and radians / 미터·라디안
    grasped: float
    offset: np.ndarray
    spread: float = 0.0  # ensemble disagreement of the correction (m) / 앙상블 불일치
    sigma: float = 0.0  # the model's own translation uncertainty (m) / 모델 자체 불확실성


class EstimatorClient:
    """Spawn and talk to the inference server. / 추론 서버를 띄우고 대화한다.

    Parameters
    ----------
    python
        Interpreter that has PyTorch (for example the simulation environment's Python).
    checkpoint
        Estimator checkpoint path.
    """

    def __init__(self, python: str, checkpoint: str, device: str = "cuda"):
        self.proc = subprocess.Popen(
            [python, "-m", "factor.perception.server", "--checkpoint", checkpoint, "--device", device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        ready = self.proc.stdout.readline().decode().split()
        if not ready or ready[0] != "READY":
            raise RuntimeError("estimator server failed to start")
        self.resolution, self.channels = int(ready[1]), int(ready[2])
        self.view_width = float(ready[3]) if len(ready) > 3 else 0.14  # canonical view width (m)
        self.label = ready[4] if len(ready) > 4 else "global"  # correction convention / 보정 규약
        self.belief_fingers = len(ready) > 5 and ready[5] == "1"  # belief render style / 렌더 방식

    def estimate(self, image: np.ndarray, grip_in_anchor: np.ndarray | None = None) -> Estimate:
        """Send an ``res x res x channels`` uint8 image and read the estimate. / 영상 전송·추정 수신.

        ``grip_in_anchor`` is the grip point the robot *believes* it has, in the (believed) anchor
        frame; estimators trained with proprioception need it. / 믿는 그립점(앵커 프레임, m).
        """
        image = np.ascontiguousarray(image, dtype=np.uint8)
        assert image.shape == (self.resolution, self.resolution, self.channels), image.shape
        extra = "" if grip_in_anchor is None else " " + " ".join(f"{float(v):.6f}" for v in grip_in_anchor)
        self.proc.stdin.write(f"OBS {self.resolution} {self.channels}{extra}\n".encode() + image.tobytes())
        self.proc.stdin.flush()
        values = [float(v) for v in self.proc.stdout.readline().split()[1:]]
        return Estimate(
            correction=np.array(values[:3] + values[9:10]),  # (dx, dy, dyaw[, dz])
            grasped=values[3],
            offset=np.array(values[4:7]),
            spread=values[7] if len(values) > 7 else 0.0,
            sigma=values[8] if len(values) > 8 else 0.0,
        )

    def close(self) -> None:
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
