"""``factor-selfcalib``: one-minute active self-calibration on the robot.

The robot shows the gripper board to the fixed camera from poses it picks itself. After every
pose the camera-in-base and board-on-flange estimate is re-solved, and the next pose is the one
with the largest predicted drop in peg-point uncertainty per second of motion. It stops when the
peg points are known to ``--target-mm`` or the budget runs out, then writes the anchor pose in the
robot base frame (``X · T_CA``) for ``factor-run --calib``.

로봇이 스스로 고른 자세들로 그리퍼 보드를 고정 카메라에 보여 준다. 자세마다 카메라-베이스·보드-플랜지
추정을 다시 풀고, "이동 1초당 기둥 점 불확실성 감소"가 가장 큰 자세로 간다. 기둥 점이 ``--target-mm``
안으로 들어오거나 시간 예산이 끝나면 멈추고, 베이스 기준 앵커 포즈(X·T_CA)를 ``factor-run --calib``
용으로 저장한다.

    factor-selfcalib --scene configs/hanoi.yaml --budget 60 --out logs/factor/selfcalib/latest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from factor.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    average_poses,
    choose_next,
    move_time,
    pose_error,
    project,
)
from factor.calib.poses import generate, near_within_limits, path_clear, search_poses
from factor.calib.simulate import peg_points


# ----------------------------------------------------------------------------- camera
def read_frame(cam_dir: str, newer_than: float = 0.0, timeout: float = 3.0):
    """Latest frame written after ``newer_than`` (camera-server clock): (rgb, t, n).

    카메라 서버가 ``newer_than`` 이후에 쓴 최신 프레임.
    """
    import cv2

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(os.path.join(cam_dir, "latest.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if float(meta.get("t", 0)) <= newer_than:
                time.sleep(0.03)
                continue
            with open(os.path.join(cam_dir, "latest_color.png"), "rb") as f:
                data = f.read()
            if meta.get("color_sha256") and hashlib.sha256(data).hexdigest() != meta["color_sha256"]:
                time.sleep(0.02)
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), float(meta["t"]), int(meta.get("n", 0))
        except (OSError, ValueError, KeyError):
            time.sleep(0.02)
    raise RuntimeError("no fresh camera frame")


class CameraSensor:
    """Frames from the camera server, both boards detected per frame. / 카메라 서버 프레임 + 두 보드 검출."""

    def __init__(self, cam_dir: str, board_det, mat_det):
        self.cam_dir, self.board_det, self.mat_det = cam_dir, board_det, mat_det

    def grab(self, newer_than: float = 0.0):
        rgb, t, _n = read_frame(self.cam_dir, newer_than)
        self.last_rgb = rgb
        board = self.board_det.detect(rgb) if self.board_det is not None else None
        return t, board, self.mat_det.detect(rgb)

    def leds(self, roi=None) -> np.ndarray:
        from factor.calib.ring import detect_leds

        return detect_leds(self.last_rgb, roi)


