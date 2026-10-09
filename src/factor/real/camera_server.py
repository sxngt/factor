"""Shared-frame camera server and recorder (``factor-camera``, ``factor-record``).

One process owns the camera and writes the newest frame to a directory; every other process
(``factor-run``, ``factor-selfcalib``, the recorder) reads the files. Files, replaced
atomically, PNG first and metadata last:

* ``latest_color.png``: RGB frame;
* ``latest.json``: ``t`` (epoch s), ``n`` (frame number), ``w``, ``h``, ``color_sha256`` (of the
  PNG bytes, so a reader that lands between the two replacements detects the mismatch and
  retries), ``intrinsics`` (``[fx, fy, cx, cy]`` or ``null``), ``backend``, ``mode``, ``owner``.

The backend is a plain UVC webcam through OpenCV (V4L2 + MJPG on Linux). Intrinsics come from
an optional JSON file (``{"width", "height", "K"}`` or ``{"width", "height", "fx", "fy", "cx",
"cy"}``) and are only reported when its resolution matches the camera's.

카메라는 한 프로세스만 열고 최신 프레임을 디렉터리에 원자적으로 쓴다(PNG 먼저, 메타 나중). 메타의
``color_sha256`` 으로 판독기가 세대 불일치를 검출한다. 백엔드는 OpenCV 일반 웹캠이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time

import numpy as np

DEFAULT_DIR = "logs/cam"


# ----------------------------------------------------------------------------- shared files


def publish_frame(
    rgb: np.ndarray, number: int, cam_dir: str = DEFAULT_DIR, owner: str = "server", **meta
) -> dict:
    """Write one frame atomically; returns the metadata written. / 프레임 하나를 원자적으로 기록."""
    import cv2

    os.makedirs(cam_dir, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = buf.tobytes()
    _atomic_write(os.path.join(cam_dir, "latest_color.png"), data)

    record = {
        "t": time.time(),
        "n": int(number),
        "w": int(rgb.shape[1]),
        "h": int(rgb.shape[0]),
        "owner": owner,
        "color_sha256": hashlib.sha256(data).hexdigest(),
        "intrinsics": None,
        "backend": "",
        "mode": "",
    }
    record.update(meta)
    _atomic_write(os.path.join(cam_dir, "latest.json"), json.dumps(record).encode("utf-8"))
    return record


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def read_meta(cam_dir: str = DEFAULT_DIR) -> dict | None:
    """``latest.json`` or ``None``. / 메타 파일 또는 None."""
    try:
        with open(os.path.join(cam_dir, "latest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def server_alive(cam_dir: str = DEFAULT_DIR, max_age: float = 2.0) -> bool:
    """Whether a server refreshed ``latest.json`` within ``max_age`` s. / 최근 갱신 여부."""
    meta = read_meta(cam_dir)
    return meta is not None and time.time() - float(meta.get("t", 0.0)) < max_age


def load_intrinsics(path: str, width: int, height: int) -> list[float] | None:
    """``[fx, fy, cx, cy]`` from a calibration JSON if its resolution matches.

    보정 JSON 의 해상도가 카메라와 같을 때만 내부 파라미터를 돌려준다.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        calib = json.load(f)
    if (int(calib["width"]), int(calib["height"])) != (int(width), int(height)):
        size = f"{calib['width']}x{calib['height']}"
        print(f"[camera] {path}: {size} != camera {width}x{height}; intrinsics not used", flush=True)
        return None
    if "K" in calib:
        k = calib["K"]
        return [float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])]
    return [float(calib[key]) for key in ("fx", "fy", "cx", "cy")]


# ----------------------------------------------------------------------------- webcam


class Webcam:
    """UVC webcam through OpenCV. / OpenCV 일반 웹캠.

    Resolution and rate are requests; the device may fall back to what it supports, which is
    reported (and used) as the actual mode.
    해상도·속도는 요청값이며 장치가 지원하는 값으로 떨어질 수 있다(실제 값을 보고).
    """

    def __init__(self, device: int = 0, width: int = 2592, height: int = 1944, fps: int = 30):
        import cv2

        self.cv2 = cv2
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {device} did not open (check the device number and other users)")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if (self.width, self.height) != (width, height):
            print(
                f"[camera] requested {width}x{height}@{fps}, got {self.width}x{self.height}@{self.fps:.0f}",
                flush=True,
            )
        self.mode = f"webcam {self.width}x{self.height}@{self.fps:.0f}"

    def read(self) -> np.ndarray:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise RuntimeError("webcam frame grab failed")
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


