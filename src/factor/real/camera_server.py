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


