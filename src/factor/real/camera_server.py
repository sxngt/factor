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


def main(argv=None) -> None:
    """``factor-camera``: own the camera and publish frames. / 카메라를 소유하고 프레임 발행."""
    ap = argparse.ArgumentParser(description="shared-frame camera server (webcam)")
    ap.add_argument(
        "--dir", default=os.environ.get("CAM_DIR", DEFAULT_DIR), help="output directory (env CAM_DIR)"
    )
    ap.add_argument("--device", type=int, default=0, help="/dev/videoN")
    ap.add_argument("--width", type=int, default=2592)
    ap.add_argument("--height", type=int, default=1944)
    ap.add_argument("--fps", type=int, default=30, help="capture rate requested from the device")
    ap.add_argument("--hz", type=float, default=10.0, help="publish rate")
    ap.add_argument("--intrinsics", default="", help="calibration JSON (width, height, K)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until Ctrl-C)")
    ap.add_argument(
        "--alive", action="store_true", help="only report whether a server is publishing (exit 0/1)"
    )
    ap.add_argument(
        "--stall-seconds", type=float, default=20.0, help="reopen the device after identical frames"
    )
    args = ap.parse_args(argv)

    if args.alive:
        sys.exit(0 if server_alive(args.dir) else 1)
    if server_alive(args.dir):
        sys.exit(f"factor-camera: {args.dir}/latest.json is being updated by another server")

    cam = Webcam(args.device, args.width, args.height, args.fps)
    intrinsics = load_intrinsics(args.intrinsics, cam.width, cam.height)
    print(f"factor-camera: {cam.mode} -> {args.dir}", flush=True)

    period = 1.0 / args.hz
    n = 0
    t0 = time.time()
    t_next = t0
    same = 0
    last_signature = None
    try:
        while True:
            rgb = cam.read()
            if time.time() < t_next:
                continue
            t_next = time.time() + period
            # Some webcams freeze and repeat one frame forever: reopen the device.
            # 일부 웹캠은 한 프레임을 반복하며 멈춘다: 장치를 다시 연다.
            signature = rgb[::97, ::89].tobytes()
            same = same + 1 if signature == last_signature else 0
            last_signature = signature
            if same >= int(args.hz * args.stall_seconds):
                print("factor-camera: identical frames, reopening the device", flush=True)
                cam.close()
                time.sleep(1.0)
                cam = Webcam(args.device, args.width, args.height, args.fps)
                same = 0
                continue
            n += 1
            publish_frame(
                rgb,
                n,
                args.dir,
                owner="factor-camera",
                intrinsics=intrinsics,
                backend="webcam",
                mode=cam.mode,
            )
            if n % max(1, int(args.hz) * 30) == 0:
                print(f"frames {n} ({n / (time.time() - t0):.1f} Hz)", flush=True)
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        print("factor-camera: closed", flush=True)


# ----------------------------------------------------------------------------- recorder


def repeats_for_gap(gap_s: float, fps: float, cap: int = 5) -> int:
    """How many times to write a frame so the video keeps wall-clock time. / 실시간 유지를 위한 반복 수.

    The camera publishes slower than the video rate; a frame that arrived ``gap_s`` after the
    previous one is written ``round(gap_s * fps)`` times (at least 1, at most ``cap``).
    """
    if gap_s <= 0.0:
        return 1
    return min(cap, max(1, int(round(gap_s * fps))))


def record_main(argv=None) -> None:
    """``factor-record``: write the shared frames to an mp4. / 공유 프레임을 mp4 로 녹화."""
    import cv2

    ap = argparse.ArgumentParser(description="record the camera server's frames to mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", default=os.environ.get("CAM_DIR", DEFAULT_DIR))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-sec", type=float, default=3600.0)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args(argv)

    stop = {"now": False}

    def request_stop(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    writer = None
    last_n = -1
    written = 0
    t0 = time.time()
    t_last = 0.0
    while not stop["now"] and time.time() - t0 < args.max_sec:
        meta = read_meta(args.dir)
        if meta is None or meta.get("n") == last_n or time.time() - float(meta.get("t", 0)) > 2.0:
            time.sleep(0.02)
            continue
        with open(os.path.join(args.dir, "latest_color.png"), "rb") as f:
            data = f.read()
        if meta.get("color_sha256") and hashlib.sha256(data).hexdigest() != meta["color_sha256"]:
            continue
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        last_n = meta.get("n")
        if args.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            size = (bgr.shape[1], bgr.shape[0])
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
            print(f"recording {args.out} {size[0]}x{size[1]}", flush=True)
        now = time.time()
        for _ in range(repeats_for_gap(now - t_last if t_last else 0.0, args.fps)):
            writer.write(bgr)
            written += 1
        t_last = now
    if writer is not None:
        writer.release()
    print(f"saved {args.out}: {written} frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
