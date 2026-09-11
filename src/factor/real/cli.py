"""``factor-run``: execute Tower of Hanoi moves on the robot with the learned estimator.

Wires the scene specification, the FR3 ROS 2 adapter, the SDK gripper, a shared-frame camera
source, the estimator server and the closed-loop skill. Runs in the ROS Python; the estimator
runs in the interpreter given by ``--torch-python``.

실기 실행 명령. 장면 명세·FR3 ROS 2 어댑터·SDK 그리퍼·공유 프레임 카메라·추정 서버·폐루프 스킬을
엮는다. ROS 파이썬에서 실행하고 추정기는 ``--torch-python`` 인터프리터에서 돈다.

Example::

    factor-run --scene configs/hanoi.yaml --checkpoint estimator.pt \\
        --stacks "left:ring0,ring1,ring2,ring3,ring4,ring5;middle:;right:" --goal right --dry
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from factor.perception.canonical import VirtualCamera
from factor.perception.client import EstimatorClient
from factor.real.run import LoopConfig, MoveSpec, RealLoop
from factor.real.safety import WorkspaceBox
from factor.scene import load_scene
from factor.tasks.hanoi import HanoiLayout, HanoiPlanner


def shared_frame_source(cam_dir: str, max_age: float = 1.5, retries: int = 10):
    """Frame source reading the latest PNG written by a camera server. / 카메라 서버의 최신 PNG.

    The server writes ``latest_color.png`` and ``latest.json`` (with the PNG's SHA-256), so a
    read that lands mid-write is detected by the checksum and retried.
    서버가 PNG 와 그 SHA-256 을 함께 쓰므로, 쓰는 도중 읽은 프레임은 체크섬으로 걸러 다시 읽는다.
    """
    import hashlib
    import io

    from PIL import Image

    def latest() -> np.ndarray:
        for _ in range(retries):
            with open(os.path.join(cam_dir, "latest.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if time.time() - float(meta.get("t", 0)) > max_age:
                raise RuntimeError("camera frame is stale")
            with open(os.path.join(cam_dir, "latest_color.png"), "rb") as f:
                data = f.read()
            expected = meta.get("color_sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                time.sleep(0.02)
                continue
            try:
                return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("could not read a consistent camera frame")

    return latest


def parse_stacks(text: str) -> dict[str, list[str]]:
    stacks = {"left": [], "middle": [], "right": []}
    for part in text.split(";"):
        if ":" not in part:
            continue
        peg, names = part.split(":", 1)
        stacks[peg.strip()] = [n.strip() for n in names.split(",") if n.strip()]
    return stacks


def anchor_to_base(anchor_in_base: np.ndarray, p: np.ndarray) -> np.ndarray:
    return anchor_in_base[:3, :3] @ np.asarray(p, dtype=float) + anchor_in_base[:3, 3]


