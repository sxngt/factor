"""Supervised training of the state estimator on twin frames.

Data come from the twin driven by the teacher (the negated sampled belief error) and,
optionally, by the student itself (DAgger-style collection: the student's estimates steer the
planner while the labels stay exact). Frames are stored as uint8 images plus float labels in
pinned host memory so that hundreds of thousands of frames fit on a workstation.

트윈 프레임으로 상태 추정기를 지도학습한다. 교사(샘플링된 믿음 오차의 역) 또는 학생 자신의 추정으로
플래너를 구동해 수집하고(라벨은 항상 정확), uint8 영상 + 라벨을 pinned 메모리에 저장한다.

Run inside the Isaac Lab Python environment::

    python -m factor.perception.train --assets <usd dir> --num_envs 96 --steps 1000 \
        --out estimator.pt --headless
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from queue import Queue

import numpy as np
import torch

from factor.perception.model import EstimatorConfig, StateEstimator, estimator_loss


@dataclass
class FrameBuffer:
    """Pinned host buffer of frames. / pinned 호스트 프레임 버퍼."""

    capacity: int
    num_envs: int
    resolution: int
    render_channels: int
    n_residual: int = 3

    def __post_init__(self):
        n, r = self.num_envs, self.resolution
        self.rgb = torch.zeros(self.capacity, n, 3, r, r, dtype=torch.uint8).pin_memory()
        self.render = (
            torch.zeros(self.capacity, n, self.render_channels, r, r, dtype=torch.uint8).pin_memory()
            if self.render_channels
            else None
        )
        self.residual = torch.zeros(self.capacity, n, self.n_residual)
        self.state = torch.zeros(self.capacity, n, dtype=torch.long)
        self.offset = torch.zeros(self.capacity, n, 3)
        self.grip = torch.zeros(self.capacity, n, 3)  # believed grip point (proprio) / 믿는 그립점
        self.filled = 0

    def push(self, frames: dict[str, torch.Tensor]) -> None:
        i = self.filled
        self.rgb[i].copy_((frames["rgb"] * 255.0).round().to(torch.uint8), non_blocking=True)
        if self.render is not None:
            self.render[i].copy_((frames["render"] * 255.0).round().to(torch.uint8), non_blocking=True)
        label = frames["correction"]
        if self.n_residual > 3:
            label = torch.cat([label, frames["correction_z"]], 1)  # + height / 높이 보정
        self.residual[i].copy_(label.cpu())
        self.state[i].copy_(frames["state"].cpu())
        self.offset[i].copy_(frames["ring_offset"].cpu())
        self.grip[i].copy_(frames["grip"].cpu())
        self.filled += 1

    @property
    def size(self) -> int:
        return self.filled * self.num_envs

    def batches(self, indices: torch.Tensor, batch: int, device):
        """Yield ``(rgb, render, residual, state)`` batches on ``device`` with a prefetch thread."""
        q: Queue = Queue(maxsize=4)
        n = self.num_envs

        def worker():
            for s in range(0, len(indices), batch):
                idx = indices[s : s + batch]
                t, e = idx // n, idx % n
                rgb = self.rgb[t, e].pin_memory().to(device, non_blocking=True).float() / 255.0
                ren = (
                    None
                    if self.render is None
                    else self.render[t, e].pin_memory().to(device, non_blocking=True).float() / 255.0
                )
                q.put(
                    (
                        rgb,
                        ren,
                        self.residual[t, e].to(device),
                        self.state[t, e].to(device),
                        self.offset[t, e].to(device),
                        self.grip[t, e].to(device),
                    )
                )
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while (item := q.get()) is not None:
            yield item


