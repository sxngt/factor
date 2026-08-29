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


def save_figures(env, out_dir: str, tag: str, model: StateEstimator | None = None, n: int = 16) -> None:
    """Save canonical-view grids for the paper. / 논문용 격자 이미지 저장.

    Writes ``<tag>_rgb.png`` (first ``n`` environments), ``<tag>_mask.png`` (robot-mask channel)
    and, when a model is given, ``<tag>_overlay.png`` with the believed (red) and estimated
    (green) anchor centres drawn on each view.
    """
    from torchvision.utils import make_grid, save_image

    os.makedirs(out_dir, exist_ok=True)
    frames = env.last_frames
    rgb = frames["rgb"][:n]
    save_image(make_grid(rgb, nrow=4, padding=2), os.path.join(out_dir, f"{tag}_rgb.png"))
    save_image(
        make_grid(env.rgb(augment=False)[:n], nrow=4, padding=2), os.path.join(out_dir, f"{tag}_clean.png")
    )
    if frames["render"] is not None:
        save_image(
            make_grid(frames["render"][:n], nrow=4, padding=2), os.path.join(out_dir, f"{tag}_mask.png")
        )
    if model is None:
        return
    with torch.no_grad():
        render = None if frames["render"] is None else frames["render"][:n]
        est = model(frames["rgb"][:n], render, frames["grip"][:n])["residual"]
    s = env.settings
    res = s.resolution
    px_per_m = res / s.view_width
    canvas = rgb.clone()
    centre = res / 2.0
    for i in range(canvas.shape[0]):
        for vec, colour in ((frames["correction"][i], (1.0, 0.2, 0.2)), (est[i], (0.2, 1.0, 0.2))):
            dx = float(vec[0]) * s.pos_scale * px_per_m
            dy = float(vec[1]) * s.pos_scale * px_per_m
            u, v = int(round(centre + dx)), int(round(centre - dy))
            u0, u1, v0, v1 = max(u - 2, 0), min(u + 3, res), max(v - 2, 0), min(v + 3, res)
            canvas[i, :, v0:v1, u0:u1] = torch.tensor(colour, device=canvas.device).view(3, 1, 1)
    save_image(make_grid(canvas, nrow=4, padding=2), os.path.join(out_dir, f"{tag}_overlay.png"))


def student_action(model: StateEstimator, frames: dict[str, torch.Tensor]) -> torch.Tensor:
    """Correction plus grasped probability from the estimator. / 추정기의 보정 + 파지 확률."""
    out = model(frames["rgb"], frames["render"], frames["grip"])
    return action_from_output(out)


def action_from_output(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Twin action ``[corr_xy, corr_yaw, p_grasped, offset_xy, corr_z]`` from estimator outputs."""
    residual = out["residual"].clamp(-1.0, 1.0)
    grasped = torch.softmax(out["state_logits"], dim=1)[:, 2:3]
    z = residual[:, 3:4] if residual.shape[1] > 3 else torch.zeros_like(grasped)
    return torch.cat([residual[:, :3], grasped, out["offset"][:, :2].clamp(-1.0, 1.0), z], 1)


def collect(
    env, buffer: FrameBuffer, steps: int, model: StateEstimator | None, beta: float, noise: float
) -> dict:
    """Run ``steps`` twin steps and push frames. / ``steps`` 스텝 수집.

    With ``model`` given, each environment follows the student's estimate with probability
    ``1 - beta`` and the teacher otherwise. ``noise`` perturbs the executed correction to widen
    the state distribution; labels are unaffected.
    """
    counts = {"succeeded": 0, "dropped": 0, "bad_release": 0, "timeout": 0}
    frames = env.last_frames
    for _ in range(steps):
        teacher = env.teacher_action()
        if model is None:
            action = teacher.clone()
        else:
            with torch.no_grad():
                est = student_action(model, frames)
            use_teacher = torch.rand(env.num_envs, device=env.device) < beta
            action = torch.where(use_teacher.unsqueeze(1), teacher, est)
        if noise > 0:
            action[:, :3] = action[:, :3] + noise * torch.randn_like(action[:, :3])
        buffer.push(frames)
        _, _, terminated, truncated, _ = env.step(action.clamp(-1.0, 1.0))
        frames = env.last_frames
        reason = env.last_term_reason
        seen = torch.zeros_like(terminated, dtype=torch.bool)
        for key in ("succeeded", "dropped", "bad_release"):
            m = reason[key] & ~seen
            counts[key] += int(m.sum())
            seen |= reason[key]
        counts["timeout"] += int((truncated & ~seen).sum())
    return counts


