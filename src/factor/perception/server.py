"""Estimator inference server over a pipe.

The real-robot loop usually runs in a ROS Python that has no PyTorch, so inference lives in a
separate process (the simulation Python, which has torch) and talks over stdin/stdout:

- request: a header line ``OBS <res> <channels> [gx gy gz]\\n`` (optional believed grip point,
  metres, anchor frame) followed by ``res*res*channels`` raw uint8
  bytes (RGB first, then the optional render channels), HWC order;
- reply: ``EST cx cy dtheta p_grasped ox oy oz spread sigma [cz]\\n`` in metres and radians;
- ``PING`` / ``QUIT`` for liveness and shutdown; ``READY <res> <channels> <view_width> <label>`` on start-up.

실기 루프(ROS 파이썬, torch 없음)와 분리된 추론 프로세스. 표준 입출력으로 영상을 받고 보정량·파지
확률·오프셋을 돌려준다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from factor.perception.model import EstimatorConfig, StateEstimator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", required=True, help="one checkpoint, or several separated by commas (ensemble)"
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # Several checkpoints form an ensemble: their estimates are averaged and their spread reported.
    # 여러 체크포인트는 앙상블로 평균하고, 서로의 퍼짐을 함께 보고한다.
    models = []
    for path in args.checkpoint.split(","):
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = EstimatorConfig(**ck["config"])
        m = StateEstimator(cfg).to(device)
        m.load_state_dict(ck["state_dict"], strict=False)
        m.eval()
        models.append(m)
    config = models[0].config
    for m in models[1:]:
        c = m.config
        if (c.resolution, c.render_channels, c.view_width, c.proprio_dim, c.label) != (
            config.resolution,
            config.render_channels,
            config.view_width,
            config.proprio_dim,
            config.label,
        ):
            raise SystemExit(
                "ensemble members must share resolution, channels, view width, proprio input and label"
            )

    res = config.resolution
    channels = config.in_channels + config.render_channels
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    out.write(
        f"READY {res} {channels} {config.view_width} {config.label} {int(config.belief_fingers)}\n".encode()
    )
    out.flush()

    while True:
        line = inp.readline()
        if not line or line.startswith(b"QUIT"):
            break
        if line.startswith(b"PING"):
            out.write(b"PONG\n")
            out.flush()
            continue
        if not line.startswith(b"OBS"):
            continue
        parts = line.split()
        r, c = int(parts[1]), int(parts[2])
        proprio = [float(v) for v in parts[3:]]  # believed grip point, metres (optional)
        n = r * r * c
        buf = inp.read(n)
        while len(buf) < n:
            more = inp.read(n - len(buf))
            if not more:
                return
            buf += more
        image = np.frombuffer(buf, np.uint8).reshape(r, r, c).copy()
        with torch.no_grad():
            x = torch.from_numpy(image).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            rgb, render = x[:, :3], (x[:, 3:] if c > 3 else None)
            p = torch.tensor([proprio], device=device) if proprio else None
            # The robot side works in metres and radians; the network in normalised units.
            # 실기 쪽은 미터·라디안, 망은 정규화 단위.
            ests, grasps, offsets, sigmas = [], [], [], []
            for m in models:
                o = m(rgb, render, p)
                c = m.config
                scale = torch.tensor(
                    [c.pos_scale, c.pos_scale, c.yaw_scale, c.pos_scale][: c.n_residual], device=device
                )
                ests.append(o["residual"][0].clamp(-1.0, 1.0) * scale)
                grasps.append(torch.softmax(o["state_logits"], dim=1)[0, 2])
                offsets.append(o["offset"][0].clamp(-1.0, 1.0) * c.offset_scale)
                # The model's own translation uncertainty (heteroscedastic head), metres.
                sigmas.append(float((0.5 * o["logvar"][0, :2]).exp().norm()) * c.pos_scale)
            stacked = torch.stack(ests)
            est = stacked.mean(0)
            spread = float(stacked[:, :2].std(0).norm()) if len(models) > 1 else 0.0  # metres
            grasped = torch.stack(grasps).mean()
            offset = torch.stack(offsets).mean(0)
        sigma = float(np.mean(sigmas))
        values = [*est[:3].tolist(), float(grasped), *offset.tolist(), spread, sigma, *est[3:].tolist()]
        out.write(("EST " + " ".join(f"{v:.5f}" for v in values) + "\n").encode())
        out.flush()


if __name__ == "__main__":
    main()
