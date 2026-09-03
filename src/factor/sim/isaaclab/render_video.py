"""Render an overlay video from the twin: what the estimator believes while the tool occludes.

For a few environments the canonical view is upscaled and overlaid with the estimated ring and
goal outlines (green), the simulator's true ring (blue), the peg axes and the interaction state.
The frames of all shown environments are tiled into one video.

트윈 롤아웃을 영상으로: 그리퍼가 물체를 가리는 동안 모델이 믿는 링·목표의 3D 윤곽(초록)과 참값(파랑),
기둥 축, 상호작용 상태를 정규화 시야 위에 그려 타일로 붙인다.

Usage (Isaac Lab Python)::

    python -m factor.sim.isaaclab.render_video --scene configs/hanoi.yaml --assets <repo> \\
        --checkpoint estimator.pt --out overlay.mp4 --envs 4 --steps 240 --headless
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="configs/hanoi.yaml")
parser.add_argument("--assets", required=True)
parser.add_argument("--checkpoint", default="", help="estimator checkpoint; empty = teacher corrections")
parser.add_argument("--out", required=True)
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--anchor_err", type=float, default=0.008)
parser.add_argument("--scale", type=int, default=4, help="upscale factor of each tile")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument(
    "--render", choices=("belief", "mask", "both", "none"), default="belief", help="render channels"
)
parser.add_argument("--view_width", type=float, default=0.14, help="metres across the canonical view")
parser.add_argument("--no_mask", action="store_true")
parser.add_argument("--offset_scale", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from factor.perception.canonical import VirtualCamera  # noqa: E402
from factor.perception.model import EstimatorConfig, StateEstimator  # noqa: E402
from factor.perception.overlay import (  # noqa: E402
    COLOUR_ESTIMATE,
    COLOUR_GOAL,
    COLOUR_TRUTH,
    STATE_NAMES,
    canonical_projector,
    draw_peg,
    draw_ring,
    label,
)
from factor.scene import load_scene  # noqa: E402
from factor.sim.contract import Randomisation  # noqa: E402
from factor.sim.isaaclab.twin import HanoiTwin, make_env_cfg, settings_from_scene  # noqa: E402
from factor.tasks.hanoi.spec import PEGS  # noqa: E402

settings = settings_from_scene(
    load_scene(args.scene),
    args.assets,
    randomisation=Randomisation(anchor_xy=args.anchor_err),
    view_width=args.view_width,
    render_mask=(args.render in ("mask", "both")) and not args.no_mask,
    render_belief=(args.render in ("belief", "both")) and not args.no_mask,
    error_resample_steps=0,
    start_mix=(1.0, 0.0, 0.0),
)
env_cfg = make_env_cfg(settings, args.envs, args.device)
env_cfg.offset_scale = args.offset_scale
env = HanoiTwin(env_cfg)
env.reset()

model = None
if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    model = StateEstimator(EstimatorConfig(**ck["config"])).to(env.device)
    model.load_state_dict(ck["state_dict"], strict=False)  # older checkpoints lack newer heads
    model.eval()
    env.cfg.offset_scale = model.config.offset_scale  # the planner must use the model's scale

lay = settings.layout
virtual = VirtualCamera(
    settings.resolution, settings.view_width, float(np.linalg.norm(settings.camera_in_anchor))
)
tile = settings.resolution * args.scale
cols = int(np.ceil(np.sqrt(args.envs)))
rows = int(np.ceil(args.envs / cols))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cols * tile, rows * tile))

scaled = VirtualCamera(tile, settings.view_width, virtual.distance)
for _ in range(args.steps):
    f = env.last_frames
    if model is None:
        action = env.teacher_action()
    else:
        with torch.no_grad():
            out = model(f["rgb"], f["render"], f["grip"])
        from factor.perception.train import action_from_output

        action = action_from_output(out)
    rgb = (env.rgb(augment=False).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for e in range(args.envs):
        img = cv2.resize(
            cv2.cvtColor(rgb[e], cv2.COLOR_RGB2BGR), (tile, tile), interpolation=cv2.INTER_NEAREST
        )
        project = canonical_projector(f["cam_pos"][e].cpu().numpy(), f["look_at"][e].cpu().numpy(), scaled)
        radius = float(f["ring_radius"][e])
        for peg in PEGS:
            base = np.array([*lay.peg_xy(peg), lay.plate_thickness])
            draw_peg(img, project, base, lay.peg_height, (160, 160, 160))
        state_id = int(f["state"][e])
        holding = bool(f["holding"][e]) or state_id >= 3  # after release the goal slot is the reference
        height = lay.rings[0].height
        draw_ring(img, project, f["ring_true"][e].cpu().numpy(), radius, height, COLOUR_TRUTH)
        if holding:
            # the held ring travels with the tool; the estimate that matters now is the goal slot
            draw_ring(img, project, f["goal_true"][e].cpu().numpy(), radius, height, COLOUR_TRUTH)
            draw_ring(img, project, f["goal_est"][e].cpu().numpy(), radius, height, COLOUR_ESTIMATE)
            err = float(np.linalg.norm((f["goal_est"][e] - f["goal_true"][e])[:2].cpu().numpy())) * 1000.0
        else:
            draw_ring(img, project, f["goal_est"][e].cpu().numpy(), radius, height, COLOUR_GOAL)
            draw_ring(img, project, f["ring_est"][e].cpu().numpy(), radius, height, COLOUR_ESTIMATE)
            err = float(np.linalg.norm((f["ring_est"][e] - f["ring_true"][e])[:2].cpu().numpy())) * 1000.0
        state = STATE_NAMES[state_id]
        label(img, f"{state}   estimate vs truth {err:.1f} mm", (6, 18))
        label(img, "green: estimate   blue: truth   yellow: goal (believed)", (6, tile - 8), (200, 200, 200))
        r, c = divmod(e, cols)
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = img
    writer.write(canvas)
    env.step(action.clamp(-1.0, 1.0))
writer.release()
print(f"[video] wrote {args.out} ({args.steps} frames, {args.envs} envs)")
env.close()
app.close()
