"""Smoke test for the twin: run the teacher and report outcomes. / 트윈 점검: 교사 실행·결과 집계.

Usage (inside the Isaac Lab Python environment)::

    python -m factor.sim.isaaclab.check --scene configs/hanoi.yaml --assets <repo root> \
        --num_envs 32 --headless

The teacher feeds the twin the negated sampled belief error, so success must be close to 100 %.
With ``--nominal`` the correction is zero (rule-only baseline).
"""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="configs/hanoi.yaml", help="scene specification (YAML)")
parser.add_argument(
    "--assets", required=True, help="root that relative asset paths in the scene resolve against"
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--nominal", action="store_true", help="zero correction (rule-only baseline)")
parser.add_argument("--anchor_err", type=float, default=0.006)
parser.add_argument("--save", default="", help="directory for a few canonical-view PNGs")
parser.add_argument(
    "--render", choices=("belief", "mask", "both", "none"), default="belief", help="render channels"
)
parser.add_argument("--view_width", type=float, default=0.14, help="metres across the canonical view")
parser.add_argument("--local_correction", action="store_true")
parser.add_argument("--belief_fingers", action="store_true")
parser.add_argument("--anchor_z", type=float, default=0.0)
parser.add_argument("--robot_recolor", type=float, default=0.0)
parser.add_argument("--no_mask", action="store_true", help="same as --render none")
parser.add_argument("--checkpoint", default="", help="run the trained estimator instead of the teacher")
parser.add_argument("--anchor_yaw_deg", type=float, default=1.0)
parser.add_argument("--camera_rand", type=float, default=0.08)
parser.add_argument("--noise", type=float, default=0.05)
parser.add_argument("--gain", type=float, default=0.4)
parser.add_argument(
    "--resample", type=int, default=0, help="in-episode belief resampling period (0 = off, as on the robot)"
)
parser.add_argument("--episode_seconds", type=float, default=8.0)
parser.add_argument("--in_hand_offset", type=float, default=0.003, help="held-start ring offset range (m)")
parser.add_argument("--offset_scale", type=float, default=0.02, help="metres per unit of the offset head")
parser.add_argument("--use_offset_estimate", action="store_true", help="use the (teacher/estimated) offset")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from factor.scene import load_scene  # noqa: E402
from factor.sim.contract import Randomisation  # noqa: E402
from factor.sim.isaaclab.twin import HanoiTwin, make_env_cfg, settings_from_scene  # noqa: E402

settings = settings_from_scene(
    load_scene(args.scene),
    args.assets,
    randomisation=Randomisation(
        anchor_xy=args.anchor_err,
        anchor_yaw=np.radians(args.anchor_yaw_deg),
        anchor_z=args.anchor_z,
        camera_xyz=args.camera_rand,
        noise=args.noise,
        gain=args.gain,
    ),
    view_width=args.view_width,
    local_correction=args.local_correction,
    belief_fingers=args.belief_fingers,
    robot_recolor=args.robot_recolor,
    render_mask=(args.render in ("mask", "both")) and not args.no_mask,
    render_belief=(args.render in ("belief", "both")) and not args.no_mask,
    error_resample_steps=args.resample,
    episode_seconds=args.episode_seconds,
    in_hand_offset=args.in_hand_offset,
    use_offset_estimate=args.use_offset_estimate,
)
env_cfg = make_env_cfg(settings, args.num_envs, args.device)
env_cfg.offset_scale = args.offset_scale
env = HanoiTwin(env_cfg)
env.reset()
model = None
if args.checkpoint:
    from factor.perception.model import EstimatorConfig, StateEstimator

    members = []  # comma-separated checkpoints form an averaged ensemble / 앙상블
    for path in args.checkpoint.split(","):
        ck = torch.load(path, map_location=env.device, weights_only=False)
        m = StateEstimator(EstimatorConfig(**ck["config"])).to(env.device)
        m.load_state_dict(ck["state_dict"], strict=False)  # older checkpoints lack newer heads
        m.eval()
        members.append(m)
    model = members[0]
    env.cfg.offset_scale = model.config.offset_scale  # the planner must use the model's scale
counts = {"succeeded": 0, "dropped": 0, "bad_release": 0, "timeout": 0}
timeout_phase = {"holding": 0, "free": 0}  # where time-outs happen / 시간 초과가 어느 단계에서 나는지
states = torch.zeros(5, dtype=torch.long)
t0 = time.time()
for k in range(args.steps):
    teacher = env.teacher_action()
    if args.nominal:
        action = torch.cat([torch.zeros(env.num_envs, 3, device=env.device), teacher[:, 3:]], 1)
    elif model is not None:
        with torch.no_grad():
            f = env.last_frames
            outs = [m(f["rgb"], f["render"], f["grip"]) for m in members]
        from factor.perception.train import action_from_output

        action = torch.stack([action_from_output(o) for o in outs]).mean(0)
    else:
        action = teacher
    holding_before = env.grip_cmd.clone()
    _, _, terminated, truncated, _ = env.step(action)
    reason = env.last_term_reason
    seen = torch.zeros_like(terminated, dtype=torch.bool)
    for key in ("succeeded", "dropped", "bad_release"):
        m = reason[key] & ~seen
        counts[key] += int(m.sum())
        seen |= reason[key]
    timed_out = truncated & ~seen
    counts["timeout"] += int(timed_out.sum())
    timeout_phase["holding"] += int((timed_out & holding_before).sum())
    timeout_phase["free"] += int((timed_out & ~holding_before).sum())
    states += torch.bincount(env.last_frames["state"].cpu(), minlength=5)
    if args.save and k in (0, 30, 90):
        from torchvision.utils import make_grid, save_image

        os.makedirs(args.save, exist_ok=True)
        save_image(
            make_grid(env.last_frames["rgb"][:16], nrow=4), os.path.join(args.save, f"rgb_{k:03d}.png")
        )
        if env.last_frames["render"] is not None:
            save_image(
                make_grid(env.last_frames["render"][:16], nrow=4),
                os.path.join(args.save, f"mask_{k:03d}.png"),
            )
episodes = sum(counts.values())
print(
    f"[twin] {args.steps} steps x {env.num_envs} envs in {time.time() - t0:.0f}s  "
    f"episodes {episodes}: {counts}  success {counts['succeeded'] / max(episodes, 1):.1%}  "
    f"state histogram {states.tolist()}  time-outs while {timeout_phase}",
    flush=True,
)
env.close()
app.close()
