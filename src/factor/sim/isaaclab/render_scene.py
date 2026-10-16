"""Still images of the twin from a free camera, with the teacher running.

An extra 1280x720 camera and a key light are added to the twin; the camera cycles through a few
views around the anchor while the teacher solves the tower. Inside the Isaac Lab Python::

    python -m factor.sim.isaaclab.render_scene --scene configs/hanoi.yaml --assets <root> --out shots/

트윈을 자유 카메라로 찍는다(교사 정책 실행 중). 1280x720 카메라와 주광을 더하고 앵커 주변 몇 시점을 돈다.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="configs/hanoi.yaml")
parser.add_argument("--assets", required=True, help="root that relative asset paths resolve against")
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=345)
parser.add_argument("--at", default="200,258,330", help="steps at which to save every view")
parser.add_argument("--dome", type=float, default=700.0, help="ambient dome light intensity")
parser.add_argument("--key", type=float, default=2600.0, help="key light intensity")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

from factor.scene import load_scene  # noqa: E402
from factor.sim.contract import Randomisation  # noqa: E402
from factor.sim.isaaclab.twin import HanoiTwin, make_env_cfg, settings_from_scene  # noqa: E402

# (eye, target) relative to the anchor centre, metres / 앵커 중심 기준 (시점, 목표)
VIEWS = {
    "wide": ((0.55, -0.62, 0.42), (0.05, 0.12, 0.10)),
    "close": ((0.26, -0.30, 0.18), (0.0, 0.0, 0.05)),
    "low": ((-0.30, -0.36, 0.10), (0.0, 0.02, 0.06)),
}


class SceneShot(HanoiTwin):
    """The twin plus a free 1280x720 camera and a key light. / 자유 카메라와 주광을 더한 트윈."""

    def _setup_scene(self):
        super()._setup_scene()
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        stage.GetPrimAtPath("/World/Light").GetAttribute("inputs:intensity").Set(args.dome)
        cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/SceneCam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
            ),
            data_types=["rgb"],
            width=1280,
            height=720,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0, horizontal_aperture=20.955, clipping_range=(0.02, 20.0)
            ),
        )
        self.scene_cam = TiledCamera(cfg)
        self.scene.sensors["scene_cam"] = self.scene_cam
        key = sim_utils.DistantLightCfg(intensity=args.key, angle=0.6, color=(1.0, 0.97, 0.92))
        key.func("/World/KeyLight", key, orientation=(0.88, 0.25, -0.35, 0.18))


def main() -> None:
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    settings = settings_from_scene(
        load_scene(args.scene),
        args.assets,
        randomisation=Randomisation(
            anchor_xy=0.004, anchor_yaw=np.radians(0.5), camera_xyz=0.0, noise=0.0, gain=0.0
        ),
        local_correction=True,
        error_resample_steps=0,
    )
    env = SceneShot(make_env_cfg(settings, 1, "cuda:0"))
    env.reset()
    origin = env.scene.env_origins[0]
    anchor = torch.tensor([*settings.anchor_xy, 0.0], device=origin.device) + origin
    names = list(VIEWS)
    marks = {int(v) for v in args.at.split(",")}
    for k in range(args.steps):
        # Two ticks per view: the second one shows the new pose. / 시점마다 두 틱(두 번째가 새 자세).
        view = names[(k // 2) % len(names)]
        eye, target = (anchor + torch.tensor(v, device=origin.device) for v in VIEWS[view])
        env.scene_cam.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        env.step(env.teacher_action())
        if k % 2 == 1 and any(m <= k < m + 2 * len(names) for m in marks):
            rgb = env.scene_cam.data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(rgb).save(os.path.join(args.out, f"scene_{k:04d}_{view}.png"))
    print(f"saved views to {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
