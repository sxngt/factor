"""Tower of Hanoi twin on Isaac Lab.

The twin is the data source for the perception layer and the training ground for skills:

- it samples a *belief error* of the anchor pose for every episode and aims the camera at the
  believed anchor centre, exactly as the real canonical view is aimed at the fiducial estimate,
  so the error is visible in the image and known as a label;
- it runs the same funnel planner and contact search as the real skill, driven by the
  corrections it is given (the teacher passes the negated sampled error, a student passes its
  estimate);
- it reports an interaction state per frame from the simulated contact outcome.

Grasping uses a kinematic attach (the held ring follows the grip point) because finger-pad
contact with thin rings is not reliable in PhysX; insertion and release are physical, so seated,
jammed and tilted outcomes come from the simulator.

Isaac Lab 위의 하노이 트윈. 에피소드마다 앵커 포즈 '믿음 오차' 를 샘플링하고 카메라를 믿는 중심으로
조준해(실기 정규화 시야와 동일 규약) 오차가 영상에 드러나고 라벨로 알려지게 한다. 실기 스킬과 같은
깔때기 플래너·접촉 탐색을 주어진 보정량으로 돌리고, 접촉 결과로 상호작용 상태를 매긴다. 파지는 운동학
attach, 삽입·해제는 물리.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform

from factor.robots.fr3 import FR3Kinematics, FR3Params, top_down_rotation
from factor.sim.contract import Randomisation
from factor.sim.isaaclab.planner import FusionState, NudgeState, apply_correction, funnel_step
from factor.tasks.hanoi.spec import PEGS, HanoiLayout

FINGER_JOINTS = ("ag95_left_finger_joint", "ag95_right_finger_joint")
ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
TCP_BODY = "tcp"
APERTURE_MM = 20.955

# Interaction states reported per frame. / 프레임별 상호작용 상태.
STATE_FREE, STATE_BETWEEN_FINGERS, STATE_GRASPED, STATE_SEATED, STATE_JAMMED = range(5)


@dataclass(frozen=True)
class TwinSettings:
    """Everything the twin needs beyond the Hanoi layout. / 하노이 기하 외에 트윈이 필요로 하는 설정."""

    robot_usd: str
    ring_usds: tuple[str, ...]
    mat_usd: str | None
    layout: HanoiLayout
    kinematics: FR3Params = field(default_factory=FR3Params)
    ik_branch: tuple[int, int, int] = (0, 0, 1)
    finger_axis: str = "y"
    finger_sign: int = 1
    finger_max: float = 0.0475
    finger_clearance: float = 0.001
    open_margin: float = 0.006
    grasp_tip_above_bottom: float = 0.002
    base_z: float = -0.026
    anchor_xy: tuple[float, float] = (-0.0744, -0.4572)
    anchor_yaw: float = 0.0  # yaw of the anchor (mat) frame in the robot base frame / 앵커 프레임 yaw
    camera_in_anchor: tuple[float, float, float] = (-0.006, -0.397, 0.707)
    look_height: float = 0.037
    resolution: int = 128
    view_width: float = 0.14
    randomisation: Randomisation = field(default_factory=Randomisation)
    start_mix: tuple[float, float, float] = (0.4, 0.3, 0.3)
    pos_scale: float = 0.01
    yaw_scale: float = math.radians(5.0)
    fusion_alpha: float = 0.9
    latch_below: float = 0.012
    holding_from_estimate: bool = (
        False  # opt-in: regrasp when the state head says not held / 추정기 파지 판정(선택)
    )
    regrasp_after: int = 3  # closed-but-not-held steps before reopening / 닫았는데 미파지면 재시도까지 스텝
    nudge_amplitude: float = 0.0007
    in_hand_offset: float = 0.003  # held-start episodes: ring offset along the finger axis (m) / 손 안 오프셋
    use_offset_estimate: bool = False  # place at goal minus the estimated in-hand offset / 오프셋 추정 사용
    error_resample_steps: int = 10
    render_mask: bool = False  # semantic robot mask (sim-only; not available on the robot)
    local_correction: bool = False  # label = translation at the active peg, yaw folded in / 국소 보정 라벨
    robot_recolor: float = 0.0  # probability of recolouring the robot pixels per frame (appearance gap)
    render_belief: bool = True  # believed-gripper render channel (works on the robot) / 믿음 렌더
    belief_fingers: bool = False  # draw both believed fingers / 핑거 실루엣
    episode_seconds: float = 8.0
    disk_mass: float = 0.010
    peg_radius: float = 0.005
    plate_size: tuple[float, float] = (0.16, 0.06)
    table_size: tuple[float, float] = (1.2, 1.3)
    table_center: tuple[float, float] = (0.0, -0.55)
    table_top_thickness: float = 0.05
    table_height: float = 0.75


def settings_from_scene(scene, assets_root: str, **overrides) -> TwinSettings:
    """Build :class:`TwinSettings` from a scene specification. / 장면 명세 → 트윈 설정.

    Asset paths in ``extras.sim`` are resolved against ``assets_root`` unless absolute.
    ``overrides`` replace individual fields (resolution, randomisation, ...).
    """
    sim = scene.extras["sim"]
    hanoi = scene.extras["hanoi"]
    grasp = scene.robot.grasp

    def path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(assets_root, p)

    anchor = scene.anchor_in_base
    fields = dict(
        robot_usd=path(sim["robot_usd"]),
        ring_usds=tuple(path(p) for p in sim["ring_usds"]),
        mat_usd=path(sim["mat_usd"]) if sim.get("mat_usd") else None,
        layout=HanoiLayout.from_scene(scene),
        kinematics=FR3Params.from_mapping(scene.robot.kinematics),
        ik_branch=tuple(int(v) for v in grasp.get("ik_branch", (0, 0, 1))),
        finger_axis=str(grasp.get("finger_axis", "y")),
        finger_sign=int(grasp.get("finger_sign", 1)),
        finger_max=float(grasp.get("finger_max", 0.0475)),
        finger_clearance=float(grasp.get("finger_clearance", 0.001)),
        open_margin=float(grasp.get("open_margin", 0.006)),
        grasp_tip_above_bottom=float(grasp.get("tip_above_bottom", 0.002)),
        base_z=float(hanoi.get("base_z", -0.026)),
        anchor_xy=(float(anchor[0, 3]), float(anchor[1, 3])),
        anchor_yaw=float(math.atan2(anchor[1, 0], anchor[0, 0])),
        camera_in_anchor=tuple(float(v) for v in scene.camera_in_anchor[:3, 3]),
    )
    fields.update(overrides)
    return TwinSettings(**fields)


def _camera_cfg(resolution: int, hfov: float, mask: bool) -> TiledCameraCfg:
    focal = APERTURE_MM / (2.0 * math.tan(hfov / 2.0))
    data_types = ["rgb", "semantic_segmentation"] if mask else ["rgb"]
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/AnchorCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=data_types,
        width=resolution,
        height=resolution,
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal, horizontal_aperture=APERTURE_MM, clipping_range=(0.05, 5.0)
        ),
    )


def make_env_cfg(s: TwinSettings, num_envs: int, device: str) -> HanoiTwinCfg:
    """Build the Isaac Lab configuration from settings. / 설정 → Isaac Lab cfg."""
    kin = FR3Kinematics(s.kinematics)
    lay = s.layout
    grip_offset = lay.rings[0].height / 2.0 - s.grasp_tip_above_bottom

    def in_base(p_anchor) -> np.ndarray:
        """Anchor-frame point -> nominal base frame. / 앵커 프레임 점 → 공칭 베이스 프레임."""
        c, s_ = math.cos(s.anchor_yaw), math.sin(s.anchor_yaw)
        x, y, z = p_anchor
        return np.array([s.anchor_xy[0] + c * x - s_ * y, s.anchor_xy[1] + s_ * x + c * y, z])

    hover_tip = in_base([*lay.peg_xy("left"), lay.hover_height - grip_offset])
    pose = np.eye(4)
    pose[:3, :3] = top_down_rotation(s.finger_axis, s.finger_sign, yaw=s.anchor_yaw)
    pose[:3, 3] = hover_tip
    home = kin.inverse(pose, s.ik_branch)
    if home is None:
        raise ValueError("hover pose unreachable; check anchor_xy and the IK branch")
    open_gap = min(lay.rings[-1].radius + s.open_margin, s.finger_max)

    cfg = HanoiTwinCfg()
    cfg.settings = s
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.episode_length_s = s.episode_seconds
    cfg.robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=s.robot_usd, activate_contact_sensors=False, semantic_tags=[("class", "robot")]
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                **{n: float(v) for n, v in zip(ARM_JOINTS, home, strict=True)},
                **{n: open_gap for n in FINGER_JOINTS},
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINTS), stiffness=4.0e5, damping=2.0e4, effort_limit_sim=3000.0
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["ag95_.*_finger_joint"],
                stiffness=1.0e4,
                damping=200.0,
                effort_limit_sim=130.0,
            ),
        },
    )
    cfg.rings = [
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Ring{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=s.disk_mass),
                semantic_tags=[("class", "ring")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(
                    in_base([*lay.peg_xy("left"), lay.plate_thickness + lay.rings[0].height * (i + 0.5)])
                )
            ),
        )
        for i, usd in enumerate(s.ring_usds)
    ]
    hfov = 2.0 * math.atan(s.view_width / (2.0 * float(np.linalg.norm(s.camera_in_anchor))))
    cfg.camera = _camera_cfg(s.resolution, hfov, s.render_mask or s.robot_recolor > 0)
    cfg.observation_space = {
        "rgb": [3, s.resolution, s.resolution],
        "render": [int(s.render_belief) + int(s.render_mask), s.resolution, s.resolution],
    }
    return cfg


@configclass
class HanoiTwinCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 8.0
    action_space = 6
    observation_space = {"rgb": [3, 128, 128], "render": [1, 128, 128]}
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=2**20, gpu_max_rigid_contact_count=2**24),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)
    robot: ArticulationCfg = None
    rings: list = []
    camera: TiledCameraCfg = None
    settings: TwinSettings = None
    max_step: float = 0.01
    attach_xy: float = 0.006
    attach_z: tuple = (-0.005, 0.006)
    release_xy: float = 0.003
    seated_xy: float = 0.002
    seated_tilt: float = math.radians(5.0)
    offset_scale: float = 0.1  # metres per unit of the ring-to-grip offset label / 링-그립 오프셋 라벨 스케일
    retry_hold_steps: int = 5  # steps to wait at hover before descending again / 재시도 전 hover 대기 스텝


class HanoiTwin(DirectRLEnv):
    """Isaac Lab environment implementing the twin. / 트윈 환경."""

    cfg: HanoiTwinCfg

    def __init__(self, cfg: HanoiTwinCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        s = self.settings = cfg.settings
        lay = self.layout = s.layout
        dev = self.device
        n = self.num_envs

        self.kin = FR3Kinematics(s.kinematics)
        self.joint_ids, _ = self.robot.find_joints(list(ARM_JOINTS), preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(list(FINGER_JOINTS), preserve_order=True)
        self.tcp_idx = self.robot.find_bodies(TCP_BODY)[0][0]

        self.n_rings = len(lay.rings)
        self.ring_h = float(lay.rings[0].height)
        self.grip_offset = self.ring_h / 2.0 - s.grasp_tip_above_bottom
        self.radii = torch.tensor([r.radius for r in lay.rings], device=dev, dtype=torch.float32)
        self.anchor = torch.tensor([*s.anchor_xy, 0.0], device=dev, dtype=torch.float32)
        # Anchor (mat) axes are rotated by ``anchor_yaw`` with respect to the robot base axes.
        # 앵커(매트) 축은 베이스 축에 대해 ``anchor_yaw`` 만큼 돌아 있다.
        self.anchor_cos = float(math.cos(s.anchor_yaw))
        self.anchor_sin = float(math.sin(s.anchor_yaw))
        self.peg_xy = (
            torch.stack(
                [
                    self.to_env_axes(
                        torch.tensor([*lay.peg_xy(p), 0.0], device=dev, dtype=torch.float32).unsqueeze(0)
                    )[0, :2]
                    for p in PEGS
                ]
            )
            + self.anchor[:2]
        )
        self.hover = float(lay.hover_height)
        self.zone_top = float(lay.zone_top)
        self.plate = float(lay.plate_thickness)
        self.tool_rot = torch.tensor(
            top_down_rotation(s.finger_axis, s.finger_sign, yaw=s.anchor_yaw), device=dev, dtype=torch.float64
        )
        self.tool_off = torch.tensor([0.0, self.grip_offset, 0.0], device=dev).repeat(n, 1)
        self.tip_off = torch.tensor([0.0, 0.0, -self.grip_offset], device=dev)
        self.home = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.targets = self.home.clone()
        ws = s.randomisation
        box = torch.tensor([[sx * 0.12, sy * 0.04, 0.0] for sx in (-1, 1) for sy in (-1, 1)], device=dev)
        box = self.to_env_axes(box)[:, :2] + self.anchor[:2]
        self.ws_lo = torch.tensor(
            [*box.min(0).values.tolist(), self.plate + self.ring_h / 2 - 0.001], device=dev
        )
        self.ws_hi = torch.tensor([*box.max(0).values.tolist(), self.hover + 0.01], device=dev)

        self.actor = torch.zeros(n, dtype=torch.long, device=dev)
        self.ring_peg = torch.zeros(self.n_rings, n, dtype=torch.long, device=dev)
        self.src = torch.zeros(n, dtype=torch.long, device=dev)
        self.dst = torch.ones(n, dtype=torch.long, device=dev)
        self.goal = torch.zeros(n, 3, device=dev)
        self.start = torch.zeros(n, 3, device=dev)
        self.close_gap = torch.zeros(n, device=dev)
        self.open_gap = torch.zeros(n, device=dev)
        self.tgt = torch.zeros(n, 3, device=dev)
        self.attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.grip_cmd = torch.zeros(n, dtype=torch.bool, device=dev)
        self.base_off = torch.zeros(n, 3, device=dev)
        self.belief_err = torch.zeros(n, 4, device=dev)  # (ex, ey, eyaw, ez) / 믿음 오차
        self.gain = torch.ones(n, 3, 1, 1, device=dev)
        self.cam_pos = torch.zeros(
            n, 3, device=dev
        )  # camera position in the anchor frame / 앵커 기준 카메라 위치
        self.look_at = torch.zeros(n, 3, device=dev)  # look-at point in the anchor frame (active peg)
        self.aim_jit = torch.zeros(n, 2, device=dev)  # detection noise on the aim point / 조준 잡음
        self.view_peg = torch.full((n,), -1, dtype=torch.long, device=dev)  # peg the view is centred on
        self.peg_x_anchor = torch.tensor([float(lay.peg_xy(p)[0]) for p in PEGS], device=dev)
        self.peg_y_anchor = float(lay.peg_xy(PEGS[0])[1])
        self.last_estimate = torch.zeros(n, 3, device=dev)  # correction applied at the last step
        self.residual = torch.zeros(n, 3, device=dev)  # skill-layer residual step (m) / 스킬 잔차 이동량
        self.hand_offset = torch.zeros(n, 3, device=dev)  # held ring centre minus grip point (env frame)
        self.retry_hold = torch.zeros(n, dtype=torch.long, device=dev)  # steps held at hover during a retry
        self.skill_context: dict[
            str, torch.Tensor
        ] = {}  # inputs of the last funnel step / 마지막 깔때기 입력
        self.bias = torch.zeros(n, 1, 1, 1, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.bad_release = torch.zeros(n, dtype=torch.bool, device=dev)
        self.released_ring = torch.zeros(n, dtype=torch.bool, device=dev)
        self.not_held_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.fusion = FusionState(n, 4, dev, s.fusion_alpha, s.latch_below, self.hover)
        self.nudge = NudgeState(n, dev, amplitude=s.nudge_amplitude)
        self.robot_class_id: int | None = None
        self.resample_belief = (
            True  # evaluation turns in-episode resampling off / 평가 시 에피소드 내 재추출 끔
        )
        self.last_frames: dict[str, torch.Tensor] = {}
        _ = ws

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        s = self.cfg.settings
        lay = s.layout
        ax, ay = s.anchor_xy
        self.robot = Articulation(self.cfg.robot)
        self.ring_objs = [RigidObject(c) for c in self.cfg.rings]
        static = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        beige = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25))
        table = sim_utils.CuboidCfg(
            size=(s.table_size[0], s.table_size[1], s.table_top_thickness), visual_material=beige, **static
        )
        table.func(
            "/World/envs/env_0/Table",
            table,
            translation=(s.table_center[0], s.table_center[1], -s.table_top_thickness / 2),
        )
        plate = sim_utils.CuboidCfg(
            size=(s.plate_size[0], s.plate_size[1], lay.plate_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            semantic_tags=[("class", "tower")],
            **static,
        )
        c, s_ = math.cos(s.anchor_yaw), math.sin(s.anchor_yaw)

        def in_env(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (ax + c * x - s_ * y, ay + s_ * x + c * y, z)

        plate.func(
            "/World/envs/env_0/Plate",
            plate,
            translation=in_env(lay.peg_row_offset[0], lay.peg_row_offset[1], lay.plate_thickness / 2),
        )
        for i, peg in enumerate(PEGS):
            cyl = sim_utils.CylinderCfg(
                radius=s.peg_radius,
                height=lay.peg_height,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.65)),
                semantic_tags=[("class", "tower")],
                **static,
            )
            cyl.func(
                f"/World/envs/env_0/Peg{i}",
                cyl,
                translation=in_env(*lay.peg_xy(peg), lay.plate_thickness + lay.peg_height / 2),
            )
        if s.mat_usd:
            mat = sim_utils.UsdFileCfg(usd_path=s.mat_usd)
            mat.func("/World/envs/env_0/Mat", mat, translation=(ax, ay, 0.0002))
        sim_utils.GroundPlaneCfg().func(
            "/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, -s.table_height)
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        for i, r in enumerate(self.ring_objs):
            self.scene.rigid_objects[f"ring{i}"] = r
        self.camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["anchor_cam"] = self.camera
        sim_utils.DomeLightCfg(intensity=1500.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=1500.0)
        )

    # ---------------------------------------------------------------- frames
    def to_env_axes(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate anchor-axis vectors into base-aligned axes. / 앵커 축 → 베이스 정렬 축."""
        c, s_ = self.anchor_cos, self.anchor_sin
        return torch.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    def to_anchor_axes(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate base-aligned vectors into anchor axes. / 베이스 정렬 축 → 앵커 축."""
        c, s_ = self.anchor_cos, self.anchor_sin
        return torch.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    def anchor_point(self, p_anchor: torch.Tensor) -> torch.Tensor:
        """Anchor-frame point -> environment point (nominal base pose). / 앵커 점 → env 점."""
        return self.to_env_axes(p_anchor) + self.anchor

    def rot_env(self, v, env_ids=None):
        yaw = self.base_off[:, 2] if env_ids is None else self.base_off[env_ids, 2]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    def rot_base(self, v, env_ids=None):
        yaw = self.base_off[:, 2] if env_ids is None else self.base_off[env_ids, 2]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    def to_base(self, p_env):
        off = torch.stack(
            [
                self.base_off[:, 0],
                self.base_off[:, 1],
                torch.full_like(self.base_off[:, 2], self.settings.base_z),
            ],
            1,
        )
        return self.rot_base(p_env - off)

    def believed(self, p_env):
        """Point in the believed anchor frame (anchor axes). / 믿는 앵커 프레임의 점.

        The belief error ``(ex, ey, eyaw)`` is expressed in anchor axes: the believed anchor sits
        at the true anchor displaced by ``(ex, ey)`` and rotated by ``eyaw``.
        """
        err = self.to_env_axes(
            torch.stack([self.belief_err[:, 0], self.belief_err[:, 1], self.belief_err[:, 3]], 1)
        )
        d = self.to_anchor_axes(self.rot_base(p_env - self.anchor - err))
        c, s_ = torch.cos(self.belief_err[:, 2]), torch.sin(self.belief_err[:, 2])
        return torch.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1], d[:, 2]], 1)

    def nominal(self, p_env):
        """Nominal point as the planner believes it (anchor axes, no error). / 플래너가 믿는 공칭 점."""
        return self.to_anchor_axes(self.rot_base(p_env - self.anchor))

    def grip_point_w(self):
        p = self.robot.data.body_pos_w[:, self.tcp_idx]
        q = self.robot.data.body_quat_w[:, self.tcp_idx]
        return p + quat_apply(q, self.tool_off)

    def ring_pos_all(self):
        return torch.stack([r.data.root_pos_w for r in self.ring_objs])

    def ring_quat_all(self):
        return torch.stack([r.data.root_quat_w for r in self.ring_objs])

    def actor_pos_w(self):
        return self.ring_pos_all().gather(0, self.actor.view(1, -1, 1).expand(1, -1, 3)).squeeze(0)

    def actor_quat_w(self):
        return self.ring_quat_all().gather(0, self.actor.view(1, -1, 1).expand(1, -1, 4)).squeeze(0)

    def teacher_action(self) -> torch.Tensor:
        """Exact correction, true grasped flag and true in-hand offset. / 정확한 보정·파지·오프셋."""
        return torch.cat(
            [
                self.teacher_correction(),
                self.attached.float().unsqueeze(1),
                self.ring_offset_label()[:, :2],
                self.teacher_correction_z(),
            ],
            1,
        )

    def ring_offset_label(self) -> torch.Tensor:
        """Ring centre minus grip point in the believed frame, normalised. / 링-그립 오프셋 라벨."""
        o = self.scene.env_origins
        grip = self.believed(self.grip_point_w() - o)
        return ((self.believed(self.actor_pos_w() - o) - grip) / self.cfg.offset_scale).clamp(-1.0, 1.0)

    def teacher_correction(self):
        """Correction that undoes the sampled belief error. / 샘플링된 오차를 되돌리는 보정량 (라벨)."""
        s = self.settings
        if s.local_correction:
            # Translation that is exact at the peg in the centre of the view (source while picking,
            # goal while holding); the yaw error is folded into it. The view only shows the belief
            # error near that peg, so this is the observable label.
            # 시야 중앙 기둥에서 정확한 평행이동(요 오차 포함). 시야로 관측 가능한 라벨이다.
            active = torch.where(self.grip_cmd, self.dst, self.src)
            peg = torch.cat([self.peg_xy[active], torch.zeros_like(self.peg_xy[active, :1])], 1)
            c = (self.believed(peg) - self.nominal(peg))[:, :2]
            return torch.stack(
                [c[:, 0] / s.pos_scale, c[:, 1] / s.pos_scale, torch.zeros_like(c[:, 0])], 1
            ).clamp(-1.0, 1.0)
        peg = torch.cat([self.peg_xy[self.src], torch.zeros_like(self.peg_xy[self.src, :1])], 1)
        true_obs = self.believed(peg)
        nom = self.nominal(peg)
        th = -self.belief_err[:, 2]
        cs, sn = torch.cos(th), torch.sin(th)
        rot_nom = torch.stack([cs * nom[:, 0] - sn * nom[:, 1], sn * nom[:, 0] + cs * nom[:, 1]], 1)
        c = true_obs[:, :2] - rot_nom
        return torch.stack([c[:, 0] / s.pos_scale, c[:, 1] / s.pos_scale, th / s.yaw_scale], 1).clamp(
            -1.0, 1.0
        )

    def teacher_correction_z(self) -> torch.Tensor:
        """Height correction ``(N, 1)``: believed minus nominal z of any point. / 높이 보정 라벨."""
        return (-self.belief_err[:, 3:4] / self.settings.pos_scale).clamp(-1.0, 1.0)

    # ---------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        s = self.settings
        o = self.scene.env_origins
        grip_env = self.grip_point_w() - o
        g_obs = self.believed(grip_env)
        cz = actions[:, 6:7] if actions.shape[1] >= 7 else torch.zeros_like(actions[:, :1])
        corr = self.fusion.update(torch.cat([actions[:, :3], cz], 1).clamp(-1.0, 1.0), g_obs[:, 2])
        held = self.grip_cmd
        if s.holding_from_estimate and actions.shape[1] >= 4:
            held_est = actions[:, 3] > 0.5
            failed = self.grip_cmd & ~held_est
            self.not_held_steps = torch.where(
                failed, self.not_held_steps + 1, torch.zeros_like(self.not_held_steps)
            )
            regrasp = self.not_held_steps >= s.regrasp_after
            held = self.grip_cmd & ~regrasp
            if regrasp.any():
                ids = regrasp.nonzero(as_tuple=False).squeeze(-1)
                self.fusion.reset(ids)
                self.nudge.reset(ids)

        n_below = (self.ring_peg == self.src.unsqueeze(0)).sum(0).float() - 1.0
        ring_nom = torch.cat(
            [self.peg_xy[self.src], (self.plate + self.ring_h * (n_below + 0.5)).unsqueeze(1)], 1
        )
        goal_nom = self.goal.clone()
        goal_nom[:, 2] += self.layout.place_clearance
        r_c = apply_correction(self.nominal(ring_nom), corr, s.pos_scale, s.yaw_scale)
        goal_c = apply_correction(self.nominal(goal_nom), corr, s.pos_scale, s.yaw_scale)

        inserting = (
            held
            & (torch.norm(g_obs[:, :2] - goal_c[:, :2], dim=1) <= 0.012)  # near the goal peg / 목표 페그 근처
            & (g_obs[:, 2] < self.zone_top + s.latch_below + 0.004)
            & (g_obs[:, 2] > self.zone_top - 0.002)
        )
        goal_c = goal_c.clone()
        if s.use_offset_estimate and actions.shape[1] >= 6:
            goal_c[:, :2] -= torch.where(
                held.unsqueeze(1), actions[:, 4:6].clamp(-1.0, 1.0) * self.cfg.offset_scale, 0.0
            )
        goal_c[:, :2] += self.nudge.update(g_obs[:, 2], inserting, held)

        step, close = funnel_step(g_obs, r_c, goal_c, held, self.hover, self.zone_top, self.cfg.max_step)
        step = self._lift_for_retry(step, g_obs[:, 2])
        step = (step + self.residual).clamp(-self.cfg.max_step, self.cfg.max_step)
        self.skill_context = {
            "grip": g_obs,
            "object": r_c,
            "goal": goal_c,
            "holding": held,
            "inserting": inserting,
        }
        self.last_estimate = corr.clone()
        self._apply_motion(self.rot_env(self.to_env_axes(step)), close)

        if (
            self.resample_belief
            and s.error_resample_steps > 0
            and self.common_step_counter > 0
            and self.common_step_counter % s.error_resample_steps == 0
        ):
            self._resample_belief(self.robot._ALL_INDICES)
        active = torch.where(self.grip_cmd, self.dst, self.src)
        self._aim((active != self.view_peg).nonzero(as_tuple=False).squeeze(-1))

    def _lift_for_retry(self, step: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
        """Exhausted contact search: rise to hover and wait for the estimate to refresh.

        접촉 탐색을 다 써 버리면 hover 까지 올려 보정값이 갱신되게 한 뒤 다시 내려간다.
        """
        retry = self.nudge.retry
        if not retry.any():
            return step
        high = grip_z > self.hover - 0.005
        self.retry_hold = torch.where(retry & high, self.retry_hold + 1, torch.zeros_like(self.retry_hold))
        done = self.retry_hold >= self.cfg.retry_hold_steps
        self.nudge.retry &= ~done
        lift = torch.zeros_like(step)
        lift[:, 2] = (self.hover - grip_z).clamp(-self.cfg.max_step, self.cfg.max_step)
        return torch.where((retry & ~done).unsqueeze(1), lift, step)

    def _apply_motion(self, delta: torch.Tensor, close: torch.Tensor):
        """Peg constraint for a held ring, workspace clamp, IK and attach/release logic."""
        ring_c = self.tgt + self.hand_offset  # the held ring, not the grip point, must clear the peg
        d_axis = (
            torch.cdist(ring_c[:, :2].unsqueeze(1), self.peg_xy.unsqueeze(0)).squeeze(1).min(dim=1).values
        )
        in_zone = self.attached & (ring_c[:, 2] < self.zone_top)
        entering = (
            self.attached
            & (ring_c[:, 2] >= self.zone_top)
            & (ring_c[:, 2] + delta[:, 2] < self.zone_top)
            & (d_axis > 0.002)
        )
        delta = delta.clone()
        delta[:, :2] = torch.where(in_zone.unsqueeze(1), torch.zeros_like(delta[:, :2]), delta[:, :2])
        delta[:, 2] = torch.where(entering, torch.zeros_like(delta[:, 2]), delta[:, 2])
        self.tgt = torch.max(torch.min(self.tgt + delta, self.ws_hi), self.ws_lo)

        poses = torch.eye(4, device=self.device, dtype=torch.float64).repeat(self.num_envs, 1, 1)
        poses[:, :3, :3] = self.tool_rot
        poses[:, :3, 3] = self.to_base(self.tgt + self.tip_off).double()
        q, valid = self.kin.inverse_batch(poses, self.settings.ik_branch)
        self.targets = torch.where(valid.unsqueeze(1), q.float(), self.targets)
        self.grip_cmd = close

        grip = self.grip_point_w()
        ring = self.actor_pos_w()
        d = grip - ring
        near = (
            (torch.norm(d[:, :2], dim=1) < self.cfg.attach_xy)
            & (d[:, 2] > self.cfg.attach_z[0])
            & (d[:, 2] < self.cfg.attach_z[1])
        )
        was = self.attached.clone()
        self.attached = torch.where(self.grip_cmd, self.attached | near, torch.zeros_like(self.attached))
        captured = (self.attached & ~was).nonzero(as_tuple=False).squeeze(-1)
        if len(captured):
            self.hand_offset[captured] = self._capture_offset(-d[captured], captured)
        released = was & ~self.attached
        ring_l = ring - self.scene.env_origins
        aligned = (torch.norm(ring_l[:, :2] - self.goal[:, :2], dim=1) <= self.cfg.release_xy) & (
            ring_l[:, 2] < self.zone_top
        )
        self.bad_release = released & ~aligned
        self.released_ring |= released & aligned

    def _capture_offset(self, ring_minus_grip: torch.Tensor, env_ids) -> torch.Tensor:
        """Offset kept after the fingers close. / 핑거가 닫힌 뒤 남는 오프셋.

        Along the finger axis the ring stays where it was; across it the fingers centre the ring
        up to the clearance. In anchor axes the finger axis is y for ``finger_axis == "y"``.
        """
        local = self.to_anchor_axes(self.rot_base(ring_minus_grip, env_ids))
        c = self.settings.finger_clearance
        if self.settings.finger_axis == "y":
            local = torch.stack([local[:, 0].clamp(-c, c), local[:, 1], torch.zeros_like(local[:, 2])], 1)
        else:
            local = torch.stack([local[:, 0], local[:, 1].clamp(-c, c), torch.zeros_like(local[:, 2])], 1)
        return self.rot_env(self.to_env_axes(local), env_ids)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)
        gap = torch.where(self.grip_cmd, self.close_gap, self.open_gap)
        self.robot.set_joint_position_target(gap.unsqueeze(1).repeat(1, 2), joint_ids=self.finger_ids)
        ids = self.attached.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            pos = self.grip_point_w()[ids] + self.hand_offset[ids]
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(ids), 1)
            for i, r in enumerate(self.ring_objs):
                sel = (self.actor[ids] == i).nonzero(as_tuple=False).squeeze(-1)
                if len(sel):
                    r.write_root_pose_to_sim(torch.cat([pos[sel], quat[sel]], 1), env_ids=ids[sel])
                    r.write_root_velocity_to_sim(
                        torch.zeros(len(sel), 6, device=self.device), env_ids=ids[sel]
                    )

    # ---------------------------------------------------------------- observation
    def interaction_state(self) -> torch.Tensor:
        """Contact-derived state per environment. / 접촉 결과에서 유도한 상태."""
        o = self.scene.env_origins
        grip = self.grip_point_w() - o
        ring = self.actor_pos_w() - o
        quat = self.actor_quat_w()
        d = grip - ring
        near = (
            (torch.norm(d[:, :2], dim=1) < self.cfg.attach_xy)
            & (d[:, 2] > self.cfg.attach_z[0])
            & (d[:, 2] < self.cfg.attach_z[1])
        )
        up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilted = up_z < math.cos(self.cfg.seated_tilt)
        on_goal_xy = torch.norm(ring[:, :2] - self.goal[:, :2], dim=1) <= self.cfg.seated_xy
        at_level = (ring[:, 2] - (self.goal[:, 2])).abs() < 0.003
        seated = self.released_ring & on_goal_xy & at_level & ~tilted
        jammed = self.released_ring & ~seated
        state = torch.full_like(self.actor, STATE_FREE)
        state = torch.where(near, torch.full_like(state, STATE_BETWEEN_FINGERS), state)
        state = torch.where(self.attached, torch.full_like(state, STATE_GRASPED), state)
        state = torch.where(seated, torch.full_like(state, STATE_SEATED), state)
        state = torch.where(jammed, torch.full_like(state, STATE_JAMMED), state)
        return state

    def rgb(self, augment: bool = True) -> torch.Tensor:
        """Canonical view in 0..1; appearance augmentation can be switched off for figures."""
        x = self.camera.data.output["rgb"][..., :3].permute(0, 3, 1, 2).float() / 255.0
        if not augment:
            return x
        if self.settings.robot_recolor > 0:
            # Real robot paint, cables and lighting differ from the twin: recolour the robot's pixels
            # (mostly dark, like the AG-95) keeping their shading. / 로봇 픽셀 색을 무작위로 바꾼다.
            m = self.robot_mask(force=True)
            if m is not None:
                n = x.shape[0]
                dark = torch.rand(n, 1, 1, 1, device=x.device) < 0.7
                colour = torch.where(
                    dark,
                    torch.rand(n, 1, 1, 1, device=x.device) * 0.25,
                    torch.rand(n, 3, 1, 1, device=x.device),
                ).expand(n, 3, 1, 1)
                shade = x.mean(1, keepdim=True)
                recoloured = (colour * (0.4 + 1.2 * shade) + 0.04 * torch.randn_like(x)).clamp(0.0, 1.0)
                pick = (torch.rand(n, 1, 1, 1, device=x.device) < self.settings.robot_recolor).float() * m
                x = pick * recoloured + (1.0 - pick) * x
        x = x * self.gain + self.bias
        r = self.settings.randomisation
        if r.blur > 0:
            # Mild defocus on a random subset (webcam autofocus, motion). / 일부 프레임에 약한 흐림.
            blurred = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
            pick = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < r.blur).float()
            x = pick * blurred + (1.0 - pick) * x
        if r.noise > 0:
            x = x + r.noise * torch.randn_like(x)
        return x.clamp(0.0, 1.0)

    def belief_render(self, grip_believed: torch.Tensor) -> torch.Tensor:
        """Believed grip point drawn into the canonical view, ``(N, 1, R, R)``. / 믿음 렌더 채널.

        Mirrors :mod:`factor.perception.belief_render`: the point the robot believes it holds,
        given in believed-anchor coordinates, is drawn as if those were true anchor coordinates,
        through the same virtual camera the canonical view uses.
        """
        from factor.perception import belief_render as br

        s = self.settings
        res = s.resolution
        dev = self.device
        t = torch.linspace(0.0, br.SEGMENT_HEIGHT, br.SEGMENT_POINTS, device=dev)
        up = t[None, :, None] * torch.tensor([0.0, 0.0, 1.0], device=dev)
        pts = grip_believed[:, None, :] + up
        if s.belief_fingers:
            # Same silhouette as factor.perception.belief_render: fingers at ±half-gap along anchor y.
            gap = torch.where(self.grip_cmd, self.close_gap, self.open_gap)
            side = torch.stack([torch.zeros_like(gap), gap, torch.zeros_like(gap)], 1)[:, None, :]
            pts = torch.cat([pts, pts + side, pts - side], 1)
        cam = self.cam_pos
        fwd = self.look_at - cam
        fwd = fwd / fwd.norm(dim=1, keepdim=True)
        down_z = torch.tensor([0.0, 0.0, -1.0], device=dev).expand_as(fwd)
        right = torch.cross(down_z, fwd, dim=1)
        right = right / right.norm(dim=1, keepdim=True)
        down = torch.cross(fwd, right, dim=1)
        rel = pts - cam[:, None, :]
        x = (rel * right[:, None, :]).sum(-1)
        y = (rel * down[:, None, :]).sum(-1)
        z = (rel * fwd[:, None, :]).sum(-1).clamp(min=1e-3)
        hfov = 2.0 * math.atan(s.view_width / (2.0 * float(np.linalg.norm(s.camera_in_anchor))))
        f = res / (2.0 * math.tan(hfov / 2.0))
        u = f * x / z + res / 2.0
        v = f * y / z + res / 2.0
        grid = torch.arange(res, device=dev, dtype=torch.float32)
        du = (grid[None, None, None, :] - u[:, :, None, None]) ** 2
        dv = (grid[None, None, :, None] - v[:, :, None, None]) ** 2
        d2 = (du + dv).amin(dim=1)
        return torch.exp(-d2 / (2.0 * br.SIGMA_PX**2)).unsqueeze(1)

    def robot_mask(self, force: bool = False) -> torch.Tensor | None:
        if not (self.settings.render_mask or force):
            return None
        seg = self.camera.data.output["semantic_segmentation"]
        if self.robot_class_id is None:
            info = self.camera.data.info
            if isinstance(info, (list, tuple)):
                info = info[0]
            labels = info["semantic_segmentation"]["idToLabels"]
            for key, label in labels.items():
                if isinstance(label, dict) and label.get("class") == "robot":
                    self.robot_class_id = int(key)
        if self.robot_class_id is None:
            return torch.zeros(self.num_envs, 1, seg.shape[1], seg.shape[2], device=self.device)
        return (seg[..., 0] == self.robot_class_id).float().unsqueeze(1)

    def _get_observations(self) -> dict:
        rgb = self.rgb()
        o = self.scene.env_origins
        grip = self.believed(self.grip_point_w() - o)
        channels = []
        if self.settings.render_belief:
            channels.append(self.belief_render(grip))
        robot = self.robot_mask()
        if robot is not None:
            channels.append(robot)
        mask = torch.cat(channels, 1) if channels else None
        self.last_frames = {
            "rgb": rgb,
            "render": mask,
            "correction": self.teacher_correction(),
            "correction_z": self.teacher_correction_z(),
            "ring_offset": self.ring_offset_label(),
            "state": self.interaction_state(),
            "grip": grip,
            "holding": self.grip_cmd.clone(),
            **self.overlay_geometry(),
        }
        obs = {"rgb": rgb}
        if mask is not None:
            obs["render"] = mask
        return {"policy": obs}

    def overlay_geometry(self) -> dict[str, torch.Tensor]:
        """Anchor-frame geometry for drawing overlays. / 오버레이용 앵커 프레임 기하.

        ``ring_true``/``goal_true`` are the simulator's truth; ``ring_est``/``goal_est`` are the
        nominal positions after the last applied correction, i.e. what the planner believes,
        mapped back into the true anchor frame so that both can be drawn on the same view.
        """
        s = self.settings
        o = self.scene.env_origins
        ring_true = self.to_anchor_axes(self.actor_pos_w() - o - self.anchor)
        n_below = (self.ring_peg == self.src.unsqueeze(0)).sum(0).float() - 1.0
        ring_nom = torch.cat(
            [self.peg_xy[self.src], (self.plate + self.ring_h * (n_below + 0.5)).unsqueeze(1)], 1
        )
        est_ring = apply_correction(self.nominal(ring_nom), self.last_estimate, s.pos_scale, s.yaw_scale)
        est_goal = apply_correction(
            self.nominal(self.goal.clone()), self.last_estimate, s.pos_scale, s.yaw_scale
        )

        def undo(p):
            c, s_ = torch.cos(self.belief_err[:, 2]), torch.sin(self.belief_err[:, 2])
            d = torch.stack([c * p[:, 0] - s_ * p[:, 1], s_ * p[:, 0] + c * p[:, 1], p[:, 2]], 1)
            err = torch.stack([self.belief_err[:, 0], self.belief_err[:, 1], self.belief_err[:, 3]], 1)
            return d + err

        return {
            "ring_true": ring_true,
            "goal_true": self.to_anchor_axes(self.goal - self.anchor),
            "ring_est": undo(est_ring),
            "goal_est": undo(est_goal),
            "ring_radius": self.radii[self.actor],
            "cam_pos": self.cam_pos.clone(),
            "look_at": self.look_at.clone(),
        }

    def skill_observation(self) -> torch.Tensor:
        """Low-dimensional state for the skill policy. / 스킬 정책용 저차원 상태.

        Mirrors :func:`factor.skills.residual.skill_features`: everything here is available on
        the robot (believed grip point, corrected targets, gripper command, search state).
        """
        from factor.skills.residual import skill_features

        c = self.skill_context
        if not c:
            o = self.scene.env_origins
            g = self.believed(self.grip_point_w() - o)
            c = {
                "grip": g,
                "object": g,
                "goal": g,
                "holding": self.grip_cmd,
                "inserting": torch.zeros_like(self.grip_cmd),
            }
        return skill_features(
            c["grip"],
            c["object"],
            c["goal"],
            c["holding"],
            c["inserting"],
            self.nudge.index,
            self.nudge.stall,
            self.residual,
            self.hover,
            self.zone_top,
            self.cfg.max_step,
        )

    def target_distance(self) -> torch.Tensor:
        """True distance from the grip point to the current target (reward shaping). / 참 목표 거리."""
        o = self.scene.env_origins
        grip = self.grip_point_w() - o
        ring = self.actor_pos_w() - o
        goal = self.goal.clone()
        target = torch.where(self.attached.unsqueeze(1), goal, ring)
        return torch.norm(grip - target, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        o = self.scene.env_origins
        ring = self.actor_pos_w() - o
        near_goal = torch.norm(ring - self.goal, dim=1) < 0.012
        self.succeeded = (~self.attached) & near_goal & (self.episode_length_buf > 5) & ~self.bad_release
        return self.succeeded.float() - 0.01

    def _get_dones(self):
        ring = self.actor_pos_w() - self.scene.env_origins
        d_axis = torch.cdist(ring[:, :2].unsqueeze(1), self.peg_xy.unsqueeze(0)).squeeze(1).min(dim=1).values
        dropped = (~self.attached) & ((ring[:, 2] < self.plate - 0.002) | (d_axis > 0.012))
        self.last_term_reason = {
            "succeeded": self.succeeded.clone(),
            "dropped": dropped.clone(),
            "bad_release": self.bad_release.clone(),
        }
        return (
            self.succeeded | dropped | self.bad_release,
            self.episode_length_buf >= self.max_episode_length - 1,
        )

    # ---------------------------------------------------------------- reset
    def _resample_belief(self, env_ids):
        r = self.settings.randomisation
        n = len(env_ids)
        e = torch.zeros(n, 4, device=self.device)
        e[:, :2] = sample_uniform(-r.anchor_xy, r.anchor_xy, (n, 2), self.device)
        e[:, 2] = sample_uniform(-r.anchor_yaw, r.anchor_yaw, (n,), self.device)
        if r.anchor_z > 0:
            e[:, 3] = sample_uniform(-r.anchor_z, r.anchor_z, (n,), self.device)
        self.belief_err[env_ids] = e
        self.fusion.reset(env_ids)
        self._place_cameras(env_ids)

    def _place_cameras(self, env_ids):
        s = self.settings
        r = s.randomisation
        n = len(env_ids)
        rel = torch.tensor(s.camera_in_anchor, device=self.device).repeat(n, 1)
        rel += sample_uniform(-r.camera_xyz, r.camera_xyz, (n, 3), self.device)
        self.cam_pos[env_ids] = rel
        self.aim_jit[env_ids] = sample_uniform(-r.aim_jitter, r.aim_jitter, (n, 2), self.device)
        self._aim(env_ids)
        self.gain[env_ids] = 1.0 + sample_uniform(-r.gain, r.gain, (n, 3, 1, 1), self.device)
        self.bias[env_ids] = sample_uniform(-r.bias, r.bias, (n, 1, 1, 1), self.device)

    def _aim(self, env_ids) -> None:
        """Centre the canonical view on the active peg (source while picking, goal while holding).

        The aim point is a nominal anchor-frame point, located through the *true* anchor as the
        fiducial detection does on the robot; the action is always in the middle of the view.
        캐노니컬 시야를 작업 중인 기둥(집기: 출발, 들고 있음: 목표)에 맞춘다. 실기의 마커 검출처럼
        참 앵커 기준의 공칭 점이다.
        """
        if len(env_ids) == 0:
            return
        o = self.scene.env_origins[env_ids]
        peg = torch.where(self.grip_cmd[env_ids], self.dst[env_ids], self.src[env_ids])
        look = torch.stack(
            [
                self.peg_x_anchor[peg] + self.aim_jit[env_ids, 0],
                torch.full_like(self.aim_jit[env_ids, 1], self.peg_y_anchor) + self.aim_jit[env_ids, 1],
                torch.full_like(self.aim_jit[env_ids, 0], self.settings.look_height),
            ],
            1,
        )
        self.camera.set_world_poses_from_view(
            o + self.anchor_point(self.cam_pos[env_ids]), o + self.anchor_point(look), env_ids=env_ids
        )
        self.look_at[env_ids] = look
        self.view_peg[env_ids] = peg

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        s = self.settings
        r = s.randomisation
        n = len(env_ids)
        dev = self.device
        nr = self.n_rings

        src = torch.randint(0, 3, (n,), device=dev)
        dst = (src + torch.randint(1, 3, (n,), device=dev)) % 3
        aux = 3 - src - dst
        actor = torch.randint(0, nr, (n,), device=dev)
        peg = torch.randint(0, 3, (nr, n), device=dev)
        j = torch.arange(nr, device=dev).unsqueeze(1)
        peg = torch.where(j > actor.unsqueeze(0), aux.unsqueeze(0), peg)
        peg = torch.where(j == actor.unsqueeze(0), src.unsqueeze(0), peg)
        self.src[env_ids], self.dst[env_ids], self.actor[env_ids], self.ring_peg[:, env_ids] = (
            src,
            dst,
            actor,
            peg,
        )
        self.close_gap[env_ids] = self.radii[actor] + s.finger_clearance
        self.open_gap[env_ids] = torch.clamp(self.radii[actor] + s.open_margin, max=s.finger_max)

        same = peg.unsqueeze(1) == peg.unsqueeze(0)
        lower = (
            torch.arange(nr, device=dev).view(-1, 1) < torch.arange(nr, device=dev).view(1, -1)
        ).unsqueeze(-1)
        level = (same & lower).sum(0)
        n_src = (peg == src.unsqueeze(0)).sum(0).float()
        n_dst = (peg == dst.unsqueeze(0)).sum(0).float()
        self.goal[env_ids] = torch.cat(
            [self.peg_xy[dst], (self.plate + self.ring_h * (n_dst + 0.5)).unsqueeze(1)], 1
        )
        self.start[env_ids] = torch.cat(
            [self.peg_xy[src], (self.plate + self.ring_h * (n_src - 0.5)).unsqueeze(1)], 1
        )
        init_peg = torch.randint(0, 3, (n,), device=dev)
        self.tgt[env_ids] = torch.cat([self.peg_xy[init_peg], torch.full((n, 1), self.hover, device=dev)], 1)
        self.tgt[env_ids] += sample_uniform(-0.01, 0.01, (n, 3), dev)

        bo = torch.zeros(n, 3, device=dev)
        bo[:, :2] = sample_uniform(-r.base_xy, r.base_xy, (n, 2), dev)
        bo[:, 2] = sample_uniform(-r.base_yaw, r.base_yaw, (n,), dev)
        self.base_off[env_ids] = bo
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = self.scene.env_origins[env_ids, :2] + bo[:, :2]
        pose[:, 2] = self.scene.env_origins[env_ids, 2] + s.base_z
        pose[:, 3] = torch.cos(bo[:, 2] / 2)
        pose[:, 6] = torch.sin(bo[:, 2] / 2)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)

        mix = torch.tensor(s.start_mix, device=dev)
        stage = torch.multinomial(mix / mix.sum(), n, replacement=True)
        pre, held = stage == 1, stage == 2
        if pre.any():
            off = sample_uniform(-0.004, 0.004, (n, 3), dev)
            off[:, 2] = sample_uniform(0.0, 0.03, (n,), dev)
            self.tgt[env_ids] = torch.where(pre.unsqueeze(1), self.start[env_ids] + off, self.tgt[env_ids])
        if held.any():
            travel = torch.rand(n, device=dev) < 0.5
            row = max(abs(float(self.layout.peg_xy(p)[0])) for p in PEGS) + 0.01
            p_tr = self.anchor_point(
                torch.stack(
                    [
                        sample_uniform(-row, row, (n,), dev),
                        sample_uniform(-0.01, 0.01, (n,), dev),
                        sample_uniform(self.zone_top + 0.005, self.hover, (n,), dev),
                    ],
                    1,
                )
            )
            gl = self.goal[env_ids]
            p_dn = torch.stack(
                [
                    gl[:, 0] + sample_uniform(-0.001, 0.001, (n,), dev),
                    gl[:, 1] + sample_uniform(-0.001, 0.001, (n,), dev),
                    sample_uniform(0.0, 1.0, (n,), dev)
                    * (self.hover - gl[:, 2] - self.layout.place_clearance)
                    + gl[:, 2]
                    + self.layout.place_clearance,
                ],
                1,
            )
            self.tgt[env_ids] = torch.where(
                held.unsqueeze(1), torch.where(travel.unsqueeze(1), p_tr, p_dn), self.tgt[env_ids]
            )

        local = torch.zeros(n, 3, device=dev)
        along = sample_uniform(-s.in_hand_offset, s.in_hand_offset, (n,), dev)  # kept along the finger axis
        across = sample_uniform(-s.finger_clearance, s.finger_clearance, (n,), dev)  # centred across it
        local[:, 0], local[:, 1] = (across, along) if s.finger_axis == "y" else (along, across)
        self.hand_offset[env_ids] = torch.where(
            held.unsqueeze(1), self.rot_env(self.to_env_axes(local), env_ids), 0.0
        )

        poses = torch.eye(4, device=dev, dtype=torch.float64).repeat(n, 1, 1)
        poses[:, :3, :3] = self.tool_rot
        poses[:, :3, 3] = self.to_base(self.tgt + self.tip_off)[env_ids].double()
        q, valid = self.kin.inverse_batch(poses, s.ik_branch)
        jp = torch.where(valid.unsqueeze(1), q.float(), self.home[env_ids]) + sample_uniform(
            -0.005, 0.005, (n, 6), dev
        )
        full = self.robot.data.default_joint_pos[env_ids].clone()
        full[:, self.joint_ids] = jp
        gap = torch.where(held, self.close_gap[env_ids], self.open_gap[env_ids])
        full[:, self.finger_ids] = gap.unsqueeze(1).repeat(1, 2)
        self.robot.write_joint_state_to_sim(full, torch.zeros_like(full), env_ids=env_ids)
        self.robot.set_joint_position_target(full, env_ids=env_ids)
        self.targets[env_ids] = jp

        for i, ring in enumerate(self.ring_objs):
            st = ring.data.default_root_state[env_ids].clone()
            st[:, :2] = self.peg_xy[peg[i]]
            st[:, 2] = self.plate + self.ring_h * (level[i].float() + 0.5)
            st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            is_actor = (actor == i) & held
            if is_actor.any():
                st[:, :3] = torch.where(
                    is_actor.unsqueeze(1), self.tgt[env_ids] + self.hand_offset[env_ids], st[:, :3]
                )
            st[:, :3] += self.scene.env_origins[env_ids]
            ring.write_root_pose_to_sim(st[:, :7], env_ids=env_ids)
            ring.write_root_velocity_to_sim(st[:, 7:], env_ids=env_ids)

        self.attached[env_ids] = held
        self.grip_cmd[env_ids] = held
        self.succeeded[env_ids] = False
        self.bad_release[env_ids] = False
        self.released_ring[env_ids] = False
        self.residual[env_ids] = 0.0
        self.nudge.reset(env_ids)
        self._resample_belief(env_ids)
