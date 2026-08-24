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


