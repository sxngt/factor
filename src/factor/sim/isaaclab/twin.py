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


