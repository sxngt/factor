"""Real-robot execution loop. / 실기 실행 루프.

One tick of the loop, at a fixed rate:

1. grab a camera frame, detect the anchor board, warp the canonical view;
2. ask the estimator for the anchor correction (and the grasped probability);
3. feed the skill the grip point, nominal object and goal positions and the correction;
4. clamp the commanded grip point into the workspace box, solve the grasp-posture IK and
   stream the joint target; command the gripper on state changes.

The loop needs only NumPy (plus OpenCV for the warp); PyTorch runs in the estimator server.

고정 주기로 프레임 → 보드 검출 → 정규화 시야 → 추정 → 스킬 → 작업공간 클램프 → IK → 서보 스트림.
torch 는 추정 서버 프로세스에만 있다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from factor.perception.canonical import CanonicalView, VirtualCamera
from factor.perception.client import EstimatorClient
from factor.perception.fusion import CorrectionFilter
from factor.planning.funnel import FunnelPlanner
from factor.planning.nudge import NudgeSearch
from factor.real.fiducial import FiducialDetector
from factor.real.robot import Gripper, Robot
from factor.real.safety import WorkspaceBox
from factor.scene.spec import SceneSpec
from factor.skills.pick_place import PickPlaceSkill, SkillObservation


@dataclass
class MoveSpec:
    """Nominal geometry of one pick-and-place in the base frame. / 한 수의 공칭 기하(베이스 프레임)."""

    object_center: np.ndarray
    goal_center: np.ndarray
    grasp_half_gap: float
    open_half_gap: float


@dataclass
class LoopConfig:
    rate_hz: float = 10.0
    max_steps: int = 400
    step_scale: float = 0.5
    grip_wait_s: float = 1.0
    detect_every: int = 20
    hover_height: float = 0.0
    zone_top: float = 0.0
    use_offset_estimate: bool = False  # place at goal minus the estimated in-hand offset / 오프셋 추정 사용
    stall_restart_steps: int = 20  # ticks without motion before the servo stream is restarted / 서보 재시작
    lead: float = 0.012  # metres the commanded point may lead the measured grip point (sets the speed)
    insert_lead: float = 0.004  # z lead while holding (gentle contact during insertion) / 삽입 중 z 선행
    place_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base-frame shift of every goal (manual trim)
    verify_grasp: bool = True  # use the estimator's grasp verdict to catch empty grasps / 비전 파지 검증
    verify_rise: float = 0.015  # metres above the grasp height before the verdict counts
    verify_frames: int = 5  # consecutive frames averaged for the verdict
    max_regrasps: int = 3
    probe: bool = True  # estimate the belief error per session by probing before acting / 능동 탐침
    probe_offsets: tuple = ((0.0, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006))
    probe_frames: int = 6  # frames per probe pose / 자세당 프레임
    probe_rounds: int = 2  # probe passes before giving up on a peg / 기둥당 최대 탐침 횟수
    max_stderr: float = 0.0008  # refuse to place above this fused standard error (m) / 놓기 허용 표준오차
    max_stderr_pick: float = 0.002  # picking tolerates more: the open fingers capture ±6 mm / 집기 허용
    use_correction: bool = True  # False: nominal geometry, estimator only observed / 보정 미적용
    record_dir: str = ""  # hindsight-labelled real data (factor.real.recorder) / 실기 자기 라벨 데이터
    seated_tol: float = 0.003  # release this close to the goal height counts as seated (m)
    # The real arm creeps near the goal (~0.3 mm per tick at 10 Hz) because the streamed target
    # is only a few mm ahead; a stall means almost no motion for a while, not "slower than sim".
    # 실기 팔은 목표 근처에서 틱당 ~0.3 mm 로 천천히 내려간다: 정체 = 한동안 거의 정지.
    # A position-servoed arm does not stall on contact, so the stall signal is meaningless on the
    # 위치 서보 팔은 접촉해도 멈추지 않아 정체 신호가 의미 없다: 기본 끔.
    nudge: bool = False
    nudge_progress: float = 0.0001
    nudge_stall_steps: int = 6
    detect_attempts: int = 20  # frames to try for the first board detection / 첫 보드 검출 시도 횟수


