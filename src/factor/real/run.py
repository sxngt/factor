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


class RealLoop:
    """Closed-loop pick-and-place on the robot. / 실기 폐루프 집기·놓기.

    ``frame_source`` returns the latest RGB frame; ``camera_in_base`` is the calibrated camera
    pose; ``anchor_in_base`` the believed anchor pose that the estimator corrects.
    """

    def __init__(
        self,
        scene: SceneSpec,
        robot: Robot,
        gripper: Gripper,
        frame_source,
        estimator: EstimatorClient,
        workspace: WorkspaceBox,
        config: LoopConfig,
        virtual: VirtualCamera,
        anchor_in_base: np.ndarray,
        residual=None,
        telemetry=None,
    ):
        self.scene = scene
        self.robot = robot
        self.gripper = gripper
        self.frames = frame_source
        self.estimator = estimator
        self.workspace = workspace
        self.config = config
        self.anchor_in_base = np.asarray(anchor_in_base, dtype=float)
        self.detector = FiducialDetector(scene.fiducials, scene.camera)
        self.view = CanonicalView(scene.camera.matrix, scene.camera.distortion, virtual)
        self.look_at = np.array([0.0, 0.0, 0.037])
        self.anchor_in_camera: np.ndarray | None = None  # detected board pose: anchors the view
        planner = FunnelPlanner(config.hover_height, config.zone_top, 0.01 * config.step_scale)
        self.skill = PickPlaceSkill(
            planner,
            self.anchor_in_base[:3, 3],
            CorrectionFilter(0.9, 0.012, config.hover_height),
            NudgeSearch(stall_steps=config.nudge_stall_steps, progress=config.nudge_progress)
            if config.nudge
            else None,
            residual,
        )
        self.log: list[dict] = []
        from factor.perception.session import SessionEstimator

        self.session = SessionEstimator()
        self.recorder = None
        if config.record_dir:
            from factor.real.recorder import MoveRecorder

            self.recorder = MoveRecorder(config.record_dir)
        self.telemetry = telemetry  # optional live sink (factor.real.telemetry) / 라이브 텔레메트리
        self.base_in_anchor = np.linalg.inv(self.anchor_in_base)
        self._last_frame = None
        self._last_view = None
        self._last_detection = None

    def observe(self, step: int, grip_in_anchor: np.ndarray | None = None) -> np.ndarray:
        """Canonical view (uint8, HWC) for the estimator, plus the belief render channel when the
        estimator expects one. / 추정기용 정규화 시야 (+ 믿음 렌더 채널)."""
        rgb = self.frames()
        if self.anchor_in_camera is None or step % self.config.detect_every == 0:
            det = self.detector.detect(rgb)
            attempts = 0
            while (det is None or det.reprojection_rms_px >= 3.0) and self.anchor_in_camera is None:
                # No pose yet: keep trying fresh frames before giving up. / 첫 포즈는 몇 프레임 더 시도.
                attempts += 1
                if attempts >= self.config.detect_attempts:
                    raise RuntimeError("anchor board not detected")
                time.sleep(0.1)
                rgb = self.frames()
                det = self.detector.detect(rgb)
            if det is not None and det.reprojection_rms_px < 3.0:
                self.anchor_in_camera = det.anchor_in_camera
                self._last_detection = det
        self._last_frame = rgb
        self._last_view = self.view.warp(rgb, self.anchor_in_camera, self.look_at)
        if self.estimator.channels > 3 and grip_in_anchor is not None:
            from factor.perception import belief_render
            from factor.perception.overlay import canonical_projector

            camera_in_anchor = np.linalg.inv(self.anchor_in_camera)
            project = canonical_projector(camera_in_anchor[:3, 3], self.look_at, self.view.virtual)
            half_gap = self.gripper.half_gap() if getattr(self.estimator, "belief_fingers", False) else None
            pixels = project(belief_render.segment_points(grip_in_anchor, half_gap))
            channel = belief_render.render(pixels, self.view.virtual.resolution)
            self._last_belief = channel
            return np.dstack([self._last_view, np.round(channel * 255.0).astype(np.uint8)])
        return self._last_view

    # ------------------------------------------------------------------ session estimation
    def _session_key(self, centre_base: np.ndarray) -> str:
        """Fusion key: one per peg for local-label estimators, one per anchor otherwise."""
        if getattr(self.estimator, "label", "global") != "local":
            return "anchor"
        c = self.base_in_anchor[:3, :3] @ centre_base + self.base_in_anchor[:3, 3]
        return f"peg{c[0]:+.3f}"

    def _goto(self, point: np.ndarray, timeout: float = 6.0) -> bool:
        """Stream to a base-frame point with the bounded lead; True once within 1 mm."""
        dt = 1.0 / self.config.rate_hz
        t0 = time.time()
        while time.time() - t0 < timeout:
            grip = self.robot.grip_point()
            if np.linalg.norm(point - grip) < 0.001:
                return True
            target = grip + np.clip(point - grip, -self.config.lead, self.config.lead)
            q = self.robot.solve_ik(self.workspace.clamp(target)[0])
            if q is not None:
                self.robot.stream_target(q)
            time.sleep(dt)
        return False

    def probe(self, centre_base: np.ndarray) -> None:
        """Look at one peg from several hover poses and feed the session estimator.

        한 기둥을 여러 hover 자세에서 보며 세션 추정기에 추정을 쌓는다.
        """
        key = self._session_key(centre_base)
        c = self.base_in_anchor[:3, :3] @ centre_base + self.base_in_anchor[:3, 3]
        self.look_at = np.array([c[0], c[1], 0.037])
        if self.telemetry is not None:
            self.telemetry.look_at = self.look_at
        for dx, dy in self.config.probe_offsets:
            p = (
                self.anchor_in_base[:3, :3] @ np.array([c[0] + dx, c[1] + dy, 0.0])
                + self.anchor_in_base[:3, 3]
            )
            p[2] = self.config.hover_height
            self._goto(p)
            time.sleep(0.3)  # let the arm settle / 정지 대기
            for _ in range(self.config.probe_frames):
                grip = self.robot.grip_point()
                grip_anchor = self.base_in_anchor[:3, :3] @ grip + self.base_in_anchor[:3, 3]
                est = self.estimator.estimate(self.observe(len(self.log), grip_anchor), grip_anchor)
                self.session.add(key, est.correction, est.sigma, est.spread)
                time.sleep(1.0 / self.config.rate_hz)
        fused = self.session.fused(key)
        print(
            f"[factor-run] probe {key}: correction {np.round(fused.correction[:2] * 1e3, 2)} mm, "
            f"stderr {fused.stderr * 1e3:.2f} mm over {fused.count} frames",
            flush=True,
        )

    def ensure_confident(self, centre_base: np.ndarray, limit: float | None = None) -> bool:
        """Probe until the fused estimate for this peg is precise enough. / 충분히 확실할 때까지 탐침."""
        limit = self.config.max_stderr if limit is None else limit
        key = self._session_key(centre_base)
        for _ in range(self.config.probe_rounds):
            fused = self.session.fused(key)
            if fused is not None and fused.stderr <= limit:
                return True
            self.probe(centre_base)
        fused = self.session.fused(key)
        return fused is not None and fused.stderr <= limit

    def _correction_in_base(self, correction: np.ndarray) -> np.ndarray:
        """Estimator correction (anchor axes) -> base axes for the skill. / 앵커 축 보정 → 베이스 축."""
        rot = self.anchor_in_base[:2, :2]
        dx, dy = rot @ np.asarray(correction[:2], dtype=float)
        out = [dx, dy, float(correction[2])]
        if len(correction) > 3:
            out.append(float(correction[3]))  # height is shared by anchor and base / 높이는 공통
        return np.array(out)

    def _to_anchor(self, point_base: np.ndarray) -> list[float]:
        p = self.base_in_anchor[:3, :3] @ np.asarray(point_base, dtype=float) + self.base_in_anchor[:3, 3]
        return [round(float(v), 4) for v in p]

    def _publish(
        self, step: int, grip, est, cmd, holding: bool, target, clamped: bool, phase: str, event: str = ""
    ) -> None:
        """Hand one tick to the telemetry sink. / 한 틱을 텔레메트리로 보낸다."""
        if self.telemetry is None:
            return
        fused = self.skill.fusion.value if self.skill.fusion is not None else None
        nudge = self.skill.nudge
        det = self._last_detection
        tick = {
            "step": step,
            "phase": phase,
            "event": event,
            "fiducial": [len(det.marker_ids), round(det.reprojection_rms_px, 2)] if det is not None else None,
            "anchor_in_camera_t": [round(float(v), 4) for v in self.anchor_in_camera[:3, 3]],
            "correction_mm_deg": [
                est.correction[0] * 1e3,
                est.correction[1] * 1e3,
                np.degrees(est.correction[2]),
            ],
            "fused_mm_deg": None if fused is None else [fused[0] * 1e3, fused[1] * 1e3, np.degrees(fused[2])],
            "grasped": float(est.grasped),
            "spread_mm": float(est.spread) * 1e3,
            "offset_mm": [float(v) * 1e3 for v in est.offset[:2]],
            "holding": bool(holding),
            "inserting": bool(holding and grip[2] < self.config.zone_top + 0.016),
            "nudge": [int(nudge.index), int(nudge.stall)] if nudge is not None else None,
            "retry": bool(nudge.retry) if nudge is not None else False,
            "grip": [round(float(v), 4) for v in grip],
            "grip_anchor": self._to_anchor(grip),
            "step_mm": [float(v) * 1e3 for v in cmd.step],
            "target": [round(float(v), 4) for v in target],
            "clamped": bool(clamped),
            "object_corrected_anchor": self._to_anchor(cmd.object_corrected),
            "goal_corrected_anchor": self._to_anchor(cmd.goal_corrected),
        }
        view = self._last_view
        belief = getattr(self, "_last_belief", None)
        if view is not None and belief is not None:
            # Show the belief channel the estimator sees, in magenta. / 추정기가 보는 믿음 채널(자홍).
            alpha = belief[..., None] * 0.8
            view = (view.astype(np.float32) * (1.0 - alpha) + np.array([255.0, 0.0, 255.0]) * alpha).astype(
                np.uint8
            )
        self.telemetry.publish(
            tick,
            self._last_frame,
            view,
            self.anchor_in_camera,
            None if det is None else det.corners,
        )

    def run_move(self, move: MoveSpec, holding: bool = False) -> str:
        """Execute one pick-and-place. Returns the termination reason. / 한 수 실행.

        ``holding`` resumes a move whose object is already in the gripper (recovery after an
        interrupted run). / 물체를 이미 들고 있는 상태에서 재개할 때 ``holding=True``.
        """
        cfg = self.config
        dt = 1.0 / cfg.rate_hz
        self.skill.reset()
        if not holding:
            self.gripper.command(move.open_half_gap, blocking=True)
            time.sleep(cfg.grip_wait_s)
        self.robot.start_streaming()
        if cfg.probe:
            # Session estimate first: act on a fused, gated correction, not on single frames.
            # 단일 프레임이 아니라 융합·검증된 보정으로 행동한다.
            # The pick tolerates more than the insertion. / 집기는 삽입보다 허용 오차가 크다.
            checks = ([] if holding else [(move.object_center, cfg.max_stderr_pick)]) + [
                (move.goal_center, cfg.max_stderr)
            ]
            for centre, limit in checks:
                if not self.ensure_confident(centre, limit):
                    self.robot.stop_streaming()
                    return "uncertain"
        was_holding = holding
        if self.recorder is not None:
            self.recorder.reset()
        grasp_z = float(self.robot.grip_point()[2])
        verdicts: list[float] = []
        regrasps = 0
        hold_until = 0.0
        target = self.robot.grip_point().copy()
        last_grip = target.copy()
        stalled = 0
        reason = "max_steps"
        try:
            t0 = time.time()
            for step in range(cfg.max_steps):
                tick = t0 + step * dt
                grip = self.robot.grip_point()
                grip_anchor = self.base_in_anchor[:3, :3] @ grip + self.base_in_anchor[:3, 3]
                # Foveate on the active peg, as in the twin. / 트윈처럼 작업 중인 기둥에 시야를 맞춘다.
                centre = move.goal_center if holding else move.object_center
                c = self.base_in_anchor[:3, :3] @ centre + self.base_in_anchor[:3, 3]
                self.look_at = np.array([c[0], c[1], 0.037])
                if self.telemetry is not None:
                    self.telemetry.look_at = self.look_at
                image = self.observe(step, grip_anchor)
                est = self.estimator.estimate(image, grip_anchor)
                if self.recorder is not None:
                    self.recorder.frame(image, grip_anchor, holding)
                correction = est.correction
                if cfg.probe:
                    fused = self.session.fused(self._session_key(centre))
                    if fused is not None:
                        correction = fused.correction
                obs = SkillObservation(
                    grip,
                    move.object_center,
                    move.goal_center + np.asarray(cfg.place_offset, dtype=float),
                    holding,
                    self._correction_in_base(correction if cfg.use_correction else np.zeros(3)),
                    est.offset if cfg.use_offset_estimate else None,
                )
                cmd = self.skill.step(obs)
                if time.time() >= hold_until:
                    # Command the planner's own target point, at most ``lead`` ahead of the
                    # measured grip point: the robot moves continuously, the command never
                    # overshoots the point, and a stopped servo cannot make it run away.
                    # 플래너 목표점을 실측 그립점에서 최대 lead 이내로 명령한다 (연속 동작, 지나침·폭주 없음).
                    desired, clamped = self.workspace.clamp(cmd.target)
                    # While inserting, lead less in z so the held object is not pressed onto the
                    # peg while the contact search moves it. / 삽입 중엔 z 선행을 줄여 누르지 않는다.
                    lead = np.array([cfg.lead, cfg.lead, cfg.insert_lead if holding else cfg.lead])
                    target = grip + np.clip(desired - grip, -lead, lead)
                else:
                    clamped = False
                event = ""
                if np.linalg.norm(grip - last_grip) < 0.0005 and np.linalg.norm(target - grip) > 0.002:
                    stalled += 1
                else:
                    stalled = 0
                last_grip = grip.copy()
                if stalled >= cfg.stall_restart_steps:
                    # The driver ends the servo stream on its own (safety trip, stale target);
                    # restart it rather than streaming into the void. / 서보 스트림이 끊겼으면 재시작.
                    self.robot.stop_streaming()
                    self.robot.start_streaming()
                    stalled = 0
                    event = "servo restarted"
                if cmd.close != holding:
                    holding = cmd.close
                    hold_until = time.time() + cfg.grip_wait_s
                    self.gripper.command(move.grasp_half_gap if holding else move.open_half_gap)
                    if holding:
                        was_holding = True
                        grasp_z = float(grip[2])
                        verdicts = []
                        if self.recorder is not None:
                            self.recorder.grasped(grip_anchor)
                            # No vision check: trust the grasp (saving still needs a seated release).
                            # 비전 확인을 끄면 파지를 믿는다 (저장은 여전히 안착된 놓기일 때만).
                            if not cfg.verify_grasp:
                                self.recorder.verified = True
                    elif self.recorder is not None:
                        self.recorder.released(grip_anchor)
                # Grasp verification from the estimator's state head: once the object should be
                # clear of its stack, a run of "not grasped" verdicts means the fingers closed on
                # nothing. Reopen and let the funnel pick again. / 비전 파지 판정으로 빈손 재시도.
                if holding and cfg.verify_grasp and grip[2] > grasp_z + cfg.verify_rise:
                    verdicts.append(float(est.grasped))
                    if len(verdicts) >= cfg.verify_frames:
                        if (
                            float(np.mean(verdicts[-cfg.verify_frames :])) < 0.5
                            and regrasps < cfg.max_regrasps
                        ):
                            regrasps += 1
                            event = f"empty grasp, retry {regrasps}"
                            holding = False
                            was_holding = False  # a retry, not a release / 재시도이지 놓기가 아님
                            if self.recorder is not None:
                                self.recorder.grasp_point = None  # the grasp was empty / 빈 파지
                            hold_until = time.time() + cfg.grip_wait_s
                            self.gripper.command(move.open_half_gap)
                            self.skill.reset()
                            verdicts = []
                        elif float(np.mean(verdicts[-cfg.verify_frames :])) >= 0.5:
                            verdicts = verdicts[-cfg.verify_frames :]
                            if self.recorder is not None:
                                self.recorder.verified = True  # the vision check saw the object held
                q = self.robot.solve_ik(target)
                if q is not None:
                    self.robot.stream_target(q)
                self.log.append(
                    {
                        "step": step,
                        "grip": grip.tolist(),
                        "correction": est.correction.tolist(),
                        "grasped": est.grasped,
                        "regrasps": regrasps,
                        "offset": est.offset.tolist(),
                        "holding": bool(holding),
                        "clamped": bool(clamped),
                    }
                )
                phase = "place" if holding else ("released" if was_holding else "pick")
                self._publish(step, grip, est, cmd, holding, target, clamped, phase, event)
                if was_holding and not holding and time.time() >= hold_until:
                    reason = "released"
                    if self.recorder is not None:
                        # Seated when released at the goal height (the ring went down the peg).
                        # 목표 높이에서 놓였으면 안착 (링이 기둥으로 내려감).
                        goal_anchor = (
                            self.base_in_anchor[:3, :3] @ move.goal_center + self.base_in_anchor[:3, 3]
                        )
                        obj_anchor = (
                            self.base_in_anchor[:3, :3] @ move.object_center + self.base_in_anchor[:3, 3]
                        )
                        rel = self.recorder.release_point
                        seated = rel is not None and abs(rel[2] - goal_anchor[2]) < cfg.seated_tol
                        path = self.recorder.save(obj_anchor, goal_anchor, seated)
                        print(f"[factor-run] recorded {path} (seated {seated})", flush=True)
                    break
                lag = tick + dt - time.time()
                if lag > 0:
                    time.sleep(lag)
            # Back to the hover height before the next move, same bounded-lead command as the loop.
            # 다음 수 전에 hover 높이로 복귀 (루프와 같은 선행 제한 명령).
            for _ in range(int(15.0 * cfg.rate_hz)):
                grip = self.robot.grip_point()
                if abs(grip[2] - cfg.hover_height) < 0.002:
                    break
                target = grip.copy()
                target[2] += float(np.clip(cfg.hover_height - grip[2], -cfg.lead, cfg.lead))
                q = self.robot.solve_ik(target)
                if q is not None:
                    self.robot.stream_target(q)
                time.sleep(dt)
        finally:
            self.robot.stop_streaming()
        return reason
