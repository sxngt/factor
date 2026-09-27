"""``factor-selfcalib``: one-minute active self-calibration on the robot.

The robot shows the gripper board to the fixed camera from poses it picks itself. After every
pose the camera-in-base and board-on-flange estimate is re-solved, and the next pose is the one
with the largest predicted drop in peg-point uncertainty per second of motion. It stops when the
peg points are known to ``--target-mm`` or the budget runs out, then writes the anchor pose in the
robot base frame (``X · T_CA``) for ``factor-run --calib``.

로봇이 스스로 고른 자세들로 그리퍼 보드를 고정 카메라에 보여 준다. 자세마다 카메라-베이스·보드-플랜지
추정을 다시 풀고, "이동 1초당 기둥 점 불확실성 감소"가 가장 큰 자세로 간다. 기둥 점이 ``--target-mm``
안으로 들어오거나 시간 예산이 끝나면 멈추고, 베이스 기준 앵커 포즈(X·T_CA)를 ``factor-run --calib``
용으로 저장한다.

    factor-selfcalib --scene configs/hanoi.yaml --budget 60 --out logs/factor/selfcalib/latest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from factor.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    average_poses,
    choose_next,
    move_time,
    pose_error,
    project,
)
from factor.calib.poses import generate, near_within_limits, path_clear, search_poses
from factor.calib.simulate import peg_points


# ----------------------------------------------------------------------------- camera
def read_frame(cam_dir: str, newer_than: float = 0.0, timeout: float = 3.0):
    """Latest frame written after ``newer_than`` (camera-server clock): (rgb, t, n).

    카메라 서버가 ``newer_than`` 이후에 쓴 최신 프레임.
    """
    import cv2

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(os.path.join(cam_dir, "latest.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if float(meta.get("t", 0)) <= newer_than:
                time.sleep(0.03)
                continue
            with open(os.path.join(cam_dir, "latest_color.png"), "rb") as f:
                data = f.read()
            if meta.get("color_sha256") and hashlib.sha256(data).hexdigest() != meta["color_sha256"]:
                time.sleep(0.02)
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), float(meta["t"]), int(meta.get("n", 0))
        except (OSError, ValueError, KeyError):
            time.sleep(0.02)
    raise RuntimeError("no fresh camera frame")


class CameraSensor:
    """Frames from the camera server, both boards detected per frame. / 카메라 서버 프레임 + 두 보드 검출."""

    def __init__(self, cam_dir: str, board_det, mat_det):
        self.cam_dir, self.board_det, self.mat_det = cam_dir, board_det, mat_det

    def grab(self, newer_than: float = 0.0):
        rgb, t, _n = read_frame(self.cam_dir, newer_than)
        self.last_rgb = rgb
        board = self.board_det.detect(rgb) if self.board_det is not None else None
        return t, board, self.mat_det.detect(rgb)

    def leds(self, roi=None) -> np.ndarray:
        from factor.calib.ring import detect_leds

        return detect_leds(self.last_rgb, roi)


class FakeSensor:
    """Synthetic camera over :class:`factor.calib.simulate.Rig` at the real frame rate.

    실제 프레임 속도로 합성 검출을 돌려주는 가짜 카메라 (러너 전체 시험용).
    """

    def __init__(self, rig, robot, board: GripperBoard, period: float = 0.375):
        import cv2

        self.rig, self.robot, self.board, self.period, self.cv2 = rig, robot, board, period, cv2
        self.t = time.time()
        self.mat_ids = tuple(m.marker_id for m in rig.scene.fiducials.markers)

    def grab(self, newer_than: float = 0.0):
        from factor.real.fiducial import FiducialDetection

        due = max(newer_than, self.t) + self.period
        time.sleep(max(0.0, due - time.time()))
        self.t = due
        q = self.robot.joint_positions()
        board = None
        hit = self.rig.detect_board(q, self.rig.fk_reading(q))
        if hit is not None:
            ids, uv = hit
            obj = self.board.object_points(ids)
            _, rvec, tvec = self.cv2.solvePnP(
                obj, uv, self.rig.k, self.rig.dist, flags=self.cv2.SOLVEPNP_IPPE
            )
            pose = np.eye(4)
            pose[:3, :3] = self.cv2.Rodrigues(rvec)[0]
            pose[:3, 3] = tvec.ravel()
            board = FiducialDetection(pose, tuple(ids), 0.3, tuple(uv.reshape(-1, 4, 2)))
        mat = FiducialDetection(self.rig.detect_mat(), self.mat_ids, 0.3, ())
        self._q = q
        return due, board, mat

    def leds(self, roi=None) -> np.ndarray:
        return self.rig.detect_leds(self._q, self.rig.fk_reading(self._q))


def merge_points(sets, radius: float = 8.0) -> np.ndarray:
    """Average blob centroids that repeat across frames of one pose. / 프레임 간 같은 점 평균."""
    pts = [p for s in sets for p in np.asarray(s).reshape(-1, 2)]
    groups: list[list] = []
    for p in pts:
        for g in groups:
            if np.linalg.norm(np.mean(g, axis=0) - p) < radius:
                g.append(p)
                break
        else:
            groups.append([p])
    need = max(1, len(sets) // 2)
    return np.array([np.mean(g, axis=0) for g in groups if len(g) >= need]).reshape(-1, 2)


def board_layout(board: GripperBoard):
    from factor.scene.spec import FiducialLayout, FiducialMarker

    return FiducialLayout(
        board.dictionary, tuple(FiducialMarker(i, (*board.centers[i], 0.0), board.size) for i in board.ids)
    )


# ----------------------------------------------------------------------------- motion
class Mover:
    """Smooth joint-space motion over the servo stream. / 서보 스트림 위의 부드러운 관절 이동."""

    def __init__(self, robot, rate: float = 30.0):
        self.robot = robot
        self.dt = 1.0 / rate
        self.target = robot.joint_positions()

    def hold(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.robot.stream_target(self.target)
            time.sleep(self.dt)

    def move(self, q1: np.ndarray) -> float:
        """Minimum-jerk path to ``q1``; returns the planned duration. / 최소 저크 경로."""
        q0 = self.target.copy()
        q1 = np.asarray(q1, dtype=float)
        dur = move_time(q1 - q0)
        t0 = time.time()
        while True:
            s = min(1.0, (time.time() - t0) / dur)
            b = s * s * s * (10 - 15 * s + 6 * s * s)
            self.target = q0 + (q1 - q0) * b
            self.robot.stream_target(self.target)
            if s >= 1.0:
                break
            time.sleep(self.dt)
        return dur

    def settle(self, tol_deg: float = 0.05, still_s: float = 0.25, timeout: float = 2.0) -> np.ndarray:
        """Hold until the measured joints sit on the target. / 실제 관절이 목표에 멈출 때까지."""
        t0 = time.time()
        since = None
        while time.time() - t0 < timeout:
            self.robot.stream_target(self.target)
            q = self.robot.joint_positions()
            if np.degrees(np.abs(q - self.target)).max() < tol_deg:
                since = since or time.time()
                if time.time() - since >= still_s:
                    break
            else:
                since = None
            time.sleep(self.dt)
        q = self.robot.joint_positions()
        if np.degrees(np.abs(q - self.target)).max() > 1.0:
            # The driver ended the stream (limit, error, safety): never keep going blind.
            # 드라이버가 스트림을 끝냈다 (한계·에러·안전): 멈춘 채 계속하지 않는다.
            off = float(np.degrees(np.abs(q - self.target)).max())
            raise RuntimeError(f"robot is not following the servo stream (off by {off:.1f}°)")
        return q


def cartesian_line(robot, mover: Mover, rotation, goal_grip, grip_offset, speed: float = 0.03) -> bool:
    """Straight grip-point line at fixed tool rotation (first approach). / 고정 자세 직선 이동."""
    kin = robot.kin
    start = kin.tool(mover.target)
    p0 = start[:3, 3] + start[:3, :3] @ grip_offset
    dist = float(np.linalg.norm(goal_grip - p0))
    dur = max(1.0, 1.875 * dist / speed)
    t0 = time.time()
    while True:
        s = min(1.0, (time.time() - t0) / dur)
        b = s * s * s * (10 - 15 * s + 6 * s * s)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = p0 + (goal_grip - p0) * b - rotation @ grip_offset
        q = kin.inverse(pose, robot.branch)
        if q is None:
            return False
        q = mover.target + (q - mover.target + math.pi) % (2 * math.pi) - math.pi
        mover.target = q
        robot.stream_target(q)
        if s >= 1.0:
            return True
        time.sleep(mover.dt)


# ----------------------------------------------------------------------------- session
def main() -> None:
    from factor.real.fiducial import FiducialDetector
    from factor.robots.fr3.kinematics import FR3Kinematics, FR3Params, top_down_rotation
    from factor.robots.fr3.ros_adapter import tool_yaw_from_anchor
    from factor.scene import load_scene

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="configs/hanoi.yaml")
    ap.add_argument("--board", default="configs/gripper_board.json")
    ap.add_argument(
        "--feature",
        choices=("ring", "board"),
        default="ring",
        help="ring = wrist LED ring, no marker on the robot; board = gripper ArUco board",
    )
    ap.add_argument("--cam-dir", default="logs/cam")
    ap.add_argument("--budget", type=float, default=60.0, help="seconds of robot time / 로봇 시간 예산 (s)")
    ap.add_argument("--target-mm", type=float, default=0.3, help="stop when the peg points' 1σ is below this")
    ap.add_argument("--min-poses", type=int, default=6)
    ap.add_argument("--frames", type=int, default=2, help="frames averaged per pose")
    ap.add_argument("--height", type=float, default=0.06, help="pose centre above the peg tops (m)")
    ap.add_argument("--out", default="logs/factor/selfcalib/latest.json")
    ap.add_argument("--dry", action="store_true", help="detect and plan only; no motion")
    ap.add_argument("--debug-dir", default="", help="save frames of search poses without a detection")
    ap.add_argument(
        "--fake",
        type=int,
        default=-1,
        help="seed: kinematic fake robot + synthetic camera (tests the runner)",
    )
    a = ap.parse_args()

    scene = load_scene(a.scene)
    board = GripperBoard.from_json(a.board)
    k, dist = scene.camera.matrix, scene.camera.distortion
    mat_det = FiducialDetector(scene.fiducials, scene.camera, min_markers=3)
    board_det = FiducialDetector(board_layout(board), scene.camera, min_markers=2)
    belief = np.asarray(scene.anchor_in_base, dtype=float)
    h = scene.extras.get("hanoi", {})
    peg_top_anchor = float(h.get("plate_thickness", 0.007)) + float(h.get("peg_height", 0.055))
    pts = peg_points(
        float(h.get("peg_spacing", 0.06)),
        float(h.get("plate_thickness", 0.007)),
        float(h.get("peg_height", 0.055)),
    )
    g = scene.robot.grasp
    branch = tuple(int(v) for v in g.get("ik_branch", (0, 0, 1)))
    kin = FR3Kinematics(FR3Params.from_mapping(scene.robot.kinematics))
    base_rot = top_down_rotation(
        str(g.get("finger_axis", "y")), int(g.get("finger_sign", 1)), yaw=tool_yaw_from_anchor(belief)
    )
    grip_offset = np.asarray(scene.robot.grip_offset, dtype=float)
    finger_reach = float(g.get("finger_max", 0.0475))

    peg_top_base = belief[:3, :3] @ [0, 0, peg_top_anchor] + belief[:3, 3]
    centre = belief[:3, :3] @ np.array([0.0, 0.0, peg_top_anchor + a.height]) + belief[:3, 3]
    min_tip_z = float(peg_top_base[2]) + 0.035
    centre_pose = np.eye(4)
    centre_pose[:3, :3] = base_rot
    centre_pose[:3, 3] = centre - base_rot @ grip_offset
    q_centre = kin.inverse(centre_pose, branch)
    if q_centre is None:
        raise SystemExit("centre pose unreachable / 중심 자세 도달 불가")

    rclpy = None
    if a.fake >= 0:
        from factor.calib.simulate import Rig
        from factor.real.fake import FakeRobot

        start = centre_pose.copy()
        start[:3, 3] += [0.02, -0.01, -0.03]
        robot = FakeRobot(kin, branch, grip_offset, base_rot, kin.inverse(start, branch))
        rig = Rig(scene, board, kin, q_centre, np.random.default_rng(a.fake))
        sensor = FakeSensor(rig, robot, board)
    else:
        sensor = CameraSensor(a.cam_dir, board_det if a.feature == "board" else None, mat_det)

    # ---- anchor seen from the camera: prior for X / 카메라가 본 앵커 → X 의 사전값
    mats, mat_w = [], []
    t_last = 0.0
    for _ in range(4):
        t_last, _b, d = sensor.grab(t_last)
        if d is not None:
            mats.append(d.anchor_in_camera)
            mat_w.append(len(d.marker_ids))
    if not mats:
        raise SystemExit("anchor mat not detected: check the camera / 앵커 매트 미검출")
    t_ca = average_poses(mats, mat_w)
    prior = belief @ np.linalg.inv(t_ca)
    print(
        f"[selfcalib] mat: {max(mat_w)} markers; "
        f"prior camera in base {np.round(prior[:3, 3] * 1e3).astype(int).tolist()} mm",
        flush=True,
    )
    ring = a.feature == "ring"
    if ring:
        from factor.calib.ring import RingHandEye

        est = RingHandEye(k, dist, prior, image_size=(scene.camera.width, scene.camera.height))
    else:
        est = ActiveHandEye(k, dist, board, prior, image_size=(scene.camera.width, scene.camera.height))

    if a.dry:
        cands = generate(kin, branch, base_rot, centre, grip_offset, q_centre, min_tip_z, finger_reach)
        _t, bd, _m = sensor.grab()
        print(
            f"[selfcalib] dry: {len(cands)} reachable candidates; "
            f"gripper board in view: {None if bd is None else bd.marker_ids}"
        )
        return

    if a.fake < 0:
        import rclpy
        from rclpy.node import Node

        from factor.robots.fr3.ros_adapter import FR3Ros

        rclpy.init()
        node = Node("factor_selfcalib")
        robot = FR3Ros(
            node,
            FR3Params.from_mapping(scene.robot.kinematics),
            branch,
            grip_offset,
            tool_yaw_from_anchor(belief),
            str(g.get("finger_axis", "y")),
            int(g.get("finger_sign", 1)),
        )
        robot.wait_ready()
    q_now = robot.joint_positions()
    tool = kin.tool(q_now)
    # Tilt of the approach axis only (tool y = world up when top-down); a turn about the vertical is fine.
    # 접근축 기울기만 본다 (수직축 회전은 허용).
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(tool[:3, 1] @ base_rot[:, 1])))))
    if tilt > 20:
        raise SystemExit(f"start posture is {tilt:.0f}° from top-down; move the arm above the tower first")
    robot.start_streaming()
    mover = Mover(robot)
    t_start = time.time()
    log = []
    try:
        # ---- 1. up, then over the tower centre / 위로, 그다음 탑 중심 위로
        grip_now = tool[:3, 3] + tool[:3, :3] @ grip_offset
        up = np.array([grip_now[0], grip_now[1], max(grip_now[2], centre[2])])
        # Straight up with the current tool rotation, then a joint move to the centre pose.
        # 현재 툴 자세 그대로 수직 상승, 그다음 관절 이동으로 중심 자세.
        if not cartesian_line(robot, mover, tool[:3, :3], up, grip_offset):
            raise SystemExit("approach path unreachable / 접근 경로 도달 불가")
        q_c = near_within_limits(kin, q_centre, mover.target)
        if q_c is None:
            raise SystemExit("centre pose too close to a joint limit / 중심 자세가 관절 한계에 가까움")
        mover.move(q_c)
        q0 = mover.settle()
        cands = generate(kin, branch, base_rot, centre, grip_offset, q0, min_tip_z, finger_reach)
        print(f"[selfcalib] {len(cands)} candidate poses; budget {a.budget:.0f} s", flush=True)

        def observe(tag: str):
            """Settle, grab fresh frames, detect both boards. / 정지 후 새 프레임에서 두 보드 검출."""
            nonlocal t_ca
            q = mover.settle()
            t_still = time.time()
            uvs, ids_ref, fk = [], None, kin.flange(q)
            led_sets = []
            t_seen = t_still + 0.35  # frames encode after capture / 캡처 후 인코딩 지연
            for _ in range(a.frames):
                t_seen, d, m = sensor.grab(t_seen)
                if m is not None and len(m.marker_ids) >= 4:
                    mats.append(m.anchor_in_camera)
                    mat_w.append(len(m.marker_ids))
                if ring:
                    led_sets.append(sensor.leds(est.roi(fk, margin=260.0)))
                    continue
                if d is None or len(d.marker_ids) < 2 or d.reprojection_rms_px > 1.5:
                    continue
                order = {int(i): c for i, c in zip(d.marker_ids, d.corners, strict=True)}
                if ids_ref is None:
                    ids_ref = sorted(order)
                if all(i in order for i in ids_ref):
                    uvs.append(np.concatenate([order[i] for i in ids_ref]))
                    pnp = d.anchor_in_camera
            t_ca = average_poses(mats, mat_w)
            if ring:
                pts_all = merge_points(led_sets)
                return (fk, pts_all) if len(pts_all) else None
            if not uvs:
                return None
            return fk, ids_ref, np.mean(uvs, axis=0), pnp

        # ---- 2. first sighting / 첫 관측
        if ring:
            o = observe("centre")
            if o is None:
                raise SystemExit("no blue LED blobs near the wrist: is the LED ring lit? / LED 미검출")
            est.add(o[0], o[1], "centre")
            print(f"[selfcalib] LED ring: {len(o[1])} blobs at the centre pose", flush=True)
        for c in [] if ring else search_poses(kin, branch, base_rot, centre, grip_offset, q0):
            mover.move(c.joints)
            o = observe(c.label)
            if o is None and a.debug_dir and getattr(sensor, "last_rgb", None) is not None:
                import cv2

                os.makedirs(a.debug_dir, exist_ok=True)
                small = cv2.resize(sensor.last_rgb, None, fx=0.5, fy=0.5)
                cv2.imwrite(os.path.join(a.debug_dir, c.label.replace(" ", "_") + ".jpg"), small[:, :, ::-1])
            if o is not None:
                fk, ids, uv, pnp = o
                est.initialise_mount(fk, pnp)
                est.add(fk, ids, uv, c.label)
                est.solve()
                print(f"[selfcalib] board found at {c.label} ({len(ids)} markers)", flush=True)
                break
        if not est.ready:
            mover.move(q0)
            mover.settle()
            raise SystemExit(
                "gripper board never seen: is it attached and facing the camera? / 그리퍼 보드 미검출"
            )

        # ---- 3. active loop / 능동 루프
        while time.time() - t_start < a.budget:
            for _ in range(6):
                c, info = choose_next(est, cands, mover.target, t_ca, pts)
                if c is None or path_clear(
                    kin, mover.target, c.joints, min_tip_z, finger_reach * 0.5, min_tip_z
                ):
                    break
                c.failed += 1
            if c is None:
                print("[selfcalib] no informative pose left", flush=True)
                break
            if time.time() - t_start + info["seconds"] > a.budget + 3:
                break
            mover.move(c.joints)
            c.used = True
            o = observe(c.label)
            if ring and o is not None:
                kept = est.add(o[0], o[1], c.label)
                if not est.aligned and len(est.observations) < 3:
                    continue  # collect three poses, then align them together / 세 자세를 모은 뒤 함께 정렬
                if not est.aligned or len(est.observations) == 5:
                    print(f"[selfcalib] coarse alignment {est.align()}", flush=True)
                    kept = len(est.observations[-1].points)
                if kept == 0:
                    print(f"[selfcalib] {c.label}: no LED near the predicted ring", flush=True)
                    c.failed += 1
                    continue
            if o is None and ring:
                c.failed += 1
                print(f"[selfcalib] {c.label}: no LED blobs", flush=True)
                continue
            if o is None:
                c.failed += 1
                # Learn the detector's viewing-angle limit: a miss at this facing angle tightens the
                # visibility prediction just above it. / 놓친 시야각 바로 위로 가시성 예측을 조인다.
                facing = est.predict(c.flange_in_base).facing_cos
                est.min_facing_cos = min(0.7, max(est.min_facing_cos, facing + 0.03))
                print(
                    f"[selfcalib] {c.label}: board not detected "
                    f"(facing {math.degrees(math.acos(facing)):.0f}°; "
                    f"limit now {math.degrees(math.acos(est.min_facing_cos)):.0f}°)",
                    flush=True,
                )
                continue
            if not ring:
                fk, ids, uv, _ = o
                est.add(fk, ids, uv, c.label)
            st = est.solve()
            sigma = est.task_uncertainty(est.information(), t_ca, pts)
            anchor = est.X @ t_ca
            moved = pose_error(belief, anchor)
            row = {
                "t": round(time.time() - t_start, 1),
                "pose": c.label,
                "n": len(est.observations),
                "sigma_mm": sigma * 1e3,
                **st,
                "anchor_shift_mm": (moved[3:] * 1e3).round(2).tolist(),
            }
            log.append(row)
            print(
                f"[selfcalib] {row['t']:5.1f}s #{row['n']:2d} {c.label:38s} "
                f"reproj {st['reproj_rms_px']:.2f}px χ² {st['chi2_per_dof']:.2f}  "
                f"peg σ {sigma * 1e3:.2f} mm (pred {info['sigma_after_mm']:.2f})  "
                f"anchor shift {row['anchor_shift_mm']} mm",
                flush=True,
            )
            if sigma * 1e3 <= a.target_mm and len(est.observations) >= a.min_poses:
                break
        # ---- 4. back over the centre, top-down / 중심 위 탑다운 자세로 복귀
        mover.move(q0)
        mover.settle()
    finally:
        try:
            robot.stop_streaming()
        except Exception as e:  # noqa: BLE001
            print(f"[selfcalib] stop_streaming: {e}")
        if rclpy is not None:
            rclpy.try_shutdown()

    if len(est.observations) < a.min_poses:
        raise SystemExit(f"only {len(est.observations)} poses observed; calibration not written")
    t_ca = average_poses(mats, mat_w)
    anchor = est.X @ t_ca
    sigma = est.task_uncertainty(est.information(), t_ca, pts)
    # Leave-one-out spread of the peg points. / 한 자세씩 빼고 다시 풀었을 때 기둥 점의 흔들림.
    loo = []
    full = (anchor[:3, :3] @ pts.T).T + anchor[:3, 3]
    for i in range(len(est.observations)):
        if ring:
            from factor.calib.ring import RingHandEye

            sub = RingHandEye(k, dist, prior, image_size=est.image_size)
            sub.X, sub.ring, sub.aligned, sub.sigma = est.X.copy(), est.ring.copy(), True, est.sigma
        else:
            sub = ActiveHandEye(k, dist, board, prior, image_size=est.image_size)
            sub.X, sub.Y, sub._mount_prior = est.X.copy(), est.Y.copy(), est._mount_prior
        sub.observations = [o for j, o in enumerate(est.observations) if j != i]
        sub.solve()
        a_i = sub.X @ t_ca
        loo.append(
            float(np.sqrt(np.mean(np.sum(((a_i[:3, :3] @ pts.T).T + a_i[:3, 3] - full) ** 2, axis=1))))
        )
    shift = pose_error(belief, anchor)
    rpy = np.degrees(
        [
            math.atan2(anchor[2, 1], anchor[2, 2]),
            -math.asin(anchor[2, 0]),
            math.atan2(anchor[1, 0], anchor[0, 0]),
        ]
    )
    result = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "anchor_in_base": anchor.tolist(),
        "anchor_xyz": anchor[:3, 3].tolist(),
        "anchor_rpy_deg": rpy.tolist(),
        "camera_in_base": est.X.tolist(),
        "feature": a.feature,
        "board_in_flange": None if ring else est.Y.tolist(),
        "ring_mm": (est.ring * 1e3).tolist() if ring else None,
        "anchor_in_camera": t_ca.tolist(),
        "n_poses": len(est.observations),
        "seconds": round(time.time() - t_start, 1),
        "peg_sigma_mm": sigma * 1e3,
        "leave_one_out_max_mm": max(loo) * 1e3,
        "reproj_rms_px": log[-1]["reproj_rms_px"] if log else None,
        "chi2_per_dof": log[-1]["chi2_per_dof"] if log else None,
        "mount_consistency": est.mount_consistency(),
        "shift_from_scene_mm": (shift[3:] * 1e3).tolist(),
        "shift_from_scene_deg": np.degrees(shift[:3]).tolist(),
        "mat_frames": len(mats),
        "log": log,
    }
    if a.fake >= 0:
        truth = (rig.anchor_true[:3, :3] @ pts.T).T + rig.anchor_true[:3, 3]
        err = float(np.sqrt(np.mean(np.sum((full - truth) ** 2, axis=1))))
        prior_anchor = rig.anchor_belief
        before = (prior_anchor[:3, :3] @ pts.T).T + prior_anchor[:3, 3]
        result["fake_truth_error_mm"] = err * 1e3
        result["fake_scene_error_mm"] = float(np.sqrt(np.mean(np.sum((before - truth) ** 2, axis=1)))) * 1e3
        print(
            f"[selfcalib] fake truth: peg-point error {err * 1e3:.2f} mm "
            f"(scene file was off by {result['fake_scene_error_mm']:.1f} mm)",
            flush=True,
        )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    # Overlay for a human check: predicted pegs over the last frame. / 확인용 오버레이.
    try:
        import cv2

        if a.fake >= 0:
            raise RuntimeError("no image in fake mode")
        rgb, _, _ = read_frame(a.cam_dir)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tower_cam = np.linalg.inv(est.X) @ anchor
        for p0, p1 in zip(pts[0::2], pts[1::2], strict=True):
            uv = project(k, dist, tower_cam, np.array([p0, p1])).astype(int)
            cv2.line(img, tuple(uv[0]), tuple(uv[1]), (255, 0, 255), 3)
        cv2.imwrite(os.path.splitext(a.out)[0] + "_overlay.png", img)
    except Exception as e:  # noqa: BLE001
        print(f"[selfcalib] overlay skipped: {e}")
    print(
        f"[selfcalib] DONE {result['n_poses']} poses in {result['seconds']} s: peg σ {sigma * 1e3:.2f} mm, "
        f"leave-one-out max {max(loo) * 1e3:.2f} mm, "
        f"mount {result['mount_consistency']}\n"
        f"           anchor xyz {np.round(anchor[:3, 3] * 1e3, 1).tolist()} mm "
        f"rpy {np.round(rpy, 2).tolist()}° (scene shift {np.round(shift[3:] * 1e3, 1).tolist()} mm, "
        f"{np.round(np.degrees(shift[:3]), 2).tolist()}°) -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
