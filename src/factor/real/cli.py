"""``factor-run``: execute Tower of Hanoi moves on the robot with the learned estimator.

Wires the scene specification, the FR3 ROS 2 adapter, the SDK gripper, a shared-frame camera
source, the estimator server and the closed-loop skill. Runs in the ROS Python; the estimator
runs in the interpreter given by ``--torch-python``.

실기 실행 명령. 장면 명세·FR3 ROS 2 어댑터·SDK 그리퍼·공유 프레임 카메라·추정 서버·폐루프 스킬을
엮는다. ROS 파이썬에서 실행하고 추정기는 ``--torch-python`` 인터프리터에서 돈다.

Example::

    factor-run --scene configs/hanoi.yaml --checkpoint estimator.pt \\
        --stacks "left:ring0,ring1,ring2,ring3,ring4,ring5;middle:;right:" --goal right --dry
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from factor.perception.canonical import VirtualCamera
from factor.perception.client import EstimatorClient
from factor.real.run import LoopConfig, MoveSpec, RealLoop
from factor.real.safety import WorkspaceBox
from factor.scene import load_scene
from factor.tasks.hanoi import HanoiLayout, HanoiPlanner


def shared_frame_source(cam_dir: str, max_age: float = 1.5, retries: int = 10):
    """Frame source reading the latest PNG written by a camera server. / 카메라 서버의 최신 PNG.

    The server writes ``latest_color.png`` and ``latest.json`` (with the PNG's SHA-256), so a
    read that lands mid-write is detected by the checksum and retried.
    서버가 PNG 와 그 SHA-256 을 함께 쓰므로, 쓰는 도중 읽은 프레임은 체크섬으로 걸러 다시 읽는다.
    """
    import hashlib
    import io

    from PIL import Image

    def latest() -> np.ndarray:
        for _ in range(retries):
            with open(os.path.join(cam_dir, "latest.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if time.time() - float(meta.get("t", 0)) > max_age:
                raise RuntimeError("camera frame is stale")
            with open(os.path.join(cam_dir, "latest_color.png"), "rb") as f:
                data = f.read()
            expected = meta.get("color_sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                time.sleep(0.02)
                continue
            try:
                return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("could not read a consistent camera frame")

    return latest


def parse_stacks(text: str) -> dict[str, list[str]]:
    stacks = {"left": [], "middle": [], "right": []}
    for part in text.split(";"):
        if ":" not in part:
            continue
        peg, names = part.split(":", 1)
        stacks[peg.strip()] = [n.strip() for n in names.split(",") if n.strip()]
    return stacks


def anchor_to_base(anchor_in_base: np.ndarray, p: np.ndarray) -> np.ndarray:
    return anchor_in_base[:3, :3] @ np.asarray(p, dtype=float) + anchor_in_base[:3, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="configs/hanoi.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--torch-python", default=os.environ.get("FACTOR_TORCH_PYTHON", "python"))
    ap.add_argument("--stacks", required=True, help='e.g. "left:ring0,ring1;middle:;right:" (bottom to top)')
    ap.add_argument("--goal", default="right")
    ap.add_argument(
        "--calib",
        default="",
        help="session self-calibration JSON (factor-selfcalib); overrides the scene anchor",
    )
    ap.add_argument("--moves", type=int, default=0, help="limit the number of moves (0 = all)")
    ap.add_argument("--cam-dir", default="logs/cam")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--step-scale", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=900, help="ticks per move before giving up")
    ap.add_argument("--no-probe", action="store_true", help="act on per-frame estimates (no session probing)")
    ap.add_argument("--record", default="", help="directory for hindsight-labelled real data ('' = off)")
    ap.add_argument(
        "--no-correction", action="store_true", help="ignore the estimator (nominal geometry); still records"
    )
    ap.add_argument(
        "--insert-lead",
        type=float,
        default=4.0,
        help="how far ahead the streamed target runs while inserting (mm)",
    )
    ap.add_argument("--nudge", action="store_true", help="enable the stall-triggered spiral insertion search")
    ap.add_argument(
        "--verify-grasp",
        action="store_true",
        help="reopen on the estimator's 'not grasped' verdict (off: its real-frame verdict is unreliable)",
    )
    ap.add_argument("--max-stderr", type=float, default=0.8, help="refuse a move above this fused error (mm)")
    ap.add_argument(
        "--place-offset", default="0,0,0", help="manual goal trim in the base frame, mm (x,y,z), e.g. 0,-2,0"
    )
    ap.add_argument(
        "--resume-holding", action="store_true", help="the first move's object is already in the gripper"
    )
    ap.add_argument(
        "--grip-wait", type=float, default=2.0, help="seconds to hold still after a gripper command"
    )
    ap.add_argument("--skill", default="", help="exported residual skill policy (.npz); empty = rule skill")
    ap.add_argument(
        "--use-offset", action="store_true", help="place at goal minus the estimated in-hand offset"
    )
    ap.add_argument("--dry", action="store_true", help="plan and observe only; no motion")
    ap.add_argument(
        "--fake",
        default="",
        help="kinematic fake robot started at these joint angles (deg, comma separated); no ROS, no motion",
    )
    ap.add_argument("--log", default="")
    ap.add_argument(
        "--live", default="logs/factor/live", help="telemetry directory for factor-view ('' = off)"
    )
    args = ap.parse_args()

    scene = load_scene(args.scene)
    layout = HanoiLayout.from_scene(scene)
    planner = HanoiPlanner(layout)
    stacks = parse_stacks(args.stacks)
    moves = planner.plan(stacks, args.goal)
    if args.moves:
        moves = moves[: args.moves]
    print(f"{len(moves)} moves planned: " + ", ".join(f"{m.item} {m.source}->{m.destination}" for m in moves))

    anchor = scene.anchor_in_base
    if args.calib:
        # Session self-calibration: the anchor measured through the robot's own kinematics this session.
        # 세션 자기보정: 이번 세션에 로봇 기구학으로 잰 앵커 포즈.
        with open(args.calib, encoding="utf-8") as f:
            cal = json.load(f)
        anchor = np.asarray(cal["anchor_in_base"], dtype=float)
        print(
            f"[factor-run] anchor from {args.calib} ({cal.get('t')}, "
            f"peg σ {cal.get('peg_sigma_mm', float('nan')):.2f} mm): "
            f"xyz {np.round(anchor[:3, 3] * 1e3, 1).tolist()} mm",
            flush=True,
        )
    virtual = VirtualCamera(128, 0.14, float(np.linalg.norm(scene.camera_in_anchor[:3, 3])))
    estimator = EstimatorClient(args.torch_python, args.checkpoint)
    virtual = VirtualCamera(estimator.resolution, estimator.view_width, virtual.distance)

    lo = anchor_to_base(anchor, [-0.12, -0.04, layout.plate_thickness + layout.rings[0].height / 2 - 0.001])
    hi = anchor_to_base(anchor, [0.12, 0.04, layout.hover_height + 0.01])
    workspace = WorkspaceBox(np.minimum(lo, hi), np.maximum(lo, hi))
    config = LoopConfig(
        rate_hz=args.rate,
        step_scale=args.step_scale,
        grip_wait_s=args.grip_wait,
        max_steps=args.max_steps,
        probe=not (args.no_probe or args.no_correction),
        use_correction=not args.no_correction,
        record_dir=os.path.abspath(args.record) if args.record else "",
        max_stderr=args.max_stderr / 1000.0,
        place_offset=tuple(float(v) / 1000.0 for v in args.place_offset.split(",")),
        hover_height=anchor_to_base(anchor, [0.0, 0.0, layout.hover_height])[2],
        zone_top=anchor_to_base(anchor, [0.0, 0.0, layout.zone_top])[2],
        use_offset_estimate=args.use_offset,
        # so the loop dropped it three times. Off unless asked. / 실기에서 잡고도 빈손 판정 → 기본 끔.
        verify_grasp=args.verify_grasp,
        nudge=args.nudge,
        insert_lead=args.insert_lead / 1000.0,
    )

    if args.dry:
        from factor.real.fiducial import FiducialDetector

        frames = shared_frame_source(args.cam_dir)
        det = FiducialDetector(scene.fiducials, scene.camera).detect(frames())
        print(
            "fiducial:",
            None if det is None else f"{len(det.marker_ids)} markers, rms {det.reprojection_rms_px:.2f} px",
        )
        estimator.close()
        return

    from factor.robots.fr3.kinematics import FR3Kinematics, FR3Params, top_down_rotation
    from factor.robots.fr3.ros_adapter import tool_yaw_from_anchor

    grasp = scene.robot.grasp
    branch = tuple(int(v) for v in grasp.get("ik_branch", (0, 0, 1)))
    finger_axis, finger_sign = str(grasp.get("finger_axis", "y")), int(grasp.get("finger_sign", 1))
    rclpy = None
    if args.fake:
        from factor.real.fake import FakeGripper, FakeRobot

        robot = FakeRobot(
            FR3Kinematics(FR3Params.from_mapping(scene.robot.kinematics)),
            branch,
            np.asarray(scene.robot.grip_offset),
            top_down_rotation(finger_axis, finger_sign, yaw=tool_yaw_from_anchor(anchor)),
            np.radians([float(v) for v in args.fake.split(",")]),
        )
        gripper = FakeGripper()
    else:
        import rclpy
        from rclpy.node import Node

        from factor.robots.fr3.ros_adapter import FR3Ros, RosGripper

        rclpy.init()
        node = Node("factor_run")
        robot = FR3Ros(
            node,
            FR3Params.from_mapping(scene.robot.kinematics),
            branch,
            np.asarray(scene.robot.grip_offset),
            tool_yaw_from_anchor(anchor),
            finger_axis,
            finger_sign,
        )
        robot.wait_ready()
        gripper = RosGripper(node)
    telemetry = None
    if args.live:
        from factor.real.telemetry import SceneGeometry, Telemetry

        telemetry = Telemetry(args.live, scene.camera.matrix, scene.camera.distortion, virtual, anchor)
        plate_size = scene.extras.get("hanoi", {}).get("plate_size", (0.16, 0.06))
        half_x, half_y = plate_size[0] / 2.0, plate_size[1] / 2.0
        z = layout.plate_thickness
        telemetry.scene = SceneGeometry(
            plate=np.array(
                [[-half_x, -half_y, z], [half_x, -half_y, z], [half_x, half_y, z], [-half_x, half_y, z]]
            ),
            pegs=[np.array([*layout.peg_xy(p), z]) for p in ("left", "middle", "right")],
            peg_height=layout.peg_height,
        )
    residual = None
    if args.skill:
        from factor.skills.residual_np import NumpyResidualPolicy

        residual = NumpyResidualPolicy.from_file(args.skill)
    loop = RealLoop(
        scene,
        robot,
        gripper,
        shared_frame_source(args.cam_dir),
        estimator,
        workspace,
        config,
        virtual,
        anchor,
        residual,
        telemetry,
    )

    try:
        for move in moves:
            obj, goal = planner.nominal_targets(stacks, move)
            ring = layout.ring(move.item)
            spec = MoveSpec(
                object_center=anchor_to_base(anchor, obj),
                goal_center=anchor_to_base(anchor, goal),
                # Real fingers must squeeze: command a width below the ring and let the force limit stop them.
                # 실기 핑거는 링보다 좁게 명령하고 힘 제한으로 멈춘다.
                grasp_half_gap=min(ring.radius, float(grasp.get("close_width", 0.012)) / 2.0),
                open_half_gap=min(
                    ring.radius + float(grasp.get("open_margin", 0.006)),
                    float(grasp.get("finger_max", 0.0475)),
                ),
            )
            print(f"== {move.item}: {move.source} -> {move.destination}")
            if telemetry is not None:
                from factor.real.telemetry import MoveGeometry

                telemetry.scene.rings = [
                    (layout.ring_center(peg, level), layout.ring(name).radius, layout.ring(name).height)
                    for peg, names in stacks.items()
                    for level, name in enumerate(names)
                ]
                telemetry.move = MoveGeometry(
                    obj, goal, ring.radius, ring.height, f"{move.item} {move.source}->{move.destination}"
                )
            reason = loop.run_move(spec, holding=args.resume_holding and move is moves[0])
            print(f"   {reason}")
            if reason == "uncertain":
                print(
                    "   refused: the fused correction stayed above --max-stderr; check the scene and camera"
                )
            if reason != "released":
                break
            stacks[move.destination].append(stacks[move.source].pop())
    finally:
        if args.log:
            with open(args.log, "w", encoding="utf-8") as f:
                json.dump(loop.log, f)
        estimator.close()
        if telemetry is not None:
            telemetry.close()
        if rclpy is not None:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
