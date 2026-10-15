# Sim-to-real log / 시뮬→실기 기록

What broke when the simulation-trained stack first met the FR3, why, and what changed.
Dated 2026-10-15; commits are in this repository unless marked `robot-arm` (the research repo
that holds the ROS 2 driver and the assets).

시뮬에서 학습한 스택을 실기에 처음 올리면서 깨진 것, 원인, 고친 것. 날짜 2026-10-15.

## 1. Mirrored camera / 거울상 카메라 (critical)

- **Symptom**: the first real canonical view showed marker id 5 at the top centre with the ring
  stack at image-left; the twin's views had +x on the right.
- **Cause**: `configs/hanoi.yaml` carried the camera at anchor (−0.006, −0.397, 0.707), computed
  without applying the anchor yaw (−178.96°). The real camera is at (+0.072, +0.391, 0.711):
  beyond the mat, looking back at the robot. The twin also ignored the anchor yaw everywhere.
- **Fix** (`1dee5ec`): `TwinSettings.anchor_yaw` and `to_env_axes` / `to_anchor_axes` at every
  anchor-frame boundary (pegs, tool rotation, workspace box, belief frames, camera placement,
  overlays, in-hand offsets); scene camera pose from the ArUco extrinsics. Verified by detecting
  marker ids in a 512 px real canonical view and in twin frames (id 5 top-centre in both).
- **Consequence**: every checkpoint trained before this commit is a mirrored-world model; see the
  caveat in `RUNS.md`. Deployment models are retrained (`dep_*`, `yaw_local_e6`).
- **Lesson**: validate the canonical view with *marker identities*, not just "it looks right".

## 2. Units on the robot side / 단위

- **Symptom** (caught before motion): the estimator returned normalised units (0.3 ≈ 3 mm) and
  the loop applied them as metres.
- **Fix** (`2ba036d`..): `EstimatorConfig.pos_scale/yaw_scale/offset_scale` travel with the
  checkpoint; the estimator server converts to metres and radians; the robot loop rotates the
  correction from anchor axes into base axes (`RealLoop._correction_in_base`).

## 3. Placeholder scene values / 자리표시자 값

- The package config had a fictitious 2×3 marker grid and camera intrinsics without distortion.
  Board reprojection RMS 141 px → 1.5 px after copying the printed mat layout
  (`config/aruco/tower_mat.json`) and the webcam calibration (`webcam.json`).

## 4. Vendor SDK next to the ROS driver / SDK 와 드라이버 공존

- **Symptom**: the first gripper command never returned (process stuck in `select`, a socket in
  SYN-SENT to port 20004).
- **Cause**: the controller serves its real-time state port to one client; the driver holds it;
  the SDK then loops on reconnect and every XML-RPC call blocks.
- **Fix**: `open_sdk_rpc()` (state thread silenced, XML-RPC only) — then replaced by driving the
  gripper through the driver's MoveIt action (`RosGripper`), so the loop has one robot interface
  and the driver handles activation after a controller power cycle (`ActGripper`).
- **Driver bug found** (`robot-arm/ros2/fr_ros2_driver.py`): the servo loop and the gripper
  shared one `xmlrpc.ServerProxy` → `CannotSendRequest` ended the servo stream exactly at
  release. Fixed with a lock around SDK calls.

## 5. Servo stream safety trips / 서보 안전 트립

- **Symptom**: "추종 편차 2.4° > 2.0°" within 0.1–0.4 s of every servo start; the robot did not
  move, or crept in bursts between restarts (the "털털털" motion).
- **Cause**: the driver ramps the commanded joints at `max_joint_speed_deg` (150°/s → 2.4° per
  16 ms tick) while ServoJ starts moving only after ~100 ms; the SafetyMonitor tolerance
  (`start_tol_deg` 2°) is smaller than one ramp step. The servo probe (±0.3°) never triggers it.
- **Fix**: driver parameters set by `robot-arm/tools/factor_real.sh` on every run:
  `start_tol_deg 8`, `max_joint_speed_deg 40`, `servo_timeout 3`. Measured afterwards: a 10 mm
  Cartesian step converges in 0.5 s with zero steady-state error.
- Also: `/fr/reset_error` before every servo start (a stopped controller ignores ServoJ), and
  the loop restarts the stream after 20 ticks without motion.

## 6. Loop-side motion quality / 루프 쪽 동작 품질

Three iterations, each diagnosed from the telemetry log:

| version | behaviour | cause |
|---------|-----------|-------|
| target += step (integrated) | target ran 30 cm away while the servo was tripped | no feedback from the measured pose |
| target = grip + step (1 step ahead) | crawl at 5 mm/s, stop-and-go chatter | each tick waits for the robot to reach a 2 mm target |
| target = integrated, lead-capped 12 mm | ±8 mm oscillation around the object (2.5 s period) | the integrated target overshot the planner's point |
| **target = planner point, ≤ 12 mm ahead of the measured grip** (`lead`) | continuous motion, stops at the point | — |

The planner's own target point (`PlannerCommand.target`, now also `SkillCommand.target`) is the
right thing to command; the lead bounds the speed (~2.4 cm/s) and runaway.

## 7. Physical contact the kinematic twin never had / 트윈에 없던 접촉

- **Symptom**: with the fingers already around the ring, lateral corrections of 3–5 mm pushed
  against the stack (the servo deviation absorbed it; the loop dithered for 30 s).
- **Cause**: in the twin the grasp is a kinematic attach; fingers never collide with the ring.
- **Fix** (`planning`: `freeze_xy_below`, in both the NumPy and the torch planner): below the
  object centre + 7.5 mm no lateral motion while picking, straight descent only (mirror of the
  insertion rule). Teacher still 100 % in the twin.

## 8. Grasp verification / 파지 검증

- The controller's gripper width feedback is unusable (reports the commanded 12 mm with a
  25 mm ring between the fingers).
- The estimator's state head was right on the robot: p(grasped) 0.98–1.0 while a ring was held,
  0.00–0.04 after an empty lift. The loop now averages five verdicts once the grip point is
  15 mm above the grasp height and reopens for a new pick if the mean is below 0.5 (≤ 3 retries,
  logged as `empty grasp, retry n`).

## 9. First successful move / 첫 성공

18:48: ring5 picked from the right peg (lateral error 0.1 mm at the grasp height, grasp at
z = 71.8 mm), lifted, carried to the middle peg and released at the goal height; 144 ticks
(14 s) at 20 % speed. The seating missed by a small margin: the ring stalled on the peg top a
few millimetres towards the robot (+y in the base frame) while the contact search ran. The
tower is attached to the mat, so a tower-versus-markers offset is a fixed calibration constant,
not a variable to randomise: `extras.hanoi.peg_row_offset` in the scene file (measured once) and
`factor-run --place-offset x,y,z` (mm, base frame) for quick trials. The 1 mm ring clearance
makes placing far more sensitive than picking (6 mm finger clearance), which is why the pick
succeeded first.

## 10. What anchors the belief / 믿음의 기준 (the decisive finding)

The twin aimed its camera at the *believed* anchor centre, so a belief error showed up as a
displaced tower in the canonical view. On the robot the view is anchored by the detected board,
so the tower is always centred and the network saw no error at all: every real placement used
the stale anchor constants unchanged (the ~5 mm misses). Two ways out were tried in the wrong
order: aiming through calibrated camera extrinsics (works only until the camera is moved, which
the user does deliberately to test placement robustness) and manual touch calibration
(defeats "calibration-free"). The right formulation, now in the twin: the camera aims at the
*true* centre as the detection does, and the belief error is observable only through the
gripper, which moves in the believed frame. The estimator learns to read the gripper's
displacement relative to the pegs and rings; no extrinsics, no pendant. Camera placement
randomisation widened to ±8 cm. Models trained before `twin: camera aims at the true anchor
centre` cannot work on the robot for this reason.

## 11. The belief must be an input / 믿음은 입력이어야 한다

With the camera aimed at the true centre (§10), an image-only estimator stays at chance (x/y MSE
≈ 0.13 for labels uniform in ±0.6): under the teacher the gripper sits exactly where it should,
so the image alone carries no trace of the belief error. The error is the difference between
where the gripper *is* (visible) and where the robot *thinks* it is (proprioception). Two inputs
now carry the belief: the believed grip point as a vector, and a **belief render channel** — the
believed finger segment drawn into the canonical view through the virtual camera, the
"kinematic render channel" of the architecture computed from the robot's own belief instead of
the true scene. Local check (48 envs, 400 steps, 25 epochs): x MSE 0.16 → 0.049 with the channel
(0.034 for y). Both are computable on the robot without camera calibration.

## 12. Height is not observable from one view / 높이는 단일 시야로 보이지 않는다

Training with a ±3 mm height belief error (`--anchor_z 0.003`, fourth output) leaves the height
estimate at chance (near-peg RMS 1.8 mm for labels with σ 1.73 mm) and worsens y (1.66 vs 1.4 mm):
with a single fixed camera looking down at ~61°, a height error and a depth (y) error move the
fingers along the same image direction. The fingertip-depth issue seen on the robot (fingers
closing on the ring's rim) is therefore a job for contact, not vision: descend until the fingers
meet resistance, or verify the grasp height from the servo deviation. Deployment models keep
three outputs.

