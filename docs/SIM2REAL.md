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

## 13. Session self-calibration / 세션 자기보정 (2026-10-16)

**Problem.** ArUco gives the tower in the camera frame (T_CA). The robot needs it in its own base
frame (T_BA = T_BC · T_CA). A constant T_BA in the scene file breaks as soon as the tower or the
camera moves. The user moved both on purpose, and it cost the 10-16 real runs about 1 cm in y.
**문제.** ArUco 는 카메라 기준 탑(T_CA)만 준다. 로봇에 필요한 것은 베이스 기준 탑(T_BA = T_BC·T_CA)이다.
장면 파일의 고정 T_BA 는 탑이나 카메라가 움직이는 순간 틀린다.

**Method** (`factor-selfcalib`, `src/factor/calib/`). A 2×2 ArUco board (ids 10–13, 25 mm) is taped
to the gripper body. Within about 60 s the robot shows it to the camera:
1. Prior: X₀ = T_BA(scene) · T_CA(now). It is weak (10°, 5 cm), so it only settles the first pose.
2. First sighting: turn about the vertical until the board is detected. The mount Y = T_FG
   initialises from it.
3. Joint estimate: Levenberg–Marquardt over X = T_BC and Y = T_FG (12 parameters) on all corner
   reprojections. Each pose is whitened by σ²I + J_f Σ_FK J_fᵀ, because a kinematic error moves all
   16 corners of a pose together. Without this the estimate is overconfident and up to 2× worse.
   The noise model inflates by χ²/dof when the data disagree.
4. Prediction: for ~400 reachable candidates (grip ±7 cm, tilt ±28° about two axes, yaw ±60°)
   predict visibility (in image, facing angle, marker size) and the information gain. The score
   is log(σ²_now / σ²_after) / (move time + dwell). σ is the RMS 1σ of the six peg points in the
   base frame through X · T_CA, the quantity the task needs, not X itself.
5. Adjustment: a predicted-visible pose that the detector misses tightens the facing-angle limit
   just above it, so the detector's real limit is learned within the session.
6. Motion: minimum-jerk joint paths (≤24°/s, ≤36°/s²) over the servo stream. The robot settles,
   then takes 2 fresh frames (the camera writes 2.7 fps).
7. Output: anchor_in_base = X · mean(T_CA), plus leave-one-out spread, mount consistency and an
   overlay image. `factor-run --calib FILE` uses it in place of the scene constant;
   `tools/factor_real.sh --self-calib` chains both.

**Why it is precise.** The gripper board and the mat are seen by the same camera at almost the
same depth. Intrinsic and depth errors therefore largely cancel in X · T_CA. Calibrating in the
task region with the task's own kinematics also absorbs local kinematic bias.
**왜 정밀한가.** 그리퍼 보드와 매트를 같은 카메라가 거의 같은 깊이에서 보므로 내부 파라미터·깊이 오차가
X·T_CA 에서 대부분 상쇄된다. 작업 영역에서 작업과 같은 기구학으로 보정하므로 국소 기구학 편향도 흡수된다.

**Synthetic result** (20 seeds; camera moved ±4 cm/3°, tower ±2 cm/2°; corners 0.3 px; FK 0.2 mm/0.02°;
peg-point RMS error vs. robot time, median (p90)):

| strategy | 20 s | 30 s | 45 s | 60 s | poses in 60 s |
|---|---|---|---|---|---|
| active (ours) | 0.60 (0.96) | 0.45 (0.81) | 0.37 (0.59) | **0.23 (0.47)** | 12.8 |
| random among predicted-visible | 1.11 (2.29) | 0.65 (1.32) | 0.44 (0.70) | 0.31 (0.63) | 10.9 |
| random (blind) | 1.07 (3.26) | 0.62 (1.14) | 0.51 (0.75) | 0.45 (0.66) | 9.7 |

Noise ablation (active, 60 s): corner noise only 0.02 mm, mat noise only 0.03 mm, FK noise only
≈1 mm before whitening. The robot's kinematic accuracy is the floor, not the camera.
End-to-end fake runs (real runner, fake robot, synthetic camera, real timing): 11–14 poses in
61–68 s, truth error 0.17–0.31 mm from scene-file errors of 15–61 mm.
Log: `logs/factor/selfcalib/sim_compare_20seeds.txt`.

**Status.** Written and tested; it has not run on the robot yet. The gripper board is not attached
(camera check 13:12). Print `assets/aruco/gripper_board.pdf` at 100 %, tape it flat on the gripper
side that faces the camera, then run `tools/factor_real.sh --calib-only`.


**2026-10-16 afternoon, real runs.** Two findings.
- **Anchor:** fingertip sighting brought the anchor to within a few mm. The user then centred the gripper on the start peg by eye, a 5.5 cm correction in base y that the sighting had missed, because its y is very sensitive to the assumed tip height. Anchor xyz is now [-0.0733, -0.4811, 0.0239].
- **Grasp verification:** with probing off (`--no-correction`), the pick worked. But the grasp head reported "not grasped" (mean p 0.32, max 0.5) while the ring was held, so the loop reopened and dropped it three times. Grasp verification is now off by default (`--verify-grasp` turns it on). The estimator's real-frame outputs (correction and grasp) are not trustworthy until the appearance gap is closed.

**2026-10-16 14:49: first clean move on the user-centred anchor.** ring5, from the start peg to the middle peg, picked first time and seated (`--no-correction`). It needed two loop fixes:
- **No-progress stall:** NUDGE's "no progress" stall read the arm's slow creep near the goal (about 0.3 mm per tick, because the streamed target is only 4 mm ahead) as contact. It hopped and searched forever 10–18 mm above the peg top. The real loop now uses less than 0.1 mm per tick for 6 ticks.
- **Grasp verification:** this stays off (see above).

## Checklist before a real run / 실기 전 점검

1. `scripts/run_real.sh --fake` (loop, camera, estimator, viewer, no motion).
2. Canonical view marker ids match the twin (`id 5` top-centre for this mat).
3. Driver up with the servo parameters above (`scripts/run_real.sh` sets them); servo probe MOVED.
4. Gripper activated (the driver does it on the first command; fingers sweep once).
5. Viewer at `http://localhost:8780`; recording in `logs/cam/videos/`.
