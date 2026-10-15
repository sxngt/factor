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

