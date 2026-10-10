#!/usr/bin/env bash
# FACTOR on the real FR3 in one script: controller -> driver -> servo probe -> camera -> viewer
# -> (self-calibration) -> recorder -> factor-run.
# 실기 원스크립트: 제어기 -> 드라이버 -> 서보 탐색 -> 카메라 -> 뷰어 -> (자기보정) -> 녹화 -> factor-run.
#
# Usage / 사용:
#   scripts/run_real.sh [--moves N] [--step-scale S] [--checkpoint a.pt,b.pt] [--goal PEG]
#                       [--stacks SPEC] [--fake] [--restart-stack] [--self-calib | --calib-only]
#                       [--calib-budget SEC] [-- extra factor-run arguments ...]
#
#   --fake          no robot: kinematic stand-in; camera, estimator and viewer are real
#   --self-calib    ~1 min active self-calibration (gripper ArUco board) before the run
#   --calib-only    self-calibration only, then exit
#   --restart-stack restart the driver this script started earlier (after changing driver code)
#
# Recommended first real runs (docs/SIM2REAL.md): check everything with --fake, then a single
# slow move without the estimator's correction, whose real-frame outputs are not trusted yet:
#   scripts/run_real.sh --fake
#   scripts/run_real.sh --moves 1 --step-scale 0.2 -- --no-correction
# Optional factor-run trims after `--`: --place-offset x,y,z (mm, base frame), --nudge, --verify-grasp.
#
# Environment / 환경 변수 (defaults in brackets):
#   ROBOT_IP [192.168.58.2]   CHECKPOINT (comma-separated estimator ensemble; or --checkpoint)
#   CAM_DIR [logs/cam]        LIVE_DIR [logs/factor/live]       LOG_DIR [logs/factor]
#   TORCH_PYTHON [python3]    Python with PyTorch for the estimator server (factor-serve)
#   ROS_SETUP [/opt/ros/humble/setup.bash]   FACTOR_PYTHON [python3]  Python that sees rclpy
#   SCENE [configs/hanoi.yaml]  BOARD [configs/gripper_board.json]  STACKS, GOAL
#   CAMERA_DEVICE [0]  CAMERA_INTRINSICS (calibration JSON)  VIEWER_PORT [8780]
#   FAKE_START_DEG (comma-separated joints for --fake)
#   Driver servo parameters: START_TOL_DEG [8] MAX_JOINT_SPEED_DEG [30] MAX_JOINT_ACCEL_DEG [45]
#   SERVO_TIMEOUT [3]  (the driver defaults 2 deg / 30 / 60 / 1.5 s trip on ServoJ's start-up lag)
set -eo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG_ROOT"

ROBOT_IP="${ROBOT_IP:-192.168.58.2}"
CHECKPOINT="${CHECKPOINT:-}"
SCENE="${SCENE:-configs/hanoi.yaml}"
BOARD="${BOARD:-configs/gripper_board.json}"
CAM_DIR="${CAM_DIR:-$PKG_ROOT/logs/cam}"
LIVE_DIR="${LIVE_DIR:-$PKG_ROOT/logs/factor/live}"
LOG_DIR="${LOG_DIR:-$PKG_ROOT/logs/factor}"
TORCH_PYTHON="${TORCH_PYTHON:-${FACTOR_TORCH_PYTHON:-python3}}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${FACTOR_PYTHON:-python3}"
CAMERA_DEVICE="${CAMERA_DEVICE:-0}"
CAMERA_INTRINSICS="${CAMERA_INTRINSICS:-}"
VIEWER_PORT="${VIEWER_PORT:-8780}"
FAKE_START_DEG="${FAKE_START_DEG:-61.48,-60.45,40.49,-70.02,-89.99,-119.55}"
STACKS="${STACKS:-left:;middle:;right:ring0,ring1,ring2,ring3,ring4,ring5}"
GOAL="${GOAL:-left}"
START_TOL_DEG="${START_TOL_DEG:-8.0}"
MAX_JOINT_SPEED_DEG="${MAX_JOINT_SPEED_DEG:-30.0}"
MAX_JOINT_ACCEL_DEG="${MAX_JOINT_ACCEL_DEG:-45.0}"
SERVO_TIMEOUT="${SERVO_TIMEOUT:-3.0}"

MOVES=1; STEP=0.3; FAKE=0; RESTART=0; SELFCAL=0; CALONLY=0; CALBUDGET=60; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --moves) MOVES="$2"; shift 2;;
    --step-scale) STEP="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --goal) GOAL="$2"; shift 2;;
    --stacks) STACKS="$2"; shift 2;;
    --fake) FAKE=1; shift;;
    --restart-stack) RESTART=1; shift;;
    --self-calib) SELFCAL=1; shift;;
    --calib-only) SELFCAL=1; CALONLY=1; shift;;
    --calib-budget) CALBUDGET="$2"; shift 2;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown argument: $1 (see --help)"; exit 2;;
  esac
done

log() { echo "[run_real $(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }

if [ -f "$ROS_SETUP" ]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP" >/dev/null
elif [ $FAKE = 0 ]; then
  die "ROS setup file not found: $ROS_SETUP (set ROS_SETUP)"
fi
# The package runs from this checkout in the ROS Python and in TORCH_PYTHON alike.
# ROS 파이썬과 TORCH_PYTHON 모두 이 체크아웃의 패키지를 쓴다.
export PYTHONPATH="$PKG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export ROBOT_IP CAM_DIR

# Package entry points (same table as [project.scripts]), run in the ROS-capable Python.
# 패키지 엔트리 포인트를 ROS 를 볼 수 있는 파이썬으로 실행.
entry_target() {
  case "$1" in
    factor-run) echo "factor.real.cli:main";;
    factor-view) echo "factor.real.viewer:main";;
    factor-selfcalib) echo "factor.calib.run:main";;
    factor-camera) echo "factor.real.camera_server:main";;
    factor-record) echo "factor.real.camera_server:record_main";;
    factor-fr3-driver) echo "factor.robots.fr3.driver:main";;
    factor-fr3-power) echo "factor.robots.fr3.tools:power_main";;
    factor-fr3-check) echo "factor.robots.fr3.tools:check_main";;
    factor-fr3-probe) echo "factor.robots.fr3.tools:probe_main";;
    *) return 1;;
  esac
}
LAUNCH='import importlib, sys
name, target = sys.argv[1:3]
del sys.argv[1:3]
sys.argv[0] = name
module, func = target.split(":")
getattr(importlib.import_module(module), func)()'
fx() {
  local target
  target="$(entry_target "$1")" || die "unknown entry point $1"
  "$PYTHON" -c "$LAUNCH" "$1" "$target" "${@:2}"
}
# Background run with a log file; $! is the process. / 로그 파일과 함께 백그라운드 실행.
bg() {
  local logfile="$1" target
  target="$(entry_target "$2")" || die "unknown entry point $2"
  nohup "$PYTHON" -c "$LAUNCH" "$2" "$target" "${@:3}" > "$logfile" 2>&1 &
}

[ -n "$CHECKPOINT" ] || die "no estimator checkpoint: set CHECKPOINT or pass --checkpoint"
mkdir -p "$CAM_DIR" "$LIVE_DIR" "$LOG_DIR/videos" "$LOG_DIR/selfcalib"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Live viewer (reused if one answers). / 라이브 뷰어(이미 있으면 재사용).
if ! curl -s -o /dev/null "http://localhost:$VIEWER_PORT/"; then
  bg "$LOG_DIR/viewer.log" factor-view --dir "$LIVE_DIR" --port "$VIEWER_PORT"
  sleep 1
fi
log "live viewer http://localhost:$VIEWER_PORT"

# Camera server (reused if one is publishing). / 카메라 서버(발행 중이면 재사용).
if ! fx factor-camera --alive --dir "$CAM_DIR"; then
  log "starting factor-camera (device $CAMERA_DEVICE)"
  CAM_ARGS=(--dir "$CAM_DIR" --device "$CAMERA_DEVICE" --hz 10)
  [ -n "$CAMERA_INTRINSICS" ] && CAM_ARGS+=(--intrinsics "$CAMERA_INTRINSICS")
  bg "$LOG_DIR/camera.log" factor-camera "${CAM_ARGS[@]}"
  for _ in $(seq 1 20); do
    sleep 1
    fx factor-camera --alive --dir "$CAM_DIR" && break
  done
  fx factor-camera --alive --dir "$CAM_DIR" || die "no camera frames (see $LOG_DIR/camera.log)"
fi

RUN=(--scene "$SCENE" --checkpoint "$CHECKPOINT" --stacks "$STACKS" --goal "$GOAL"
     --moves "$MOVES" --step-scale "$STEP" --cam-dir "$CAM_DIR" --live "$LIVE_DIR"
     --torch-python "$TORCH_PYTHON" --log "$LOG_DIR/real_$STAMP.json")

if [ $FAKE = 1 ]; then
  log "FAKE mode: no robot"
  fx factor-run "${RUN[@]}" --fake "$FAKE_START_DEG" "${EXTRA[@]}"
  exit $?
fi

# Controller: reset, AUTO, enable (XML-RPC; works while a driver holds the state port).
# 제어기: 리셋, AUTO, enable (XML-RPC 라 드라이버가 상태 포트를 잡고 있어도 동작).
if command -v ping >/dev/null; then
  ping -c1 -W1 "$ROBOT_IP" >/dev/null || die "controller $ROBOT_IP does not answer (power, Ethernet link)"
fi
fx factor-fr3-power reset --ip "$ROBOT_IP" >/dev/null || true
fx factor-fr3-power auto --ip "$ROBOT_IP" >/dev/null || die "switch to AUTO failed"
fx factor-fr3-power enable --ip "$ROBOT_IP" >/dev/null || die "servo enable failed"

# Driver (only the one this script started is ever restarted). / 드라이버(이 스크립트가 띄운 것만 재시작).
PID_FILE="$LOG_DIR/driver.pid"
if [ $RESTART = 1 ] && [ -f "$PID_FILE" ]; then
  log "stopping the driver started earlier (pid $(cat "$PID_FILE"))"
  kill -INT "$(cat "$PID_FILE")" 2>/dev/null || true
  sleep 3
  rm -f "$PID_FILE"
fi
if ! fx factor-fr3-probe --wait 3 --no-motion >/dev/null 2>&1; then
  fx factor-fr3-check --ip "$ROBOT_IP" || die "preflight NOT READY"
  log "starting factor-fr3-driver ($ROBOT_IP)"
  bg "$LOG_DIR/driver.log" factor-fr3-driver --ip "$ROBOT_IP" --ros-args \
    -p start_tol_deg:="$START_TOL_DEG" -p max_joint_speed_deg:="$MAX_JOINT_SPEED_DEG" \
    -p max_joint_accel_deg:="$MAX_JOINT_ACCEL_DEG" -p servo_timeout:="$SERVO_TIMEOUT"
  echo $! > "$PID_FILE"
  fx factor-fr3-probe --wait 20 --no-motion || die "driver did not come up (see $LOG_DIR/driver.log)"
fi

# Servo parameters (also for a reused driver), then a ±0.3 deg j6 probe.
# 서보 파라미터(재사용 드라이버 포함) 설정 후 j6 ±0.3° 탐색.
fx factor-fr3-probe \
  --set start_tol_deg="$START_TOL_DEG" --set max_joint_speed_deg="$MAX_JOINT_SPEED_DEG" \
  --set max_joint_accel_deg="$MAX_JOINT_ACCEL_DEG" --set servo_timeout="$SERVO_TIMEOUT" \
  || die "servo stream did not move the arm: factor-fr3-power reset, then retry"

if [ $SELFCAL = 1 ]; then
  CAL="$LOG_DIR/selfcalib/calib_$STAMP.json"
  log "self-calibration (budget ${CALBUDGET} s) -> $CAL"
  fx factor-selfcalib --scene "$SCENE" --board "$BOARD" --cam-dir "$CAM_DIR" --budget "$CALBUDGET" --out "$CAL" \
    || die "self-calibration failed"
  ln -sf "$CAL" "$LOG_DIR/selfcalib/latest.json"
  [ $CALONLY = 1 ] && exit 0
  RUN+=(--calib "$CAL")
fi

VIDEO="$LOG_DIR/videos/factor_$STAMP.mp4"
bg "$LOG_DIR/record.log" factor-record --out "$VIDEO" --dir "$CAM_DIR" --fps 10 --max-sec 3600
REC=$!
trap 'kill -INT $REC 2>/dev/null; wait $REC 2>/dev/null; log "recording -> $VIDEO"' EXIT

log "factor-run ${RUN[*]} ${EXTRA[*]}"
fx factor-run "${RUN[@]}" "${EXTRA[@]}"
