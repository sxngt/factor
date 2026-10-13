# FACTOR

**Fiducial-Anchored Canonical views for calibration-free Transfer Of precise Robot manipulation**

FACTOR is a hybrid learning architecture for millimetre-precision pick-and-place with a single
fixed RGB camera. It separates the problem into three layers:

| Layer | Responsibility | Learning |
|-------|----------------|----------|
| Perception | Fiducial-anchored canonical views, kinematic (URDF) render channels, model-based estimation of occluded object state | Supervised, labels from the simulation twin |
| Skills | Pick / insert micro-motions, search on contact, grasp and release timing | Rule-based initialisation, reinforcement-learning fine-tuning in a contact simulation |
| Task | Scene specification, symbolic planning (what goes where) | Task plugin, e.g. Tower of Hanoi |

The Tower of Hanoi task is the reference example; the layers below the task plugin are
object-agnostic and are configured from a scene specification (CAD / URDF, fiducial layout,
cameras). The reference hardware is a FAIRINO FR3 arm with a DH-Robotics AG-95 gripper and a
fixed USB webcam.

FACTOR 는 고정된 일반 RGB 카메라 한 대로 mm 급 픽앤플레이스를 수행하기 위한 하이브리드 학습
아키텍처입니다. 인식(지도학습)·스킬(규칙 초기화 + 강화학습 미세조정)·태스크(플러그인)의 세 층으로
나뉘며, 하노이 탑은 첫 번째 예시 태스크일 뿐입니다. 태스크 아래 층은 물체와 무관하고 장면 명세
(CAD/URDF, 피듀셜 배치, 카메라)로 구성됩니다. 기준 하드웨어는 FAIRINO FR3 + DH AG-95 + 고정 웹캠입니다.

## Install / 설치

FACTOR is a [uv](https://docs.astral.sh/uv/) project (Python 3.10–3.12).

```
git clone https://github.com/sxngt/factor && cd factor
uv sync --extra dev                  # core (numpy, pyyaml) + pytest, ruff
uv sync --extra dev --extra real     # + OpenCV, Pillow (camera, fiducials, real loop)
uv sync --extra perception           # + PyTorch (estimator training and serving)
uv run pytest -q
```

Three dependencies cannot come from PyPI:

* **Isaac Lab** (simulation twin, training): install it following its documentation and install
  FACTOR into its Python (`python -m pip install -e .`).
* **ROS 2** (robot driver and the real loop): `rclpy`, `sensor_msgs`, `std_srvs`,
  `control_msgs`, `rcl_interfaces` come from the ROS 2 install (Humble tested); source its
  setup file. The `ros` extra only documents this.
* **FAIRINO SDK** (`fairino` package, FAIRINO's `fairino-python-sdk`) at the tag that matches
  the controller firmware (tested: SDK v2.1.6, firmware v3.8.6). Only the real driver and the
  `factor-fr3-*` tools import it; `factor-fr3-driver --fake` runs without it.

## Quick start in simulation / 시뮬레이션 빠른 시작

The twin needs the robot, ring and mat USD files; [docs/ASSETS.md](docs/ASSETS.md) explains
what they are and how to build them from the vendor descriptions. With an assets root `<root>`
(the directory containing `assets/usd/...`), inside the Isaac Lab Python:

```
# teacher smoke test: success close to 100 %
python -m factor.sim.isaaclab.check --scene configs/hanoi.yaml --assets <root> --num_envs 32 --headless

# train the state estimator on twin frames (teacher collection + DAgger rounds)
python -m factor.perception.train --scene configs/hanoi.yaml --assets <root> \
    --num_envs 96 --steps 1000 --dagger_rounds 2 --out estimator.pt --headless

# evaluate it in the twin
python -m factor.sim.isaaclab.check --scene configs/hanoi.yaml --assets <root> \
    --checkpoint estimator.pt --num_envs 32 --headless
```

Training recipes and results of every run: [docs/RUNS.md](docs/RUNS.md).

